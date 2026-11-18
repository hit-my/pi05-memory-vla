"""RGB-only bowl tracking: Grounding DINO + SAM2.1 on the first frame, then causal CUTIE.

Follows labeling/mask_centroid/run_a.py (same models, tiling, thresholds and CUTIE settings). Additions
needed for closed loop: frame-by-frame stepping, automatic bowl selection and optional fixes for a bowl
mask that swallows its plate. Never sees simulator segmentation or object IDs.

Bowl selection uses only prior knowledge of the task and the fixed camera: how many bowls the task has,
and that anything above WORK_TOP is not on the table. Bowls are returned left to right (index 1..K).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
TILES = [(0, 0, SIZE, SIZE)] + [(x, y, x + 128, y + 128) for y in (0, 64, 128) for x in (0, 64, 128)]


@dataclass
class Options:
    bowl_query: str = "bowl."
    selection: str = "detector" # detector: run_a.py rule. appearance: many low-score boxes, pick by mask shape and colour
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    box_nms: float = 0.50
    mask_nms: float = 0.85
    work_top: int = 60          # image rows above this are cabinets, not the table
    bowl_mask: str = "single"   # single: as run_a.py. tight: smallest good SAM proposal, so plate stays out
    surround_px: int = 0        # >0: ring around each bowl becomes a distractor object for CUTIE
    keep_plates: bool = True
    mem_every: int = 5
    max_mem_frames: int = 5
    long_term: bool = False
    guard: bool = False         # identity guard: conflict checks, redetection and re-seeding, RGB only
    check_every: int = 10       # frames between redetections while a bowl id is lost or colliding
    lost_after: int = 5         # frames without a mask before a bowl id counts as lost
    conflict_px: float = 14.0   # two bowl ids this close are on one bowl


def box_iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def mask_features(rgb, mask):
    area = int(mask.sum())
    if area < 30:
        return None
    ys, xs = np.nonzero(mask)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[mask]
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask][:, 1]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    box_area = int(xs.max() - xs.min() + 1) * int(ys.max() - ys.min() + 1)
    return {"area": area, "fill": area / box_area, "sat": float(saturation.mean()),
            "solidity": cv2.contourArea(largest) / max(cv2.contourArea(cv2.convexHull(largest)), 1),
            "circ": 4 * np.pi * cv2.contourArea(largest) / max(perimeter ** 2, 1),
            "gray_mean": float(gray.mean()), "gray_std": float(gray.std()), "white": float((gray > 190).mean()),
            "dark": float((gray < 80).mean()), "cx": float(xs.mean()), "cy": float(ys.mean())}


def looks_like_bowl(f):
    """Fitted on 22 first frames of training-state replays (tools/tracker_init_features.py). Specific to this scene."""
    return (400 <= f["area"] <= 1500 and f["sat"] <= 20 and f["solidity"] >= 0.85 and f["fill"] >= 0.55
            and 120 <= f["gray_mean"] <= 170 and f["gray_std"] >= 22 and f["white"] <= 0.25 and f["dark"] <= 0.25)


class TrackerCore:
    def __init__(self, root, options: Options | None = None):
        self.root = Path(root).resolve()
        self.options = options or Options()
        for folder in reversed([self.root / "deps", self.root / "sources/sam2", self.root / "sources/Cutie"]):
            sys.path.insert(0, str(folder))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra import compose, initialize_config_dir
        from cutie.model.cutie import CUTIE
        from cutie.model.utils import resnet
        from cutie.inference.inference_core import InferenceCore

        self.torch, self.Image, self.InferenceCore = torch, Image, InferenceCore
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp = self.device == "cuda"  # FP16 autocast as in run_a.py on GPU, plain FP32 on CPU
        torch.set_num_threads(int(os.environ.get("TRACKER_THREADS", "2")))
        torch.manual_seed(0)
        np.random.seed(0)
        cv2.setNumThreads(1)
        detector_path = self.root / "models/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(str(detector_path), local_files_only=True, use_fast=False)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(detector_path), local_files_only=True, disable_custom_kernels=True).to(self.device).eval()
        self.segmenter = SAM2ImagePredictor(build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml", str(self.root / "models/sam2.1_hiera_tiny.pt"),
            device=self.device, apply_postprocessing=False))
        # SAM2 claims Hydra's global state on import; run_a.py avoided this with separate processes.
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3.2", config_dir=str(self.root / "sources/Cutie/cutie/config")):
            cfg = compose(config_name="eval_config")
        cfg.amp = self.amp
        cfg.max_internal_size = -1
        cfg.use_long_term = self.options.long_term
        cfg.mem_every = self.options.mem_every
        cfg.max_mem_frames = self.options.max_mem_frames
        cfg.weights = str(self.root / "models/cutie-base-mega.pth")
        self.cfg = cfg
        factories = {name: getattr(resnet, name) for name in ("resnet18", "resnet50")}

        def offline(factory):
            def construct(*values, **kwargs):
                kwargs["pretrained"] = False
                return factory(*values, **kwargs)
            return construct
        try:
            for name, factory in factories.items():
                setattr(resnet, name, offline(factory))
            network = CUTIE(cfg)
        finally:
            for name, factory in factories.items():
                setattr(resnet, name, factory)
        network.load_state_dict(torch.load(cfg.weights, map_location="cpu", weights_only=True), strict=True)
        self.network = network.to(self.device).eval()
        self.core = None
        self.n_bowls = 0

    def _detect(self, rgb, queries=None, threshold=None):
        torch, options = self.torch, self.options
        candidates = []
        box_threshold = options.box_threshold if threshold is None else threshold
        text_threshold = options.text_threshold if threshold is None else threshold
        with torch.inference_mode():
            for query in (queries or (options.bowl_query, "plate.")):
                for x0, y0, x1, y1 in TILES:
                    inputs = self.processor(images=self.Image.fromarray(rgb[y0:y1, x0:x1]), text=query, return_tensors="pt").to(self.device)
                    raw = self.detector(**inputs)
                    found = self.processor.post_process_grounded_object_detection(
                        raw, inputs.input_ids, threshold=box_threshold,
                        text_threshold=text_threshold, target_sizes=[(y1 - y0, x1 - x0)])[0]
                    for box, score, text in zip(found["boxes"].cpu().numpy(), found["scores"].cpu().numpy(), found["text_labels"]):
                        absolute = [float(box[0] + x0), float(box[1] + y0), float(box[2] + x0), float(box[3] + y0)]
                        cut = (x0 > 0 and absolute[0] <= x0 + 1) or (y0 > 0 and absolute[1] <= y0 + 1) \
                            or (x1 < SIZE and absolute[2] >= x1 - 1) or (y1 < SIZE and absolute[3] >= y1 - 1)
                        label = text.strip().lower().rstrip(".")
                        label = "bowl" if "bowl" in label else "plate" if "plate" in label else label
                        if cut or label not in ("bowl", "plate"):
                            continue
                        clipped = [max(0.0, min(float(SIZE), v)) for v in absolute]
                        candidates.append({"class": label, "score": float(score), "box": clipped})
        kept = []
        for candidate in sorted(candidates, key=lambda c: -c["score"]):
            if any(candidate["class"] == k["class"] and box_iou(candidate["box"], k["box"]) > options.box_nms for k in kept):
                continue
            kept.append(candidate)
        return kept

    def _segment(self, rgb, kept):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            self.segmenter.set_image(rgb)
            for candidate in kept:
                box = np.asarray(candidate["box"], dtype=np.float32)
                if candidate["class"] == "bowl" and self.options.bowl_mask == "tight":
                    masks, quality, _ = self.segmenter.predict(box=box, multimask_output=True)
                    masks = np.asarray(masks).reshape(-1, SIZE, SIZE).astype(bool)
                    quality = np.asarray(quality).reshape(-1)
                    areas = masks.sum(axis=(1, 2))
                    usable = [i for i in range(len(masks)) if areas[i] >= 150 and quality[i] >= 0.5 * quality.max()]
                    index = min(usable, key=lambda i: areas[i]) if usable else int(quality.argmax())
                else:
                    masks, quality, _ = self.segmenter.predict(box=box, multimask_output=False)
                    masks = np.asarray(masks).reshape(-1, SIZE, SIZE).astype(bool)
                    quality = np.asarray(quality).reshape(-1)
                    index = 0
                candidate["mask"] = masks[index]
                candidate["quality"] = float(quality[index])
                candidate["priority"] = candidate["score"] * max(0.0, candidate["quality"])

    def _appearance_bowls(self, rgb, n_bowls):
        """Boxes are only proposals. A bowl here is a solid, grey, unsaturated blob of bowl size on the table."""
        torch = self.torch
        boxes = []
        found = self._detect(rgb, ("bowl.",), 0.08) + self._detect(rgb, ("black bowl.",), 0.08)  # NMS within a prompt only
        for candidate in sorted(found, key=lambda c: -c["score"]):
            if candidate["class"] != "bowl" or (candidate["box"][1] + candidate["box"][3]) / 2 < self.options.work_top:
                continue
            if not any(box_iou(candidate["box"], b["box"]) > 0.8 for b in boxes):
                boxes.append(candidate)
        proposals = []
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            self.segmenter.set_image(rgb)
            for candidate in boxes:
                masks, quality, _ = self.segmenter.predict(box=np.asarray(candidate["box"], np.float32), multimask_output=True)
                masks = np.asarray(masks).reshape(-1, SIZE, SIZE).astype(bool)
                for mask, q in zip(masks, np.asarray(quality).reshape(-1)):
                    shape = mask_features(rgb, mask)
                    if shape is not None and looks_like_bowl(shape):
                        proposals.append({"class": "bowl", "score": candidate["score"], "box": candidate["box"], "mask": mask,
                                          "quality": float(q), "priority": shape["circ"] - shape["sat"] / 50,
                                          "cx": shape["cx"], "cy": shape["cy"]})
        chosen = []
        for proposal in sorted(proposals, key=lambda p: -p["priority"]):
            if all((proposal["cx"] - c["cx"]) ** 2 + (proposal["cy"] - c["cy"]) ** 2 > 15 ** 2 for c in chosen):
                chosen.append(proposal)
            if len(chosen) == n_bowls:
                break
        return chosen, len(boxes), len(proposals)

    def initialize(self, rgb, n_bowls):
        """Start a new episode from its first frame. Returns a report; bowls may be fewer than asked."""
        started = time.perf_counter()
        torch, options = self.torch, self.options
        extra = {}
        if options.selection == "appearance":
            picked, n_boxes, n_proposals = self._appearance_bowls(rgb, n_bowls)
            extra = {"candidate_boxes": n_boxes, "bowl_like_proposals": n_proposals}
            kept = [c for c in self._detect(rgb, ("plate.",)) if c["class"] == "plate"]
            self._segment(rgb, kept)
            kept = picked + kept
        else:
            kept = self._detect(rgb)
            self._segment(rgb, kept)
        retained = []
        for candidate in sorted(kept, key=lambda c: -c["priority"]):
            if not candidate["mask"].any():
                continue
            duplicate = False
            for other in retained:
                if other["class"] == candidate["class"]:
                    union = np.logical_or(candidate["mask"], other["mask"]).sum()
                    if np.logical_and(candidate["mask"], other["mask"]).sum() / max(union, 1) > options.mask_nms:
                        duplicate = True
                        break
            if not duplicate:
                retained.append(candidate)
        table = [c for c in retained if c["class"] == "bowl" and (c["box"][1] + c["box"][3]) / 2 >= options.work_top]
        if options.selection == "appearance":
            table = [c for c in kept if c["class"] == "bowl"]
            retained = table + [c for c in retained if c["class"] == "plate"]
        bowls = sorted(table[:n_bowls], key=lambda c: (c["box"][0] + c["box"][2]) / 2)
        labels = np.zeros((SIZE, SIZE), np.int64)
        occupied = np.zeros((SIZE, SIZE), bool)
        for index, bowl in enumerate(bowls, 1):
            visible = bowl["mask"] & ~occupied
            labels[visible] = index
            occupied |= visible
        next_id = len(bowls) + 1
        distractors = []
        if options.keep_plates:
            distractors += [c["mask"] for c in retained if c["class"] == "plate"]
        if options.surround_px > 0:
            size = 2 * options.surround_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            for bowl in bowls:
                distractors.append(cv2.dilate(bowl["mask"].astype(np.uint8), kernel).astype(bool))
        for mask in distractors:
            visible = mask & ~occupied
            if visible.sum() < 20:
                continue
            labels[visible] = next_id
            occupied |= visible
            next_id += 1
        self.n_bowls = len(bowls)
        report = {**extra, "bowls_found": len(bowls), "bowls_asked": n_bowls, "objects": next_id - 1,
                  "bowls": [{"box": [round(v, 1) for v in b["box"]], "score": round(b["score"], 3),
                             "quality": round(b["quality"], 3), "pixels": int((labels == i).sum())}
                            for i, b in enumerate(bowls, 1)],
                  "rejected_bowls": [[round(v, 1) for v in c["box"]] for c in retained
                                     if c["class"] == "bowl" and c not in bowls],
                  "plates": sum(c["class"] == "plate" for c in retained)}
        self.core = None
        self.objects = list(range(1, next_id))
        self.frame = 0
        self.history = {i: [] for i in range(1, self.n_bowls + 1)}
        self.lost_frames = {i: 0 for i in range(1, self.n_bowls + 1)}
        self.last_check = -99
        self.suppressed = set()
        self.guard_events = {"checks": 0, "conflicts": 0, "reassigned": 0, "suppressed_frames": 0}
        if next_id > 1:
            self.core = self.InferenceCore(self.network, cfg=self.cfg)
            result = self._finish(self._raw(rgb, labels, self.objects))
        else:
            result = np.zeros((SIZE, SIZE), np.uint8)
        report["seconds"] = round(time.perf_counter() - started, 3)
        return report, result

    def _raw(self, rgb, seed=None, objects=None):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(device=self.device, dtype=torch.float32).div_(255)
            mask = torch.from_numpy(seed).to(self.device) if seed is not None else None
            probabilities = self.core.step(image, mask, objects)
            output = self.core.output_prob_to_mask(probabilities).to(device="cpu", dtype=torch.int32).numpy()
        return output

    def _finish(self, output):
        result = output.copy()
        result[result > self.n_bowls] = 0  # distractors only help CUTIE; they are not reported
        for bowl in self.suppressed:
            result[result == bowl] = 0
        self.guard_events["suppressed_frames"] += len(self.suppressed)
        return result.astype(np.uint8)

    def _centres(self, output):
        centres = {}
        for bowl in range(1, self.n_bowls + 1):
            ys, xs = np.nonzero(output == bowl)
            centres[bowl] = np.array([xs.mean(), ys.mean()]) if len(xs) >= 80 else None
        return centres

    def _past(self, bowl):
        """Where this bowl was before any recent trouble."""
        past = self.history[bowl]
        return past[-min(len(past), 15)] if past else None

    def _check(self, rgb, output, centres, conflict, lost):
        """Only a bowl id that vanished or collided may be re-seeded; any other id is left alone."""
        self.guard_events["checks"] += 1
        found, _, _ = self._appearance_bowls(rgb, 8)
        claims = {}
        for bowl in range(1, self.n_bowls + 1):
            mask = output == bowl
            best, best_iou = None, 0.25
            for index, det in enumerate(found):
                union = (mask | det["mask"]).sum()
                iou = (mask & det["mask"]).sum() / max(union, 1)
                if iou >= best_iou:
                    best, best_iou = index, iou
            claims.setdefault(best, []).append(bowl)
        settled, loose = {}, []
        for bowl in claims.pop(None, []):
            if bowl in lost or bowl in conflict:
                loose.append(bowl)
        for index, bowls in claims.items():
            det = np.array([found[index]["cx"], found[index]["cy"]])

            def distance(b):
                past = self._past(b)
                return float(np.linalg.norm(past - det)) if past is not None else 1e9
            keep = min(bowls, key=distance)
            settled[index] = keep
            loose += [b for b in bowls if b != keep and (b in lost or b in conflict)]
        # A bowl id with no detection is only loose if a free bowl exists; a hidden bowl under the gripper stays put.
        free = [i for i in range(len(found)) if i not in settled]
        moves = {}
        for bowl in sorted(loose, key=lambda b: 0 if b in conflict else 1):
            if not free:
                break
            past = self._past(bowl)
            target = min(free, key=lambda i: float(np.linalg.norm(past - np.array([found[i]["cx"], found[i]["cy"]]))) if past is not None else 0.0)
            moves[bowl] = target
            free.remove(target)
        self.suppressed = {b for b in loose if b not in moves and b in conflict}
        if not moves:
            return output
        seed = output.copy()
        for bowl in moves:
            seed[seed == bowl] = 0
        for bowl, index in moves.items():
            seed[found[index]["mask"]] = bowl
            self.history[bowl] = [np.array([found[index]["cx"], found[index]["cy"]])]
        self.guard_events["reassigned"] += len(moves)
        return self._raw(rgb, seed, self.objects)

    def step(self, rgb):
        if self.core is None:
            return np.zeros((SIZE, SIZE), np.uint8)
        output = self._raw(rgb)
        self.frame += 1
        if not self.options.guard:
            return self._finish(output)
        centres = self._centres(output)
        conflict = set()
        for a in centres:
            for b in centres:
                if a < b and centres[a] is not None and centres[b] is not None and np.linalg.norm(centres[a] - centres[b]) < self.options.conflict_px:
                    conflict |= {a, b}
        if conflict:
            self.guard_events["conflicts"] += 1
        for bowl, centre in centres.items():
            self.lost_frames[bowl] = self.lost_frames[bowl] + 1 if centre is None else 0
        lost = {b for b, n in self.lost_frames.items() if n >= self.options.lost_after}
        # Redetection is expensive and these conditions last many frames, so throttle it.
        if (conflict or lost) and self.frame - self.last_check >= self.options.check_every:
            self.last_check = self.frame
            output = self._check(rgb, output, centres, conflict, lost)
            centres = self._centres(output)
        for bowl, centre in centres.items():
            if centre is not None and bowl not in conflict and bowl not in self.suppressed:
                self.history[bowl].append(centre)
                del self.history[bowl][:-60]
        return self._finish(output)
