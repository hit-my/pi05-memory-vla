"""Online tracker server for RoboMemArena, v2: SAM3 "method B" first-frame init + SAM3 frame-by-frame tracking.

Reproduces, one frame at a time, what the offline pipeline xxxx/rma/trackv/tv_full_b.py does up to the masks:
  frame 0 : SAM3 image detector with generic prompts (initB/initB_det.py) -> candidates 20..700 px, de-duplicated,
            assigned to the task's fixed clicks with linear_sum_assignment, R = 30 px (initB/initB_eval.py, method_b)
            -> objects without a detection fall back to the legacy click mask (rma_tracker.NamedTracker.initialize_named,
            as initB/initB_cur.py) if its area is 20..700 -> Sam3Stream.reset(rgb, first masks)
  frame t : Sam3Stream.step(rgb)                                   (keep=None, as the offline run)
The gripper veto is NOT done here: it needs the proprioceptive state and runs in rma_perception_v2 / rma_online_veto.

Code reuse: Sam3Stream, NamedTracker, rma_spec are imported. initB_det.py and initB_eval.py do their work at import
time (they read sys.argv, build models and loop over all episodes), so they cannot be imported; the functions
`cen`, `dedupe`, `method_b` and the bodies marked VERBATIM below are copied from them unchanged.

Protocol on stdin/stdout (binary, same style as rma_tracker.py). K = number of tracked objects of the task.
  RESET {"task": <id>}\n + 256*256*3 RGB  -> one JSON line (report), then K*256*32 bytes = np.packbits of the K
                                             first-frame (method B) masks, then K*256*32 bytes = the K masks SAM3
                                             returns for frame 0 (these are what the veto sees, as offline)
  STEP\n + 256*256*3 RGB                   -> K*256*32 bytes = np.packbits(masks [K,256,256], axis=-1)
  STAT\n                                   -> one JSON line (frames, GPU memory of this process as seen by torch)
Masks, not a label image: the K SAM3 tracks are independent and may overlap, and the veto needs each mask as it is.
Everything a library prints goes to stderr (file descriptor 1 is redirected), so the protocol stream stays clean.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import rma_spec  # noqa: E402  (code_causal_v2/rma_spec.py; identical to xxxx/rma/trackv/code/rma_spec.py)

TV = Path("xxxx/rma/trackv")
sys.path.insert(1, str(TV))
from rma_sam3_stream import Sam3Stream  # noqa: E402  (puts the SAM3 source tree and its deps on sys.path)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

SIZE = 256
PROMPTS = ["box", "bottle", "carton", "small object", "food package"]  # initB_det.py
LO, HI = 20, 700  # initB_eval.py
R_BEST = 30  # the radius chosen for the final run (initB_eval_all.py 30)
YS, XS = np.mgrid[0:256, 0:256]


# ---------------------------------------------------------------- VERBATIM from initB/initB_eval.py
def cen(m):
    return np.array([XS[m].mean(), YS[m].mean()])


def dedupe(masks, scores):
    keep = []
    for i in np.argsort(-scores):
        if all((masks[i] & masks[j]).sum() / max(1, (masks[i] | masks[j]).sum()) <= 0.5 for j in keep):
            keep.append(i)
    return keep


def method_b(clicks, cand, cc, cur, R):
    """-> labels uint8, source per object ('det' | 'fallback' | 'none'), chosen candidate index per object"""
    K = len(clicks)
    labels = np.zeros((256, 256), np.uint8)
    src, pick = ["none"] * K, [None] * K
    if len(cand):
        d = np.hypot(cc[None, :, 0] - np.asarray(clicks)[:, None, 0], cc[None, :, 1] - np.asarray(clicks)[:, None, 1])
        cost = np.where(d <= R, d, 1e4)  # maximise the number of objects matched within R, then minimise total distance
        for k, j in zip(*linear_sum_assignment(cost)):
            if d[k, j] <= R:
                pick[k] = int(j)
    for k in range(K):  # detections first, in object order; an earlier object keeps contested pixels
        if pick[k] is not None:
            m = cand[pick[k]] & (labels == 0)
            if m.sum() >= LO:
                labels[m] = k + 1
                src[k] = "det"
    for k in range(K):
        if src[k] == "none":
            m = cur == k + 1
            if LO <= m.sum() <= HI:
                m = m & (labels == 0)
                if m.sum() >= LO:
                    labels[m] = k + 1
                    src[k] = "fallback"
    return labels, src, pick
# ---------------------------------------------------------------- end VERBATIM


class TrackerV2:
    def __init__(self, root, legacy="lazy"):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.root = root
        model = build_sam3_image_model(checkpoint_path=str(TV / "src/sam3.pt"), load_from_HF=False)  # initB_det.py
        self.proc = Sam3Processor(model, confidence_threshold=0.15)
        self.stream = Sam3Stream()  # keep=None: verified pixel-identical to the whole-video run
        self.named = None
        self.frames = 0
        if legacy == "eager":
            self._named()

    def _named(self):
        if self.named is None:  # Grounding DINO + SAM2.1 + CUTIE, only needed when a click has no detection
            from rma_tracker import NamedTracker
            from tracker_core import Options
            self.named = NamedTracker(self.root, Options(selection="detector", max_mem_frames=10))  # initB_cur.py
        return self.named

    def detect(self, rgb):
        """Body of the episode loop of initB/initB_det.py (VERBATIM apart from the return)."""
        proc = self.proc
        masks, scores, pid, boxes = [], [], [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(Image.fromarray(rgb).resize((1008, 1008), Image.BICUBIC))
            for p, prompt in enumerate(PROMPTS):
                o = proc.set_text_prompt(state=state, prompt=prompt)
                if len(o["scores"]) == 0:
                    continue
                m = F.interpolate(o["masks_logits"].float(), (256, 256), mode="area")[:, 0] > 0.5
                masks.append(m.cpu().numpy())
                scores.append(o["scores"].float().cpu().numpy())
                boxes.append(o["boxes"].float().cpu().numpy() / 1008 * 256)
                pid.append(np.full(len(m), p))
        masks = np.concatenate(masks) if masks else np.zeros((0, 256, 256), bool)
        area = masks.sum(axis=(1, 2))
        keep = (area >= 8) & (area <= 2000)  # drop table/background-sized masks early; the 20..700 rule is applied in initB_eval.py
        return masks[keep], (np.concatenate(scores) if scores else np.zeros(0, np.float32))[keep], len(area)

    def legacy(self, rgb, task):
        """initB/initB_cur.py: fixed click -> SAM2.1 -> CUTIE frame 0 (tracker.initialize_named)."""
        _, labels = self._named().initialize_named(rgb.copy(), rma_spec.detector_objects(task))
        return np.asarray(labels).astype(np.uint8)

    def first_frame(self, rgb, task):
        """Method B labels for frame 0 (the per-episode part of initB/initB_eval.py with R = 30)."""
        clicks = rma_spec.CLICKS[task]
        start = time.time()
        masks, scores, n_raw = self.detect(rgb)
        t_det = time.time() - start
        # VERBATIM from initB_eval.py (z["masks"] / z["scores"] are the arrays returned by detect)
        area = masks.sum(axis=(1, 2))
        ok = np.nonzero((area >= LO) & (area <= HI))[0]
        keep = [ok[i] for i in dedupe(masks[ok], scores[ok])] if len(ok) else []
        cand = masks[keep]
        cc = np.array([cen(m) for m in cand]).reshape(-1, 2)
        # the legacy mask `cur` only enters method_b for objects left without a detection, so it is computed only then
        cur = np.zeros((256, 256), np.uint8)
        labels, src, pick = method_b(clicks, cand, cc, cur, R_BEST)
        t_leg = 0.0
        if "none" in src:
            start = time.time()
            cur = self.legacy(rgb, task)
            t_leg = time.time() - start
            labels, src, pick = method_b(clicks, cand, cc, cur, R_BEST)
        report = {"task": task, "k": len(clicks), "src": src, "pick": pick, "n_raw": int(n_raw), "n_cand": int(len(cand)),
                  "legacy_used": bool(t_leg), "pixels": [int((labels == j + 1).sum()) for j in range(len(clicks))],
                  "detect_s": round(t_det, 3), "legacy_s": round(t_leg, 3)}
        return labels, report

    def reset(self, rgb, task):
        k = len(rma_spec.TASKS[task]["objects"])
        labels, report = self.first_frame(rgb, task)
        first = np.stack([labels == j + 1 for j in range(k)])  # tv_full_b.py
        start = time.time()
        ms = self.stream.reset(rgb, [first[j] for j in range(k)])
        report["stream_reset_s"] = round(time.time() - start, 3)
        self.frames = 1
        return report, first, np.stack(ms).astype(bool)

    def step(self, rgb):
        self.frames += 1
        return np.stack(self.stream.step(rgb)).astype(bool)


def main():
    # keep the protocol stream private: anything printed by a library goes to stderr
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "xxxx/pi05-abc/tracker/assets"
    legacy = sys.argv[sys.argv.index("--legacy") + 1] if "--legacy" in sys.argv else "lazy"
    torch.set_num_threads(int(os.environ.get("TRACKER_THREADS", "2")))
    tracker = TrackerV2(root, legacy)
    out.write(b"READY\n")
    out.flush()
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        header = header.decode().strip()
        if header == "STAT":
            out.write((json.dumps({"frames": tracker.frames, "legacy_loaded": tracker.named is not None,
                                   "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20),
                                   "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20),
                                   "gpu_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20)}) + "\n").encode())
            out.flush()
            continue
        data = stdin.read(SIZE * SIZE * 3)
        if len(data) != SIZE * SIZE * 3:
            return
        rgb = np.frombuffer(data, np.uint8).reshape(SIZE, SIZE, 3).copy()
        if header.startswith("RESET"):
            report, first, masks = tracker.reset(rgb, int(json.loads(header[len("RESET "):])["task"]))
            out.write((json.dumps(report) + "\n").encode())
            out.write(np.packbits(first, axis=-1).tobytes())
        elif header == "STEP":
            masks = tracker.step(rgb)
        else:
            raise ValueError(f"unknown command {header!r}")
        out.write(np.packbits(masks, axis=-1).tobytes())
        out.flush()


if __name__ == "__main__":
    main()
