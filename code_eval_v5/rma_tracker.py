"""RGB-only tracking of named objects for RoboMemArena (Grounding DINO + SAM2.1 on frame 0, then CUTIE).

The same models and CUTIE settings as tracker_core.py (max_mem_frames 10). Detection: one text prompt per
object; the best box per prompt over the full image and 128-px tiles, never reusing a box another object
took. Never sees simulator segmentation.

Server protocol on stdin/stdout (like tracker_server.py):
  RESET <json list of prompts>\n + 256*256*3 RGB  -> one JSON line report, then 256*256 uint8 labels
  STEP\n + RGB                                     -> 256*256 uint8 labels (0 background, k = prompt k)
"""
from __future__ import annotations

import json
import sys

import numpy as np

import tracker_core
from tracker_core import SIZE, TILES, Options, TrackerCore, box_iou


class NamedTracker(TrackerCore):
    def _best_boxes(self, rgb, objects, threshold=0.10):
        """objects: list of {"prompts": [...], "max_area": px}. Best-scoring small box over all its prompts."""
        torch = self.torch
        chosen = []
        with torch.inference_mode():
            for obj in objects:
                found = []
                for prompt, (x0, y0, x1, y1) in ((p, tile) for p in obj["prompts"] for tile in TILES):
                    inputs = self.processor(images=self.Image.fromarray(rgb[y0:y1, x0:x1]), text=prompt,
                                            return_tensors="pt").to(self.device)
                    raw = self.detector(**inputs)
                    result = self.processor.post_process_grounded_object_detection(
                        raw, inputs.input_ids, threshold=threshold, text_threshold=threshold,
                        target_sizes=[(y1 - y0, x1 - x0)])[0]
                    for box, score in zip(result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()):
                        b = [float(box[0] + x0), float(box[1] + y0), float(box[2] + x0), float(box[3] + y0)]
                        cut = (x0 > 0 and b[0] <= x0 + 1) or (y0 > 0 and b[1] <= y0 + 1) \
                            or (x1 < SIZE and b[2] >= x1 - 1) or (y1 < SIZE and b[3] >= y1 - 1)
                        area = (b[2] - b[0]) * (b[3] - b[1])
                        if cut or not 20 <= area <= obj["max_area"] or (b[1] + b[3]) / 2 < 50:
                            continue  # these objects are small; the back wall is never one
                        found.append((float(score), b))
                found.sort(key=lambda item: -item[0])
                pick = next((item for item in found
                             if all(c["box"] is None or box_iou(item[1], c["box"]) < 0.3 for c in chosen)), None)
                chosen.append({"prompt": obj["prompts"][0], "score": pick[0] if pick else 0.0, "box": pick[1] if pick else None})
        return chosen

    def initialize_named(self, rgb, prompts):
        """prompts: list of {"point": [x, y]} (a click on the first frame) or {"prompts": [...], "max_area": n}."""
        if all("point" in p for p in prompts):
            chosen = [{"prompt": "click", "score": 1.0, "box": None, "point": p["point"]} for p in prompts]
        else:
            chosen = self._best_boxes(rgb, prompts)
        labels = np.zeros((SIZE, SIZE), np.int64)
        occupied = np.zeros((SIZE, SIZE), bool)
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            self.segmenter.set_image(rgb)
            for index, c in enumerate(chosen, 1):
                if c.get("point") is not None:
                    masks, quality, _ = self.segmenter.predict(point_coords=np.asarray([c["point"]], np.float32),
                                                               point_labels=np.ones(1, np.int32), multimask_output=True)
                    masks = np.asarray(masks).reshape(-1, SIZE, SIZE).astype(bool)
                    quality = np.asarray(quality).reshape(-1)
                    areas = masks.sum(axis=(1, 2))
                    # The object, not its support: the best mask that is small and contains the click.
                    x, y = int(round(c["point"][0])), int(round(c["point"][1]))
                    usable = [i for i in range(len(masks)) if 8 <= areas[i] <= 700 and masks[i][y, x]]
                    k = max(usable, key=lambda i: quality[i]) if usable else int(np.argmin(areas))
                    visible = masks[k] & ~occupied
                    labels[visible] = index
                    occupied |= visible
                    ys, xs = np.nonzero(visible)
                    c["box"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None
                    c["pixels"] = int(visible.sum())
                    continue
                if c["box"] is None:
                    c["pixels"] = 0
                    continue
                masks, quality, _ = self.segmenter.predict(box=np.asarray(c["box"], np.float32), multimask_output=True)
                masks = np.asarray(masks).reshape(-1, SIZE, SIZE).astype(bool)
                quality = np.asarray(quality).reshape(-1)
                # The smallest good proposal: a box around a small object often makes SAM take its support too.
                areas = masks.sum(axis=(1, 2))
                bw = (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1])
                usable = [i for i in range(len(masks)) if 8 <= areas[i] <= 1.3 * bw and quality[i] >= 0.5 * quality.max()]
                k = min(usable, key=lambda i: -quality[i]) if usable else int(quality.argmax())
                visible = masks[k] & ~occupied
                labels[visible] = index
                occupied |= visible
                c["pixels"] = int(visible.sum())
        self.n_bowls = len(prompts)
        self.objects = list(range(1, len(prompts) + 1))
        self.frame, self.last_check, self.suppressed = 0, -99, set()
        self.history = {i: [] for i in self.objects}
        self.lost_frames = {i: 0 for i in self.objects}
        self.guard_events = {"checks": 0, "conflicts": 0, "reassigned": 0, "suppressed_frames": 0}
        self.core = self.InferenceCore(self.network, cfg=self.cfg)
        result = self._finish(self._raw(rgb, labels, self.objects))
        report = {"objects": [{"prompt": c["prompt"], "score": round(c["score"], 3),
                               "box": [round(v, 1) for v in c["box"]] if c["box"] else None, "pixels": c["pixels"]}
                              for c in chosen]}
        return report, result


def main():
    root = sys.argv[sys.argv.index("--root") + 1]
    tracker = NamedTracker(root, Options(selection="detector", max_mem_frames=10))
    out = sys.stdout.buffer
    out.write(b"READY\n")
    out.flush()
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        header = header.decode().strip()
        data = stdin.read(SIZE * SIZE * 3)
        rgb = np.frombuffer(data, np.uint8).reshape(SIZE, SIZE, 3).copy()
        if header.startswith("RESET"):
            report, labels = tracker.initialize_named(rgb, json.loads(header[len("RESET "):]))
            out.write((json.dumps(report) + "\n").encode())
        else:
            labels = tracker.step(rgb)
        out.write(np.ascontiguousarray(labels, np.uint8).tobytes())
        out.flush()


if __name__ == "__main__":
    main()
