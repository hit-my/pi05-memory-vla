"""Bowl tracker shared by dataset labelling (real_track.py) and deployment (real_deploy.py): one code path, causal.

Library:  BowlTracker().reset(rgb256) -> (centres [K,2], report);  .step(rgb256) -> centres [K,2]   (nan = not seen)
Server:   real_tracker.py --serve      binary protocol on stdin/stdout, so the SAM3 environment can live in its own venv:
            -> b"RESET\n" + 256*256*3 bytes      <- one JSON line {"centres": [[x,y]|null,...], "area": [...], "report": {...}}
            -> b"STEP\n"  + 256*256*3 bytes      <- one JSON line {"centres": ..., "area": ...}
          The first line the server prints is READY.
Needs: SAM3 source tree + sam3.pt (TRACKER_SAM3_ROOT, default /data/lgd/rma/trackv: src/sam3-<rev>/, src/sam3.pt,
rma_sam3_stream.py), torch with CUDA.

Why this is more than "SAM3 video tracking": the two bowls look the same. A bowl disappears under the gripper while
it is carried, and plain SAM3 then either never finds it again or re-attaches its track to the OTHER bowl when that
one is put where the lost bowl used to stand. Two rules repair this, using only the current frame and the past:
  * collision: when both tracks cover the same pixels, the track that was lost more recently is dropped;
  * merge: when a bowl is put down next to the other one, its mask can grow over both (area > MERGE_AREA). The frame
    is re-detected and the track restarts on the detected bowl that overlaps its own last single-bowl mask most;
  * re-acquisition by exclusion: while a bowl is lost, the first frame is re-run through the text detector every
    REDETECT_EVERY frames; if exactly one detected bowl is not the bowl the other track holds, the lost track
    restarts on it. (With two bowls, "the bowl that is not the other one" is unambiguous.)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402

TV = Path(os.environ.get("TRACKER_SAM3_ROOT", "/data/lgd/rma/trackv"))
LOST_AFTER = 5        # frames without the bowl before re-detection starts
REDETECT_EVERY = 5    # frames between re-detections while lost
REACQUIRE_AREA = 400  # a re-acquired bowl must be mostly visible (a free bowl is 600..1700 px, larger near the camera)
MERGE_AREA = 1900     # one track covering more than this holds two touching bowls (measured: 2100..2400)


def iou(a, b):
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def make_stream():
    sys.path.insert(1, str(TV))
    from rma_sam3_stream import NUM_FRAMES, Sam3Stream  # also puts the SAM3 source tree on sys.path
    import torch

    class ObjectStream(Sam3Stream):
        """Sam3Stream with one clock per object, so a single object can be restarted from a new mask."""

        def _start(self, k, img, mask):
            if not mask.any():
                self.states[k] = None
                return np.zeros(self.hw, bool)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = self.p.init_state(video_height=self.hw[0], video_width=self.hw[1], num_frames=NUM_FRAMES)
                st["images"] = {0: img}
                self.p.add_new_mask(inference_state=st, frame_idx=0, obj_id=1, mask=torch.from_numpy(mask.astype(np.float32)))
            self.states[k], self.clock[k] = st, 0
            return self._run(st, 0, preflight=True)

        def reset(self, rgb, masks):
            self.hw = rgb.shape[:2]
            self.states, self.clock = [None] * len(masks), [0] * len(masks)
            img = self._prep(rgb)
            return [self._start(k, img, m) for k, m in enumerate(masks)]

        def step(self, rgb):
            img = self._prep(rgb)
            out = []
            for k, st in enumerate(self.states):
                if st is None:
                    out.append(np.zeros(self.hw, bool))
                    continue
                self.clock[k] += 1
                st["images"] = {self.clock[k]: img}
                out.append(self._run(st, self.clock[k]))
            return out

        def restart(self, k, rgb, mask):
            return self._start(k, self._prep(rgb), mask)

    return ObjectStream()


class BowlTracker:
    def __init__(self):
        self.stream = make_stream()
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.proc = Sam3Processor(build_sam3_image_model(checkpoint_path=str(TV / "src/sam3.pt"), load_from_HF=False),
                                  confidence_threshold=0.15)
        self.k = len(S.NAMES)
        self.area = [0] * self.k
        self.events = []

    def detect(self, rgb, min_area):
        """SAM3 text detection -> bowl masks, best score first, duplicates removed; and a report."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(Image.fromarray(rgb).resize((1008, 1008), Image.BICUBIC))
            o = self.proc.set_text_prompt(state=state, prompt=S.DETECTOR_PROMPT)
        if len(o["scores"]) == 0:
            return [], {"candidates": 0}
        masks = (F.interpolate(o["masks_logits"].float(), (S.SOURCE, S.SOURCE), mode="area")[:, 0] > 0.5).cpu().numpy()
        scores = o["scores"].float().cpu().numpy()
        area = masks.sum(axis=(1, 2))
        keep = []
        for i in [i for i in np.argsort(-scores) if min_area <= area[i] <= 3000]:
            if all(iou(masks[i], masks[j]) <= 0.5 for j in keep):
                keep.append(i)
        report = {"candidates": int(len(scores)), "kept": len(keep), "scores": [round(float(scores[i]), 3) for i in keep[:4]],
                  "areas": [int(area[i]) for i in keep[:4]]}
        return [masks[i] for i in keep], report

    def _finish(self, masks):
        self.masks = masks
        out = np.full((self.k, 2), np.nan, np.float32)
        for j, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            self.area[j] = int(len(xs))
            seen = len(xs) >= S.MIN_PIXELS
            if seen:
                out[j] = (xs.mean(), ys.mean())
            self.lost[j] = 0 if seen else self.lost[j] + 1
            self.healthy[j] = self.healthy[j] + 1 if seen else 0
            if seen and len(xs) <= MERGE_AREA:
                self.single[j] = m  # the last mask that held one bowl
        return out

    def reset(self, rgb):
        rgb = np.ascontiguousarray(rgb, np.uint8)
        found, report = self.detect(rgb, S.MIN_PIXELS)  # the gripper may cover part of a bowl at the start
        if len(found) < self.k:
            raise RuntimeError(f"bowl detection failed: {report}")
        chosen = sorted(found[:self.k], key=lambda m: -np.nonzero(m)[1].mean())  # object order: largest x first
        self.frame, self.lost, self.healthy, self.events = 0, [0] * self.k, [0] * self.k, []
        self.single = [None] * self.k
        return self._finish(self.stream.reset(rgb, chosen)), report

    def step(self, rgb):
        rgb = np.ascontiguousarray(rgb, np.uint8)
        self.frame += 1
        masks = self.stream.step(rgb)
        seen = [int(m.sum()) >= S.MIN_PIXELS for m in masks]
        for a in range(self.k):  # collision: two tracks on the same bowl -> drop the one that was lost more recently
            for b in range(a + 1, self.k):
                if seen[a] and seen[b] and iou(masks[a], masks[b]) > 0.5:
                    drop = a if self.healthy[a] < self.healthy[b] else b
                    masks[drop], seen[drop] = np.zeros_like(masks[drop]), False
                    self.events.append((self.frame, drop, "collision"))
        for j in range(self.k):  # merge: one track over two touching bowls -> back to the bowl it was on
            if seen[j] and int(masks[j].sum()) > MERGE_AREA and self.single[j] is not None:
                found, _ = self.detect(rgb, REACQUIRE_AREA)
                inside = [m for m in found if float((m & masks[j]).sum()) / max(1.0, float(m.sum())) > 0.6]
                if len(inside) >= 2:
                    own = max(inside, key=lambda m: iou(m, self.single[j]))
                    if iou(own, self.single[j]) > 0.2:
                        masks[j] = self.stream.restart(j, rgb, own)
                        seen[j] = int(masks[j].sum()) >= S.MIN_PIXELS
                        self.events.append((self.frame, j, "split"))
        for j in range(self.k):  # re-acquisition by exclusion
            lost = self.lost[j] + 1 if not seen[j] else 0
            if lost >= LOST_AFTER and (lost - LOST_AFTER) % REDETECT_EVERY == 0 and all(seen[i] for i in range(self.k) if i != j):
                found, _ = self.detect(rgb, REACQUIRE_AREA)
                free = [m for m in found if all(iou(m, masks[i]) < 0.3 for i in range(self.k) if i != j)]
                if len(free) == 1:
                    masks[j] = self.stream.restart(j, rgb, free[0])
                    self.events.append((self.frame, j, "reacquired"))
        return self._finish(masks)


def serve():
    real_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything a library prints goes to stderr; the protocol owns the real stdout
    tracker = BowlTracker()
    real_out.write(b"READY\n"), real_out.flush()
    size = S.SOURCE * S.SOURCE * 3
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        rgb = np.frombuffer(stdin.read(size), np.uint8).reshape(S.SOURCE, S.SOURCE, 3)
        reply = {}
        try:
            if header.strip() == b"RESET":
                centres, reply["report"] = tracker.reset(rgb)
            else:
                centres = tracker.step(rgb)
            reply["centres"] = [None if not np.isfinite(c).all() else [float(c[0]), float(c[1])] for c in centres]
            reply["area"] = list(tracker.area)
        except Exception as exc:  # reported to the caller, which decides
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        real_out.write((json.dumps(reply) + "\n").encode()), real_out.flush()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
