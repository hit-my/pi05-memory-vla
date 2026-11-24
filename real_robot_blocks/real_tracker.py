"""Object tracker shared by dataset labelling (real_track.py) and deployment (real_deploy*.py): one code path, causal.

Objects (real_spec.NAMES): the three cups (far, middle, near) and the three numbered blocks (0, 1, 2).
Library:  ObjectTracker().reset(rgb256) -> (centres [K,2], report);  .step(rgb256, closed) -> centres [K,2]   (nan = not seen)
Server:   real_tracker.py --serve      same binary protocol as real_robot/real_tracker.py (RESET / STEP + 256*256*3 bytes
          -> one JSON line {"centres": [[x,y]|null,...], "area": [...]}); STEP carries the gripper flag: b"STEP 1\n" = closed.
          The first line printed is READY.
Needs: SAM3 source tree + sam3.pt (TRACKER_SAM3_ROOT), torch with CUDA.

First frame: SAM3 text detection with several prompts per kind; the cups are the three candidates that form the right
column, the blocks the three that form the left column (real_spec.COLUMNS), each with the highest total score.
After that SAM3 video tracking, one stream per object. The raw mask centre is NOT what is reported, because it lies:
when the arm passes over a standing object, the visible part of its mask shrinks to one side (or leaks onto the gripper)
and the centre slides along with the arm, which draws a trajectory for an object that never moved. In this task an
object only moves while the gripper holds it, so the tracker is told every frame whether the gripper is commanded
closed (labelling: the recorded command; deployment: the last executed action), and:
  * standing (the default): the object is reported at the centre it had when it last came to rest, however much of it
    is covered and wherever its mask centre is; with fewer than MIN_PIXELS visible it is "not seen" (nothing is drawn);
  * carried: while the gripper is closed, the object whose mask centre has been more than LIFT_PIXELS from its standing
    centre for LIFT_FRAMES frames and has left the place where it stood (at most LEFT_INSIDE of its mask is still there) is the carried one (the furthest, if several); only it reports its mask centre;
  * settling: after the gripper opens the carried object keeps reporting its mask centre until the mask is steady for
    SETTLE_FRAMES frames; that centre is where it stands from then on;
  * a carried centre that jumps more than JUMP_PIXELS in one frame is not believed (not seen) unless it stays.All of it uses the current frame and the past only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402

TV = Path(os.environ.get("TRACKER_SAM3_ROOT", "/path/to/trackv"))
LIFT_PIXELS = {"cup": 20.0, "block": 10.0}   # gripper closed: an object whose mask centre is this far from where it stands
                                             # is being carried (a block is about 26 px across, a cup about 40)
LEFT_INSIDE = {"cup": 0.3, "block": 0.6}     # ... and at most this share of its mask still lies on the place where it stood (it has left it;
                         #     a standing object that the arm covers keeps most of its visible mask on that place)
COLOUR_TOLERANCE = 45.0  # ... and its mask still has the object's own colour (RGB distance to the first frame; a mask that
                         #     slipped onto the arm or onto another object has not)
LIFT_FRAMES = 4          # ... for this many consecutive frames
SWITCH_PIXELS = 25.0     # another object further away than the carried one by this much takes over (wrong first guess)
SETTLE_FRAMES = 8        # gripper open again: frames with a steady mask before the object stands at its new place
SETTLE_PIXELS = 1.5      # "steady": centre change per frame
SETTLE_LIMIT = 45        # ... and at the latest after this many frames
REFINE_GAIN = 1.15       # standing, gripper open: a steady mask this much larger than the one the standing centre was taken
REFINE_LIMIT = 1.3       # from (the gripper has moved off the object), but not larger than this times its first-frame area
                         # (a mask that leaked onto something), and of the object's colour, replaces the standing centre
JUMP_PIXELS = 45.0       # a carried object's centre this far from the last reported one in a single frame is not believed ...
JUMP_FRAMES = 5          # ... unless it stays there this many frames

def iou(a, b):
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def centre_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()], np.float32)

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


class ObjectTracker:
    def __init__(self):
        self.stream = make_stream()
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.proc = Sam3Processor(build_sam3_image_model(checkpoint_path=str(TV / "src/sam3.pt"), load_from_HF=False),
                                  confidence_threshold=S.DETECTOR_CONFIDENCE)
        self.k = len(S.NAMES)
        self.area = [0] * self.k
        self.events = []

    def detect(self, rgb, prompts, area_range):
        """SAM3 text detection over `prompts` -> masks (best score first, duplicates removed), scores, report."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        masks, scores = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(Image.fromarray(rgb).resize((1008, 1008), Image.BICUBIC))
            for prompt in prompts:
                o = self.proc.set_text_prompt(state=state, prompt=prompt)
                if len(o["scores"]) == 0:
                    continue
                m = (F.interpolate(o["masks_logits"].float(), (S.SOURCE, S.SOURCE), mode="area")[:, 0] > 0.5).cpu().numpy()
                masks += list(m)
                scores += [float(v) for v in o["scores"].float().cpu().numpy()]
        keep = []
        for i in np.argsort(-np.array(scores)) if scores else []:
            if area_range[0] <= masks[i].sum() <= area_range[1] and all(iou(masks[i], masks[j]) <= 0.5 for j in keep):
                keep.append(int(i))
        report = {"candidates": len(scores), "kept": len(keep), "scores": [round(scores[i], 3) for i in keep[:6]]}
        return [masks[i] for i in keep], [scores[i] for i in keep], report

    @staticmethod
    def column(found, scores, rule):
        """The three candidates that stand in a column (rule) with the highest total score, top first; or None."""
        import itertools
        centre = [centre_of(m) for m in found]
        ok = [i for i in range(len(found)) if rule["x"][0] <= centre[i][0] <= rule["x"][1]]
        best = None
        for triple in itertools.combinations(ok, 3):
            a, b, c = sorted(triple, key=lambda i: centre[i][1])
            g1, g2 = centre[b][1] - centre[a][1], centre[c][1] - centre[b][1]
            xs = [centre[i][0] for i in triple]
            if (max(xs) - min(xs) <= rule["x_spread"] and rule["gap"][0] <= g1 <= rule["gap"][1]
                    and rule["gap"][0] <= g2 <= rule["gap"][1] and abs(g1 - g2) <= rule["gap_difference"]):
                total = sum(scores[i] for i in triple)
                if best is None or total > best[0]:
                    best = (total, [a, b, c])
        return None if best is None else [found[i] for i in best[1]]

    def _stand(self, j, mask):
        self.home[j], self.foot[j], self.stand_area[j] = centre_of(mask), mask.copy(), int(mask.sum())

    def _report(self, masks, closed, rgb):
        """Raw masks + gripper flag -> reported centres (the rules of the module text)."""
        centre = [centre_of(m) if int(m.sum()) >= S.MIN_PIXELS[j] else None for j, m in enumerate(masks)]
        away = [float(np.hypot(*(c - self.home[j]))) if c is not None else 0.0 for j, c in enumerate(centre)]
        for j, m in enumerate(masks):
            self.area[j] = int(m.sum())
            left = centre[j] is not None and float((m & self.foot[j]).sum()) / max(1, self.area[j]) <= LEFT_INSIDE[S.OBJECTS[j]]
            left = left and float(np.linalg.norm(rgb[m].astype(np.float32).mean(0) - self.colour[j])) <= COLOUR_TOLERANCE
            self.lifted[j] = self.lifted[j] + 1 if closed and away[j] > LIFT_PIXELS[S.OBJECTS[j]] and left else 0
        if closed:
            self.settle = 0
            ready = [j for j in range(self.k) if self.lifted[j] >= LIFT_FRAMES]
            if ready:
                best = max(ready, key=lambda j: away[j])
                if self.carried is None or (best != self.carried and away[best] > away[self.carried] + SWITCH_PIXELS):
                    self.carried, self.reported, self.jumped = best, None, 0
                    self.events.append((self.frame, best, "carried"))
        elif self.carried is not None:  # released: wait until it lies still
            j, c = self.carried, centre[self.carried]
            steady = c is not None and self.previous is not None and float(np.hypot(*(c - self.previous))) <= SETTLE_PIXELS
            self.still = self.still + 1 if steady else 0
            self.settle += 1
            if c is not None and (self.still >= SETTLE_FRAMES or self.settle >= SETTLE_LIMIT):
                self._stand(j, masks[j])
                self.events.append((self.frame, j, "standing"))
                self.carried, self.still = None, 0
        for j in range(self.k):  # a standing object that is now seen better gets a better standing centre
            calm = centre[j] is not None and self.last[j] is not None and float(np.hypot(*(centre[j] - self.last[j]))) <= SETTLE_PIXELS
            self.calm[j] = self.calm[j] + 1 if calm and not closed and j != self.carried else 0
            self.last[j] = centre[j]
            if (self.calm[j] >= SETTLE_FRAMES and REFINE_GAIN * self.stand_area[j] <= self.area[j] <= REFINE_LIMIT * self.full_area[j]
                    and float(np.linalg.norm(rgb[masks[j]].astype(np.float32).mean(0) - self.colour[j])) <= COLOUR_TOLERANCE):
                self._stand(j, masks[j])
                self.events.append((self.frame, j, "refined"))
        out = np.full((self.k, 2), np.nan, np.float32)
        for j in range(self.k):
            if centre[j] is None:
                continue
            if j != self.carried:
                out[j] = self.home[j]
                continue
            c = centre[j]
            if self.reported is not None and float(np.hypot(*(c - self.reported))) > JUMP_PIXELS and self.jumped < JUMP_FRAMES:
                self.jumped += 1
                continue
            self.jumped, self.reported = 0, c.copy()
            out[j] = c
        self.previous = centre[self.carried] if self.carried is not None else None
        return out
    def reset(self, rgb):
        rgb = np.ascontiguousarray(rgb, np.uint8)
        chosen, report = [], {}
        for kind, prompts, area, rule in S.KINDS:
            found, scores, rep = self.detect(rgb, prompts, area)
            triple = self.column(found, scores, rule)
            if triple is None:
                raise RuntimeError(f"{kind} detection failed: {rep}")
            chosen += triple if kind == "cup" else triple[::-1]   # cups: far, middle, near; blocks: 0 (bottom), 1, 2 (top)
            report[kind] = rep
        report["centres"] = [[round(float(v)) for v in centre_of(m)] for m in chosen]
        self.start(chosen, rgb)
        self.raw = self.stream.reset(rgb, chosen)
        return self._report(self.raw, False, rgb), report

    @classmethod
    def offline(cls):
        """The reporting rules without SAM3: start(first_masks, rgb), then replay(masks, closed, rgb) per frame."""
        self = cls.__new__(cls)
        self.k, self.area, self.events = len(S.NAMES), [0] * len(S.NAMES), []
        return self

    def start(self, chosen, rgb):
        self.chosen = [np.asarray(m, bool) for m in chosen]
        self.frame, self.events = 0, []
        self.home, self.foot, self.lifted = [None] * self.k, [None] * self.k, [0] * self.k
        self.stand_area, self.full_area = [0] * self.k, [int(m.sum()) for m in self.chosen]
        self.last, self.calm = [None] * self.k, [0] * self.k
        self.carried, self.previous, self.reported, self.jumped, self.still, self.settle = None, None, None, 0, 0, 0
        self.colour = [rgb[m].astype(np.float32).mean(0) for m in self.chosen]
        for j in range(self.k):
            self._stand(j, self.chosen[j])

    def replay(self, masks, closed, rgb, first=False):
        if not first:
            self.frame += 1
        return self._report([np.asarray(m, bool) for m in masks], bool(closed) and not first, np.asarray(rgb))
    def step(self, rgb, closed=False):
        rgb = np.ascontiguousarray(rgb, np.uint8)
        self.frame += 1
        self.raw = self.stream.step(rgb)
        return self._report(self.raw, bool(closed), rgb)

def serve():
    real_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything a library prints goes to stderr; the protocol owns the real stdout
    tracker = ObjectTracker()
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
            if header.strip().startswith(b"RESET"):
                centres, reply["report"] = tracker.reset(rgb)
            else:
                centres = tracker.step(rgb, header.strip().endswith(b"1"))
            reply["centres"] = [None if not np.isfinite(c).all() else [float(c[0]), float(c[1])] for c in centres]
            reply["area"] = list(tracker.area)
        except Exception as exc:  # reported to the caller, which decides
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        real_out.write((json.dumps(reply) + "\n").encode()), real_out.flush()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
