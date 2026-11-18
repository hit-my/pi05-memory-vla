"""Online tracker server for RoboMemArena, v5 (evaluation side of the v5 tracks in /data/lgd/rma/trackv/all/final_v5).

Reproduces, one frame at a time, the MASKS of the offline v5 tracking (the rule layer OnlineVetoV5 runs in the caller,
rma_perception_v5.py, as for v2):
  tasks 1-17, 20-24  frame 0: SAM3 generic detections -> method B (fixed clicks, R = 30, legacy click mask as fallback)
                     = rma_tracker_v2.TrackerV2.first_frame, unchanged (offline: initB/initB_det.py + initB_eval_all.py 30
                     -> all/final via tv_full_b.py); then the SAM3 stream frame by frame.
  tasks 25, 26       frame 0: method B, then B2 = box-like detections first (v4/initB2.py; select() below is its body on
                     the in-memory detections instead of initB/det/<ep>.npz) -> all/final_fix via v4/tv_full_fix.py.
                     Every object has its own SAM3 state, so tracking all objects from the B2 masks equals the offline
                     "re-track the changed objects, copy the unchanged ones" of tv_full_fix.py.
  tasks 18, 19       /data/lgd/rma/trackv/v5/track5b.py track_episode with CHECKS=split (= track5.py + split check), run
                     causally: nothing before T0 (2 / 10), selection by appearance at T0, second look at frame 6 (task 18),
                     late acquisition / re-acquisition every 10 frames, identity check at the first grasp ("pick up the
                     <object>" item + gripper closed and moved), split check while carried, Stream5.add for every restart.
                     The functions and constants are IMPORTED from track5.py / track5b.py (same code as offline); the
                     classifier / dmax / usual areas / beta are those of ff_clf_<task>_d.json (= the constants file
                     rma_veto_v5_constants.json "first_frame_appearance", checked at start).
Inputs per frame: the RGB frame, the 8-d robot state (gripper pixel = tv_common_v4 projection), the gripper command issued
before the frame ("closed"), the object named by the current checklist item when it is "pick up the <object>" (None
otherwise). No simulator state.

Protocol on stdin/stdout (binary). K = number of tracked objects of the task.
  RESET {"task": id, "state": [8]}\n + 256*256*3 RGB -> one JSON line (report), then K*256*32 bytes = np.packbits of the K
                                                       masks of frame 0 (what the rules see)
  STEP {"state": [8], "closed": bool, "pick": k|null}\n + RGB -> one JSON line (frame info), then K*256*32 bytes
  STAT\n                                              -> one JSON line
Everything a library prints goes to stderr.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import rma_tracker_v2 as v2  # noqa: E402  (method B, SAM3 detector, legacy fallback; puts the SAM3 tree on sys.path)

TV = Path("/data/lgd/rma/trackv")
sys.path.insert(1, str(TV / "v5"))
import track5  # noqa: E402  (Stream5, candidates, probs, select_first, T0, RECHECK, constants)
import track5b  # noqa: E402  (split_part, mask_far, crop_at, overlap, gripper_candidate, C_RUN, CHECKS)
import ff_lib  # noqa: E402
import det5  # noqa: E402
from tv_common_v4 import eef_pixels  # noqa: E402

rma_spec = v2.rma_spec
SIZE = 256
APPEARANCE_TAG = "d"  # ff_clf_18_d.json / ff_clf_19_d.json: the classifier the final v5 tracks were made with
assert track5b.CHECKS == {"split"}, f"track5b CHECKS must be the v5 default 'split', got {track5b.CHECKS}"

# ---------------------------------------------------------------- VERBATIM from v4/initB2.py
LO, HI, R = 20, 700, 30
BOX_AREA, BOX_FILL, BOX_SCORE = (40, 220), 0.72, 0.5
YS, XS = np.mgrid[0:256, 0:256]


def box_like(m, score):
    ys, xs = np.nonzero(m)
    a = len(xs)
    fill = a / float((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
    return BOX_AREA[0] <= a <= BOX_AREA[1] and fill >= BOX_FILL and score >= BOX_SCORE
# ---------------------------------------------------------------- end VERBATIM


def b2_candidates(masks, scores):
    """initB2.candidates on the in-memory detections (initB/det/<ep>.npz holds exactly TrackerV2.detect's output)"""
    area = masks.sum(axis=(1, 2))
    ok = np.nonzero((area >= LO) & (area <= HI))[0]
    keep = []
    for i in ok[np.argsort(-scores[ok])]:
        if all((masks[i] & masks[j]).sum() / max(1, (masks[i] | masks[j]).sum()) <= 0.5 for j in keep):
            keep.append(i)
    return masks[keep], scores[keep]


def b2_select(clicks, cand, scores, old):
    """initB2.select (body unchanged) with cand/scores/old given: old = method-B labels of this episode"""
    clicks = np.asarray(clicks, float)
    k = len(clicks)
    boxes = [i for i in range(len(cand)) if box_like(cand[i], scores[i])]
    labels = np.zeros((256, 256), np.uint8)
    src = ["none"] * k
    if boxes:
        cc = np.array([[XS[cand[i]].mean(), YS[cand[i]].mean()] for i in boxes])
        d = np.hypot(cc[None, :, 0] - clicks[:, None, 0], cc[None, :, 1] - clicks[:, None, 1])
        from scipy.optimize import linear_sum_assignment
        for a, b in zip(*linear_sum_assignment(np.where(d <= R, d, 1e4))):
            if d[a, b] <= R:
                m = cand[boxes[b]] & (labels == 0)
                if m.sum() >= LO:
                    labels[m] = a + 1
                    src[a] = "box"
    for j in range(k):
        if src[j] == "none":
            m = (old == j + 1) & (labels == 0)
            if m.sum() >= LO:
                labels[m] = j + 1
                src[j] = "methodB"
    changed = [bool(((labels == j + 1) != (old == j + 1)).any()) for j in range(k)]
    return labels, src, changed


def appearance_constants(task):
    """(classifier, dmax, usual areas, beta) of ff_clf_<task>_d.json; must equal the constants file next to this code"""
    J = json.load(open(TV / f"v5/ff_clf_{task}_{APPEARANCE_TAG}.json"))
    C = json.load(open(CODE / "rma_veto_v5_constants.json"))["first_frame_appearance"][str(task)]
    sel = C["selection"]
    assert J["clf"] == C["classifier"] and float(J["dmax"]) == float(sel["pair_distance_max_px"]) \
        and list(J["atyp"]) == list(sel["usual_area_px"]) and float(J["beta"]) == float(sel["area_exponent"]), \
        f"ff_clf_{task}_{APPEARANCE_TAG}.json differs from rma_veto_v5_constants.json"
    assert int(C["t0"]) == track5.T0[task] and C.get("second_look_frame") == track5.RECHECK.get(task)
    return ff_lib.Softmax.from_json(J["clf"]), float(J["dmax"]), J["atyp"], float(J["beta"])


class Appearance1819:
    """track5b.track_episode (CHECKS = split) as a causal per-frame object. The loop body below is the body of
    track_episode's `for t in range(t0 + 1, n)` with the episode arrays replaced by the per-frame inputs
    (rgb[t] -> rgb, e[t] -> e, closed[t] / closed[t-1] -> closed / self.closed_prev, picks[t] -> pick) and the
    branches of the inactive checks (carry, release: not in CHECKS) left out."""

    def __init__(self, stream, proc, task, C):
        self.stream, self.proc, self.task = stream, proc, task
        self.clf, self.dmax, self.atyp, self.beta = C
        self.K = len(rma_spec.TASKS[task]["objects"])
        self.t0 = track5.T0[task]
        self.t = -1
        self.closed_prev = False
        self.info = None
        self.started = False

    def _cand(self, rgb, crop=None):
        self.info["detections"] += 1
        return track5b.candidates(self.proc, rgb, self.task, crop) if crop is not None else track5b.candidates(self.proc, rgb, self.task)

    def frame(self, rgb, e, closed, pick):
        """-> masks [K,256,256] of this frame"""
        self.t += 1
        t, K, stream = self.t, self.K, self.stream
        out = None
        if t < self.t0:
            out = np.zeros((K, SIZE, SIZE), bool)
        elif t == self.t0:
            self.info = dict(t0=self.t0, detections=0, grasp_fix=[], dropped=[], reacquired=[], recheck=[], split_fix=[])
            cand, sc = self._cand(rgb)
            P = track5.probs(self.clf, rgb, cand, K)
            sel = track5.select_first(P, cand, K, self.dmax, self.atyp, self.beta)
            first = np.stack([cand[sel[k]] if sel[k] >= 0 else np.zeros((256, 256), bool) for k in range(K)])
            self.info.update(p_first=[float(P[sel[k], k]) if sel[k] >= 0 else None for k in range(K)],
                             acquired=[t if sel[k] >= 0 else None for k in range(K)], p_late=[None] * K, n_cand=int(len(cand)))
            self.pending, self.watch = None, None
            self.lost = np.zeros(K, int)
            self.last_c = [ff_lib.centroid(first[k]) if sel[k] >= 0 else None for k in range(K)]
            self.init_c = [ff_lib.centroid(first[k]) if sel[k] >= 0 else None for k in range(K)]
            self.first = first
            ms = stream.reset(rgb, [first[k] for k in range(K)])
            out = np.stack(ms).astype(bool)
            self.started = True
        else:
            out = self._step(rgb, e, closed, pick)
        self.closed_prev = bool(closed)
        return out

    def _step(self, rgb, e, closed, pick):
        t, K, stream, info = self.t, self.K, self.stream, self.info
        clf, dmax, atyp, beta = self.clf, self.dmax, self.atyp, self.beta
        masks = np.stack(stream.step(rgb)).astype(bool)   # masks[t] of track_episode

        def restart(k, m, key, p):
            for j in range(K):
                if j != k and stream.states[j] is not None and track5b.overlap(m, masks[j], 0.3):
                    stream.states[j] = None  # that track was on this object: it is looked for again
                    masks[j] = False
                    info["dropped"].append([int(j), int(t)])
            masks[k] = stream.add(k, rgb, m)
            info[key].append([int(k), int(t), float(p)])

        # ---- second look (task 18)
        if t == track5.RECHECK.get(self.task):
            cand, sc = self._cand(rgb)
            P = track5.probs(clf, rgb, cand, K)
            sel2 = track5.select_first(P, cand, K, dmax, atyp, beta)
            for k in range(K):
                if sel2[k] < 0:
                    continue
                m, mk = cand[sel2[k]], masks[k]
                if stream.states[k] is not None and mk.sum() >= track5.LO and track5b.overlap(m, mk, 0.2):
                    continue
                restart(k, m, "recheck", P[sel2[k], k])
                self.init_c[k] = ff_lib.centroid(m)
        # ---- grasp: first look (track5) and the watch of the carried object (split check)
        if closed and not self.closed_prev:
            self.pending, self.watch = None, None
            if pick is not None:
                self.pending = dict(k=pick, t=t, e=e.copy())
        if not closed:
            self.pending, self.watch = None, None
        if self.pending is not None and t - self.pending["t"] >= track5.G_WAIT and np.hypot(*(e - self.pending["e"])) >= track5.G_MOVE:
            k = self.pending["k"]
            mk = masks[k]
            if stream.states[k] is None:
                need, pth = True, track5.G_PMIN
            elif mk.sum() < track5.LO:
                need, pth = True, track5.G_PLOST
            else:
                need, pth = track5b.mask_far(mk, e), track5.G_PMIN
            self.pending = None
            fixed = False
            if need:
                cand, sc = self._cand(rgb, crop=track5b.crop_at(e))
                if len(cand):
                    P = track5.probs(clf, rgb, cand, K)
                    i = track5b.gripper_candidate(P, cand, k, e, pth)
                    if i is not None:
                        restart(k, cand[i], "grasp_fix", P[i, k])
                        fixed = True
            if not fixed:
                self.watch = dict(k=k, run=0, tries=0, last=t, srun=0)
        elif self.watch is not None:
            w = self.watch
            k = w["k"]
            part = track5b.split_part(masks[k], e) if stream.states[k] is not None else None
            w["srun"] = w["srun"] + 1 if part is not None else 0
            if w["srun"] >= track5b.C_RUN and "split" in track5b.CHECKS:
                restart(k, part, "split_fix", 1.0)
                w["srun"] = 0
            w["run"] = w["run"] + 1 if track5b.mask_far(masks[k], e) else 0   # (only read by the inactive carry check)
        # ---- late acquisition / re-acquisition (track5)
        for k in range(K):
            if masks[k].sum() >= track5.LO:
                self.last_c[k] = ff_lib.centroid(masks[k])
            self.lost[k] = self.lost[k] + 1 if (stream.states[k] is not None and masks[k].sum() < track5.LO) else 0
        missing = [k for k in range(K) if stream.states[k] is None or self.lost[k] >= track5.LOST_N]
        if missing and (t - self.t0) % track5.EVERY == 0:
            cand, sc = self._cand(rgb)
            if len(cand):
                P = track5.probs(clf, rgb, cand, K)
                for k in missing:
                    ratio = np.array([min(1.0, float(m.sum()) / atyp[k]) for m in cand])
                    for i in np.argsort(-(np.log(P[:, k] + 1e-9) + beta * np.log(ratio))):
                        if P[i, k] < track5.PLATE or ratio[i] < 0.4:
                            continue
                        m = cand[i]
                        c = ff_lib.centroid(m)
                        if np.hypot(*(c - e)) < 15:
                            continue
                        if stream.states[k] is not None:
                            if self.last_c[k] is None or np.hypot(*(c - self.last_c[k])) > track5.NEAR_LAST:
                                continue
                        else:
                            others = [self.init_c[j] for j in range(K) if j != k and self.init_c[j] is not None]
                            if others and min(float(np.hypot(*(c - o))) for o in others) > dmax + track5.NEAR_INIT:
                                continue
                        if any(j != k and track5b.overlap(m, masks[j], 0.3) for j in range(K)):
                            continue
                        if stream.states[k] is not None:
                            info["reacquired"].append([int(k), int(t), float(P[i, k])])
                        else:
                            info["acquired"][k], info["p_late"][k] = int(t), float(P[i, k])
                        masks[k] = stream.add(k, rgb, m)
                        self.lost[k] = 0
                        break
        return masks


class TrackerV5:
    def __init__(self, root, legacy="lazy"):
        self.base = v2.TrackerV2.__new__(v2.TrackerV2)      # TrackerV2 with a Stream5 (Sam3Stream + add) as its stream
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.base.root = root
        model = build_sam3_image_model(checkpoint_path=str(TV / "src/sam3.pt"), load_from_HF=False)
        self.base.proc = Sam3Processor(model, confidence_threshold=0.15)   # initB_det.py; det5.THR is 0.15 as well
        assert det5.THR == 0.15
        self.base.stream = track5.Stream5()
        self.base.named = None
        self.base.frames = 0
        if legacy == "eager":
            self.base._named()
        self.app = None
        self.task = None
        self.frames = 0
        self.consts = {}

    @staticmethod
    def eef(task, state):
        return eef_pixels(f"t{int(task):02d}_online", np.asarray(state, np.float64).reshape(1, -1))[0]

    def reset(self, rgb, task, state):
        self.task = int(task)
        k = len(rma_spec.TASKS[self.task]["objects"])
        self.frames = 1
        start = time.time()
        if self.task in (18, 19):
            if self.task not in self.consts:
                self.consts[self.task] = appearance_constants(self.task)
            self.app = Appearance1819(self.base.stream, self.base.proc, self.task, self.consts[self.task])
            masks = self.app.frame(rgb, self.eef(self.task, state), False, None)
            report = {"task": self.task, "k": k, "first_frame": "appearance", "t0": self.app.t0}
        else:
            self.app = None
            labels, report = self.base.first_frame(rgb, self.task)            # method B (+ legacy fallback)
            report["first_frame"] = "B"
            if self.task in (25, 26):
                dm, ds, _ = self.base.detect(rgb)                               # same detections as initB/det/<ep>.npz
                cand, sc = b2_candidates(dm, ds)
                labels, src2, changed = b2_select(rma_spec.CLICKS[self.task], cand, sc, labels)
                report.update(first_frame="B2", src_b2=src2, changed_b2=changed)
            first = np.stack([labels == j + 1 for j in range(k)])
            ms = self.base.stream.reset(rgb, [first[j] for j in range(k)])
            masks = np.stack(ms).astype(bool)
            report["pixels_first"] = [int(first[j].sum()) for j in range(k)]
        report["reset_s"] = round(time.time() - start, 3)
        return report, masks

    def step(self, rgb, state, closed, pick):
        self.frames += 1
        if self.app is not None:
            n0 = self.app.info["detections"] if self.app.info else 0
            masks = self.app.frame(rgb, self.eef(self.task, state), bool(closed), None if pick is None else int(pick))
            info = {"t": self.app.t, "detections": (self.app.info["detections"] - n0) if self.app.info else 0}
            return info, masks
        return {}, np.stack(self.base.stream.step(rgb)).astype(bool)

    def stat(self):
        import torch
        s = {"frames": self.frames, "task": self.task, "legacy_loaded": self.base.named is not None,
             "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20),
             "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20),
             "gpu_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
        if self.app is not None and self.app.info is not None:
            s["info_1819"] = self.app.info
        return s


def main():
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "/data/lgd/pi05-abc/tracker/assets"
    legacy = sys.argv[sys.argv.index("--legacy") + 1] if "--legacy" in sys.argv else "lazy"
    import torch
    # >= 2 CPU threads: with ONE torch thread the SAM3 stream's masks differ from the offline runs (measured on
    # t04_seed100: 287/1020 frames differ with 1 thread, bit-identical with 2, 3 or 90 threads)
    torch.set_num_threads(max(2, int(os.environ.get("TRACKER_THREADS", "2"))))
    tracker = TrackerV5(root, legacy)
    out.write(b"READY\n")
    out.flush()
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        header = header.decode().strip()
        if header == "STAT":
            out.write((json.dumps(tracker.stat()) + "\n").encode())
            out.flush()
            continue
        data = stdin.read(SIZE * SIZE * 3)
        if len(data) != SIZE * SIZE * 3:
            return
        rgb = np.frombuffer(data, np.uint8).reshape(SIZE, SIZE, 3).copy()
        if header.startswith("RESET "):
            arg = json.loads(header[len("RESET "):])
            info, masks = tracker.reset(rgb, int(arg["task"]), arg["state"])
        elif header.startswith("STEP "):
            arg = json.loads(header[len("STEP "):])
            info, masks = tracker.step(rgb, arg["state"], bool(arg["closed"]), arg.get("pick"))
        else:
            raise ValueError(f"unknown command {header!r}")
        out.write((json.dumps(info) + "\n").encode())
        out.write(np.packbits(masks, axis=-1).tobytes())
        out.flush()


if __name__ == "__main__":
    main()
