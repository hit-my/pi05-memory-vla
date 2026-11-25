"""v5 tracking for tasks 18/19 (first-frame selection by appearance + late acquisition), causal, RGB + robot state only.

  frame T0 (2 for task 18, 10 for task 19: the objects are still dropping before; nothing is drawn before T0):
      zoomed SAM3 text detections (det5.detect) -> appearance classifier (ff_clf_<task>_<TAG>.json, learned on the selection
      half) -> one detection per object (ff_lib.select_joint: probability >= PMIN, no overlapping choices, a penalty when
      two chosen detections are farther apart than the objects ever stand on cabinet 1) -> SAM3 tracker started from these masks.
  later, every EVERY frames, for each object that has no track yet (it was hidden, e.g. the butter behind the chocolate) or
      whose mask has been (nearly) empty for LOST_N frames: detect again; a detection with probability >= PLATE and at least
      0.4 x the object's usual area that does not overlap another object's current mask and is not at the gripper (re)starts
      the track.
  at the first grasp of each object (checklist item "pick up the <object>", gripper closes and moves away): if the mask of
      that object did not travel with the gripper (or the object has no track), it is re-started from the detection at the
      gripper; a track of another object that sat on the carried object is dropped and looked for again.
Output v5/track/<ep>.npz: same keys as all/final (masks packed, raw_centres, raw_visibility, area, first) + info (json).
track5.py SHARD NSHARDS TAG TASK[,TASK] [ep ...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("xxxx/rma/trackv")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE / "v5"))
import rma_spec  # noqa: E402
from rma_sam3_stream import NUM_FRAMES, Sam3Stream  # noqa: E402
from tv_common import centres_from_masks, load, task_of  # noqa: E402
from tv_common_v4 import eef_pixels  # noqa: E402
import det5  # noqa: E402
import ff_lib  # noqa: E402
import torch  # noqa: E402

T0 = {18: 2, 19: 10}
RECHECK = {18: 6}  # frame of a second look: the selection is repeated; an object whose new detection is not where its mask is, is re-started
PMIN, PLATE, EVERY, LO, HI = 0.5, 0.7, 10, 20, 700
GRASP_FIX = True   # identity check at the first grasp of each object (uses the checklist item "pick up the <object>")
G_WAIT, G_MOVE, G_FAR, G_EDGE, G_PMIN, G_PLOST = 15, 18.0, 12.0, 6.0, 0.02, 0.5
NEAR_LAST, NEAR_INIT = 15.0, 8.0   # re-acquisition: within NEAR_LAST px of where the mask was last seen; first acquisition:
#                                    within dmax + NEAR_INIT px of where another object was selected (they stand together)
LOST_N = 10        # a track whose mask has had < LO pixels for this many frames is looked for again


class Stream5(Sam3Stream):
    def add(self, j, rgb, mask):
        """start a track for object j at the current frame (late acquisition)"""
        img = self._prep(rgb)
        t = self.t
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = self.p.init_state(video_height=self.hw[0], video_width=self.hw[1], num_frames=NUM_FRAMES)
            st["images"] = {t: img}
            self.p.add_new_mask(inference_state=st, frame_idx=t, obj_id=1, mask=torch.from_numpy(mask.astype(np.float32)))
        self.states[j] = st
        return self._run(st, t, preflight=True)


def candidates(proc, rgb, task, crop=None):
    masks, scores, pid = det5.detect(proc, rgb, task, crop)
    area = masks.sum(axis=(1, 2)) if len(masks) else np.zeros(0)
    ok = np.nonzero((area >= LO) & (area <= HI))[0]
    keep = []
    for i in ok[np.argsort(-scores[ok], kind="stable")]:
        if all((masks[i] & masks[j]).sum() / max(1, (masks[i] | masks[j]).sum()) <= 0.7 for j in keep):
            keep.append(i)
    return masks[keep], scores[keep]


def probs(clf, rgb, cand, K):
    if len(cand) == 0:
        return np.zeros((0, K + 1))
    return clf.predict_proba(np.array([ff_lib.feat(rgb, m) for m in cand]))


def select_first(P, cand, K, dmax, atyp, beta, pmin=PMIN):
    if len(cand) == 0:
        return [-1] * K
    cen = np.array([ff_lib.centroid(m) for m in cand])
    return ff_lib.select_joint(P, cen, cand, K, pmin, dmax, atyp=atyp, beta=beta)


if __name__ == "__main__":
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    shard, shards, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    tasks = [int(x) for x in sys.argv[4].split(",")]
    out = HERE / "v5/track"
    out.mkdir(parents=True, exist_ok=True)
    claims = HERE / "v5/claims"; claims.mkdir(exist_ok=True)
    eps = sys.argv[5:]
    if not eps:  # selection half first (even positions of each task), tasks interleaved
        per = {t: sorted(p.stem for p in (HERE / "all/final").glob(f"t{t:02d}_seed*.npz")) for t in tasks}
        eps = sorted((ep for t in tasks for ep in per[t]), key=lambda ep: (per[task_of(ep)].index(ep) % 2, per[task_of(ep)].index(ep), task_of(ep)))
    order = eps[shard::shards] + eps
    clfs = {t: ff_lib.Softmax.from_json(json.load(open(HERE / f"v5/ff_clf_{t}_{TAG}.json"))["clf"]) for t in tasks}
    dmaxs = {t: float(json.load(open(HERE / f"v5/ff_clf_{t}_{TAG}.json"))["dmax"]) for t in tasks}
    atyps = {t: json.load(open(HERE / f"v5/ff_clf_{t}_{TAG}.json"))["atyp"] for t in tasks}
    betas = {t: float(json.load(open(HERE / f"v5/ff_clf_{t}_{TAG}.json"))["beta"]) for t in tasks}
    stream = Stream5()
    proc = Sam3Processor(build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False), confidence_threshold=det5.THR)
    for ep in order:
        if (out / f"{ep}.npz").exists():
            continue
        try:
            (claims / ep).mkdir()
        except FileExistsError:
            continue
        start = time.time()
        task = task_of(ep)
        rgb, states, actions = load(ep)
        n = len(rgb)
        K = len(rma_spec.TASKS[task]["objects"])
        e = eef_pixels(ep, states)
        t0 = T0[task]
        masks = np.zeros((n, K, 256, 256), bool)
        cand, sc = candidates(proc, rgb[t0], task)
        P = probs(clfs[task], rgb[t0], cand, K)
        sel = select_first(P, cand, K, dmaxs[task], atyps[task], betas[task])
        first = np.stack([cand[sel[k]] if sel[k] >= 0 else np.zeros((256, 256), bool) for k in range(K)])
        info = dict(t0=t0, p_first=[float(P[sel[k], k]) if sel[k] >= 0 else None for k in range(K)], acquired=[t0 if sel[k] >= 0 else None for k in range(K)],
                    p_late=[None] * K, n_cand=int(len(cand)), detections=1)
        meta = np.load(f"xxxx/rma/lr_eps/meta/{ep}.npz")
        sub_text = {int(s): str(x).lower() for s, x in zip(meta["subtask_order"], meta["subtask_texts"])}
        sub = meta["subtask"]
        closed = np.r_[False, actions[:-1, 6] > 0]
        pending = None
        info["grasp_fix"], info["dropped"], info["reacquired"], info["recheck"] = [], [], [], []
        lost = np.zeros(K, int)
        last_c = [ff_lib.centroid(first[k]) if sel[k] >= 0 else None for k in range(K)]   # where the mask of k was last seen
        init_c = [ff_lib.centroid(first[k]) if sel[k] >= 0 else None for k in range(K)]   # where k was selected
        ms = stream.reset(rgb[t0], [first[k] for k in range(K)])
        for k in range(K):
            masks[t0, k] = ms[k]
        for t in range(t0 + 1, n):
            ms = stream.step(rgb[t])
            for k in range(K):
                masks[t, k] = ms[k]
            # ---- second look (task 18, frame 6: the arm has moved away from the cabinet a little)
            if t == RECHECK.get(task):
                cand, sc = candidates(proc, rgb[t], task)
                info["detections"] += 1
                P = probs(clfs[task], rgb[t], cand, K)
                sel2 = select_first(P, cand, K, dmaxs[task], atyps[task], betas[task])
                for k in range(K):
                    if sel2[k] < 0:
                        continue
                    m, mk = cand[sel2[k]], masks[t, k]
                    if stream.states[k] is not None and mk.sum() >= LO and (m & mk).sum() > 0.2 * min(m.sum(), mk.sum()):
                        continue  # the track of k is on the detection chosen for k
                    for j in range(K):
                        if j != k and stream.states[j] is not None and (m & masks[t, j]).sum() > 0.3 * min(m.sum(), max(1, masks[t, j].sum())):
                            stream.states[j] = None
                            masks[t, j] = False
                            info["dropped"].append([int(j), int(t)])
                    masks[t, k] = stream.add(k, rgb[t], m)
                    init_c[k] = ff_lib.centroid(m)
                    info["recheck"].append([int(k), int(t), float(P[sel2[k], k])])
            # ---- identity check at the grasp: during "pick up the <object k>" the gripper closes and moves away; if the mask of k
            #      has not travelled with it (or k has no track), k is re-started from the detection at the gripper
            if GRASP_FIX and closed[t] and not closed[t - 1]:
                s = sub_text[int(sub[t])]
                k = rma_spec.subtask_object(task, s) if s.startswith("pick") else None
                pending = None
                if k is not None:
                    pending = dict(k=k, t=t, e=e[t].copy())
            if pending is not None and not closed[t]:
                pending = None
            if pending is not None and t - pending["t"] >= G_WAIT and np.hypot(*(e[t] - pending["e"])) >= G_MOVE:
                # the closed gripper has moved away from where it closed: the object named by the checklist item is in the hand
                k = pending["k"]
                mk = masks[t, k]
                ys_, xs_ = np.nonzero(mk)
                if stream.states[k] is None:
                    need, pth = True, G_PMIN
                elif len(xs_) < LO:
                    need, pth = True, G_PLOST   # its mask is empty: only a confident detection at the gripper restarts it
                else:
                    need, pth = float(np.min(np.hypot(xs_ - e[t][0], ys_ - e[t][1]))) > G_FAR, G_PMIN   # its mask is somewhere else
                pending = None
                if need:
                    cx, cy = int(np.clip(e[t][0] - 48, 0, 160)), int(np.clip(e[t][1] - 48, 0, 160))
                    cand, sc = candidates(proc, rgb[t], task, crop=(cx, cy, cx + 96, cy + 96))
                    info["detections"] += 1
                    if len(cand):
                        P = probs(clfs[task], rgb[t], cand, K)
                        ys, xs = np.mgrid[0:256, 0:256]
                        for i in np.argsort(-P[:, k]):
                            if P[i, k] < pth:
                                break
                            m = cand[i]
                            if float(np.min(np.hypot(xs[m] - e[t][0], ys[m] - e[t][1]))) >= G_EDGE:
                                continue  # not at the gripper
                            for j in range(K):
                                if j != k and stream.states[j] is not None and (m & masks[t, j]).sum() > 0.3 * min(m.sum(), max(1, masks[t, j].sum())):
                                    stream.states[j] = None  # that track was on the object in the hand: it is looked for again
                                    masks[t, j] = False
                                    info["dropped"].append([int(j), int(t)])
                            masks[t, k] = stream.add(k, rgb[t], m)
                            info["grasp_fix"].append([int(k), int(t), float(P[i, k])])
                            break
            # an object without a track, or whose mask has been (nearly) empty for LOST_N frames, is looked for (again)
            for k in range(K):
                if masks[t, k].sum() >= LO:
                    last_c[k] = ff_lib.centroid(masks[t, k])
                lost[k] = lost[k] + 1 if (stream.states[k] is not None and masks[t, k].sum() < LO) else 0
            missing = [k for k in range(K) if stream.states[k] is None or lost[k] >= LOST_N]
            if missing and (t - t0) % EVERY == 0:
                cand, sc = candidates(proc, rgb[t], task)
                info["detections"] += 1
                if len(cand):
                    P = probs(clfs[task], rgb[t], cand, K)
                    for k in missing:
                        ratio = np.array([min(1.0, float(m.sum()) / atyps[task][k]) for m in cand])
                        for i in np.argsort(-(np.log(P[:, k] + 1e-9) + betas[task] * np.log(ratio))):
                            if P[i, k] < PLATE or ratio[i] < 0.4:
                                continue
                            m = cand[i]
                            c = ff_lib.centroid(m)
                            if np.hypot(*(c - e[t])) < 15:
                                continue  # at the gripper
                            if stream.states[k] is not None:
                                if last_c[k] is None or np.hypot(*(c - last_c[k])) > NEAR_LAST:
                                    continue  # a lost track is only re-started where its mask was last seen
                            else:
                                others = [init_c[j] for j in range(K) if j != k and init_c[j] is not None]
                                if others and min(float(np.hypot(*(c - o))) for o in others) > dmaxs[task] + NEAR_INIT:
                                    continue  # a first acquisition must be next to where another object was selected
                            if any(j != k and (m & masks[t, j]).sum() > 0.3 * min(m.sum(), max(1, masks[t, j].sum())) for j in range(K)):
                                continue  # another tracked object is there
                            if stream.states[k] is not None:
                                info["reacquired"].append([int(k), int(t), float(P[i, k])])
                            else:
                                info["acquired"][k], info["p_late"][k] = int(t), float(P[i, k])
                            masks[t, k] = stream.add(k, rgb[t], m)
                            lost[k] = 0
                            break
        raw_c, raw_v = centres_from_masks(masks)
        np.savez_compressed(out / f"{ep}.tmp.npz", centres=raw_c.astype(np.float32), visibility=raw_v, raw_centres=raw_c.astype(np.float32), raw_visibility=raw_v,
                            area=masks.reshape(n, K, -1).sum(-1).astype(np.int32), first=np.packbits(first, axis=-1), masks=np.packbits(masks, axis=-1),
                            events=json.dumps([]), info=json.dumps(info))
        (out / f"{ep}.tmp.npz").rename(out / f"{ep}.npz")
        print(f"{ep} frames {n} K {K} {json.dumps(info)} raw drawn {100 * raw_v.mean():.0f}% {(time.time() - start) / 60:.1f} min", flush=True)
    print("TRACK5_DONE", shard, flush=True)
