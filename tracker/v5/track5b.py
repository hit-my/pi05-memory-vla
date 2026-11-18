"""v5 tracking for tasks 18/19, second pass = track5.py + two additions (causal, RGB + robot state + checklist item only).

track5.py (unchanged steps): selection at frame T0 by appearance, second look (task 18), late acquisition / re-acquisition every
EVERY frames, first look at the grasp ("pick up the <object>": gripper closed and moved G_MOVE px -> if the object's mask is not
at the gripper, or it has no track, it is re-started from the detection at the gripper).
Additions:
  carry check continued: after that first look the object named by "pick up ..." is watched while the gripper stays closed: if its
      mask is farther than G_FAR px from the gripper pixel for C_RUN consecutive frames (or it still has no track), the detection
      at the gripper is tried again (every C_EVERY frames, at most C_TRIES times).
  release check: R_WAIT frames after the gripper opened, one detection on the crop centred on the release pixel: a detection of
      the released object (probability >= R_PMIN, >= 0.4 x its usual area, within R_NEAR px of the release pixel, not at the
      gripper) that the object's mask does not overlap re-starts its track (the mask drifted off while it was carried).
  split check: while the object named by "pick up ..." is carried (same watch), if its mask has a part at the gripper and another
      part of >= LO px farther than S_FAR px from the gripper pixel for C_RUN consecutive frames, the track is re-started from
      the part at the gripper alone (the first-frame mask had covered a second object standing behind it; that one is then free
      to be acquired).
v5 keeps only the split check (CHECKS=split, the default): on the selection half carry / release made 1 / 4 episodes much worse
(they re-started tracks on background patches) while split repaired 3. Until a check fires the computation is the one of track5.py, so only the episodes where they fire (found by the
scan mode on the stored track5 masks) are tracked again.
  track5b.py scan SHARD NSHARDS TAG            -> v5/scan/<ep>.json for every episode in v5/track
  track5b.py run  SHARD NSHARDS TAG ep [ep...] -> v5/track_fix/<ep>.npz (same keys as v5/track)"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE / "v5"))
import rma_spec  # noqa: E402
from tv_common import centres_from_masks, load, task_of  # noqa: E402
from tv_common_v4 import eef_pixels  # noqa: E402
import ff_lib  # noqa: E402
import track5  # noqa: E402
from track5 import (EVERY, G_EDGE, G_FAR, G_MOVE, G_PLOST, G_PMIN, G_WAIT, LO, LOST_N, NEAR_INIT, NEAR_LAST, PLATE, RECHECK, T0, candidates, probs,  # noqa: E402
                    select_first)

C_RUN, C_EVERY, C_TRIES = 3, 5, 5                                   # carry check continued
R_WAIT, R_PMIN, R_NEAR, R_GRIP, R_OVER = 20, 0.8, 25.0, 15.0, 0.2   # release check
S_FAR = 15.0                                                        # split check
import os  # noqa: E402
CHECKS = set(os.environ.get("CHECKS", "split").split(","))  # active additions; v5 uses only "split" (carry / release were worse on the selection half)
YS, XS = np.mgrid[0:256, 0:256]


def mask_far(mk, e):
    """the mask (>= LO px) is farther than G_FAR px from the gripper pixel"""
    ys, xs = np.nonzero(mk)
    return len(xs) >= LO and float(np.min(np.hypot(xs - e[0], ys - e[1]))) > G_FAR


def split_part(mk, e):
    """the mask has a part at the gripper (within G_EDGE px of the gripper pixel, >= LO px) and another part (>= LO px) farther than
    S_FAR px from it: returns the part at the gripper (bool mask), else None"""
    import cv2
    if mk.sum() < 2 * LO:
        return None
    num, lab = cv2.connectedComponents(mk.astype(np.uint8), connectivity=8)
    if num < 3:
        return None
    near, far = None, False
    for i in range(1, num):
        m = lab == i
        if m.sum() < LO:
            continue
        d = float(np.min(np.hypot(XS[m] - e[0], YS[m] - e[1])))
        if d < G_EDGE and near is None:
            near = m
        elif d > S_FAR:
            far = True
    return near if (near is not None and far) else None


def gripper_candidate(P, cand, k, e, pth):
    """index of the most probable detection of object k that touches the gripper pixel, or None (as in track5.py)"""
    for i in np.argsort(-P[:, k]):
        if P[i, k] < pth:
            break
        m = cand[i]
        if float(np.min(np.hypot(XS[m] - e[0], YS[m] - e[1]))) < G_EDGE:
            return int(i)
    return None


def release_candidate(P, cand, k, e_rel, e_now, atyp):
    """index of a confident whole detection of object k next to the release pixel (and not at the gripper), or None"""
    for i in np.argsort(-P[:, k]):
        if P[i, k] < R_PMIN:
            break
        m = cand[i]
        c = ff_lib.centroid(m)
        if m.sum() >= 0.4 * atyp[k] and np.hypot(*(c - e_rel)) <= R_NEAR and np.hypot(*(c - e_now)) >= R_GRIP:
            return int(i)
    return None


def crop_at(p):
    cx, cy = int(np.clip(p[0] - 48, 0, 160)), int(np.clip(p[1] - 48, 0, 160))
    return (cx, cy, cx + 96, cy + 96)


def overlap(m, mk, frac):
    return (m & mk).sum() > frac * min(m.sum(), max(1, mk.sum()))


def episode_inputs(ep):
    task = task_of(ep)
    rgb, states, actions = load(ep)
    meta = np.load(f"/data/lgd/rma/lr_eps/meta/{ep}.npz")
    sub_text = {int(s): str(x).lower() for s, x in zip(meta["subtask_order"], meta["subtask_texts"])}
    picks = [None] * len(rgb)   # object named by the checklist item "pick up the <object>" at each frame
    for t in range(len(rgb)):
        s = sub_text[int(meta["subtask"][t])]
        picks[t] = rma_spec.subtask_object(task, s) if s.startswith("pick") else None
    return task, rgb, eef_pixels(ep, states), np.r_[False, actions[:-1, 6] > 0], picks


def track_episode(ep, stream, proc, C):
    """the whole episode with track5 + the two additions; returns masks, first, info"""
    task, rgb, e, closed, picks = episode_inputs(ep)
    n = len(rgb)
    K = len(rma_spec.TASKS[task]["objects"])
    clf, dmax, atyp, beta = C[task]
    t0 = T0[task]
    masks = np.zeros((n, K, 256, 256), bool)
    cand, sc = candidates(proc, rgb[t0], task)
    P = probs(clf, rgb[t0], cand, K)
    sel = select_first(P, cand, K, dmax, atyp, beta)
    first = np.stack([cand[sel[k]] if sel[k] >= 0 else np.zeros((256, 256), bool) for k in range(K)])
    info = dict(t0=t0, p_first=[float(P[sel[k], k]) if sel[k] >= 0 else None for k in range(K)], acquired=[t0 if sel[k] >= 0 else None for k in range(K)],
                p_late=[None] * K, n_cand=int(len(cand)), detections=1, grasp_fix=[], dropped=[], reacquired=[], recheck=[], carry_fix=[], release_fix=[], split_fix=[])
    pending, watch, rel = None, None, None
    lost = np.zeros(K, int)
    last_c = [ff_lib.centroid(first[k]) if sel[k] >= 0 else None for k in range(K)]
    init_c = [ff_lib.centroid(first[k]) if sel[k] >= 0 else None for k in range(K)]
    carried = None
    ms = stream.reset(rgb[t0], [first[k] for k in range(K)])
    for k in range(K):
        masks[t0, k] = ms[k]

    def restart(k, t, m, key, p):
        for j in range(K):
            if j != k and stream.states[j] is not None and overlap(m, masks[t, j], 0.3):
                stream.states[j] = None  # that track was on this object: it is looked for again
                masks[t, j] = False
                info["dropped"].append([int(j), int(t)])
        masks[t, k] = stream.add(k, rgb[t], m)
        info[key].append([int(k), int(t), float(p)])

    for t in range(t0 + 1, n):
        ms = stream.step(rgb[t])
        for k in range(K):
            masks[t, k] = ms[k]
        # ---- second look (task 18)
        if t == RECHECK.get(task):
            cand, sc = candidates(proc, rgb[t], task)
            info["detections"] += 1
            P = probs(clf, rgb[t], cand, K)
            sel2 = select_first(P, cand, K, dmax, atyp, beta)
            for k in range(K):
                if sel2[k] < 0:
                    continue
                m, mk = cand[sel2[k]], masks[t, k]
                if stream.states[k] is not None and mk.sum() >= LO and overlap(m, mk, 0.2):
                    continue
                restart(k, t, m, "recheck", P[sel2[k], k])
                init_c[k] = ff_lib.centroid(m)
        # ---- grasp: first look (track5) and continued carry check
        if closed[t] and not closed[t - 1]:
            pending, watch, rel = None, None, None
            if picks[t] is not None:
                pending = dict(k=picks[t], t=t, e=e[t].copy())
                carried = picks[t]
        if not closed[t]:
            pending, watch = None, None
        if pending is not None and t - pending["t"] >= G_WAIT and np.hypot(*(e[t] - pending["e"])) >= G_MOVE:
            k = pending["k"]
            mk = masks[t, k]
            if stream.states[k] is None:
                need, pth = True, G_PMIN
            elif mk.sum() < LO:
                need, pth = True, G_PLOST
            else:
                need, pth = mask_far(mk, e[t]), G_PMIN
            pending = None
            fixed = False
            if need:
                cand, sc = candidates(proc, rgb[t], task, crop=crop_at(e[t]))
                info["detections"] += 1
                if len(cand):
                    P = probs(clf, rgb[t], cand, K)
                    i = gripper_candidate(P, cand, k, e[t], pth)
                    if i is not None:
                        restart(k, t, cand[i], "grasp_fix", P[i, k])
                        fixed = True
            if not fixed:
                watch = dict(k=k, run=0, tries=0, last=t, srun=0)
        elif watch is not None:
            k = watch["k"]
            part = split_part(masks[t, k], e[t]) if stream.states[k] is not None else None
            watch["srun"] = watch["srun"] + 1 if part is not None else 0
            if watch["srun"] >= C_RUN and "split" in CHECKS:
                restart(k, t, part, "split_fix", 1.0)
                watch["srun"] = 0
            none = stream.states[k] is None
            watch["run"] = watch["run"] + 1 if mask_far(masks[t, k], e[t]) else 0
            if "carry" in CHECKS and (none or watch["run"] >= C_RUN) and watch["tries"] < C_TRIES and t - watch["last"] >= C_EVERY:
                watch["tries"] += 1; watch["last"] = t
                cand, sc = candidates(proc, rgb[t], task, crop=crop_at(e[t]))
                info["detections"] += 1
                if len(cand):
                    P = probs(clf, rgb[t], cand, K)
                    i = gripper_candidate(P, cand, k, e[t], G_PMIN)
                    if i is not None:
                        restart(k, t, cand[i], "carry_fix", P[i, k])
                        watch = None
        # ---- release check
        if not closed[t] and closed[t - 1] and carried is not None:
            rel = dict(k=carried, t=t, e=e[t].copy())
            carried = None
        if rel is not None and t - rel["t"] == R_WAIT and "release" in CHECKS:
            k = rel["k"]
            cand, sc = candidates(proc, rgb[t], task, crop=crop_at(rel["e"]))
            info["detections"] += 1
            if len(cand):
                P = probs(clf, rgb[t], cand, K)
                i = release_candidate(P, cand, k, rel["e"], e[t], atyp)
                if i is not None and not (masks[t, k].sum() >= LO and overlap(cand[i], masks[t, k], R_OVER)):
                    restart(k, t, cand[i], "release_fix", P[i, k])
            rel = None
        # ---- late acquisition / re-acquisition (track5)
        for k in range(K):
            if masks[t, k].sum() >= LO:
                last_c[k] = ff_lib.centroid(masks[t, k])
            lost[k] = lost[k] + 1 if (stream.states[k] is not None and masks[t, k].sum() < LO) else 0
        missing = [k for k in range(K) if stream.states[k] is None or lost[k] >= LOST_N]
        if missing and (t - t0) % EVERY == 0:
            cand, sc = candidates(proc, rgb[t], task)
            info["detections"] += 1
            if len(cand):
                P = probs(clf, rgb[t], cand, K)
                for k in missing:
                    ratio = np.array([min(1.0, float(m.sum()) / atyp[k]) for m in cand])
                    for i in np.argsort(-(np.log(P[:, k] + 1e-9) + beta * np.log(ratio))):
                        if P[i, k] < PLATE or ratio[i] < 0.4:
                            continue
                        m = cand[i]
                        c = ff_lib.centroid(m)
                        if np.hypot(*(c - e[t])) < 15:
                            continue
                        if stream.states[k] is not None:
                            if last_c[k] is None or np.hypot(*(c - last_c[k])) > NEAR_LAST:
                                continue
                        else:
                            others = [init_c[j] for j in range(K) if j != k and init_c[j] is not None]
                            if others and min(float(np.hypot(*(c - o))) for o in others) > dmax + NEAR_INIT:
                                continue
                        if any(j != k and overlap(m, masks[t, j], 0.3) for j in range(K)):
                            continue
                        if stream.states[k] is not None:
                            info["reacquired"].append([int(k), int(t), float(P[i, k])])
                        else:
                            info["acquired"][k], info["p_late"][k] = int(t), float(P[i, k])
                        masks[t, k] = stream.add(k, rgb[t], m)
                        lost[k] = 0
                        break
    return masks, first, info


def scan_episode(ep, proc, C):
    """would one of the two additions fire on the stored track5 masks? (the masks are those of track5 up to the first firing)"""
    task, rgb, e, closed, picks = episode_inputs(ep)
    z = np.load(HERE / "v5/track" / f"{ep}.npz")
    masks = np.unpackbits(z["masks"], axis=-1)[..., :256].astype(bool)
    info = json.loads(str(z["info"]))
    n, K = masks.shape[:2]
    clf, dmax, atyp, beta = C[task]
    fixed_at = {(int(k), int(t)) for k, t, _ in info.get("grasp_fix", [])}
    acquired = info["acquired"]
    reasons = []
    pending, watch, rel, carried = None, None, None, None
    # does object k have a track at frame t? (from the stored info: tracks start at T0 / late acquisition / fixes, end when dropped)
    ev = {k: [] for k in range(K)}
    for k in range(K):
        if info["p_first"][k] is not None:
            ev[k].append((T0[task], 1))
        if acquired[k] is not None:
            ev[k].append((int(acquired[k]), 1))
    for key in ("grasp_fix", "recheck", "reacquired"):
        for k, t, _ in info.get(key, []):
            ev[int(k)].append((int(t), 1))
    for k, t in info.get("dropped", []):
        ev[int(k)].append((int(t) - 0.5, 0))

    def no_track(k, t):
        last = [s for tt, s in sorted(ev[k]) if tt <= t]
        return not last or last[-1] == 0
    for t in range(T0[task] + 1, n):
        if closed[t] and not closed[t - 1]:
            pending, watch, rel = None, None, None
            if picks[t] is not None:
                pending = dict(k=picks[t], t=t, e=e[t].copy())
                carried = picks[t]
        if not closed[t]:
            pending, watch = None, None
        if pending is not None and t - pending["t"] >= G_WAIT and np.hypot(*(e[t] - pending["e"])) >= G_MOVE:
            k = pending["k"]
            pending = None
            if (k, t) not in fixed_at:
                watch = dict(k=k, run=0, last=t, srun=0)
        elif watch is not None:
            k = watch["k"]
            watch["srun"] = watch["srun"] + 1 if (not no_track(k, t) and split_part(masks[t, k], e[t]) is not None) else 0
            if watch["srun"] >= C_RUN and "split" in CHECKS:
                reasons.append(["split", int(k), int(t), "mask in two parts while carried"])
                break
            none = no_track(k, t)
            watch["run"] = watch["run"] + 1 if mask_far(masks[t, k], e[t]) else 0
            if "carry" in CHECKS and (none or watch["run"] >= C_RUN) and t - watch["last"] >= C_EVERY:
                reasons.append(["carry", int(k), int(t), "no track" if none else "mask far from the gripper"])
                break
        if not closed[t] and closed[t - 1] and carried is not None:
            rel = dict(k=carried, t=t, e=e[t].copy())
            carried = None
        if rel is not None and t - rel["t"] == R_WAIT and "release" in CHECKS:
            k = rel["k"]
            cand, sc = candidates(proc, rgb[t], task, crop=crop_at(rel["e"]))
            if len(cand):
                P = probs(clf, rgb[t], cand, K)
                i = release_candidate(P, cand, k, rel["e"], e[t], atyp)
                if i is not None and not (masks[t, k].sum() >= LO and overlap(cand[i], masks[t, k], R_OVER)):
                    reasons.append(["release", int(k), int(t), float(P[i, k])])
                    break
            rel = None
    return dict(ep=ep, trigger=bool(reasons), reasons=reasons)


if __name__ == "__main__":
    import det5
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    mode, shard, shards, TAG = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    C = {}
    for t in (18, 19):
        J = json.load(open(HERE / f"v5/ff_clf_{t}_{TAG}.json"))
        C[t] = (ff_lib.Softmax.from_json(J["clf"]), float(J["dmax"]), J["atyp"], float(J["beta"]))
    proc = Sam3Processor(build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False), confidence_threshold=det5.THR)
    if mode == "scan":
        out = HERE / ("v5/scan" if CHECKS != {"split"} else "v5/scan_split"); out.mkdir(exist_ok=True)
        eps = sorted(p.stem for p in (HERE / "v5/track").glob("t1?_seed*.npz") if "tmp" not in p.name)
        for i, ep in enumerate(eps):
            if i % shards != shard or (out / f"{ep}.json").exists():
                continue
            r = scan_episode(ep, proc, C)
            json.dump(r, open(out / f"{ep}.json", "w"))
            print("scan", ep, r["trigger"], r["reasons"], flush=True)
        print("SCAN_DONE", shard, flush=True)
    else:
        out = HERE / ("v5/track_fix" if CHECKS != {"split"} else "v5/track_split"); out.mkdir(exist_ok=True)
        claims = HERE / ("v5/claims_fix" if CHECKS != {"split"} else "v5/claims_split"); claims.mkdir(exist_ok=True)
        eps = sys.argv[5:]
        stream = track5.Stream5()
        for ep in eps[shard::shards] + eps:
            if (out / f"{ep}.npz").exists():
                continue
            try:
                (claims / ep).mkdir()
            except FileExistsError:
                continue
            start = time.time()
            masks, first, info = track_episode(ep, stream, proc, C)
            n, K = masks.shape[:2]
            raw_c, raw_v = centres_from_masks(masks)
            np.savez_compressed(out / f"{ep}.tmp.npz", centres=raw_c.astype(np.float32), visibility=raw_v, raw_centres=raw_c.astype(np.float32), raw_visibility=raw_v,
                                area=masks.reshape(n, K, -1).sum(-1).astype(np.int32), first=np.packbits(first, axis=-1), masks=np.packbits(masks, axis=-1),
                                events=json.dumps([]), info=json.dumps(info))
            (out / f"{ep}.tmp.npz").rename(out / f"{ep}.npz")
            print(f"{ep} frames {n} K {K} {json.dumps(info)} raw drawn {100 * raw_v.mean():.0f}% {(time.time() - start) / 60:.1f} min", flush=True)
        print("TRACK5B_DONE", shard, flush=True)
