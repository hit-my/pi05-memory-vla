"""First-frame identity by appearance, v2.

Truth (reliable, from the demonstration itself): the FOOTPRINT of object k = pixels that change between the start of the
subtask in which k is first grasped and the end of the following subtask (k has been carried away), near the grasp, with the
projected gripper region removed. Objects do not move before their first grasp, so this is where k is in frame 0.

Method: colour model per object type (HSV histogram of footprint pixels in template demonstrations, never the test ones).
In a new first frame, SAM3 generic detection gives the candidate pixels (union of small detections); every candidate pixel
is assigned to the task's object type whose colour model explains it best (if better than the background model); the
object's seed point is the centre of its best connected blob. A seed inside the true footprint (3 px tolerance) is a hit.
Baseline: the fixed click of the current pipeline.
Usage: tv_ident2.py PER_TASK_TEST TASKS [debug]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, str(HERE / "src/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"))
sys.path.append(str(HERE / "pydeps"))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
import cv2  # noqa: E402
import rma_spec  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rma_fuse import FusedTracks  # noqa: E402
from tv_common import CAM, LR, ROOT  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

per_task = int(sys.argv[1])
only = [int(x) for x in sys.argv[2].split(",")]
debug = len(sys.argv) > 3
PROMPTS = ["box", "bottle", "carton", "small object"]
BINS = (18, 6, 6)
S = 1008
model = build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False)
proc = Sam3Processor(model, confidence_threshold=0.15)
metas = {}
for p in sorted((LR / "meta").glob("t*_seed*.npz")):
    metas.setdefault(int(p.stem[1:3]), []).append(p.stem)


def candidates(rgb):
    """union of SAM3 small generic detections, bool [256,256]"""
    union = np.zeros((256, 256), bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(Image.fromarray(rgb).resize((S, S), Image.BICUBIC))
        for prompt in PROMPTS:
            o = proc.set_text_prompt(state=state, prompt=prompt)
            masks = o["masks"].float().cpu().numpy().reshape(-1, S, S) > 0.5
            for m in masks:
                small = cv2.resize(m.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA) > 0
                if 20 <= small.sum() <= 900:
                    union |= small
    return union


def footprints(ep):
    """-> rgb0, [footprint mask or None per object], robot region at frame 0"""
    task = int(ep[1:3])
    m = np.load(LR / "meta" / f"{ep}.npz")
    rgb = np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")
    n = min(len(rgb), len(m["states"]))
    sub, closed = m["subtask"][:n], m["actions"][:n, 6] > 0
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    fuser = FusedTracks(CAM[str(task)]["matrix"])
    P = np.asarray(CAM[str(task)]["matrix"])
    k_objs = len(rma_spec.TASKS[task]["objects"])
    out = [None] * k_objs
    for t in range(1, n):
        if closed[t] and not closed[t - 1]:
            k = rma_spec.subtask_object(task, text[int(sub[t])])
            if k is None or out[k] is not None:
                continue
            a = int(np.nonzero(sub == sub[t])[0][0])
            later = [s for s in range(t, n) if sub[s] != sub[t]]
            if not later:
                continue
            nxt = later[0]
            b = next((s for s in range(nxt, n) if sub[s] != sub[nxt]), n) - 1
            diff = np.abs(rgb[a].astype(np.int16) - rgb[b].astype(np.int16)).sum(2) > 50
            for f in (a, b):
                r, _ = fuser.robot_region(m["states"][f])
                diff &= ~cv2.dilate(r.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            h = P @ np.append(m["states"][t, :3], 1.0)
            gx, gy = h[0] / h[2], h[1] / h[2]
            ys, xs = np.mgrid[0:256, 0:256]
            diff &= np.hypot(xs - gx, ys - gy) < 40
            num, lab, stats, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8), connectivity=8)
            comps = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= 12]
            if comps:
                best = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                out[k] = lab == best
    r0, _ = fuser.robot_region(m["states"][0])
    return np.ascontiguousarray(rgb[0]), out, r0


def hist_of(hsv, mask):
    h = cv2.calcHist([hsv], [0, 1, 2], mask.astype(np.uint8), list(BINS), [0, 180, 0, 256, 0, 256])
    return h


def bins_of(hsv):
    return (hsv[..., 0].astype(int) * BINS[0] // 180, hsv[..., 1].astype(int) * BINS[1] // 256, hsv[..., 2].astype(int) * BINS[2] // 256)


# ---- colour models from the last 6 demonstrations of every task
model_h, bg = {}, np.zeros(BINS, np.float64)
for task, eps in sorted(metas.items()):
    for ep in eps[-6:]:
        rgb0, fps, _ = footprints(ep)
        hsv = cv2.cvtColor(rgb0, cv2.COLOR_RGB2HSV)
        allfp = np.zeros((256, 256), bool)
        for k, fp in enumerate(fps):
            if fp is not None:
                noun = rma_spec.TASKS[task]["objects"][k][0]
                model_h[noun] = model_h.get(noun, 0) + hist_of(hsv, fp)
                allfp |= fp
        bg += hist_of(hsv, ~cv2.dilate(allfp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
model_h = {n: (h + 0.5) / (h + 0.5).sum() for n, h in model_h.items()}
bg = (bg + 0.5) / (bg + 0.5).sum()
print("colour models:", sorted(model_h), flush=True)

tot = dict(n=0, app=0, fix=0, objs=0, app_o=0, fix_o=0)
for task in only:
    eps = metas[task][:-6]
    eps = eps[::max(1, len(eps) // per_task)][:per_task]
    nouns = [o[0] for o in rma_spec.TASKS[task]["objects"]]
    st = dict(n=0, skip=0, app=0, fix=0, objs=0, app_o=0, fix_o=0)
    for ei, ep in enumerate(eps):
        rgb0, fps, r0 = footprints(ep)
        if any(fp is None for fp in fps):
            st["skip"] += 1
            continue
        hsv = cv2.cvtColor(rgb0, cv2.COLOR_RGB2HSV)
        cand = candidates(rgb0) & ~r0
        bi = bins_of(hsv)
        like = np.stack([model_h[nn][bi] for nn in nouns])  # [K,256,256]
        owner = like.argmax(0)
        good = cand & (like.max(0) > 2.0 * bg[bi])
        seeds = []
        for k in range(len(nouns)):
            mk = (good & (owner == k)).astype(np.uint8)
            mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            num, lab, stats, cent = cv2.connectedComponentsWithStats(mk, connectivity=8)
            if num <= 1:
                seeds.append(None)
                continue
            best = max(range(1, num), key=lambda i: float(like[k][lab == i].sum()))
            seeds.append(cent[best])
        tol = [cv2.dilate(fp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) for fp in fps]
        hit = lambda p, k: p is not None and tol[k][int(np.clip(round(p[1]), 0, 255)), int(np.clip(round(p[0]), 0, 255))]
        oa = [bool(hit(seeds[k], k)) for k in range(len(nouns))]
        of = [bool(hit(rma_spec.CLICKS[task][k], k)) for k in range(len(nouns))]
        st["n"] += 1; st["app"] += all(oa); st["fix"] += all(of); st["objs"] += len(nouns); st["app_o"] += sum(oa); st["fix_o"] += sum(of)
        if debug and ei < 4:
            z = 4
            im = cv2.resize(rgb0[..., ::-1], (256 * z, 256 * z), interpolation=cv2.INTER_CUBIC)
            cols = [(255, 120, 0), (0, 140, 255), (200, 0, 200)]
            for k, fp in enumerate(fps):
                cs, _ = cv2.findContours(cv2.resize(fp.astype(np.uint8), (256 * z, 256 * z), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, cols[k], 2)
                if seeds[k] is not None:
                    cv2.circle(im, (int(seeds[k][0] * z), int(seeds[k][1] * z)), 7, cols[k], -1)
                    cv2.circle(im, (int(seeds[k][0] * z), int(seeds[k][1] * z)), 7, (255, 255, 255), 1)
                q = rma_spec.CLICKS[task][k]
                cv2.drawMarker(im, (int(q[0] * z), int(q[1] * z)), cols[k], cv2.MARKER_CROSS, 18, 2)
            ys, xs = np.nonzero(np.any(fps, axis=0))
            x0, y0 = max(0, int(xs.mean() * z) - 200), max(0, int(ys.mean() * z) - 200)
            crop = im[y0:y0 + 400, x0:x0 + 400]
            cv2.putText(crop, f"{ep} outline=true object, dot=appearance, +=fixed click", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imwrite(str(ROOT / "init" / f"id2_{ep}.jpg"), crop)
    n = max(st["n"], 1)
    print(f"task {task:2d} {nouns} usable {st['n']} (skipped {st['skip']})  ALL objects right: appearance {100 * st['app'] / n:5.1f}%  fixed click {100 * st['fix'] / n:5.1f}%   "
          f"per object: appearance {100 * st['app_o'] / max(st['objs'], 1):5.1f}%  fixed click {100 * st['fix_o'] / max(st['objs'], 1):5.1f}%", flush=True)
    for a, b in (("n", "n"), ("app", "app"), ("fix", "fix"), ("objs", "objs"), ("app_o", "app_o"), ("fix_o", "fix_o")):
        tot[a] += st[b]
print(f"ALL usable {tot['n']}: all objects right appearance {100 * tot['app'] / max(tot['n'], 1):.1f}%  fixed click {100 * tot['fix'] / max(tot['n'], 1):.1f}%  "
      f"per object appearance {100 * tot['app_o'] / max(tot['objs'], 1):.1f}%  fixed {100 * tot['fix_o'] / max(tot['objs'], 1):.1f}%")
