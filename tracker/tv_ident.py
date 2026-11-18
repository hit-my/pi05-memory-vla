"""First-frame identity by appearance: SAM3 generic detection finds the small objects, a few template crops per object
type (taken from OTHER demonstrations) decide which detection is which named object.

Truth per episode: the gripper pixel at each object's first grasp (clicks26.json; refined blob centre when available);
the truth detection of an object is the detection nearest to that point (<= 12 px). Episodes where two objects map to the
same detection are ambiguous and skipped.
Usage: tv_ident.py PER_TASK_TEST [TASKS]   templates: 6 episodes per task (never among the test episodes)
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
from scipy.optimize import linear_sum_assignment  # noqa: E402
from tv_common import LR, ROOT  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

per_task = int(sys.argv[1])
only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 27))
PROMPTS = ["box", "bottle", "carton", "small object"]
S = 1008
model = build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False)
proc = Sam3Processor(model, confidence_threshold=0.15)
clicks = json.load(open("/pfs/pfs-7jnepv/lgd/pi05-abc/rma/clicks26.json"))["episodes"]
by_task = {}
for ep in sorted(clicks):
    by_task.setdefault(int(ep[1:3]), []).append(ep)


def detect(ep):
    """-> list of dict(centre [x,y] in 256 px, hist, mean colour, area)"""
    rgb = np.ascontiguousarray(np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")[0])
    big = np.asarray(Image.fromarray(rgb).resize((S, S), Image.BICUBIC))
    hsv = cv2.cvtColor(big, cv2.COLOR_RGB2HSV)
    found = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(Image.fromarray(big))
        for prompt in PROMPTS:
            o = proc.set_text_prompt(state=state, prompt=prompt)
            masks = o["masks"].float().cpu().numpy().reshape(-1, S, S) > 0.5
            for m, s in zip(masks, o["scores"].float().cpu().numpy()):
                area = m.sum() / (S / 256) ** 2
                if 20 <= area <= 900:
                    found.append((float(s), m))
    found.sort(key=lambda x: -x[0])
    keep = []
    for s, m in found:  # drop duplicates of the same object found under several prompts
        if all((m & k["mask"]).sum() / max((m | k["mask"]).sum(), 1) < 0.5 for k in keep):
            ys, xs = np.nonzero(m)
            hist = cv2.calcHist([hsv], [0, 1, 2], m.astype(np.uint8), [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            keep.append(dict(mask=m, score=s, centre=[xs.mean() * 256 / S, ys.mean() * 256 / S], hist=hist / max(hist.sum(), 1),
                             area=float(m.sum() / (S / 256) ** 2)))
    return keep


def truth_points(ep):
    return [None if p is None else (p[2:] if len(p) == 4 else p[:2]) for p in clicks[ep]]


def truth_detection(dets, point):
    if point is None or not dets:
        return None
    d = [np.hypot(k["centre"][0] - point[0], k["centre"][1] - point[1]) for k in dets]
    return int(np.argmin(d)) if min(d) <= 12 else None


# ---- templates from the last 6 episodes of every task (the test set uses the first ones)
templates = {}
for task, eps in sorted(by_task.items()):
    for ep in eps[-6:]:
        dets = detect(ep)
        for k, (noun, _, _) in enumerate(rma_spec.TASKS[task]["objects"]):
            i = truth_detection(dets, truth_points(ep)[k])
            if i is not None:
                templates.setdefault(noun, []).append((dets[i]["hist"], dets[i]["area"]))
print("templates per object type:", {n: len(v) for n, v in templates.items()}, flush=True)


def similarity(det, noun):
    best = 0.0
    for hist, area in templates.get(noun, []):
        s = float(np.minimum(det["hist"], hist).sum())  # histogram intersection, 0..1
        s *= min(det["area"], area) / max(det["area"], area) ** 0.5 / min(det["area"], area) ** 0.5  # mild size agreement
        best = max(best, s)
    return best


rows = []
for task in only:
    eps = by_task[task][:-6]
    eps = eps[::max(1, len(eps) // per_task)][:per_task]
    stat = dict(n=0, amb=0, miss=0, app=0, fixed=0, objs=0, app_obj=0, fixed_obj=0, cands=[])
    for ep in eps:
        dets = detect(ep)
        nouns = [o[0] for o in rma_spec.TASKS[task]["objects"]]
        tr = [truth_detection(dets, p) for p in truth_points(ep)]
        if len(sys.argv) > 3 and eps.index(ep) < 4:  # debug picture: detections (numbered), truth points (x), fixed clicks (+)
            rgb0 = np.ascontiguousarray(np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")[0])
            pts = [p for p in truth_points(ep) if p is not None]
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            x0, y0 = int(np.clip(cx - 40, 0, 176)), int(np.clip(cy - 40, 0, 176))
            z = 10
            im = cv2.resize(rgb0[y0:y0 + 80, x0:x0 + 80, ::-1], (80 * z, 80 * z), interpolation=cv2.INTER_CUBIC)
            for i, dd in enumerate(dets):
                m = cv2.resize(dd["mask"].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)[y0:y0 + 80, x0:x0 + 80]
                cs, _ = cv2.findContours(cv2.resize(m, (80 * z, 80 * z), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, (0, 255, 255), 1)
                px, py = int((dd["centre"][0] - x0) * z), int((dd["centre"][1] - y0) * z)
                if 0 <= px < 80 * z and 0 <= py < 80 * z:
                    cv2.putText(im, str(i), (px - 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cols = [(255, 120, 0), (0, 140, 255), (200, 0, 200)]
            for k2, p in enumerate(truth_points(ep)):
                if p is not None:
                    cv2.drawMarker(im, (int((p[0] - x0) * z), int((p[1] - y0) * z)), cols[k2], cv2.MARKER_TILTED_CROSS, 22, 3)
                q = rma_spec.CLICKS[task][k2]
                cv2.drawMarker(im, (int((q[0] - x0) * z), int((q[1] - y0) * z)), cols[k2], cv2.MARKER_CROSS, 22, 2)
            cv2.putText(im, f"{ep} {nouns} x=truth +=fixed click", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(ROOT / "init" / f"dbg_{ep}.jpg"), im)
        if any(t is None for t in tr):
            stat["miss"] += 1
            continue
        if len(set(tr)) < len(tr):
            stat["amb"] += 1
            continue
        cost = np.array([[-similarity(d, n) for d in dets] for n in nouns])
        r, c = linear_sum_assignment(cost)
        pred = dict(zip(r.tolist(), c.tolist()))
        fixed = [truth_detection(dets, list(p)) for p in rma_spec.CLICKS[task]]
        stat["n"] += 1
        stat["cands"].append(len(dets))
        ok_app = [pred.get(k) == tr[k] for k in range(len(nouns))]
        ok_fix = [fixed[k] == tr[k] for k in range(len(nouns))]
        stat["app"] += all(ok_app)
        stat["fixed"] += all(ok_fix)
        stat["objs"] += len(nouns)
        stat["app_obj"] += sum(ok_app)
        stat["fixed_obj"] += sum(ok_fix)
    rows.append(dict(task=task, **{k: v for k, v in stat.items() if k != "cands"}, cands=float(np.mean(stat["cands"])) if stat["cands"] else 0))
    n = max(stat["n"], 1)
    print(f"task {task:2d} usable {stat['n']:3d} (no detection on an object {stat['miss']}, ambiguous truth {stat['amb']})  candidates/frame {rows[-1]['cands']:4.1f}  "
          f"ALL objects right: appearance {100 * stat['app'] / n:5.1f}%  fixed click {100 * stat['fixed'] / n:5.1f}%   per object: appearance {100 * stat['app_obj'] / max(stat['objs'], 1):5.1f}%  fixed {100 * stat['fixed_obj'] / max(stat['objs'], 1):5.1f}%", flush=True)
    (ROOT / "init" / "ident.json").write_text(json.dumps(rows))
tot = {k: sum(r[k] for r in rows) for k in ("n", "app", "fixed", "objs", "app_obj", "fixed_obj")}
print(f"ALL usable {tot['n']}: all objects right appearance {100 * tot['app'] / max(tot['n'], 1):.1f}%  fixed {100 * tot['fixed'] / max(tot['n'], 1):.1f}%")
