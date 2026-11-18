"""First-frame click points from the demonstrations: project the end effector at the first grasp of each object.

For every episode and tracked object k: the first gripper-close onset inside a subtask whose text names object k
(rma_spec.subtask_object); the end-effector position there is projected with the agentview camera matrix
(rma_camera.py; no row flip, verified against the hand-read pilot clicks). Objects do not move before their
first grasp, so this is where object k sits in frame 0.

Usage: rma_clicks.py LR_EPS CAMERA.json OUT.json
OUT: {"episodes": {ep: [[x, y] | null, ...]}, "tasks": {task: {"median": [[x, y], ...], "spread": [px, ...], "n": n}}}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rma_spec  # noqa: E402

eps_dir, camera, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
episodes, per_task = {}, {}
for path in sorted(glob.glob(str(eps_dir / "meta" / "*.npz"))):
    m = np.load(path)
    task = int(m["task_id"]); ep = Path(path).stem
    matrix = np.asarray(camera[str(task)]["matrix"])
    sub, order, texts = m["subtask"], list(m["subtask_order"]), [str(x) for x in m["subtask_texts"]]
    text_of = {int(s): t for s, t in zip(order, texts)}
    closed = m["actions"][:, 6] > 0
    points = [None] * len(rma_spec.TASKS[task]["objects"])
    grasp = [None] * len(points)
    for t in range(1, len(closed)):
        if closed[t] and not closed[t - 1]:
            k = rma_spec.subtask_object(task, text_of[int(sub[t])])
            if k is not None and points[k] is None:
                p = matrix @ np.append(m["states"][t, :3], 1.0)
                points[k] = [round(float(p[0] / p[2]), 1), round(float(p[1] / p[2]), 1)]
                grasp[k] = t
    # Refine: the object's footprint is where frame 0 differs from the frame after the object was carried
    # away (end of the subtask that follows the grasp's subtask), near the projected grasp point.
    front = np.load(eps_dir / "front" / f"{ep}.npy", mmap_mode="r")
    first = front[0].astype(np.int16)
    for k, t in enumerate(grasp):
        if t is None:
            continue
        later = [s for s in range(t, len(sub)) if sub[s] != sub[t]]
        if not later:
            continue
        nxt = later[0]
        end = next((s for s in range(nxt, len(sub)) if sub[s] != sub[nxt]), len(sub) - 1)
        diff = np.abs(first - front[end].astype(np.int16)).sum(axis=2) > 60
        x0, y0, r = int(points[k][0]), int(points[k][1]), 20
        bx, by = max(0, x0 - r), max(0, y0 - r)
        labels, count = ndimage.label(diff[by:y0 + r + 1, bx:x0 + r + 1])
        best = None
        for c in range(1, count + 1):  # the changed blob nearest the projected grasp point is the object
            ys, xs = np.nonzero(labels == c)
            if not 12 <= len(xs) <= 500:
                continue
            dist = np.min(np.hypot(xs + bx - points[k][0], ys + by - points[k][1]))
            if dist <= 10 and (best is None or dist < best[0]):
                best = (dist, xs.mean() + bx, ys.mean() + by)
        if best is not None:
            points[k] = points[k] + [round(float(best[1]), 1), round(float(best[2]), 1)]  # [proj_x, proj_y, ref_x, ref_y]
    episodes[ep] = points
    per_task.setdefault(task, []).append(points)
tasks = {}
for task, rows in sorted(per_task.items()):
    med, spread = [], []
    refined_med, n_refined = [], []
    for k in range(len(rma_spec.TASKS[task]["objects"])):
        pts = np.array([r[k][:2] for r in rows if r[k] is not None])
        ref = np.array([r[k][2:] for r in rows if r[k] is not None and len(r[k]) == 4])
        if len(pts) == 0:
            med.append(None); spread.append(None); refined_med.append(None); n_refined.append(0); continue
        c = np.median(pts, axis=0)
        med.append([round(float(c[0])), round(float(c[1]))])
        spread.append(round(float(np.percentile(np.linalg.norm(pts - c, axis=1), 90)), 1))
        n_refined.append(len(ref))
        refined_med.append([round(float(np.median(ref[:, 0]))), round(float(np.median(ref[:, 1])))] if len(ref) >= 20 else None)
    missing = sum(1 for r in rows for p in r if p is None)
    tasks[task] = {"median": med, "refined": refined_med, "n_refined": n_refined, "spread90": spread, "n": len(rows), "missing": missing}
    print(task, rma_spec.names(task), "proj", med, "refined", refined_med, "n_ref", n_refined, "spread90", spread, flush=True)
out.write_text(json.dumps({"episodes": episodes, "tasks": tasks}))
