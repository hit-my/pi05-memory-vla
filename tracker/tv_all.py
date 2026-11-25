"""All 2600 demos: current tracks (dataset centres) vs fused tracks, scored against a label-based reference.

Reference (uses the subtask annotation, which the online tracker never sees): object k rests at its frame-0 position;
from the gripper-close inside a subtask that names k until the next gripper-open it moves rigidly with the gripper pixel.
Scores per method: share of its visible frames farther than 15 px from the reference; share of grasps credited to the
right object. Usage: tv_all.py SHARD NSHARDS  ->  xxxx/rma/trackv/all/part_SHARD.json, tracks in all/fuse/<ep>.npz
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
from tv_common import DS, LR, ROOT, eef_pixels, project, task_of  # noqa: E402
from tv_fuse import fuse_centres  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])


def STATE_KW(states):
    """TV_STATE=thr,settle: derive "gripper closing" from the finger opening alone (no commanded action needed)."""
    if not os.environ.get("TV_STATE"):
        return {}
    thr, settle = os.environ["TV_STATE"].split(",")
    return dict(closed=states[:, 6] < float(thr), settle=int(settle))
out = ROOT / "all"
(out / "fuse").mkdir(parents=True, exist_ok=True)


def robot_masks(ep, states):
    n = len(states)
    res = np.zeros((n, 256, 256), bool)
    for t in range(n):
        pos = states[t, :3]
        R, _ = cv2.Rodrigues(states[t, 3:6])
        pts = project(ep, np.stack([pos + 0.03 * R[:, 2], pos - 0.24 * R[:, 2], pos + 0.05 * R[:, 0], pos + 0.05 * R[:, 1], pos]))
        radius = int(np.clip(max(np.hypot(*(pts[2] - pts[4])), np.hypot(*(pts[3] - pts[4]))), 6, 30))
        canvas = np.zeros((256, 256), np.uint8)
        if np.all(np.isfinite(pts[:2])) and np.all(np.abs(pts[:2]) < 2000):
            cv2.line(canvas, tuple(int(round(v)) for v in pts[0]), tuple(int(round(v)) for v in pts[1]), 1, thickness=2 * radius)
        res[t] = canvas > 0
    return res


def reference(task, m, start, e):
    n, k = len(e), len(start)
    sub = m["subtask"][:n]
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    closed = m["actions"][:n, 6] > 0
    ref = np.tile(np.asarray(start, float), (n, 1, 1))
    held_ref = np.full(n, -1)
    held, off = None, None
    for t in range(1, n):
        if closed[t] and not closed[t - 1] and held is None:
            j = rma_spec.subtask_object(task, text[int(sub[t])])
            if j is not None and np.all(np.isfinite(ref[t - 1, j])):
                held, off = j, ref[t - 1, j] - e[t]
        ref[t] = ref[t - 1]
        if held is not None:
            ref[t, held] = e[t] + off
            held_ref[t] = held
            if not closed[t]:
                held = None
    return ref, held_ref


files = sorted((LR / "meta").glob("t*_seed*.npz"))
rows = []
for i, path in enumerate(files):
    if i % shards != shard:
        continue
    ep = path.stem
    task = task_of(ep)
    m = np.load(path)
    z = np.load(DS / f"{ep}.npz")
    n = min(len(z["centres"]), len(m["states"]))
    states, actions = m["states"][:n].astype(float), m["actions"][:n].astype(float)
    rgb = np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")
    c, v = z["centres"][:n].astype(float), z["visibility"][:n].astype(bool)
    e = eef_pixels(ep, states)
    robot = robot_masks(ep, states)
    start = np.where(v[0][:, None], c[0], np.nan)
    fc, fv, events = fuse_centres(ep, np.where(v[..., None], c, np.nan), v, start, [None] * v.shape[1], robot, rgb, states, actions,
                                     grasp_px=float(os.environ.get("TV_GRASP", "45")), hide=os.environ.get("TV_HIDE", "0") == "1", **STATE_KW(states))
    ref, held_ref = reference(task, m, start, e)
    np.savez_compressed(out / "fuse" / f"{ep}.npz", centres=fc.astype(np.float32), visibility=fv, reference=ref.astype(np.float32))
    row = dict(ep=ep, task=task, frames=n)
    for name, (cc, vv) in (("cur", (c, v)), ("fuse", (fc, fv))):
        d = np.hypot(*(np.nan_to_num(cc) - np.nan_to_num(ref)).transpose(2, 0, 1))
        ok = vv & np.isfinite(ref[..., 0])
        row[name] = dict(vis=float(vv.mean()), bad=int((ok & (d > 15)).sum()), seen=int(ok.sum()),
                         held_bad=int((ok & (d > 15))[held_ref >= 0].sum()), held_seen=int(ok[held_ref >= 0].sum()))
    grasps_ref = sorted({(int(t), int(h)) for t, h in zip(np.nonzero(np.diff(held_ref, prepend=-1) != 0)[0], held_ref[np.nonzero(np.diff(held_ref, prepend=-1) != 0)[0]]) if h >= 0})
    got = [(t, j) for t, kind, j in events if kind == "grasp"]
    right = sum(any(abs(t - tr) <= 20 and j == jr for t, j in got) for tr, jr in grasps_ref)
    miss = []
    for tr, jr in grasps_ref:
        if not any(abs(t - tr) <= 20 and j == jr for t, j in got):
            d = [float(np.hypot(*(fc[tr - 1, j] - e[tr]))) if np.all(np.isfinite(fc[tr - 1, j])) else None for j in range(fc.shape[1])]
            miss.append(dict(t=tr, obj=jr, dist=d, vis=[bool(x) for x in fv[tr - 1]], width=float(states[min(tr + 12, n - 1), 6]),
                             start_ok=bool(np.isfinite(start[jr, 0])), got=[(t, j) for t, j in got if abs(t - tr) <= 20],
                             ev=[(t, kind, j) for t, kind, j in events if abs(t - tr) <= 20 or (kind == "hidden" and j == jr and t < tr)]))
    row["miss"] = miss
    row.update(grasps=len(grasps_ref), grasps_right=right, grasps_extra=len(got) - right,
               not_a_grasp=sum(kind == "not_a_grasp" for _, kind, _ in events))
    rows.append(row)
    if len(rows) % 20 == 0:
        print(shard, len(rows), ep, flush=True)
        (out / f"part_{shard}.json").write_text(json.dumps(rows))
(out / f"part_{shard}.json").write_text(json.dumps(rows))
print("ALL_DONE", shard, len(rows))
