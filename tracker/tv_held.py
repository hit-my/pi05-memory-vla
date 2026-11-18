"""While an object is carried: does each visual tracker still see the object itself, and where?
Carried intervals come from the label-based reference (gripper-close in a subtask naming the object -> next open).
For every tracker: share of carried frames with a mask, and distance of the mask centre from (gripper pixel + grasp offset)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
from tv_common import LR, ROOT, centres_from_masks, eef_pixels, load, load_masks, task_of  # noqa: E402

methods = ["cutie", "sam3", "o_cutie", "o_sam3"]
from tv_fuse import observe_only  # noqa: E402
from tv_render import robot_masks  # noqa: E402
acc = {m: dict(n=0, seen=0, d=[]) for m in methods}
rest = {m: dict(n=0, seen=0, d=[]) for m in methods}
eps = [p.stem for p in sorted((ROOT / "out/sam3").glob("*.npz"))]
for ep in eps:
    task = task_of(ep)
    if task in (18, 19):  # first-frame identity is unreliable there
        continue
    rgb, states, actions = load(ep)
    m = np.load(LR / "meta" / f"{ep}.npz")
    n = len(states)
    e = eef_pixels(ep, states)
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    sub, closed = m["subtask"][:n], actions[:, 6] > 0
    base, _ = load_masks("cutie", ep)
    c0, _ = centres_from_masks(base[:1])
    k = base.shape[1]
    ref = np.tile(c0[0], (n, 1, 1))
    held = np.full(n, -1)
    cur, off = None, None
    for t in range(1, n):
        if closed[t] and not closed[t - 1] and cur is None:
            j = rma_spec.subtask_object(task, text[int(sub[t])])
            if j is not None and np.all(np.isfinite(ref[t - 1, j])):
                cur, off = j, ref[t - 1, j] - e[t]
        ref[t] = ref[t - 1]
        if cur is not None:
            ref[t, cur] = e[t] + off
            held[t] = cur
            if not closed[t]:
                cur = None
    for name in methods:
        f = ROOT / "out" / name.replace("o_", "") / f"{ep}.npz"
        if not f.exists():
            continue
        if name.startswith("o_"):
            c, v, _ = observe_only(ep, load_masks(name[2:], ep)[0][:n], robot_masks(ep, states), rgb, states, actions)
        else:
            c, v = centres_from_masks(load_masks(name, ep)[0][:n])
        for j in range(k):
            h = held == j
            # carried, and the object has actually left its start position (lifted)
            moving = h & (np.hypot(*(ref[:, j] - c0[0, j]).T) > 15)
            acc[name]["n"] += int(moving.sum())
            acc[name]["seen"] += int((moving & v[:, j]).sum())
            acc[name]["d"] += np.hypot(*(c[:, j] - ref[:, j]).T)[moving & v[:, j]].tolist()
            still = (held != j) & (np.arange(n) < (np.nonzero(h)[0][0] if h.any() else n))  # before its first grasp
            rest[name]["n"] += int(still.sum())
            rest[name]["seen"] += int((still & v[:, j]).sum())
            rest[name]["d"] += np.hypot(*(c[:, j] - c0[0, j]).T)[still & v[:, j]].tolist()
print(len(eps), "episodes (tasks 18/19 skipped)")
for title, A in (("WHILE CARRIED (object lifted off its start position)", acc), ("AT REST before its first grasp", rest)):
    print(title)
    for name in methods:
        a = A[name]
        d = np.array(a["d"])
        if a["n"]:
            print(f"  {name:8s} frames {a['n']:6d}  has a mask {100 * a['seen'] / a['n']:5.1f}%   of those: within 10px {100 * np.mean(d < 10):5.1f}%  within 20px {100 * np.mean(d < 20):5.1f}%  "
                  f"median {np.median(d):5.1f}px  p90 {np.percentile(d, 90):5.1f}px")
