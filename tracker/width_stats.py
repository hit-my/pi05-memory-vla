"""Finger opening N frames after each gripper-close command, split by what the subtask handles (all 2600 demos)."""
import glob, re, numpy as np
obj, other = {4: [], 8: [], 12: []}, {4: [], 8: [], 12: []}
dur_obj, dur_other = [], []
for f in sorted(glob.glob("xxxx/rma/lr_eps/meta/t*_seed*.npz")):
    m = np.load(f); s, a, sub = m["states"], m["actions"], m["subtask"]
    text = {int(k): str(v) for k, v in zip(m["subtask_order"], m["subtask_texts"])}
    closed = a[:, 6] > 0
    for t in range(1, len(closed) - 13):
        if closed[t] and not closed[t - 1]:
            tx = text[int(sub[t])]
            handle = bool(re.search(r"drawer|door|microwave", tx)) and not tx.startswith(("pick", "place"))
            end = next((u for u in range(t, len(closed)) if not closed[u]), len(closed))
            for d in (4, 8, 12):
                if t + 1 + d < end:
                    (other if handle else obj)[d].append(s[t + 1 + d, 6])
            (dur_other if handle else dur_obj).append(end - t)
for name, g in (("object grasps", obj), ("handle / door grasps", other)):
    for d in (4, 8, 12):
        v = np.array(g[d]); print(f"{name:22s} +{d:2d} frames n={len(v):5d}  min {v.min():.4f}  p1 {np.percentile(v,1):.4f}  p50 {np.median(v):.4f}  p99 {np.percentile(v,99):.4f}  max {v.max():.4f}")
print("closed duration: objects p1/p50", np.percentile(dur_obj, [1, 50]), " handles p1/p50", np.percentile(dur_other, [1, 50]))
