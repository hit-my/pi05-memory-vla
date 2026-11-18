"""For one episode: where does an accepted point travel while the tracker does not consider the object held, and why was it accepted?"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
from tv_common import LR, eef_pixels, load  # noqa: E402

ep = sys.argv[1]
z = np.load(HERE / "all/final" / f"{ep}.npz")
rgb, states, actions = load(ep)
c, v, area = z["centres"], z["visibility"], z["area"]
events = json.loads(str(z["events"]))
n, k = v.shape
e = eef_pixels(ep, states)
m = np.load(LR / "meta" / f"{ep}.npz")
text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
sub = m["subtask"][:n]
print("events", events)
print("subtasks", [(int(np.argmax(sub == s)), text[int(s)]) for s in m["subtask_order"]])
held = np.full(n, -1)
cur = -1
ev = {t: (kind, j) for t, kind, j in events if kind in ("grasp", "release")}
for t in range(n):
    if t in ev:
        cur = ev[t][1] if ev[t][0] == "grasp" else -1
    held[t] = cur
reacq = {(t, j) for t, kind, j in events if kind == "reacquired"}
for j in range(k):
    print(f"--- object {j}: first-frame area {int(area[0, j])}")
    last = None
    seg = []
    for t in range(n):
        if not v[t, j]:
            continue
        if last is not None and held[t] != j:
            step = float(np.hypot(*(c[t, j] - c[last, j])))
            if step > 3 or (t, j) in reacq:
                seg.append((t, last, round(step, 1), [round(float(x)) for x in c[t, j]], round(float(np.hypot(*(c[t, j] - e[t]))), 0), int(area[t, j]),
                            "REACQ" if (t, j) in reacq else "", "closed" if actions[max(t - 1, 0), 6] > 0 else "open"))
        last = t
    print("moves while not held (frame, previous drawn frame, step px, position, dist to gripper px, area, tag, gripper):")
    for s in seg[:60]:
        print("  ", s)
    print("   total such moves", len(seg))
