"""Grasp attribution of the veto on every finished episode, judged by the demo's subtask labels (not seen by the tracker):
for each label grasp (gripper closes inside a subtask that names object j) which object did the veto consider held?"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402

LR = Path("/data/lgd/rma/lr_eps")
per = defaultdict(Counter)
bad = []
for f in sorted((HERE / "all/final").glob("t*_seed*.npz")):
    if "tmp" in f.name:
        continue
    ep = f.stem
    task = int(ep[1:3])
    z = np.load(f)
    events = json.loads(str(z["events"]))
    m = np.load(LR / "meta" / f"{ep}.npz")
    n = len(z["visibility"])
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    sub, cmd = m["subtask"][:n], m["actions"][:n, 6] > 0
    held = np.full(n, -1)
    cur = -1
    ev = {t: (kind, j) for t, kind, j in events if kind in ("grasp", "release")}
    for t in range(n):
        if t in ev:
            cur = ev[t][1] if ev[t][0] == "grasp" else -1
        held[t] = cur
    t = 1
    while t < n:
        if cmd[t] and not cmd[t - 1]:
            end = t
            while end < n and cmd[end]:
                end += 1
            j = rma_spec.subtask_object(task, text[int(sub[t])])
            seen = [int(x) for x in np.unique(held[t:end]) if x >= 0]
            if j is None:
                per[task]["handle_ok" if not seen else "handle_taken_as_object"] += 1
            elif seen == [j]:
                per[task]["right"] += 1
            elif not seen:
                per[task]["missed"] += 1
            else:
                per[task]["wrong_object"] += 1
                bad.append(ep)
            t = end
        else:
            t += 1
keys = ["right", "wrong_object", "missed", "handle_ok", "handle_taken_as_object"]
tot = Counter()
print("task | " + " | ".join(keys))
for task in sorted(per):
    tot.update(per[task])
    print(f"{task:>3} | " + " | ".join(f"{per[task][k]:5d}" for k in keys))
print("ALL | " + " | ".join(f"{tot[k]:5d}" for k in keys))
obj = tot["right"] + tot["wrong_object"] + tot["missed"]
print(f"object grasps {obj}: right {100 * tot['right'] / obj:.1f}%  wrong object {100 * tot['wrong_object'] / obj:.1f}%  missed {100 * tot['missed'] / obj:.1f}%")
print("episodes with a wrong-object grasp:", len(set(bad)))
