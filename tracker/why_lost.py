"""Why does the veto tracker drop points while an object is carried? Classify every carried frame without a point."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ["TV_EPS"] = "eps104.json"
HERE = Path("xxxx/rma/trackv")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
import tv_fuse  # noqa: E402
from tv_common import LR, ROOT, centres_from_masks, eef_pixels, episodes, load, load_masks, task_of  # noqa: E402
from tv_render import robot_masks  # noqa: E402

tasks = {int(x) for x in sys.argv[1].split(",")}
total = Counter()
for ep in episodes():
    task = task_of(ep)
    if task not in tasks or not (ROOT / "out/sam3" / f"{ep}.npz").exists():
        continue
    rgb, states, actions = load(ep)
    n = len(states)
    e = eef_pixels(ep, states)
    cmd = actions[:, 6] > 0
    m = np.load(LR / "meta" / f"{ep}.npz")
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    sub = m["subtask"][:n]
    masks = load_masks("sam3", ep)[0][:n]
    c_in, v_in = centres_from_masks(masks)
    held_true = np.full(n, -1)
    cur = None
    for t in range(1, n):
        if cmd[t] and not cmd[t - 1] and cur is None:
            cur = rma_spec.subtask_object(task, text[int(sub[t])])
        if cur is not None:
            held_true[t] = cur
            if not cmd[t]:
                cur = None
    c, v, events = tv_fuse.observe_only(ep, masks, robot_masks(ep, states), rgb, states, actions)
    grasp = {}  # frame -> object the tracker believes is held
    belief = np.full(n, -1)
    cur = -1
    ev = {t: (kind, j) for t, kind, j in events if kind in ("grasp", "release")}
    for t in range(n):
        if t in ev:
            cur = ev[t][1] if ev[t][0] == "grasp" else -1
        belief[t] = cur
    why = Counter()
    for j in range(masks.shape[1]):
        lost = (held_true == j) & ~v[:, j]
        for t in np.nonzero(lost)[0]:
            if not v_in[t, j]:
                why["SAM3 has no mask"] += 1
            elif belief[t] != j:
                d0 = np.hypot(*(c_in[t, j] - c_in[max(0, t - 1), j])) if v_in[max(0, t - 1), j] else -1
                why["tracker does not believe it is held (finger width %.3f)" % round(float(states[t, 6]), 3)] += 1
            else:
                why["held but too far from gripper edge"] += 1
    carried = int((held_true >= 0).sum())
    grouped = Counter()
    for k, val in why.items():
        grouped[k.split(" (")[0]] += val
    widths = sorted({k.split("width ")[1][:-1] for k in why if "width" in k})
    print(ep, "carried frames", carried, dict(grouped), "finger widths when not believed:", widths[:6], "grasp events", [(t, k, j) for t, k, j in events if k != "reacquired"][:8], flush=True)
    total.update(grouped)
print("TOTAL", dict(total))
