"""Self-check for drawn points that sit where the object cannot be. Independent reference from the subtask labels:
before its first grasp an object rests at its first-frame position; after a release it rests where the gripper let go.
A drawn point of a NOT-carried object farther than TOL px from that resting place is a ghost.
Usage: ghost_check.py [EP_FOR_DETAIL]"""
import os
import sys
from collections import defaultdict
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

TOL = 30
detail = sys.argv[1] if len(sys.argv) > 1 else None
per_task = defaultdict(lambda: [0, 0, 0, 0])  # frames not carried & drawn, ghosts (veto), same for raw sam3
worst = []
for ep in episodes():
    task = task_of(ep)
    if not (ROOT / "out/sam3" / f"{ep}.npz").exists():
        continue
    rgb, states, actions = load(ep)
    n = len(states)
    e = eef_pixels(ep, states)
    cmd = actions[:, 6] > 0
    m = np.load(LR / "meta" / f"{ep}.npz")
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    sub = m["subtask"][:n]
    masks = load_masks("sam3", ep)[0][:n]
    raw_c, raw_v = centres_from_masks(masks)
    c, v, events = tv_fuse.observe_only(ep, masks, robot_masks(ep, states), rgb, states, actions)
    k = masks.shape[1]
    rest = np.tile(raw_c[0], (n, 1, 1))  # expected resting place per frame
    carried = np.zeros((n, k), bool)
    cur = None
    for t in range(1, n):
        rest[t] = rest[t - 1]
        if cmd[t] and not cmd[t - 1] and cur is None:
            cur = rma_spec.subtask_object(task, text[int(sub[t])])
        if cur is not None:
            carried[t, cur] = True
            rest[t, cur] = e[t]
            if not cmd[t]:
                cur = None
    ep_ghost = 0
    for j in range(k):
        settle = np.zeros(n, bool)  # 15 frames after a release the object may still be settling
        rel = np.nonzero(carried[:-1, j] & ~carried[1:, j])[0]
        for r in rel:
            settle[r:r + 15] = True
        free = ~carried[:, j] & ~settle & np.isfinite(rest[:, j, 0])
        for name, cc, vv, col in (("veto", c, v, 0), ("raw", raw_c, raw_v, 2)):
            drawn = free & vv[:, j]
            d = np.hypot(*(np.nan_to_num(cc[:, j]) - rest[:, j]).T)
            ghost = drawn & (d > TOL)
            per_task[task][col] += int(drawn.sum())
            per_task[task][col + 1] += int(ghost.sum())
            if name == "veto":
                ep_ghost += int(ghost.sum())
                if detail == ep and ghost.any():
                    g = np.nonzero(ghost)[0]
                    print("object", j, "ghost frames", len(g), "first", g[0], "last", g[-1], "drawn at", c[g[0], j].round(0), "expected", rest[g[0], j].round(0),
                          "raw sam3 centre there", raw_c[g[0], j].round(0), "mask px", int(masks[g[0], j].sum()))
    if detail == ep:
        print("events", events)
    worst.append((ep_ghost, ep))
if not detail:
    print("task | not-carried drawn frames | ghost frames veto (%) | ghost frames raw SAM3 (%)")
    T = [0, 0, 0, 0]
    for task in sorted(per_task):
        a = per_task[task]
        T = [x + y for x, y in zip(T, a)]
        print(f"{task:>3} | {a[0]:6d} | {a[1]:5d} ({100 * a[1] / max(a[0], 1):4.1f}%) | {a[3]:5d} ({100 * a[3] / max(a[2], 1):4.1f}%)")
    print(f"ALL | {T[0]:6d} | {T[1]:5d} ({100 * T[1] / max(T[0], 1):4.1f}%) | {T[3]:5d} ({100 * T[3] / max(T[2], 1):4.1f}%)")
    print("worst episodes:", sorted(worst, reverse=True)[:14])
