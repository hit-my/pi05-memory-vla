"""Risk audit on the unbiased sample (eps104.json: 4 fixed episodes per task): current CUTIE vs SAM3 vs the chosen
"SAM3 seen-only + gripper veto". Per task: strict error frames, share of frames with a drawn point, share of carried
frames with a point, longest gap without a point, number of re-acquisitions."""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TV_EPS"] = "eps104.json"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
from tv_common import LR, ROOT, centres_from_masks, eef_pixels, episodes, load, load_masks, task_of  # noqa: E402
from tv_fuse import observe_only  # noqa: E402
from tv_render import robot_masks, strict  # noqa: E402

rows = []
for ep in episodes():
    if not (ROOT / "out/sam3" / f"{ep}.npz").exists() or not (ROOT / "out/cutie" / f"{ep}.npz").exists():
        continue
    task = task_of(ep)
    rgb, states, actions = load(ep)
    n = len(states)
    e = eef_pixels(ep, states)
    cmd = actions[:, 6] > 0
    robot = robot_masks(ep, states)
    m = np.load(LR / "meta" / f"{ep}.npz")
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    sub = m["subtask"][:n]
    base = load_masks("cutie", ep)[0][:n]
    k = base.shape[1]
    held = np.full(n, -1)
    cur = None
    for t in range(1, n):
        if cmd[t] and not cmd[t - 1] and cur is None:
            cur = rma_spec.subtask_object(task, text[int(sub[t])])
        if cur is not None:
            held[t] = cur
            if not cmd[t]:
                cur = None
    row = dict(ep=ep, task=task, frames=n, k=k)
    sam = load_masks("sam3", ep)[0][:n]
    oc, ov, ev = observe_only(ep, sam, robot, rgb, states, actions)
    cc, cv, _ = observe_only(ep, base, robot, rgb, states, actions)
    for name, (c, v) in (("cutie", centres_from_masks(base)), ("sam3", centres_from_masks(sam)), ("veto", (oc, ov)), ("vetoc", (cc, cv))):
        s = strict(c, v, e, cmd)
        carried = sum(int((held == j).sum()) for j in range(k))
        carried_seen = sum(int(((held == j) & v[:, j]).sum()) for j in range(k))
        gap = 0
        for j in range(k):
            run = 0
            for x in v[:, j]:
                run = 0 if x else run + 1
                gap = max(gap, run)
        row[name] = dict(A=s["A"], B=s["B"], C=s["C"], D=s["D"], vis=float(v.mean()), carried=carried, carried_seen=carried_seen, gap=gap)
    row["reacq"] = sum(1 for x in ev if x[1] == "reacquired")
    row["grasps"] = sum(1 for x in ev if x[1] == "grasp")
    row["start_missing"] = int((~centres_from_masks(sam[:1])[1][0]).sum())
    rows.append(row)
(ROOT / "all" / "audit104.json").write_text(json.dumps(rows))
print(len(rows), "episodes")
print("task | CUTIE A B D | SAM3 A B D | veto A B D | drawn% cutie/sam3/veto | carried frames with a point % sam3/veto | longest gap (veto) | re-acquired | objects without a first-frame mask")
T = {}
for r in rows:
    T.setdefault(r["task"], []).append(r)
def line(name, rs):
    f = lambda m, key: sum(r[m][key] for r in rs)
    cs = lambda m: 100 * f(m, "carried_seen") / max(f(m, "carried"), 1)
    return (f"{name:>4} | {f('cutie','A'):5d} {f('cutie','B'):5d} {f('cutie','D'):5d} | {f('sam3','A'):4d} {f('sam3','B'):4d} {f('sam3','D'):4d} | {f('veto','A'):4d} {f('veto','B'):4d} {f('veto','D'):4d} | "
            f"{100*np.mean([r['cutie']['vis'] for r in rs]):3.0f} {100*np.mean([r['sam3']['vis'] for r in rs]):3.0f} {100*np.mean([r['veto']['vis'] for r in rs]):3.0f} | {cs('sam3'):5.1f} {cs('veto'):5.1f} | "
            f"{max(r['veto']['gap'] for r in rs):5d} | {sum(r['reacq'] for r in rs):3d} | {sum(r['start_missing'] for r in rs)}")
for task, rs in sorted(T.items()):
    print(line(str(task), rs))
print(line("ALL", rows))
print("CUTIE + the same veto (A B D | drawn% | carried with a point%):")
for task, rs in sorted(T.items()) + [("ALL", rows)]:
    f = lambda key: sum(r["vetoc"][key] for r in rs)
    print(f"{str(task):>4} | {f('A'):4d} {f('B'):4d} {f('D'):4d} | {100*np.mean([r['vetoc']['vis'] for r in rs]):3.0f} | {100*f('carried_seen')/max(f('carried'),1):5.1f}")
bad = sorted(rows, key=lambda r: -(r["veto"]["A"] + r["veto"]["B"] + r["veto"]["D"]))[:8]
print("worst episodes for the chosen method:", [(r["ep"], r["veto"]["A"], r["veto"]["B"], r["veto"]["D"]) for r in bad])
