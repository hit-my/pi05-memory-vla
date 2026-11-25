"""Checklist parsing / signal tests (CPU, no simulator):  test_checklist_v5.py [PER_TASK]
 A  exactness: per frame a memory text whose ledger has the demo's current subtask as its first open item (items ended
    before t ticked; rendered exactly as the Ledger Driver renders it, causal "?" items included) -> ChecklistSignalV5
    must give the same retirement flags and picks as LabelSignal on the subtask labels (= what built final_v5 / track5b).
 B  realistic cadence: the same memory refreshed only every 20 frames (as in the eval loop) -> no exception; the
    retirement frames are reported (they lag by < 20 frames + the skipped-item replay).
 C  robustness: garbage / None / truncated / wrong-length / regressing memories interleaved -> never raises, keeps state.
"""
import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from causal_labels import ledger_at, prefix_at, task_prior  # noqa: E402
from ledger_v2 import render_ledger  # noqa: E402
from rma_checklist_v5 import ChecklistSignalV5, LabelSignal, current_item  # noqa: E402

TV = Path("xxxx/rma/trackv")
per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 3
by = {}
for p in sorted((TV / "v5/feat5").glob("t*_seed*.npz")):
    if "tmp" not in p.name:
        by.setdefault(p.stem[:3], []).append(p.stem)
eps = []
for task, names in sorted(by.items()):
    rng = np.random.default_rng(5000 + int(task[1:]))
    eps += sorted(rng.choice(names, min(per_task, len(names)), replace=False).tolist())


def episode(ep):
    m = np.load(f"xxxx/rma/lr_eps/meta/{ep}.npz")
    sub = m["subtask"]
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    starts = [0] + [t for t in range(1, len(sub)) if sub[t] != sub[t - 1]]
    items = [text[int(sub[t])] for t in starts]
    lasts = [s - 1 for s in starts[1:]] + [len(sub) - 1]       # last frame of each item
    return m, sub, text, items, lasts


priors = {}
for task in sorted({int(e[1:3]) for e in eps}):
    lists = []
    for p in sorted((Path("xxxx/rma/lr_eps/meta")).glob(f"t{task:02d}_seed*.npz")):
        _, _, _, items, _ = episode(p.stem)
        lists.append(items)
    priors[task] = task_prior(lists)

bad = 0
for ep in eps:
    task = int(ep[1:3])
    m, sub, text, items, lasts = episode(ep)
    n = len(sub)
    actions, states = m["actions"], m["states"]
    lab, chk, chk20, chkg = LabelSignal(task), ChecklistSignalV5(task), ChecklistSignalV5(task), ChecklistSignalV5(task)
    r_lab, r_chk, r20 = np.zeros((n, 8), bool), np.zeros((n, 8), bool), np.zeros((n, 8), bool)
    p_lab, p_chk = [], []
    mem20 = "Action:none\nSubgoal:Not initialized.\nTask:Incomplete"
    rng = np.random.default_rng(7)
    garbage = [None, 17, "", "Action:x", "Subgoal:[ ]1", "Action:a\nSubgoal:[x]1 pick up the; [ ]2\nTask:Incomplete",
               "Action:a\nSubgoal:[x]1 a; [x]2 b; [x]3 c; [x]4 d; [x]5 e; [x]6 f; [x]7 g\nTask:Complete",
               "Action:a\nSubgoal:[ ]1 pick up the cookies\nTask:Incomplete", "\x00\xff"]
    errors = 0
    for t in range(n):
        cmd = None if t == 0 else float(actions[t - 1, 6])
        # ledger whose first open item is the demo's current subtask at t: items whose last frame is < t are ticked
        done = sum(1 for e in lasts if e < t)
        # causal "?" texts as the labeler writes them (started = items begun by t)
        led = ledger_at(priors[task], prefix_at(items, lasts, t))
        for it in led.items:
            it.done = it.index <= done
        led.task_status = "Incomplete"
        mem = "Action:none\n" + render_ledger(led)
        assert current_item(mem)[0] == done + 1, (ep, t, mem)
        a, pa = lab.update(text[int(sub[t])], cmd, states[t])
        b, pb = chk.update(mem, cmd, states[t])
        r_lab[t, a], r_chk[t, b] = True, True
        p_lab.append(pa); p_chk.append(pb)
        if t % 20 == 0:
            mem20 = mem
        c, _ = chk20.update(mem20, cmd, states[t])
        r20[t, c] = True
        try:
            chkg.update(garbage[rng.integers(len(garbage))] if rng.random() < 0.3 else mem20, cmd, states[t])
        except Exception:
            errors += 1
    first = lambda r: [int(np.argmax(r[:, j])) if r[:, j].any() else None for j in range(r.shape[1]) if r[:, j].any()]
    ok_a = bool(np.array_equal(r_lab, r_chk)) and p_lab == p_chk
    bad += not ok_a or errors > 0
    print("OK  " if ok_a and not errors else "FAIL", json.dumps(dict(ep=ep, frames=n, items=len(items), exact=ok_a,
          picks_equal=p_lab == p_chk, retired_labels_from=first(r_lab), retired_every20_from=first(r20),
          garbage_exceptions=errors, garbage_log=chkg.log[:3])), flush=True)
print("CHECKLIST_TEST failed", bad, "of", len(eps), flush=True)
