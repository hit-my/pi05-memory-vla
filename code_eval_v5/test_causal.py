"""Causal labels end to end: label -> step_memory (strict driver) and -> EvidenceGate (deployed path)."""
import random

import numpy as np

import ledger_v2
import memory_format as mf
from causal_labels import episode_labels, task_prior
from memory_format import INITIAL_MEMORY, step_memory
from rma_evidence import EvidenceGate

rng = random.Random(0)
checks = [f"{a} the {lvl} drawer" for lvl in ("top", "middle", "bottom") for a in ("open", "close")]
demos = []
for answer in ("top", "middle", "bottom"):
    items = checks + [f"open the {answer} drawer again", f"place the butter into the {answer} drawer",
                      f"close the {answer} drawer"]
    demos.append((items, sorted(rng.sample(range(25, 600), len(items)))))
prior = task_prior([d[0] for d in demos])
act = "blue bowl: still"

for items, ends in demos:
    n = ends[-1] + 1
    labels = episode_labels(prior, items, ends, n, [act] * n)
    # strict driver, round by round (every 20 frames), exactly as at test time
    memory = INITIAL_MEMORY
    for t in range(0, n, 20):
        previous, delta, now = labels[t]
        assert previous == memory, (t, previous, memory)
        r = step_memory(memory, f"Action:{act}\n{delta}", "A", expected_total=prior.length)
        assert r["memory_valid"], (t, delta)
        memory = r["next_memory_text"]
    # a tick of an unknown item is rejected by the strict driver
    bad = step_memory(labels[0][1] and mf.render_previous(act, labels[0][2]),
                      f"Action:{act}\nSubgoal:done 1,2,3,4,5,6,7\nTask:Incomplete", "A")
    assert bad["memory_valid"] is False

# deployed gate: init with "?" items, reveal later, ticks of "?" vetoed
gate = EvidenceGate(4, lambda text: None, np.eye(3, 4), mf, ledger_v2)
init = "Subgoal:init [" + "; ".join(checks + ["?", "?", "?"]) + "]\nTask:Incomplete"
memory, complete, info = gate.apply(f"Action:{act}\n{init}")
assert "[ ]7 ?" in memory and not complete
memory, complete, info = gate.apply(f"Action:{act}\nSubgoal:done 1,2,3,4,5,6,7\nTask:Incomplete")
assert info.get("vetoed_unknown") == 7 or info.get("vetoed"), info
memory, complete, info = gate.apply(f"Action:{act}\nSubgoal:none reveal 7=open the top drawer again\nTask:Incomplete")
assert "open the top drawer again" in memory and info["revealed"] == [7], (memory, info)
assert gate.tracker.kinds[6][0] == "articulated"
print("test_causal: all checks passed")
