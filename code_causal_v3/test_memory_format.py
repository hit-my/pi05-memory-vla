"""Offline checks of the memory formats and the ledger step. Run: python test_memory_format.py"""
from memory_format import (INITIAL_MEMORY, LAT, VERBS, action_entities, ledger_from_text, render_action,
                           render_memory_text, step_memory, verb_targets)
from ledger_v2 import render_ledger

ACTION = "blue bowl: up ; picked | orange bowl: still | purple bowl: occluded"
assert render_action(ACTION, "A") == ACTION
assert render_action(ACTION, "C") == f"blue bowl: up ; picked{LAT} | orange bowl: still{LAT} | purple bowl: occluded{LAT}"
assert render_action(ACTION, "B") == f"blue bowl:{LAT} | orange bowl:{LAT} | purple bowl:{LAT}"
for variant in "ABC":
    assert action_entities(render_action(ACTION, variant), variant) == ["blue bowl", "orange bowl", "purple bowl"]
assert action_entities(render_action(ACTION, "A"), "B") is None
assert action_entities(render_action(ACTION, "B"), "A") is None
assert action_entities("blue bowl: still | blue bowl: still", "A") is None
assert render_memory_text(INITIAL_MEMORY, "C") == INITIAL_MEMORY

rows = verb_targets(ACTION)
assert [VERBS[i] for i, v in enumerate(rows[0]) if v] == ["up", "picked"]
assert [VERBS[i] for i, v in enumerate(rows[2]) if v] == ["occluded"]
assert len(VERBS) == 15
arrow = verb_targets("blue bowl: up -> right ; put-in ; taken-out")[0]
assert [VERBS[i] for i, v in enumerate(arrow) if v] == ["up", "right", "put-in", "taken-out"]

for variant in "ABC":
    act = render_action("blue bowl: still", variant)
    init = f"Action:{act}\nSubgoal:init [blue bowl -> same plate; blue bowl -> same plate]\nTask:Incomplete"
    first = step_memory(INITIAL_MEMORY, init, variant, expected_total=2)
    assert first["memory_valid"] and first["certified_count"] == 0
    assert step_memory(INITIAL_MEMORY, init, variant, expected_total=7)["memory_valid"] is False
    assert step_memory(INITIAL_MEMORY, init, variant, eos_terminated=False)["memory_valid"] is False

    done1 = step_memory(first["next_memory_text"], f"Action:{act}\nSubgoal:done 1\nTask:Incomplete", variant, expected_total=2)
    assert done1["certified_count"] == 1 and "[x]1 blue bowl -> same plate; [ ]2" in done1["next_memory_text"]
    recovered = ledger_from_text(done1["next_memory_text"])
    assert render_ledger(recovered) == "\n".join(done1["next_memory_text"].split("\n")[1:])
    # Driver rejections: repeat tick, out-of-order tick, early Complete, re-init, tick before init.
    for sub, task in (("done 1", "Incomplete"), ("done 3", "Incomplete"), ("none", "Complete"),
                      ("init [a -> b]", "Incomplete")):
        bad = f"Action:{act}\nSubgoal:{sub}\nTask:{task}"
        assert step_memory(done1["next_memory_text"], bad, variant)["memory_valid"] is False, sub
    assert step_memory(INITIAL_MEMORY, f"Action:{act}\nSubgoal:done 1\nTask:Incomplete", variant)["memory_valid"] is False
    # A multi-index edit, and Task:Complete only when every item is ticked.
    full = step_memory(first["next_memory_text"], f"Action:{act}\nSubgoal:done 1,2\nTask:Complete", variant, expected_total=2)
    assert full["sequence_complete"] and full["certified_count"] == 2
    # All ticked may still be Incomplete: the physical final state is not reached yet.
    held = step_memory(first["next_memory_text"], f"Action:{act}\nSubgoal:done 1,2\nTask:Incomplete", variant, expected_total=2)
    assert held["memory_valid"] and not held["sequence_complete"]
print("memory_format: all checks passed")
