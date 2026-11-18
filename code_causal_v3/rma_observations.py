"""RoboMemArena adapter: what a check step observes, written into the checklist right when it becomes visible.

causal_labels.py is benchmark-agnostic; a benchmark supplies observations(task_id, items) -> [(k, text)]: the
outcome of item k (0-based), visible in the camera at the last frame of item k. causal_labels inserts it as its
own item right after k, revealed and ticked at that frame, so it is written exactly when it is seen and never
before.

Drawer checks (tasks 4 and 5): at the end of a check-round "open the X drawer" the open drawer's content is
visible in the front camera (a small object inside, or nothing; verified on demo frames, r84_zoom.py). Task 4 puts
butter into "the drawer that already contains an object" and task 5 into "the empty drawer", so the drawer the
demonstrator finally opens again tells which drawer held what; that is what the camera showed when it was open.
"""
import re

LEVELS = ("top", "middle", "bottom")
_AGAIN = re.compile(r"^open the (top|middle|bottom) drawer again$")
_CHECK = re.compile(r"^open the (top|middle|bottom) drawer$")
OBSERVED = {"has": "has object", "empty": "empty"}


def observations(task_id, items):
    if task_id not in (4, 5):
        return []
    answers = [m.group(1) for m in map(_AGAIN.match, items) if m]
    if len(answers) != 1:
        raise ValueError(f"task {task_id}: expected one 'open the X drawer again', got {items}")
    answer = answers[0]
    out = []
    for k, text in enumerate(items):
        m = _CHECK.match(text)
        if not m:
            continue
        level = m.group(1)
        has = (level == answer) if task_id == 4 else (level != answer)
        out.append((k, f"{level} drawer: {OBSERVED['has' if has else 'empty']}"))
    if len(out) != 3:
        raise ValueError(f"task {task_id}: expected three check-round opens, got {items}")
    return out
