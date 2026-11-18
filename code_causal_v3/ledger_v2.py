"""Ledger driver v2 = claude_handoff/ledger_driver.py + multi-index `done k,l` edits.

Only change vs v1: `Subgoal:done 2,3` confirms several items in one round; for an
ordered ledger the indices must be strictly increasing and equal to the next
pending items. Rendering, Task rule and all other checks are unchanged.
Training labels and deployment must import render_ledger/parse_delta from here.

Causal extension (code_causal): an item whose text is not yet known is rendered "?" and named later with
`reveal k=text` appended to the Subgoal line ("Subgoal:done 6 reveal 7=open the top drawer again"). A ledger whose
length is not known ends in "; ..." and a reveal of item len+1 appends to it. A "?" item cannot be ticked.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Optional


UNKNOWN = "?"
MORE = "..."


@dataclass
class SubgoalItem:
    index: int
    text: Optional[str]  # None = not yet known (rendered "?")
    done: bool = False
    done_round: Optional[int] = None


@dataclass
class Ledger:
    ordered: bool = True
    initialized: bool = False
    items: list[SubgoalItem] = field(default_factory=list)
    task_status: str = "Incomplete"
    open_ended: bool = False  # more items may follow (rendered "; ...")

    def find(self, k):
        return next((it for it in self.items if it.index == k), None)

    def pending(self):
        return [it for it in self.items if not it.done]

    def all_done(self):
        return self.initialized and all(it.done for it in self.items)


def render_ledger(ledger: Ledger) -> str:
    if not ledger.initialized:
        sub = "Subgoal:Not initialized."
    else:
        parts = [f"[{'x' if it.done else ' '}]{it.index} {UNKNOWN if it.text is None else it.text}" for it in ledger.items]
        if ledger.open_ended:
            parts.append(MORE)
        sub = "Subgoal:" + "; ".join(parts)
    return f"{sub}\nTask:{ledger.task_status}"


TASK_STATES = ("Incomplete", "Complete")
_SUB_NONE = re.compile(r"^Subgoal:none$")
_SUB_DONE = re.compile(r"^Subgoal:done (\d+(?:,\d+)*)$")
_SUB_INIT = re.compile(r"^Subgoal:init \[(.+)\]$")
# v2 (correctable progress): "Subgoal:at k" = items 1..k are done and the rest are not; k may be lower than the
# ledger's count, so a wrongly ticked previous memory can be corrected.
_SUB_AT = re.compile(r"^Subgoal:at (\d+)$")
_TASK = re.compile(rf"^Task:({'|'.join(TASK_STATES)})$")


class InvalidDelta(Exception):
    pass


@dataclass
class Delta:
    kind: str
    task: str
    indices: Optional[list[int]] = None
    items: Optional[list[Optional[str]]] = None  # init: None = "?"
    open_ended: bool = False                     # init ended in "..."
    reveals: dict = field(default_factory=dict)  # k -> text


_REVEAL = re.compile(r"^(\d+)=([^=|\[\];]+)$")


def _parse_reveals(text):
    out = {}
    for part in text.split("|"):
        m = _REVEAL.match(part.strip())
        if not m or not m.group(2).strip() or m.group(2).strip() in (UNKNOWN, MORE):
            raise InvalidDelta(f"bad reveal: {part!r}")
        k = int(m.group(1))
        if k in out:
            raise InvalidDelta(f"item {k} revealed twice")
        out[k] = m.group(2).strip()
    return out


def parse_delta(text: str) -> Delta:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 2:
        raise InvalidDelta(f"need exactly two lines, got {len(lines)}")
    sub_line, task_line = lines
    m = _TASK.match(task_line)
    if not m:
        raise InvalidDelta(f"bad Task line: {task_line!r}")
    task = m.group(1)
    reveals = {}
    if " reveal " in sub_line:
        sub_line, rest = sub_line.split(" reveal ", 1)
        reveals = _parse_reveals(rest)
    if _SUB_NONE.match(sub_line):
        return Delta("none", task, reveals=reveals)
    if m := _SUB_DONE.match(sub_line):
        return Delta("done", task, indices=[int(x) for x in m.group(1).split(",")], reveals=reveals)
    if m := _SUB_AT.match(sub_line):
        return Delta("at", task, indices=[int(m.group(1))], reveals=reveals)
    if m := _SUB_INIT.match(sub_line):
        if reveals:
            raise InvalidDelta("init carries its items itself")
        items = [s.strip() for s in m.group(1).split(";") if s.strip()]
        open_ended = bool(items) and items[-1] == MORE
        if open_ended:
            items = items[:-1]
        if MORE in items:
            raise InvalidDelta("'...' only at the end of init")
        if not items and not open_ended:
            raise InvalidDelta("empty init list")
        return Delta("init", task, items=[None if s == UNKNOWN else s for s in items], open_ended=open_ended)
    raise InvalidDelta(f"bad Subgoal line: {sub_line!r}")


def apply_reveals(ledger: Ledger, reveals: dict) -> None:
    """In place. A reveal names a "?" item, or appends item len+1 to an open-ended ledger."""
    for k in sorted(reveals):
        item = ledger.find(k)
        if item is not None:
            if item.text is not None:
                raise InvalidDelta(f"item {k} is already known")
            item.text = reveals[k]
        elif ledger.open_ended and k == len(ledger.items) + 1:
            ledger.items.append(SubgoalItem(k, reveals[k]))
        else:
            raise InvalidDelta(f"no item {k} to reveal")


def apply_delta(ledger: Ledger, delta: Delta, round_idx: int) -> Ledger:
    new = copy.deepcopy(ledger)
    if delta.kind == "init":
        if new.initialized:
            raise InvalidDelta("ledger already initialized")
        new.items = [SubgoalItem(i + 1, t) for i, t in enumerate(delta.items)]
        new.open_ended = delta.open_ended
        new.initialized = True
    else:
        if not new.initialized:
            raise InvalidDelta("first round must init")
        apply_reveals(new, delta.reveals)
        if delta.kind == "at":
            k = delta.indices[0]
            if not 0 <= k <= len(new.items):
                raise InvalidDelta(f"at {k} outside 0..{len(new.items)}")
            for item in new.items:
                done = item.index <= k
                if done and not item.done:
                    item.done_round = round_idx
                item.done = done
        if delta.kind == "done":
            idx = delta.indices
            if len(set(idx)) != len(idx) or idx != sorted(idx):
                raise InvalidDelta(f"indices must be strictly increasing: {idx}")
            for k in idx:
                item = new.find(k)
                if item is None:
                    raise InvalidDelta(f"no item {k}")
                if item.done:
                    raise InvalidDelta(f"item {k} already done")
            if new.ordered:
                expected = [it.index for it in new.pending()[:len(idx)]]
                if idx != expected:
                    raise InvalidDelta(f"ordered ledger expects {expected}, got {idx}")
            for k in idx:
                item = new.find(k)
                item.done, item.done_round = True, round_idx
        if any(it.done and it.text is None for it in new.items):
            raise InvalidDelta("an unknown item cannot be ticked")
    if delta.task == "Complete" and not new.all_done():
        raise InvalidDelta("Task cannot be Complete with pending items")
    if delta.task == "Complete":
        new.open_ended = False
    new.task_status = delta.task
    return new


def ledger_from_count(items: list[str], confirmed: int, ordered=True) -> Ledger:
    led = Ledger(ordered=ordered, initialized=True,
                 items=[SubgoalItem(i + 1, t, done=i < confirmed) for i, t in enumerate(items)])
    led.task_status = "Complete" if confirmed == len(items) else "Incomplete"
    return led


def delta_text(prev_confirmed: Optional[int], now_confirmed: int, items: list[str]) -> str:
    """prev_confirmed=None -> previous ledger not initialized (t<20)."""
    total = len(items)
    task = "Complete" if now_confirmed == total else "Incomplete"
    if prev_confirmed is None:
        if now_confirmed:
            raise ValueError("confirmation inside the first window; init cannot also confirm")
        return f"Subgoal:init [{'; '.join(items)}]\nTask:{task}"
    if now_confirmed < prev_confirmed:
        raise ValueError("regressing confirmation count")
    if now_confirmed == prev_confirmed:
        return f"Subgoal:none\nTask:{task}"
    return f"Subgoal:done {','.join(str(k) for k in range(prev_confirmed + 1, now_confirmed + 1))}\nTask:{task}"
