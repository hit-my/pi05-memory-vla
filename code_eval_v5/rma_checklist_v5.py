"""The policy's checklist -> the per-frame signals the v5 perception needs (evaluation side).

Where the checklist comes from: the adapter keeps the Ledger-Driver memory text of the policy (`self.memory`, rendered by
memory_format.render_previous after each accepted infer() call, i.e. once per replan = every 20 frames):
    Action:<...>
    Subgoal:[x]1 pick up the cookies; [ ]2 place the cookies into the basket; [ ]3 ?; ...
    Task:Incomplete
The CURRENT checklist item = the first item not ticked ("[ ]"), as in the training labels, where item k is the current
subtask from its first frame on and is ticked at its last frame (causal_labels.ledger_at). Its text may be "?" (not yet
revealed): an item with unknown text carries no trigger, but moving onto it still ends the item before.
"Subgoal:Not initialized." -> no current item. Task:Complete / every item of a closed list ticked -> END (the last item ended).
The memory read at frame t is the one produced by the infer() call of an EARLIER frame (observe() runs before infer()).

Signals (same rules as offline, tv_fuse_v5.retired_from_labels_v5 / track5b picks):
  retired  RetireSignalV5 (rma_online_veto_v5.py, unchanged) fed with the current item text every frame.
           Two cases the demo labels never contain:
             * several items ticked in one round (k -> k+2): the skipped items are fed one after the other at that frame
               (each starts and ends there), so a trigger armed in a skipped "place ..." item fires, as it would at its end;
             * the ledger goes back ("Subgoal:at k" lower than before, or reopened after END): nothing fires; the triggers
               of the item now current are armed again (retirement stays sticky).
  pick     the object named by the current item when its text starts with "pick" (rma_spec.subtask_object), else None.
Robustness: an unreadable memory (non-string, no Subgoal line, no items) = no new information: the current item stays
what it was. Nothing in update() raises.
"""
from __future__ import annotations

import re

from rma_online_veto_v5 import RetireSignalV5

_ITEM = re.compile(r"\[( |x)\](\d+) (.*?)(?=; \[|; \.\.\.$|$)")
END = "__checklist_complete__"


def parse_checklist(memory_text):
    """-> None (not initialized) or (items [(k, text|None, done)], complete, open_ended); ValueError when unreadable"""
    if not isinstance(memory_text, str):
        raise ValueError("memory is not a string")
    lines = memory_text.split("\n")
    sub = next((ln for ln in lines if ln.startswith("Subgoal:")), None)
    if sub is None:
        raise ValueError("no Subgoal line")
    body = sub[len("Subgoal:"):].strip()
    if body == "Not initialized.":
        return None
    items = [(int(k), None if t.strip() in ("?", "") else t.strip(), m == "x") for m, k, t in _ITEM.findall(body)]
    if not items:
        raise ValueError(f"no checklist items in {body[:80]!r}")
    complete = any(ln.strip() == "Task:Complete" for ln in lines)
    return items, complete, body.endswith("...")


def current_item(memory_text):
    """-> (index, text): (None, None) not initialized; (-1, END) all done; text None = '?'"""
    parsed = parse_checklist(memory_text)
    if parsed is None:
        return None, None
    items, complete, open_ended = parsed
    for k, text, done in items:
        if not done:
            return k, text
    if open_ended and not complete:
        return max(k for k, _, _ in items) + 1, None   # every listed item ticked, the next one is not revealed yet
    return -1, END


class ChecklistSignalV5:
    """update(memory_text, last_gripper_cmd, state) once per frame -> (retired object indices (sticky), pick or None)."""

    def __init__(self, task_id):
        self.task_id = int(task_id)
        self.sig = RetireSignalV5(self.task_id)
        self.reset()

    def reset(self):
        self.sig.reset()
        self.index, self.text = None, None   # current item as last understood
        self.texts = {}                       # index -> text, every item text seen in the memory
        self.log = []                         # (frame, message) for the adapter log
        self.frame = -1
        self.errors = 0

    @staticmethod
    def _key(index, text):
        """the string RetireSignalV5 sees for an item: its text, a per-index placeholder when unknown, or END"""
        if text == END:
            return END
        if index is None:
            return ""
        return text.lower() if text else f"?#{index}"

    def _move(self, s, closed):
        """RetireSignalV5.update's 'a new item became current' step, without advancing its frame counter"""
        sig = self.sig
        if s != sig.text:
            for a in sig.active:
                sig.retired.add(a["obj"])
            sig._start(s, closed)
            sig.text = s

    def update(self, memory_text, last_gripper_cmd=None, state=None):
        self.frame += 1
        sig = self.sig
        try:
            parsed = parse_checklist(memory_text)
            index, text = current_item(memory_text)
            if parsed is not None:
                for k, t, _ in parsed[0]:
                    if t:
                        self.texts[k] = t
        except Exception as exc:  # unreadable: keep what we had
            self.errors += 1
            if self.errors <= 20:
                self.log.append((self.frame, f"unparsable checklist ({type(exc).__name__}: {str(exc)[:80]})"))
            index, text = self.index, self.text
        try:
            closed = bool(sig.t > 0 and last_gripper_cmd is not None and float(last_gripper_cmd) > 0)
            old = self.index
            if (index, text) != (old, self.text):
                if old is not None and old != -1 and (index == -1 or index > old + 1):
                    last = max([old] + list(self.texts)) if index == -1 else index - 1
                    for j in range(old + 1, last + 1):   # items ticked in the same round: each starts and ends now
                        self._move(self._key(j, self.texts.get(j)), closed)
                    self.log.append((self.frame, f"item {old}->{index} (skipped {list(range(old + 1, last + 1))}): {text}"))
                elif old is not None and index is not None and index != -1 and (old == -1 or index < old):
                    sig.active = []                      # back: re-arm the item now current, fire nothing
                    sig._start(self._key(index, text), closed)
                    sig.text = self._key(index, text)
                    self.log.append((self.frame, f"item {old}->{index} (back): {text}"))
                else:
                    self.log.append((self.frame, f"item {old}->{index}: {text}"))
                self.index, self.text = index, text
            retired = sig.update(self._key(self.index, self.text), last_gripper_cmd, state)
        except Exception as exc:  # never let the signal end an episode
            self.errors += 1
            self.log.append((self.frame, f"signal error {type(exc).__name__}: {str(exc)[:80]}"))
            sig.t += 1
            retired = sorted(sig.retired)
        pick = None
        try:
            if self.text and self.text != END and self.text.lower().startswith("pick"):
                pick = sig._obj(self.text.lower())
        except Exception:
            pick = None
        return retired, pick


class LabelSignal:
    """Test stand-in: the demo's per-frame subtask text fed straight to RetireSignalV5 (as test_veto_v5.py) and the
    track5b picks (as track5b.episode_inputs). Same return as ChecklistSignalV5.update."""

    def __init__(self, task_id):
        self.task_id = int(task_id)
        self.sig = RetireSignalV5(self.task_id)

    def reset(self):
        self.sig.reset()

    def update(self, text, last_gripper_cmd=None, state=None):
        retired = self.sig.update(text, last_gripper_cmd, state)
        s = str(text).lower()
        pick = self.sig._obj(s) if s.startswith("pick") else None
        return retired, pick
