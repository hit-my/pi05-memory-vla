"""Memory text formats for groups A/B/C and the stateless ledger step.

Stored labels (dataset A-v3) are always the A format. Groups B and C are derived:
  A: blue bowl: up ; picked | orange bowl: still
  C: blue bowl: up ; picked<unused0> | orange bowl: still<unused0>
  B: blue bowl:<unused0> | orange bowl:<unused0>
<unused0> is the slot of one latent vector. Previous memory = the previous Action
line plus the driver-rendered ledger, so the ledger can be recovered from its text
and every step is stateless.
"""
from __future__ import annotations

import re

from ledger_v2 import (MORE, UNKNOWN, Ledger, SubgoalItem, InvalidDelta, apply_delta, parse_delta,
                       render_ledger)

LAT = "<unused0>"
VARIANTS = ("A", "B", "C")
LATENT_DIM = 64
SEM_DIM = 16
MAX_ENTITIES = 3
INITIAL_MEMORY = "Action:none\nSubgoal:Not initialized.\nTask:Incomplete"
DIR_WORDS = ("up", "down", "forward", "backward", "left", "right")
EVENT_WORDS = ("picked", "placed", "put-in", "taken-out", "opened", "closed", "pressed")
VERBS = DIR_WORDS + ("still", "occluded") + EVENT_WORDS
COLOURS = ("blue", "orange", "purple", "green", "magenta", "ochre", "indigo", "brown")

_DIR = "|".join(DIR_WORDS)
_EV = "|".join(EVENT_WORDS)
# v3 (scheme B): accumulated chain of direction/event words within the current subgoal
_WORD = rf"(?:{_DIR}|{_EV})"
MAX_CHAIN = 12          # words kept per object when the Action line is handed to the next round
_MAX_PARSE = 24         # words accepted in a generated clause (longer -> invalid)
_BODY = rf"(?:occluded|still|{_WORD}(?:(?: -> | ; ){_WORD}){{0,{_MAX_PARSE - 1}}})"
_NAME = rf"(?P<name>(?:{'|'.join(COLOURS)}) [a-z]+)"
_CLAUSE = {"A": re.compile(rf"^{_NAME}: {_BODY}$"),
           "C": re.compile(rf"^{_NAME}: {_BODY}{re.escape(LAT)}$"),
           "B": re.compile(rf"^{_NAME}:{re.escape(LAT)}$")}
_ITEM = re.compile(r"\[( |x)\](\d+) (.*?)(?=; \[|$)")


def render_action(action_text, variant):
    """A-format Action text -> this variant's Action text."""
    if variant == "A":
        return action_text
    clauses = action_text.split(" | ")
    if variant == "C":
        return " | ".join(clause + LAT for clause in clauses)
    if variant == "B":
        return " | ".join(clause.split(": ", 1)[0] + ":" + LAT for clause in clauses)
    raise ValueError(f"Unknown variant {variant!r}")


def render_memory_text(text, variant):
    """Apply render_action to the Action line of any stored three-line memory."""
    lines = text.split("\n")
    if not lines[0].startswith("Action:"):
        raise ValueError(f"Not a memory text: {text!r}")
    action = lines[0][len("Action:"):]
    if action != "none":
        lines[0] = "Action:" + render_action(action, variant)
    return "\n".join(lines)


def action_entities(action, variant):
    """Entity names of a generated Action line, or None if its grammar is invalid."""
    names = []
    for clause in action.split(" | "):
        match = _CLAUSE[variant].match(clause)
        if match is None:
            return None
        names.append(match.group("name"))
    return names if len(set(names)) == len(names) else None


def verb_targets(action_text):
    """A-format Action text -> multi-hot rows [K][len(VERBS)]; order is not encoded here."""
    rows = []
    for clause in action_text.split(" | "):
        body = clause.split(": ", 1)[1]
        words = set(re.findall(r"[a-z]+(?:-[a-z]+)*", body))  # keeps put-in, skips the -> arrow
        unknown = words.difference(VERBS)
        if unknown:
            raise ValueError(f"Unknown verbs {sorted(unknown)} in {clause!r}")
        rows.append([float(verb in words) for verb in VERBS])
    return rows


def ledger_from_text(previous_memory_text):
    """Recover the driver's ledger from a previous-memory string."""
    lines = previous_memory_text.split("\n")
    if len(lines) != 3 or not lines[1].startswith("Subgoal:") or not lines[2].startswith("Task:"):
        raise InvalidDelta(f"Bad previous memory: {previous_memory_text!r}")
    subgoal, task = lines[1][len("Subgoal:"):], lines[2][len("Task:"):]
    if subgoal == "Not initialized.":
        return Ledger(task_status=task)
    open_ended = subgoal == MORE or subgoal.endswith("; " + MORE)
    if open_ended:
        subgoal = subgoal[:-len(MORE)].rstrip("; ")
    items = [SubgoalItem(int(k), None if text == UNKNOWN else text, done=mark == "x")
             for mark, k, text in _ITEM.findall(subgoal)]
    if not items and not open_ended:
        raise InvalidDelta(f"Bad rendered ledger: {subgoal!r}")
    return Ledger(initialized=True, items=items, task_status=task, open_ended=open_ended)


def cap_action(action_line, limit=MAX_CHAIN):
    """Keep the last `limit` words of every clause chain (any variant: a trailing latent slot is kept)."""
    if action_line == "none":
        return action_line
    out = []
    for clause in action_line.split(" | "):
        suffix = LAT if clause.endswith(LAT) else ""
        core = clause[:-len(suffix)] if suffix else clause
        if ": " in core:
            name, body = core.split(": ", 1)
            words = re.split(r" -> | ; ", body)
            if len(words) > limit:
                core = f"{name}: " + " -> ".join(words[-limit:])
        out.append(core + suffix)
    return " | ".join(out)


def accumulate_action(previous_chain, window_text, reset):
    """Scheme-B label rule (A format). previous_chain: this subgoal's Action line from 20 frames ago (None at the start
    of an episode); window_text: rma_build.action_text of the last 20 frames; reset: a subgoal was confirmed in this
    round -> the chain restarts from this window. Objects not moved/picked in this subgoal show their window state
    (still/occluded); moved ones show the chain of direction/event words, consecutive repeats merged, no pauses."""
    prev = {}
    if previous_chain is not None and not reset:
        for clause in previous_chain.split(" | "):
            name, body = clause.split(": ", 1)
            prev[name] = [] if body in ("still", "occluded") else re.split(r" -> | ; ", body)
    out = []
    for clause in window_text.split(" | "):
        name, body = clause.split(": ", 1)
        words = list(prev.get(name, []))
        if body not in ("still", "occluded"):
            for w in re.split(r" -> | ; ", body):
                if not words or words[-1] != w:
                    words.append(w)
        out.append(f"{name}: " + (" -> ".join(words[-MAX_CHAIN:]) if words else body))
    return " | ".join(out)


def render_previous(action_line, ledger):
    """Previous memory for the next round: this round's Action line (chains capped) + rendered ledger."""
    return f"Action:{cap_action(action_line)}\n{render_ledger(ledger)}"


def step_memory(previous_memory_text, generated_text, variant, *, eos_terminated=True, expected_total=None,
                round_index=0):
    """Validate one generated memory and apply it to the ledger, as the Ledger Driver does.

    Any failure returns memory_valid=False and next_memory_text=None, and the caller
    terminates with invalid_model_memory; there is no repair and no simulator truth.
    """
    invalid = {"memory_valid": False, "sequence_complete": False, "certified_count": None,
               "next_memory_text": None, "action_text": None, "subgoal_delta": None, "task_text": None}
    if not eos_terminated or not isinstance(generated_text, str) or not generated_text.isascii():
        return invalid
    lines = generated_text.split("\n")
    if (len(lines) != 3 or not lines[0].startswith("Action:") or not lines[1].startswith("Subgoal:")
            or not lines[2].startswith("Task:")):
        return invalid
    action = lines[0][len("Action:"):]
    if action_entities(action, variant) is None:
        return invalid
    try:
        previous = ledger_from_text(previous_memory_text)
        ledger = apply_delta(previous, parse_delta("\n".join(lines[1:])), round_index)
    except InvalidDelta:
        return invalid
    # expected_total 0 = the instruction's length varies across demos (open-ended ledger)
    if expected_total and not ledger.open_ended and len(ledger.items) != expected_total:
        return invalid
    return {"memory_valid": True, "sequence_complete": ledger.task_status == "Complete",
            "certified_count": sum(item.done for item in ledger.items),
            "next_memory_text": render_previous(action, ledger), "action_text": action,
            "subgoal_delta": lines[1][len("Subgoal:"):], "task_text": ledger.task_status}
