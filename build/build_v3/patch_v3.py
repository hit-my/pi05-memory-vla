"""Create ~/pi05-abc/code_causal_v3 edits (run after `cp -a code_causal_v2 code_causal_v3`).
Scheme B accumulated Action line (user-approved 11-15):
  * memory_format: Action clause body = still | occluded | chain of direction/event words joined by ' -> ' (or the
    legacy ' ; '), up to 24 words accepted from the model; render_previous keeps at most the last MAX_CHAIN (12) words
    per object (identity on every training label); accumulate_action() is the single label rule.
  * model_data: memory_target.window_action_text (the per-20-frame Action line) is a required field and feeds the
    scheme-C verb head, so verbs stay aligned with the per-window latent z*; PROMPT_VERSION bumped.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(name, pairs):
    p = root / name
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:60], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", name)


patch("memory_format.py", [
    ('''_BODY = (rf"(?:occluded|still|(?:(?:{_DIR})(?: -> (?:{_DIR})){{0,2}}(?: ; (?:{_EV}))*)|"
         rf"(?:(?:{_EV})(?: ; (?:{_EV}))*))")''',
     '''# v3 (scheme B): accumulated chain of direction/event words within the current subgoal
_WORD = rf"(?:{_DIR}|{_EV})"
MAX_CHAIN = 12          # words kept per object when the Action line is handed to the next round
_MAX_PARSE = 24         # words accepted in a generated clause (longer -> invalid)
_BODY = rf"(?:occluded|still|{_WORD}(?:(?: -> | ; ){_WORD}){{0,{_MAX_PARSE - 1}}})"'''),
    ('''def render_previous(action_line, ledger):
    """Previous memory for the next round: this round's Action line + rendered ledger."""
    return f"Action:{action_line}\\n{render_ledger(ledger)}"''',
     '''def cap_action(action_line, limit=MAX_CHAIN):
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
    return f"Action:{cap_action(action_line)}\\n{render_ledger(ledger)}"'''),
])

patch("model_data.py", [
    ('PROMPT_VERSION = "native-state8-ledger-previous-memory-abc-v2-rawfront3"',
     'PROMPT_VERSION = "native-state8-ledger-previous-memory-abc-v2-rawfront3-accB"'),
    ('''    "action_text", "subgoal_delta", "task_text", "confirmed_goals", "total_goals", "action_reliable",
)''',
     '''    "action_text", "subgoal_delta", "task_text", "confirmed_goals", "total_goals", "action_reliable",
    "window_action_text",  # v3: per-20-frame Action line, the verb-head target aligned with z*
)'''),
    ('            parsed = verb_targets(row["memory_target"]["action_text"])',
     '            parsed = verb_targets(row["memory_target"]["window_action_text"])'),
])
