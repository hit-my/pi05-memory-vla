"""Causal Subgoal labels: the memory label at frame t is a function of the episode prefix up to t only.

The rule has no task categories and no benchmark-specific knowledge:
  * TaskPrior, computed once from all training episodes of one instruction: item k's text is "fixed" when every
    demo has the same text at position k (it then carries only what the instruction says, nothing about this
    episode); the length is fixed when every demo has the same number of items.
  * At frame t an item's text may be written only when it is fixed or its subtask has started (start <= t).
    Otherwise it is "?" (fixed length) or not listed yet, behind "..." (variable length).
  * Item k is ticked from the last frame of subtask k, Task:Complete from the last frame of the episode.
The labeler (ledger_at) only ever receives Prefix(t), so it cannot read the future. audit() checks it anyway by
future perturbation: rewriting everything after t (texts, timing, length) must not change any label at frames
<= t. The same audit run on the old labels (rma_build: whole subtask list in the init) reports their leaks.

Usage:
  causal_labels.py selftest
  causal_labels.py report  BUILT                 # per task: fixed/hidden items, old-label leaks, audit
  causal_labels.py relabel BUILT OUT             # OUT/rows/<ep>.jsonl with causal Subgoal labels (Action kept)
BUILT = an rma_build output dir (rows/, episodes/, log/).
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger_v2 import MORE, UNKNOWN, Ledger, SubgoalItem, apply_delta, parse_delta, render_ledger  # noqa: E402
from memory_format import INITIAL_MEMORY, render_previous  # noqa: E402

INTERVAL = 20


@dataclass(frozen=True)
class TaskPrior:
    fixed: dict             # position (0-based) -> text, identical in every training demo of the instruction
    length: Optional[int]   # number of items when every demo has the same, else None
    demos: int = 0


def task_prior(item_lists, observed=()) -> TaskPrior:
    """observed: positions of observation items. They are never fixed: an observation is written only once seen,
    even when every training demo happened to see the same thing."""
    item_lists = [list(x) for x in item_lists]
    lengths = {len(x) for x in item_lists}
    shortest = min(lengths)
    fixed = {k: item_lists[0][k] for k in range(shortest)
             if k not in observed and all(x[k] == item_lists[0][k] for x in item_lists)}
    return TaskPrior(fixed, lengths.pop() if len(lengths) == 1 else None, len(item_lists))


@dataclass(frozen=True)
class Prefix:
    started: tuple   # texts of the subtasks that have started by frame t, in order
    done: int        # subtasks ended by frame t
    finished: bool   # episode over (last subtask ended) by frame t


def prefix_at(items, ends, t) -> Prefix:
    starts = [0] + [e + 1 for e in ends[:-1]]
    n_started = sum(1 for s in starts if s <= t)
    n_done = sum(1 for e in ends if e <= t)
    return Prefix(tuple(items[:n_started]), n_done, n_done == len(items))


def ledger_at(prior: TaskPrior, p: Prefix) -> Ledger:
    """The only labeler. Sees the prior (instruction-level) and the prefix, nothing else."""
    if prior.length is not None:
        texts = [p.started[k] if k < len(p.started) else prior.fixed.get(k) for k in range(prior.length)]
        open_ended = False
    else:
        texts = list(p.started)
        k = len(texts)
        while k in prior.fixed:  # fixed items right after the known ones; a gap would leak the length
            texts.append(prior.fixed[k])
            k += 1
        open_ended = not p.finished
    items = [SubgoalItem(k + 1, text, done=k < p.done) for k, text in enumerate(texts)]
    return Ledger(initialized=True, items=items, task_status="Complete" if p.finished else "Incomplete",
                  open_ended=open_ended)


def delta_between(before: Optional[Ledger], after: Ledger) -> str:
    """Subgoal+Task lines that turn `before` (None = not initialized) into `after`."""
    task = after.task_status
    if before is None:
        if any(it.done for it in after.items):
            raise ValueError("confirmation inside the first window; init cannot also confirm")
        parts = [UNKNOWN if it.text is None else it.text for it in after.items] + ([MORE] if after.open_ended else [])
        return f"Subgoal:init [{'; '.join(parts)}]\nTask:{task}"
    old = {it.index: it for it in before.items}
    reveals = {it.index: it.text for it in after.items
               if it.text is not None and (it.index not in old or old[it.index].text is None)}
    ticks = [it.index for it in after.items if it.done and not (it.index in old and old[it.index].done)]
    head = f"done {','.join(map(str, ticks))}" if ticks else "none"
    tail = (" reveal " + "|".join(f"{k}={v}" for k, v in sorted(reveals.items()))) if reveals else ""
    return f"Subgoal:{head}{tail}\nTask:{task}"


def episode_labels(prior, items, ends, n, actions):
    """-> per frame (previous_memory_text, target_subgoal_task_text, ledger_now); actions[t] = A-format Action."""
    out = []
    for t in range(n):
        now = ledger_at(prior, prefix_at(items, ends, t))
        if t < INTERVAL:
            before, previous = None, INITIAL_MEMORY
        else:
            before = ledger_at(prior, prefix_at(items, ends, t - INTERVAL))
            previous = render_previous(actions[t - INTERVAL], before)
        delta = delta_between(before, now)
        # the deployed driver must reach exactly this ledger from the previous memory and the label
        start = before if before is not None else Ledger()
        if render_ledger(apply_delta(start, parse_delta(delta), t)) != render_ledger(now):
            raise AssertionError(f"driver mismatch at t={t}: {delta!r}")
        out.append((previous, delta, now))
    return out


# ---------------------------------------------------------------- audit

def old_labels(items, ends, n):
    """What rma_build wrote: the whole subtask list in the init (for the leak report)."""
    from ledger_v2 import delta_text
    confirmed = [sum(1 for e in ends if e <= t) for t in range(n)]
    return [delta_text(None if t < INTERVAL else confirmed[t - INTERVAL], confirmed[t], items) for t in range(n)]


def perturb_after(items, ends, t, rng):
    """Same past, different future: texts, timing and length of everything not started by t are rewritten."""
    starts = [0] + [e + 1 for e in ends[:-1]]
    keep = sum(1 for s in starts if s <= t)
    if ends[-1] <= t:  # already over: the end of the episode is observed, there is no future to rewrite
        return list(items), list(ends)
    new_items = list(items[:keep]) + [f"zz future {rng.randrange(10**6)}" for _ in range(len(items) - keep + rng.choice((0, 1)))]
    new_ends, last = [], 0
    for k in range(len(new_items)):
        if k < len(ends) and ends[k] <= t:
            last = ends[k]
        else:
            last = max(last, t, INTERVAL) + rng.randrange(5, 60)
        new_ends.append(last)
    return new_items, new_ends


def audit(labeler, items, ends, n, rng, probes=12):
    """Number of probes where some label at frame <= t changed when only the future changed."""
    base = labeler(items, ends, n)
    leaks = 0
    for t in sorted(rng.sample(range(n), min(probes, n))):
        items2, ends2 = perturb_after(items, ends, t, rng)
        other = labeler(items2, ends2, max(n, ends2[-1] + 1))
        if any(base[s] != other[s] for s in range(t + 1)):
            leaks += 1
    return leaks, min(probes, n)


def causal_labeler(prior):
    def run(items, ends, n):
        return [render_ledger(ledger_at(prior, prefix_at(items, ends, t))) for t in range(n)]
    return run


# ---------------------------------------------------------------- data

def with_observations(items, ends, observed):
    """Insert observation items: (k, text) -> a one-frame item right after item k, ending one frame after it, so
    it is revealed and ticked exactly when item k is over (the observation is in view), never earlier."""
    items, ends = list(items), list(ends)
    for k, text in sorted(observed, reverse=True):
        at = ends[k] + 1
        if k + 1 < len(ends) and ends[k + 1] <= at:
            raise ValueError(f"no room for an observation after item {k}")
        items.insert(k + 1, text)
        ends.insert(k + 1, at)
    return items, ends


def observation_positions(observed):
    """0-based positions of the inserted observation items in the augmented list."""
    return {k + 1 + i for i, (k, _) in enumerate(sorted(observed))}


def load_built(built: Path, observe=None):
    """observe(task_id, items) -> [(k, text)], a benchmark's observation adapter (e.g. rma_observations)."""
    import numpy as np
    eps = {}
    for log in sorted((built / "log").glob("*.json")):
        ep = log.stem
        items = json.loads(log.read_text())["items"]
        npz = np.load(built / "episodes" / f"{ep}.npz")
        ends = [int(e) for e in npz["ends"]]
        task = int(ep[1:3]) if ep.startswith("t") else None
        if observe is not None:
            observed = observe(task, items)
            items, ends = with_observations(items, ends, observed)
            obs_at = observation_positions(observed)
        else:
            obs_at = set()
        eps[ep] = {"task": task, "items": items, "ends": ends, "n": int(len(npz["confirmed_goals"])),
                   "rows": built / "rows" / f"{ep}.jsonl", "observed": obs_at}
    return eps


def priors_by_task(eps):
    groups, observed = defaultdict(list), defaultdict(set)
    for e in eps.values():
        groups[e["task"]].append(e["items"])
        observed[e["task"]] |= set(e.get("observed", ()))
    return {task: task_prior(lists, observed[task]) for task, lists in groups.items()}


def report(built: Path, observe=None):
    eps = load_built(built, observe)
    priors = priors_by_task(eps)
    rng = random.Random(0)
    print("task demos length hidden_items | old: frames naming a hidden item before it starts | causal leak probes")
    for task in sorted(priors):
        prior = priors[task]
        members = [e for e in eps.values() if e["task"] == task]
        length = prior.length if prior.length is not None else "var"
        shortest = min(len(e["items"]) for e in members)
        hidden = [k + 1 for k in range(max(len(e["items"]) for e in members)) if k not in prior.fixed]
        old_frames = frames = new_leaks = probes = 0
        for e in members:
            starts = [0] + [x + 1 for x in e["ends"][:-1]]
            last_hidden_start = max([starts[k - 1] for k in hidden if k - 1 < len(starts)], default=0)
            old_frames += last_hidden_start  # rma_build's memory lists every item from frame 0
            frames += e["n"]
        for e in members[:20]:
            b, m = audit(causal_labeler(prior), e["items"], e["ends"], e["n"], rng)
            new_leaks, probes = new_leaks + b, probes + m
        print(f"{task:4d} {prior.demos:5d} {str(length):>6} {hidden} | {100 * old_frames / frames:5.1f}% | {new_leaks}/{probes}")
        variants = defaultdict(lambda: defaultdict(int))
        for e in members:
            for k in hidden:
                if k - 1 < len(e["items"]):
                    variants[k][e["items"][k - 1]] += 1
        for k in hidden:
            print(f"       item {k}: " + "; ".join(f"{c}x {txt}" for txt, c in sorted(variants[k].items(), key=lambda x: -x[1])))
        for k, (n_states, n_ambiguous) in sufficiency(prior, members).items():
            print(f"       item {k}: memory before the reveal decides it in {n_states - n_ambiguous}/{n_states} "
                  f"memory states{'  <-- INSUFFICIENT: memory does not hold what decides it' if n_ambiguous else ''}")


def sufficiency(prior, members):
    """Per hidden item: the Subgoal memory one round before its reveal, across demos. When one memory state is
    followed by different revealed texts, what decided the item is not in the (text) memory. The Action line is
    left out (it differs between demos for unrelated reasons); its content is checked separately."""
    seen = defaultdict(lambda: defaultdict(set))
    for e in members:
        starts = [0] + [x + 1 for x in e["ends"][:-1]]
        for k, s in enumerate(starts):
            if k in prior.fixed or k in e.get("observed", ()):  # an observation is decided by the camera
                continue
            t = max(0, (s // INTERVAL) * INTERVAL - INTERVAL)
            state = render_ledger(ledger_at(prior, prefix_at(e["items"], e["ends"], t)))
            seen[k + 1][state].add(e["items"][k])
    return {k: (len(states), sum(len(v) > 1 for v in states.values())) for k, states in sorted(seen.items())}


def relabel(built: Path, out: Path, observe=None):
    eps = load_built(built, observe)
    priors = priors_by_task(eps)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    changed = total = 0
    for ep, e in sorted(eps.items()):
        rows = [json.loads(line) for line in open(e["rows"])]
        actions = [r["memory_target"]["action_text"] for r in rows]
        labels = episode_labels(priors[e["task"]], e["items"], e["ends"], len(rows), actions)
        prior = priors[e["task"]]
        with open(out / "rows" / f"{ep}.jsonl", "w") as stream:
            for r, (previous, delta, now) in zip(rows, labels):
                sub, task = delta.split("\n")
                total += 1
                changed += r["memory_target_text"] != f"Action:{r['memory_target']['action_text']}\n{delta}"
                r["previous_memory_text"] = previous
                r["memory_target_text"] = f"Action:{r['memory_target']['action_text']}\n{delta}"
                r["memory_target"].update(subgoal_delta=sub[len("Subgoal:"):], task_text=task[len("Task:"):],
                                          confirmed_goals=sum(it.done for it in now.items),
                                          total_goals=prior.length or 0)
                r["label_rule"] = "causal_v1"
                stream.write(json.dumps(r) + "\n")
    (out / "priors.json").write_text(json.dumps({str(k): {"fixed": {str(i): t for i, t in v.fixed.items()},
                                                          "length": v.length, "demos": v.demos}
                                                 for k, v in sorted(priors.items())}, indent=1))
    print(f"episodes {len(eps)} rows {total} rows_changed {changed}")


def build_dataset(src: Path, dst: Path, observe=None):
    """dst = src with causal labels: everything but the row indices is a symlink into src; train/val.jsonl keep
    src's order and split; tasks without hidden items must come out byte-identical (regression check)."""
    import hashlib
    import os
    eps = load_built(src, observe)
    priors = priors_by_task(eps)
    dst.mkdir(parents=True, exist_ok=False)
    own = {"rows", "train.jsonl", "val.jsonl", "split_manifest.json"}
    for entry in src.iterdir():
        if entry.name not in own:
            # hard links: the trainer refuses paths that resolve outside the dataset root (symlinks do)
            import shutil
            import subprocess
            if os.name == "posix":
                subprocess.run(["cp", "-al", str(entry), str(dst / entry.name)], check=True)
            else:  # local tests on Windows
                (shutil.copytree if entry.is_dir() else shutil.copy)(entry, dst / entry.name)
    (dst / "rows").mkdir()
    changed_eps, changed_rows = defaultdict(int), 0
    for ep, e in sorted(eps.items()):
        prior = priors[e["task"]]
        rows = [json.loads(line) for line in open(e["rows"])]
        actions = [r["memory_target"]["action_text"] for r in rows]
        labels = episode_labels(prior, e["items"], e["ends"], len(rows), actions)
        has_hidden = prior.length is None or len(prior.fixed) < prior.length
        with open(dst / "rows" / f"{ep}.jsonl", "w") as stream:
            for t, (r, (previous, delta, now)) in enumerate(zip(rows, labels)):
                assert r["observation_index"] == t, (ep, t)
                text = f"Action:{r['memory_target']['action_text']}\n{delta}"
                differs = r["previous_memory_text"] != previous or r["memory_target_text"] != text
                if differs and not has_hidden:
                    raise AssertionError(f"{ep} t={r['observation_index']}: label changed in a task without hidden items")
                if differs:
                    changed_rows += 1
                    changed_eps[e["task"]] += 1
                sub, task = delta.split("\n")
                r["previous_memory_text"], r["memory_target_text"] = previous, text
                r["memory_target"].update(subgoal_delta=sub[len("Subgoal:"):], task_text=task[len("Task:"):],
                                          confirmed_goals=sum(it.done for it in now.items),
                                          total_goals=prior.length or 0)
                stream.write(json.dumps(r) + "\n")
    manifest = json.loads((src / "split_manifest.json").read_text())
    for part in ("train", "val"):
        current, cache, n = None, None, 0
        h = hashlib.sha256()
        with open(src / f"{part}.jsonl") as old, open(dst / f"{part}.jsonl", "w") as new:
            for line in old:
                r = json.loads(line)
                if r["episode_id"] != current:
                    current = r["episode_id"]
                    cache = {json.loads(x)["sample_id"]: x for x in open(dst / "rows" / f"{current}.jsonl")}
                out = cache[r["sample_id"]]
                new.write(out)
                h.update(out.encode())
                n += 1
        assert n == manifest["splits"][part]["queries"], (part, n)
        manifest["splits"][part]["index_sha256"] = h.hexdigest()
    manifest["label_rule"] = "causal_v1 (causal_labels.py): an item's text appears only once its subtask has started, " \
                             "unless it is identical in every training demo of the instruction"
    (dst / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    (dst / "priors.json").write_text(json.dumps({str(k): {"fixed": {str(i): t for i, t in v.fixed.items()},
                                                          "length": v.length, "demos": v.demos}
                                                 for k, v in sorted(priors.items())}, indent=1))
    print("BUILT", dst, "changed rows", changed_rows, "by task", dict(changed_eps))


# ---------------------------------------------------------------- self test

def selftest():
    rng = random.Random(1)
    checks = [f"{a} the {lvl} drawer" for lvl in ("top", "middle", "bottom") for a in ("open", "close")]
    demos = []
    for answer in ("top", "middle", "bottom") * 3:
        items = checks + [f"open the {answer} drawer again", f"place the butter into the {answer} drawer",
                          f"close the {answer} drawer"]
        ends = sorted(rng.sample(range(25, 600), len(items)))
        demos.append((items, ends, ends[-1] + 1))
    prior = task_prior([d[0] for d in demos])
    assert prior.length == 9 and sorted(prior.fixed) == [0, 1, 2, 3, 4, 5], prior
    for items, ends, n in demos:
        acts = ["butter: still"] * n
        labels = episode_labels(prior, items, ends, n, acts)
        init = labels[0][1]
        assert "?" in init and items[6] not in init, init           # the answer is not in the init
        leaks, _ = audit(causal_labeler(prior), items, ends, n, rng)
        assert leaks == 0
        old, _ = audit(old_labels, items, ends, n, rng)
        assert old > 0                                                # the old labels do leak
    # variable length (e.g. "pour until ..."): hidden behind "..."
    var = [(["pick the cup", "pour once"], [30, 90]), (["pick the cup", "pour once", "pour again"], [30, 90, 150])]
    prior = task_prior([v[0] for v in var])
    assert prior.length is None
    for items, ends in var:
        n = ends[-1] + 1
        labels = episode_labels(prior, items, ends, n, ["cup: still"] * n)
        assert labels[0][1].startswith("Subgoal:init [pick the cup; pour once; ...]"), labels[0][1]
        assert audit(causal_labeler(prior), items, ends, n, rng)[0] == 0
    # with the drawer observations written when seen, the memory decides the answer (tasks 4 and 5)
    from rma_observations import observations
    for task in (4, 5):
        members = []
        for answer in ("top", "middle", "bottom") * 2:
            items = checks + [f"open the {answer} drawer again", f"place the butter into the {answer} drawer",
                              f"close the {answer} drawer"]
            ends = sorted(rng.sample(range(25, 600), len(items)))
            ends = [e + 2 * i for i, e in enumerate(ends)]  # room for the one-frame observation items
            observed = observations(task, items)
            items, ends = with_observations(items, ends, observed)
            members.append({"items": items, "ends": ends, "n": ends[-1] + 1, "observed": observation_positions(observed)})
        prior = task_prior([m["items"] for m in members], members[0]["observed"])
        # an observation every demo agrees on is still hidden until seen
        assert task_prior([m["items"] for m in members[:1]] * 2, members[0]["observed"]).fixed.keys().isdisjoint(members[0]["observed"])
        assert prior.length == 12 and sorted(prior.fixed) == [0, 2, 3, 5, 6, 8], prior
        assert all(amb == 0 for _, amb in sufficiency(prior, members).values()), sufficiency(prior, members)
        for m in members:
            n = m["n"]
            labels = episode_labels(prior, m["items"], m["ends"], n, ["blue bowl: still"] * n)
            assert "drawer: " not in labels[0][1] and audit(causal_labeler(prior), m["items"], m["ends"], n, rng)[0] == 0
            seen = [d for _, d, _ in labels if "reveal 2=" in d]
            assert seen and seen[0].startswith("Subgoal:done 1,2 reveal 2=top drawer: ") or "done 2 reveal 2=" in seen[0], seen[:1]
    print("selftest ok")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-observe"]
    observe = None
    if "--no-observe" not in sys.argv[1:]:
        from rma_observations import observations as observe  # this benchmark's observation adapter
    cmd = args[0]
    if cmd == "selftest":
        selftest()
    elif cmd == "report":
        report(Path(args[1]), observe)
    elif cmd == "build":
        build_dataset(Path(args[1]), Path(args[2]), observe)
    elif cmd == "relabel":
        relabel(Path(args[1]), Path(args[2]), observe)
    else:
        raise SystemExit(__doc__)
