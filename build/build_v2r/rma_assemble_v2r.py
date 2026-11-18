"""Assemble a v2r dataset from rma_build_v2r.py output: causal Subgoal labels, split, normalization, manifest, contract.

Stages (all functions that define labels are imported from the training code directory, nothing is re-implemented):
  1. causal labels = causal_labels.py (task_prior / episode_labels, observation items from rma_observations), applied
     to OUT/rows_base/<ep>.jsonl -> OUT/rows/<ep>.jsonl. The Action line is kept as written by the row writer.
     The per-instruction prior (which checklist items are identical in every demo) is computed from the subtask texts
     of ALL episodes of the task in RAW/meta, whether built or not, so a partial (rehearsal) build gets the same
     labels as the full build. In a full build this equals the old behaviour (prior over all built episodes).
  2. split / train.jsonl / val.jsonl / norm_stats.json / split_manifest.json = rma_dataset.py's rules:
     the last seed of each task (among the built episodes) is validation; normalization from training episodes only.
  3. contract: --contract-in (contract_C_v2.json) with dataset.latents_dir set to --latents-dir and source.tasks
     counts taken from this dataset; written to --contract-out (never into a code directory).

Usage (~/SANE/.venv python):
  rma_assemble_v2r.py --out DATASET --raw LR_EPS --tracks TRACKS --latents-dir DIR --contract-out FILE
                      [--code ~/pi05-abc/code_causal_v2] [--contract-in CODE/contract_C_v2.json] [--partial]
Without --partial every episode of RAW/meta must have been built.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--raw", type=Path, required=True)
p.add_argument("--tracks", type=Path, required=True)
p.add_argument("--latents-dir", type=Path, required=True)
p.add_argument("--contract-out", type=Path, required=True)
p.add_argument("--code", type=Path, default=Path.home() / "pi05-abc/code_causal_v2")
p.add_argument("--contract-in", type=Path)
p.add_argument("--partial", action="store_true", help="rehearsal: a subset of the episodes is enough")
args = p.parse_args()
dataset, code = args.out, args.code.resolve()
sys.path.insert(0, str(code))
import causal_labels as cl  # noqa: E402
import rma_spec  # noqa: E402
from rma_observations import observations  # noqa: E402

if code in args.contract_out.resolve().parents:
    raise SystemExit("--contract-out must not be inside the code directory")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. causal labels
built = sorted(path.stem for path in (dataset / "rows_base").glob("*.jsonl"))
every = sorted(path.stem for path in (args.raw / "meta").glob("t*_seed*.npz") if (args.raw / "front" / f"{path.stem}.npy").exists())
if not built:
    raise SystemExit("nothing built")
if not args.partial and built != every:
    raise SystemExit(f"incomplete build: {len(built)} of {len(every)} episodes (use --partial for a rehearsal)")
tasks_built = sorted({int(ep[1:3]) for ep in built})

# prior: item lists of every episode of the task (texts only; observation items inserted as in causal_labels.load_built)
item_lists, observed_at = defaultdict(list), defaultdict(set)
for ep in every:
    task = int(ep[1:3])
    if task not in tasks_built:
        continue
    with np.load(args.raw / "meta" / f"{ep}.npz") as m:
        texts = [str(x) for x in m["subtask_texts"]]
    obs = observations(task, texts)
    items, _ = cl.with_observations(texts, [1000 * (i + 1) for i in range(len(texts))], obs)
    item_lists[task].append(items)
    observed_at[task] |= cl.observation_positions(obs)
priors = {task: cl.task_prior(lists, observed_at[task]) for task, lists in item_lists.items()}

(dataset / "rows").mkdir(exist_ok=True)
changed_by_task, rows_total = defaultdict(int), 0
goals = {}
for ep in built:
    task = int(ep[1:3])
    prior = priors[task]
    log = json.loads((dataset / "log" / f"{ep}.json").read_text())
    with np.load(dataset / "episodes" / f"{ep}.npz") as z:
        ends = [int(e) for e in z["ends"]]
    obs = observations(task, log["items"])
    items, ends = cl.with_observations(log["items"], ends, obs)
    rows = [json.loads(line) for line in open(dataset / "rows_base" / f"{ep}.jsonl")]
    actions = [r["memory_target"]["action_text"] for r in rows]
    labels = cl.episode_labels(prior, items, ends, len(rows), actions)
    has_hidden = prior.length is None or len(prior.fixed) < prior.length
    temporary = dataset / "rows" / f"{ep}.jsonl.tmp"
    with open(temporary, "w") as stream:
        for t, (r, (previous, delta, now)) in enumerate(zip(rows, labels)):
            assert r["observation_index"] == t, (ep, t)
            text = f"Action:{r['memory_target']['action_text']}\n{delta}"
            differs = r["previous_memory_text"] != previous or r["memory_target_text"] != text
            if differs and not has_hidden:
                raise AssertionError(f"{ep} t={t}: label changed in a task without hidden items")
            changed_by_task[task] += differs
            sub, task_line = delta.split("\n")
            r["previous_memory_text"], r["memory_target_text"] = previous, text
            r["memory_target"].update(subgoal_delta=sub[len("Subgoal:"):], task_text=task_line[len("Task:"):],
                                      confirmed_goals=sum(it.done for it in now.items), total_goals=prior.length or 0)
            stream.write(json.dumps(r) + "\n")
            rows_total += 1
    temporary.replace(dataset / "rows" / f"{ep}.jsonl")
    goals[task] = rows[0]["memory_target"]["total_goals"]
(dataset / "priors.json").write_text(json.dumps({str(k): {"fixed": {str(i): t for i, t in v.fixed.items()},
                                                          "length": v.length, "demos": v.demos}
                                                 for k, v in sorted(priors.items())}, indent=1))
print("CAUSAL rows", rows_total, "rows changed by the causal rule, by task", dict(changed_by_task), flush=True)

# ---------------------------------------------------------------- 2. split, index, normalization, manifest (rma_dataset.py)
episodes = defaultdict(list)
for ep in built:
    episodes[int(ep[1:3])].append(ep)
split = {}
for task, names in episodes.items():
    names.sort(key=lambda n: int(n.split("seed")[1]))
    for name in names:
        split[name] = "val" if name == names[-1] else "train"
counts = {name: sum(1 for _ in open(dataset / "rows" / f"{name}.jsonl")) for name in split}
for part in ("train", "val"):
    with open(dataset / f"{part}.jsonl.tmp", "w") as out:
        for name in sorted(n for n in split if split[n] == part):
            out.write(open(dataset / "rows" / f"{name}.jsonl").read())
    os.replace(dataset / f"{part}.jsonl.tmp", dataset / f"{part}.jsonl")

states, actions = [], []
for name in split:
    if split[name] == "train":
        z = np.load(dataset / "episodes" / f"{name}.npz")
        states.append(z["states"]), actions.append(z["actions"])
stats = {}
for key, values in (("state", np.concatenate(states)), ("actions", np.concatenate(actions))):
    stats[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                  "q01": np.quantile(values, 0.01, axis=0).tolist(), "q99": np.quantile(values, 0.99, axis=0).tolist(),
                  "min": values.min(0).tolist(), "max": values.max(0).tolist()}
(dataset / "norm_stats.json").write_text(json.dumps(stats, indent=1))

records = [{"task_id": int(n[1:3]), "episode_id": n, "split": split[n], "actions": counts[n], "queries": counts[n],
            "episode_npz": f"episodes/{n}.npz"} for n in sorted(split)]
splits = {}
for part in ("train", "val"):
    chosen = [r for r in records if r["split"] == part]
    tasks = {}
    for task in sorted(episodes):
        members = [r for r in chosen if r["task_id"] == task]
        tasks[str(task)] = {"episodes": len(members), "actions": sum(r["actions"] for r in members),
                            "queries": sum(r["queries"] for r in members), "episode_ids": [r["episode_id"] for r in members]}
    splits[part] = {"index": f"{part}.jsonl", "index_sha256": sha(dataset / f"{part}.jsonl"), "episodes": len(chosen),
                    "actions": sum(r["actions"] for r in chosen), "queries": sum(r["queries"] for r in chosen), "tasks": tasks}
track_shas = hashlib.sha256()
for ep in built:
    track_shas.update((ep + ":" + json.loads((dataset / "log" / f"{ep}.json").read_text())["tracks"]["sha256"] + "\n").encode())
manifest = {"schema_version": 1, "complete": True, "splits": splits,
            "normalization": {"sha256": sha(dataset / "norm_stats.json"), "fit_split": "train",
                              "episodes": [{"task_id": r["task_id"], "episode_id": r["episode_id"]} for r in records if r["split"] == "train"],
                              "dense_rows": splits["train"]["actions"]},
            "integrity": {"episode_disjoint": True},
            "episodes": records,
            "source": {"episodes": len(records), "actions": sum(r["actions"] for r in records),
                       "queries": sum(r["queries"] for r in records)},
            "label_rule": "causal_v1 (causal_labels.py): an item's text appears only once its subtask has started, "
                          "unless it is identical in every training demo of the instruction",
            "v2r": {"builder": "rma_build_v2r + rma_assemble_v2r", "tracks_dir": str(args.tracks),
                    "track_files_sha256": track_shas.hexdigest(), "partial": bool(args.partial),
                    "raw_image": "images/<ep>/NNNNNN_raw.png: clean front frame, 256 -> 224 PIL BILINEAR",
                    "overlay": "CanonicalRenderer on stored tracks; only seen positions; retired objects not drawn, trail cleared",
                    "action_line": "rma_build.action_text; 'occluded' recomputed from the stored tracks' drawn flag (frames t-20..t)"}}
(dataset / "split_manifest.json").write_text(json.dumps(manifest, indent=1))

# ---------------------------------------------------------------- 3. contract
contract = json.loads((args.contract_in or code / "contract_C_v2.json").read_text(encoding="utf-8"))
old_tasks = {int(t["task_id"]): t for t in contract["source"]["tasks"]}
tasks = []
for task in sorted(episodes):
    members = [r for r in records if r["task_id"] == task]
    entry = dict(old_tasks[task])
    # required_goals is kept as the contract has it (the training code does not read it)
    new = {"metadata_episode_count": len(members), "metadata_frame_count": sum(r["actions"] for r in members)}
    if not args.partial and any(entry.get(k) != v for k, v in new.items()):
        raise SystemExit(f"task {task}: dataset {new} differs from the contract's source.tasks entry {entry}")
    entry.update(new)
    tasks.append(entry)
if not args.partial and set(old_tasks) != set(episodes):
    raise SystemExit("task membership differs from the contract")
contract["source"]["tasks"] = tasks
contract["dataset"]["latents_dir"] = str(args.latents_dir)
args.contract_out.parent.mkdir(parents=True, exist_ok=True)
args.contract_out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("DATASET", {q: splits[q]["episodes"] for q in splits}, {q: splits[q]["queries"] for q in splits},
      "tasks", len(tasks), "contract", args.contract_out, flush=True)
print("ASSEMBLE_DONE", flush=True)
