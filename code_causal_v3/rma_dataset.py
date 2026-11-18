"""Assemble the RoboMemArena pilot dataset from rma_build.py output: train/val split (the last seed of each
task is validation), training-only normalization, split manifest, and contracts for A and C.

Usage: rma_dataset.py DATASET STEPS   (run in ~/SANE/.venv; writes contracts into this code directory)
"""
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import rma_spec

dataset, steps = Path(sys.argv[1]), int(sys.argv[2])
CODE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


episodes = defaultdict(list)
for path in sorted((dataset / "rows").glob("*.jsonl")):
    task = int(path.stem[1:3])
    episodes[task].append(path.stem)
split = {}
for task, names in episodes.items():
    names.sort(key=lambda n: int(n.split("seed")[1]))
    for name in names:
        split[name] = "val" if name == names[-1] else "train"

counts = {}
for name in split:
    counts[name] = sum(1 for _ in open(dataset / "rows" / f"{name}.jsonl"))
for part in ("train", "val"):
    with open(dataset / f"{part}.jsonl", "w") as out:
        for name in sorted(n for n in split if split[n] == part):
            out.write(open(dataset / "rows" / f"{name}.jsonl").read())

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
manifest = {"schema_version": 1, "complete": True, "splits": splits,
            "normalization": {"sha256": sha(dataset / "norm_stats.json"), "fit_split": "train",
                              "episodes": [{"task_id": r["task_id"], "episode_id": r["episode_id"]} for r in records if r["split"] == "train"],
                              "dense_rows": splits["train"]["actions"]},
            "integrity": {"episode_disjoint": True},
            "episodes": records,
            "source": {"episodes": len(records), "actions": sum(r["actions"] for r in records),
                       "queries": sum(r["queries"] for r in records)}}
(dataset / "split_manifest.json").write_text(json.dumps(manifest, indent=1))

tasks = []
for task in sorted(episodes):
    members = [r for r in records if r["task_id"] == task]
    first = json.loads(open(dataset / "rows" / f"{members[0]['episode_id']}.jsonl").readline())
    folder = rma_spec.TASKS[task]["folder"] or f"task{task}"
    tasks.append({"task_id": task, "name": folder, "filename": folder,
                  "metadata_episode_count": len(members), "metadata_frame_count": sum(r["actions"] for r in members),
                  "required_goals": first["memory_target"]["total_goals"]})
for group in ("A", "C"):
    contract = json.loads((CODE / f"contract_{group}_ft.json").read_text())
    contract["project"] = f"pi05_ledger_{group}_rma"
    contract["source"]["tasks"] = tasks
    contract["source"]["split_policy"] = "RoboMemArena pilot: last downloaded seed of each task is validation."
    contract["dataset"].pop("failure_index", None)
    contract["memory"].pop("failure_rows", None)
    contract["memory"]["minimum_visible_pixels_for_motion_label"] = rma_spec.MIN_PIXELS
    if group == "C":
        contract["dataset"]["latents_dir"] = str(Path.home() / "pi05-abc/rma/latents/C")
    t = contract["training"]
    devices = int(os.environ.get("PI05_DEVICES", 4))
    interval = int(os.environ.get("CKPT_INTERVAL", steps // 2))
    suffix = os.environ.get("CONTRACT_SUFFIX", "rma")
    t.update(optimizer_steps=steps, checkpoint_interval=interval, validation_interval=interval,
             learning_rate=float(os.environ.get("LR", 2.5e-5)), final_learning_rate=float(os.environ.get("FINAL_LR", 2.5e-6)),
             num_devices=devices, global_batch_size=16 * devices, per_device_batch_size=16)
    t["wandb"]["name"] = f"pi05-ledger-{group}-{suffix}"
    if group == "C" and os.environ.get("LATENTS_DIR"):
        contract["dataset"]["latents_dir"] = os.environ["LATENTS_DIR"]
    (CODE / f"contract_{group}_{suffix}.json").write_text(json.dumps(contract, indent=2) + "\n")
print("DATASET", {p: splits[p]["episodes"] for p in splits}, {p: splits[p]["queries"] for p in splits},
      [(t["task_id"], t["metadata_episode_count"], t["required_goals"]) for t in tasks])
