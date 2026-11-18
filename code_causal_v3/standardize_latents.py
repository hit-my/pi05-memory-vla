"""Remove the common mean of z* so that cosine alignment is informative.

Raw encoder output is dominated by a shared direction (group B: cos of random window
pairs = 0.99), so 1-cos(z, z*) could be near zero for a constant z. Per-dimension
standardization with train-only, verb-balanced statistics is an invertible linear map,
so no trajectory information is lost. Same procedure for B and C.

Usage: standardize_latents.py GROUP   (moves the raw tables to latents_raw/GROUP once)
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from target_encoder import knn_accuracy

group = sys.argv[1]
base_name = sys.argv[2] if len(sys.argv) > 2 else "latents"
root = Path.home() / "pi05-abc"
latents, raw = root / base_name / group, root / f"{base_name}_raw" / group
if not raw.exists():
    raw.parent.mkdir(exist_ok=True)
    shutil.move(str(latents), str(raw))
latents.mkdir(parents=True, exist_ok=True)

windows = {"train": defaultdict(list), "val": defaultdict(list)}
for split in ("train", "val"):
    tables = {}
    for line in open(root / "dataset" / f"{split}.jsonl"):
        row = json.loads(line)
        name = row["sample_id"].rsplit("_", 1)[0]
        if name not in tables:
            with np.load(raw / f"{name}.npz") as archive:
                tables[name] = (archive["z"], archive["valid"])
        z, valid = tables[name]
        t = row["observation_index"]
        for k, clause in enumerate(row["memory_target"]["action_text"].split(" | ")):
            if valid[t, k]:
                windows[split][clause.split(": ", 1)[1]].append(z[t, k])

rng = np.random.default_rng(0)
balanced = []
for values in windows["train"].values():
    values = np.asarray(values)
    balanced.append(values[rng.choice(len(values), 2000)])
balanced = np.concatenate(balanced)
mean, std = balanced.mean(axis=0), balanced.std(axis=0) + 1e-6

count = 0
for path in sorted(raw.glob("*.npz")):
    with np.load(path) as archive:
        z, valid = archive["z"], archive["valid"]
    out = np.zeros_like(z)
    scaled = (z[valid] - mean) / std
    out[valid] = scaled / (np.linalg.norm(scaled, axis=1, keepdims=True) + 1e-8) * 8.0
    np.savez_compressed(latents / path.name, z=out.astype(np.float32), valid=valid)
    count += 1


def transform(x):
    scaled = (np.asarray(x) - mean) / std
    return scaled / (np.linalg.norm(scaled, axis=-1, keepdims=True) + 1e-8)


def sample(split, keep, n=4000):
    pool = [v for key, values in windows[split].items() if keep(key) for v in values]
    return transform(np.asarray(pool)[rng.choice(len(pool), n)])


still = lambda key: key == "still"
moving = lambda key: key not in ("still", "occluded")
labels = {key: i for i, key in enumerate(sorted(set(windows["train"]) | set(windows["val"])))}


def labelled(split, cap):
    xs, ys = [], []
    for key, values in windows[split].items():
        if moving(key):
            values = np.asarray(values)
            xs.append(values[rng.choice(len(values), min(cap, len(values)), replace=False)])
            ys.append(np.full(len(xs[-1]), labels[key]))
    return transform(np.concatenate(xs)), np.concatenate(ys)


train_x, train_y = labelled("train", 3000)
val_x, val_y = labelled("val", 100000)
report = {"group": group, "episodes": count, "dim_std_ratio_max_min": float(std.max() / std.min()),
          "cos_still_still": float(np.mean(np.sum(sample("train", still) * sample("train", still), axis=1))),
          "cos_moving_moving": float(np.mean(np.sum(sample("train", moving) * sample("train", moving), axis=1))),
          "cos_still_moving": float(np.mean(np.sum(sample("train", still) * sample("train", moving), axis=1))),
          "knn5_verb_sequence_acc_val_moving": knn_accuracy(train_x, train_y, val_x, val_y)}
(latents / "standardize.json").write_text(json.dumps({**report, "mean": mean.tolist(), "std": std.tolist()}))
print("STANDARDIZED " + json.dumps(report), flush=True)
