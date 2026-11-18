"""z* for group C on the RoboMemArena pilot: the same encoder recipe as target_encoder.py (C: rec + cons + SupCon),
trained on the tracker's centre tracks of the pilot data, then standardized as standardize_latents.py does.

Usage: rma_latents.py DATASET OUT   -> OUT/C/<episode>.npz (z[N,K,64], valid[N,K]) and OUT/C/standardize.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import target_encoder as te

dataset, out = Path(sys.argv[1]), Path(sys.argv[2])
split_of = {}
for split in ("train", "val"):
    for line in open(dataset / f"{split}.jsonl"):
        row = json.loads(line)
        split_of[row["episode_id"]] = split


def load_windows():
    xs, keys, verbs, reliable, splits, shapes = [], [], [], [], [], {}
    for path in sorted((dataset / "episodes").glob("*.npz")):
        name = path.stem
        if name not in split_of:
            continue
        z = np.load(path)
        centres, visible, texts = z["centres"].astype(np.float32), z["visibility"].astype(bool), z["action_text"]
        n, k_count = visible.shape
        shapes[name] = (n, k_count)
        for t in range(n):
            for k, clause in enumerate(str(texts[t]).split(" | ")):
                frames = np.arange(t - te.WINDOW + 1, t + 1)
                inside = frames >= 0
                vis = np.zeros(te.WINDOW, bool)
                vis[inside] = visible[frames[inside], k]
                pos = np.zeros((te.WINDOW, 2), np.float32)
                pos[inside] = centres[frames[inside], k]
                if vis.any():
                    pos = pos - pos[np.argmax(vis)]
                pos = np.where(vis[:, None], pos, 0.0) / te.PIXEL_SCALE
                xs.append(np.concatenate((pos, vis[:, None].astype(np.float32)), axis=1))
                keys.append((name, t, k))
                verbs.append(clause.split(": ", 1)[1])
                reliable.append(True)
                splits.append(split_of[name])
    return np.stack(xs).astype(np.float32), keys, verbs, np.array(reliable), np.array(splits), shapes


te.load_windows = load_windows
raw = out / "raw"
sys.argv = [sys.argv[0], "--group", "C", "--out", str(raw), "--steps", "3000", "--rec-weight", "30", "--sup-weight", "3"]
te.main()

# Standardize with train-only, verb-balanced statistics (standardize_latents.py).
windows = defaultdict(list)
tables = {}
for path in sorted((raw / "C").glob("*.npz")):
    with np.load(path) as archive:
        tables[path.stem] = (archive["z"], archive["valid"])
for name, (z, valid) in tables.items():
    if split_of.get(name) != "train":
        continue
    texts = np.load(dataset / "episodes" / f"{name}.npz")["action_text"]
    for t in range(len(z)):
        for k, clause in enumerate(str(texts[t]).split(" | ")):
            if valid[t, k]:
                windows[clause.split(": ", 1)[1]].append(z[t, k])
rng = np.random.default_rng(0)
balanced = np.concatenate([np.asarray(v)[rng.choice(len(v), 2000)] for v in windows.values()])
mean, std = balanced.mean(axis=0), balanced.std(axis=0) + 1e-6
final = out / "C"
final.mkdir(parents=True, exist_ok=True)
for name, (z, valid) in tables.items():
    result = np.zeros_like(z)
    scaled = (z[valid] - mean) / std
    result[valid] = scaled / (np.linalg.norm(scaled, axis=1, keepdims=True) + 1e-8) * 8.0
    np.savez_compressed(final / f"{name}.npz", z=result.astype(np.float32), valid=valid)
(final / "standardize.json").write_text(json.dumps({"episodes": len(tables), "mean": mean.tolist(), "std": std.tolist()}))
print("LATENTS_DONE", len(tables), flush=True)
