"""Outcome-aware latent target: what happened to the bowl relative to the plates and the hand.

Per frame and bowl, from the 21-frame window (observable signals only: mask centres, end-effector
state, finger gap): displacement, path, end speed, distance to the nearest plate, distance to the
hand, whether it moved with the hand, finger gap, visibility. Standardized on training data with
moving and still windows weighted equally, lifted to 64-d by a fixed orthonormal map (which keeps
cosines unchanged and the existing model code), and scaled to norm 8. No training, no encoder.

Writes OUT/<episode>.npz with z[N,K,64], valid[N,K]. Usage: physical_targets.py OUT
"""
import json
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
INPUTS = HOME / "pi05-ledger-A/inputs"
DATASET = HOME / "pi05-abc/dataset"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
WINDOW, DIM = 21, 11
index = {e["name"]: e for e in json.loads((INPUTS / "index.json").read_text())["episodes"]}
camera = json.loads((HOME / "pi05-ledger-A/probe/camera_probe.json").read_text())
manifest = json.loads((DATASET / "split_manifest.orig.json").read_text())
train = {f"t{int(e['task_id']):02d}_{e['episode_id']}" for e in manifest["episodes"] if e["split"] == "train"}


def project(points, task_id):
    cam = camera[str(task_id)]
    K, E = np.asarray(cam["K"]), np.asarray(cam["cam_to_world"])
    w2c = np.linalg.inv(E)
    pc = w2c[:3, :3] @ points.T + w2c[:3, 3:4]
    uv = (K @ pc)[:2] / pc[2:3]
    return np.stack([uv[0] - 0.5, uv[1] - 0.5], -1)


def episode_features(z, meta):
    reg = meta["registry"]
    sids = [str(s) for s in z["sids"]]
    bowls = sorted([s for s in sids if reg[s]["category"] == "bowl"], key=lambda s: reg[s]["segmentation_id"])
    plates = [s for s in sids if reg[s]["category"] == "plate"]
    centres, visible = z["centres"].astype(float), z["visibility"].astype(bool)
    states = z["states"].astype(float)
    n = len(states)
    eef = project(states[:, :3], meta["task_id"])
    width = states[:, 6] - states[:, 7]
    plate_at = []
    for p in plates:
        j = sids.index(p)
        if visible[:, j].any():
            plate_at.append(centres[int(np.argmax(visible[:, j])), j])
    plate_at = np.asarray(plate_at)
    features = np.zeros((n, len(bowls), DIM), np.float32)
    valid = np.zeros((n, len(bowls)), bool)
    for k, sid in enumerate(bowls):
        j = sids.index(sid)
        for t in range(n):
            a = max(0, t - WINDOW + 1)
            frames = np.flatnonzero(visible[a:t + 1, j]) + a
            if not len(frames):
                continue
            seen, hand = centres[frames, j], eef[frames]
            steps, hand_steps = np.diff(seen, axis=0), np.diff(hand, axis=0)
            together = 0.0
            if len(steps):
                norm = np.linalg.norm(steps) * np.linalg.norm(hand_steps)
                together = float(np.sum(steps * hand_steps) / norm) if norm > 1e-6 else 0.0
            lengths = np.linalg.norm(steps, axis=1) if len(steps) else np.zeros(1)
            nearest = float(np.min(np.linalg.norm(plate_at - seen[-1], axis=1))) if len(plate_at) else 0.0
            features[t, k] = (
                (seen[-1, 0] - seen[0, 0]) / 50, (seen[-1, 1] - seen[0, 1]) / 50, lengths.sum() / 50, lengths[-5:].mean(),
                nearest / 50, np.linalg.norm(seen[-1] - hand[-1]) / 50, together,
                np.linalg.norm(seen[-1] - seen[0]) / max(float(np.linalg.norm(hand[-1] - hand[0])), 1.0),
                width[t] * 10, width[a:t + 1].mean() * 10, len(frames) / WINDOW)
            valid[t, k] = True
    return features, valid


tables = {}
paths = [(p.stem, p, index[p.stem]) for p in sorted(INPUTS.glob("t0*.npz"))]
paths += [(p.stem, p, index[p.stem.replace("_empty_hand", "").replace("_drop", "")]) for p in sorted((DATASET / "failures/episodes").glob("*.npz"))]
for number, (name, path, meta) in enumerate(paths):
    with np.load(path) as z:
        tables[name] = episode_features(z, meta)
    if number % 50 == 0:
        print("episode", number, "/", len(paths), flush=True)

# Training statistics, moving and still windows weighted equally.
samples = {"moving": [], "still": []}
for name, (features, valid) in tables.items():
    base = name.replace("_empty_hand", "").replace("_drop", "")
    if base in train:
        rows = features[valid]
        moving = rows[:, 2] > 3 / 50
        samples["moving"].append(rows[moving])
        samples["still"].append(rows[~moving])
rng = np.random.default_rng(0)
parts = []
for values in samples.values():
    values = np.concatenate(values)
    parts.append(values[rng.choice(len(values), 50000)])
balanced = np.concatenate(parts)
mean, std = balanced.mean(axis=0), balanced.std(axis=0) + 1e-6
lift, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(64, DIM)))
for name, (features, valid) in tables.items():
    table = np.zeros((*valid.shape, 64), np.float32)
    scaled = (features[valid] - mean) / std
    lifted = scaled @ lift.T
    table[valid] = lifted / (np.linalg.norm(lifted, axis=1, keepdims=True) + 1e-8) * 8.0
    np.savez_compressed(out / f"{name}.npz", z=table, valid=valid)
pairs = balanced / std - mean / std
a, b = pairs[rng.choice(len(pairs), 4000)], pairs[rng.choice(len(pairs), 4000)]
cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
(out / "stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "random_pair_cos": float(cos.mean())}))
print("PHYSICAL_DONE", len(tables), "random pair cosine", round(float(cos.mean()), 3), flush=True)
