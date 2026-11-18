"""Per-bowl memory representations on held-out rows, for t-SNE. Teacher history and teacher text.

B, C : the projected latent at each <LAT> slot (full z, and C/B first 16 raw dims "z_sem").
A    : no latent exists; the 2048-d hidden state at the end of each Action clause (same place in
       the sequence), which is the closest comparable representation, not a latent.
Every group uses the same verb-balanced sample of (row, bowl) windows; unreliable rows excluded.
Usage: extract_latents.py GROUP CHECKPOINT DATASET OUT.npz
"""
import json
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from model_data import FullCorpus, JointTokenizer, read_json
from policy import bootstrap_openpi

group, checkpoint, dataset, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
PER_CATEGORY, BATCH = 500, 16
metadata = read_json(checkpoint / "metadata.json")
bootstrap_openpi(metadata)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from joint_model import load_full_checkpoint  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402

model, metadata = load_full_checkpoint(checkpoint)
contract = deepcopy(metadata["contract"])
contract["dataset"]["index"] = "val.jsonl"
tokenizer = JointTokenizer(contract["model"]["tokenizer"], metadata["capacities"])
corpus = FullCorpus(dataset, contract, tokenizer)
inputs = Path.home() / "pi05-ledger-A/inputs"
index = {e["name"]: e for e in json.loads((inputs / "index.json").read_text())["episodes"]}


def category(body):
    """One colour per window: an event if present, else the latest direction, else still/occluded."""
    if body in ("still", "occluded"):
        return body
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", body)
    for event in ("placed", "picked"):
        if event in words:
            return event
    latest = words[-1]
    return {"forward": "forward/backward", "backward": "forward/backward",
            "left": "left/right", "right": "left/right"}.get(latest, latest)


tracks = {}
def displacement(name, t, k):
    """Largest image displacement (px) of this bowl in the window, from recorded centres."""
    if name not in tracks:
        z, reg = np.load(inputs / f"{name}.npz"), index[name]["registry"]
        sids = [str(s) for s in z["sids"]]
        bowls = sorted([s for s in sids if reg[s]["category"] == "bowl"], key=lambda s: reg[s]["segmentation_id"])
        columns = [sids.index(s) for s in bowls]
        tracks[name] = (z["centres"][:, columns].astype(float), z["visibility"][:, columns].astype(bool))
    centres, visible = tracks[name]
    a = max(0, t - 20)
    points = centres[a:t + 1, k][visible[a:t + 1, k]]
    return float(np.max(np.linalg.norm(points - points[0], axis=1))) if len(points) else 0.0


windows = defaultdict(list)
for i, row in enumerate(corpus.rows):
    if not row["memory_target"]["action_reliable"]:
        continue
    for k, clause in enumerate(row["memory_target"]["action_text"].split(" | ")):
        windows[category(clause.split(": ", 1)[1])].append((i, k))
rng = np.random.default_rng(42)
chosen = []
for name in sorted(windows):
    pool = windows[name]
    picks = rng.choice(len(pool), min(PER_CATEGORY, len(pool)), replace=False)
    chosen += [(pool[j][0], pool[j][1], name) for j in picks]
print({k: len(v) for k, v in sorted(windows.items())}, "chosen", len(chosen), flush=True)
by_row = defaultdict(list)
for i, k, name in chosen:
    by_row[i].append((k, name))
rows = sorted(by_row)

forward = nnx_utils.module_jit(model.joint_forward)
key = jax.random.PRNGKey(0)
sep_id, newline_id = tokenizer.sp.piece_to_id("▁|"), tokenizer.newline_id
vectors, sems, categories, tasks, motions, ids = [], [], [], [], [], []
for start in range(0, len(rows), BATCH):
    part = rows[start:start + BATCH]
    batch = corpus.batch_indices(np.asarray(part + [-1] * (BATCH - len(part)), np.int64))
    noisy = jax.random.normal(key, batch["actions"].shape)
    _, hidden = forward(jax.tree.map(jnp.asarray, batch["observation"]), noisy, jnp.full((BATCH,), 0.5, jnp.float32),
                        jnp.asarray(batch["target_ids"]), jnp.asarray(batch["target_mask"]),
                        None if group == "A" else jnp.asarray(batch["prev_latents"]),
                        None if group == "A" else jnp.asarray(batch["z_star"]))
    hidden = np.asarray(hidden.astype(jnp.float32))
    for b, i in enumerate(part):
        tokens = [int(x) for x, m in zip(batch["target_ids"][b], batch["target_mask"][b]) if m]
        row = corpus.rows[i]
        count = len(row["memory_target"]["action_text"].split(" | "))
        end = tokens.index(newline_id)
        if group == "A":
            positions = [j for j, t in enumerate(tokens[:end]) if t == sep_id] + [end]
        else:
            positions = [j for j, t in enumerate(tokens[:end]) if t == tokenizer.lat_id]
        if len(positions) != count:
            raise ValueError(f"{row['sample_id']}: {len(positions)} positions for {count} clauses")
        name = row["sample_id"].rsplit("_", 1)[0]
        for k, label in by_row[i]:
            h = jnp.asarray(hidden[b, positions[k]])[None]
            if group == "A":
                vector, sem = np.asarray(h[0]), np.zeros(1, np.float32)
            else:
                raw, z = model._project_latent(h)
                vector, sem = np.asarray(z[0]), np.asarray(raw[0, :model.sem_dim])
            vectors.append(vector); sems.append(sem); categories.append(label)
            tasks.append(int(row["task_id"])); ids.append(f"{row['sample_id']}/{k}")
            motions.append(displacement(name, int(row["observation_index"]), k))
    if start % (BATCH * 20) == 0:
        print("batch", start, "/", len(rows), flush=True)
np.savez_compressed(out, vectors=np.stack(vectors), sems=np.stack(sems), categories=np.array(categories),
                    tasks=np.array(tasks), motions=np.array(motions), ids=np.array(ids))
print("EXTRACTED", group, np.stack(vectors).shape, flush=True)
