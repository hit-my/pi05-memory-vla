"""Do the latents actually matter? Ablations on a saved B or C checkpoint, held-out rows, fixed noise.

1. Previous-round latents into the prefix: true z*(t-20) vs zeros vs shuffled across rows.
   Effect on sampled actions (error against the expert, and change from the true-latent output),
   and on text cross-entropy by section.
2. Current-round latents read by later text: true z* vs zeros vs shuffled. Effect on Subgoal/Task CE.
3. Beyond verb text: cos(z, z*) against the best a verb-only code could do (mean z* of the same
   verb sequence), on moving windows.
Usage: latent_effect.py CHECKPOINT DATASET [rows=320]
"""
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from model_data import FullCorpus, JointTokenizer, read_json
from policy import bootstrap_openpi

checkpoint, dataset = Path(sys.argv[1]), Path(sys.argv[2])
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 320
BATCH = 32
metadata = read_json(checkpoint / "metadata.json")
bootstrap_openpi(metadata)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from joint_model import load_full_checkpoint  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
import optax  # noqa: E402

model, metadata = load_full_checkpoint(checkpoint)
contract = deepcopy(metadata["contract"])
contract["dataset"]["index"] = "val.jsonl"
tokenizer = JointTokenizer(contract["model"]["tokenizer"], metadata["capacities"])
corpus = FullCorpus(dataset, contract, tokenizer)
rng = np.random.default_rng(0)

# Half of the rows have a moving bowl in the previous window, where the previous latent has content.
moving, still = [], []
for i, row in enumerate(corpus.rows):
    if int(row["observation_index"]) < 20:
        continue
    text = row["memory_target"]["action_text"]
    (moving if any(not c.endswith(("still", "occluded")) for c in text.split(" | ")) else still).append(i)
chosen = np.r_[rng.choice(moving, ROWS // 2, replace=False), rng.choice(still, ROWS // 2, replace=False)]
rng.shuffle(chosen)
chosen = chosen[:ROWS // BATCH * BATCH].tolist()

prefix_fn = nnx_utils.module_jit(model.prefix)
sample_fn = nnx_utils.module_jit(model.sample_from_prefix)
forward_fn = nnx_utils.module_jit(model.joint_forward)
decode = nnx_utils.module_jit(lambda h: model.target_ce_logits(h)) if hasattr(model, "target_ce_logits") else None


def logits_of(hidden):
    from joint_model import tied_decode
    return model.PaliGemma.llm(hidden, method=tied_decode).astype(jnp.float32)


logits_fn = nnx_utils.module_jit(logits_of) if False else None


def section_ce(hidden, batch):
    """Token CE split into Action / Subgoal / Task lines."""
    from joint_model import tied_decode
    logits = np.asarray(jax.device_get(model.PaliGemma.llm(hidden, method=tied_decode).astype(jnp.float32)))
    ids, mask = batch["target_ids"], batch["target_mask"]
    losses = np.asarray(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(logits), jnp.asarray(np.where(mask, ids, 0))))
    total = defaultdict(float)
    count = defaultdict(int)
    for b in range(ids.shape[0]):
        line = 0
        for t in range(ids.shape[1]):
            if not mask[b, t]:
                continue
            name = ("action", "subgoal", "task")[min(line, 2)]
            total[name] += float(losses[b, t])
            count[name] += 1
            if ids[b, t] == tokenizer.newline_id:
                line += 1
    return total, count


results = defaultdict(lambda: defaultdict(float))
counts = defaultdict(lambda: defaultdict(int))
pairs_true, pairs_other = [], []
key = jax.random.PRNGKey(7)
for start in range(0, len(chosen), BATCH):
    part = chosen[start:start + BATCH]
    batch = corpus.batch_indices(np.asarray(part, np.int64))
    observation = jax.tree.map(jnp.asarray, batch["observation"])
    true_prev, z_star = jnp.asarray(batch["prev_latents"]), jnp.asarray(batch["z_star"])
    variants_prev = {"true": true_prev, "zero": jnp.zeros_like(true_prev), "shuffled": jnp.roll(true_prev, 1, axis=0)}
    actions_true = None
    for name, latents in variants_prev.items():
        prefix, valid = prefix_fn(observation, latents)
        keys = key
        sampled = np.asarray(sample_fn(keys, observation, prefix, valid))
        mask = batch["action_mask"][..., None]
        target = batch["actions"]
        error = float(np.sum(((sampled[..., :7] - target[..., :7]) ** 2) * mask) / (mask.sum() * 7))
        results["prev_action_mse"][name] += error
        counts["prev_action_mse"][name] += 1
        if name == "true":
            actions_true = sampled
        else:
            results["prev_action_change_from_true"][name] += float(np.mean((sampled[..., :7] - actions_true[..., :7]) ** 2))
            counts["prev_action_change_from_true"][name] += 1
        noisy = jax.random.normal(key, batch["actions"].shape)
        _, hidden = forward_fn(observation, noisy, jnp.full((BATCH,), 0.5, jnp.float32), jnp.asarray(batch["target_ids"]),
                               jnp.asarray(batch["target_mask"]), latents, z_star)
        total, count = section_ce(hidden, batch)
        for section in total:
            results[f"prev_ce_{section}"][name] += total[section]
            counts[f"prev_ce_{section}"][name] += count[section]
    variants_cur = {"true": z_star, "zero": jnp.zeros_like(z_star), "shuffled": jnp.roll(z_star, 1, axis=0)}
    for name, latents in variants_cur.items():
        noisy = jax.random.normal(key, batch["actions"].shape)
        _, hidden = forward_fn(observation, noisy, jnp.full((BATCH,), 0.5, jnp.float32), jnp.asarray(batch["target_ids"]),
                               jnp.asarray(batch["target_mask"]), true_prev, latents)
        total, count = section_ce(hidden, batch)
        for section in total:
            results[f"current_ce_{section}"][name] += total[section]
            counts[f"current_ce_{section}"][name] += count[section]
        if name == "true":
            ids, mask = batch["target_ids"], batch["target_mask"]
            is_lat = (ids == tokenizer.lat_id) & mask
            h = np.asarray(hidden.astype(jnp.float32))
            for b, i in enumerate(part):
                row = corpus.rows[i]
                clauses = row["memory_target"]["action_text"].split(" | ")
                slots = np.flatnonzero(is_lat[b])
                valid_z = corpus.z_valid(row)
                for k, clause in enumerate(clauses):
                    if not valid_z[k] or k >= len(slots):
                        continue
                    _, z = model._project_latent(jnp.asarray(h[b, slots[k]])[None])
                    pairs_true.append((clause.split(": ", 1)[1], np.asarray(z[0]) / 8.0, np.asarray(batch["z_star"][b, k]) / 8.0))
    print("batch", start, flush=True)

report = {"group": corpus.variant, "rows": len(chosen)}
for metric, values in results.items():
    report[metric] = {name: round(value / max(counts[metric][name], 1), 5) for name, value in values.items()}

# Beyond verbs: the best verb-only code predicts the mean z* of the same verb sequence.
table = defaultdict(list)
for verb, _, target in pairs_true:
    table[verb].append(target)
means = {verb: np.mean(values, axis=0) for verb, values in table.items()}
for name, chooser in (("moving", lambda v: v not in ("still", "occluded")), ("still", lambda v: v == "still")):
    rows = [p for p in pairs_true if chooser(p[0])]
    model_cos = [float(z @ t) for _, z, t in rows]
    verb_cos = [float(means[v] @ t / (np.linalg.norm(means[v]) + 1e-9)) for v, _, t in rows]
    shuffle_cos = [float(rows[(i + 1) % len(rows)][1] @ rows[i][2]) for i in range(len(rows))]
    report[f"z_{name}"] = {"windows": len(rows), "cos_model_vs_zstar": round(float(np.mean(model_cos)), 4),
                           "cos_verb_mean_vs_zstar": round(float(np.mean(verb_cos)), 4),
                           "cos_other_rows_z_vs_zstar": round(float(np.mean(shuffle_cos)), 4)}
print("LATENT_EFFECT " + json.dumps(report, indent=1), flush=True)
