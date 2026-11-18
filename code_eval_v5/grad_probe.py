"""Per-loss gradient magnitudes on a saved checkpoint, and z* statistics, for choosing loss weights.

Usage: grad_probe.py CHECKPOINT DATASET [batch=16]
"""
import json
import sys
from pathlib import Path

import numpy as np

from model_data import FullCorpus, JointTokenizer, read_json
from policy import bootstrap_openpi

checkpoint, dataset = Path(sys.argv[1]), Path(sys.argv[2])
size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
metadata = read_json(checkpoint / "metadata.json")
bootstrap_openpi(metadata)

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import traverse_util  # noqa: E402
from joint_model import load_full_checkpoint, parameter_component  # noqa: E402

model, metadata = load_full_checkpoint(checkpoint)
contract = metadata["contract"]
tokenizer = JointTokenizer(contract["model"]["tokenizer"], metadata["capacities"])
corpus = FullCorpus(dataset, contract, tokenizer)
rng = np.random.default_rng(0)
batch = corpus.batch_indices(rng.choice(len(corpus.rows), size, replace=False))
batch = jax.tree.map(jnp.asarray, batch)
graph, params, other = nnx.split(model, nnx.Param, ...)
terms = {"A": ("flow7", "text_ce"), "B": ("flow7", "text_ce", "align"),
         "C": ("flow7", "text_ce", "align", "verb")}[corpus.variant]
weights = {"flow7": 1.0, "text_ce": contract["training"]["text_loss_weight"],
           "align": contract["training"].get("align_loss_weight", 0.0),
           "verb": contract["training"].get("verb_loss_weight", 0.0)}


def term_value(state, name):
    return nnx.merge(graph, state, other).loss(jax.random.PRNGKey(0), batch)[1][name]


report = {"group": corpus.variant, "step": metadata["sampler"]["optimizer_step"], "terms": {}}
for name in terms:
    value, gradient = jax.value_and_grad(lambda state: term_value(state, name))(params)
    flat = traverse_util.flatten_dict(gradient.to_pure_dict())
    groups = {}
    for path, leaf in flat.items():
        component = parameter_component(path)
        groups[component] = groups.get(component, 0.0) + float(jnp.sum(jnp.square(leaf)))
    norm = {key: value_ ** 0.5 for key, value_ in groups.items()}
    report["terms"][name] = {"value": float(value), "weight": weights[name],
                             "grad_norm_by_component": {k: round(v, 4) for k, v in norm.items()},
                             "weighted_vlm_grad": round(weights[name] * norm.get("vlm", 0.0), 4),
                             "weighted_vision_grad": round(weights[name] * norm.get("vision", 0.0), 4)}
if corpus.variant != "A":
    # Is z* informative, or nearly constant (which would make L_align trivially small)?
    z, moving = [], []
    for row in corpus.rows[::7]:
        clauses = row["memory_target"]["action_text"].split(" | ")
        table, valid = corpus.z_star(row), corpus.z_valid(row)
        for k, clause in enumerate(clauses):
            if valid[k]:
                z.append(table[k] / 8.0)
                moving.append(not clause.endswith("still"))
    z, moving = np.asarray(z), np.asarray(moving)
    pick = lambda mask, n=4000: z[rng.choice(np.flatnonzero(mask), n)]
    report["zstar"] = {"windows": len(z), "moving_fraction": float(moving.mean()),
                       "cos_still_still": float(np.mean(np.sum(pick(~moving) * pick(~moving), axis=1))),
                       "cos_moving_moving": float(np.mean(np.sum(pick(moving) * pick(moving), axis=1))),
                       "cos_still_moving": float(np.mean(np.sum(pick(~moving) * pick(moving), axis=1)))}
print("GRAD_PROBE " + json.dumps(report), flush=True)
