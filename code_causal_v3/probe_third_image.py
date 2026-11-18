"""v2 check: is the clean front image (right_wrist_0_rgb slot) really embedded and attended to?

Usage: probe_third_image.py CONTRACT DATASET [CHECKPOINT] [--rows N]
One GPU. Loads the base weights of the contract (or a finished checkpoint), builds N real rows and runs the
same joint forward several times with fixed noise/time/rng, changing ONLY the third image:
  zeros      clean image replaced by zeros (a grey image after the [-1,1] transform)
  swapped    clean image replaced by another row's clean image
  as_overlay clean image replaced by the row's own overlay image (what a silent fallback would feed)
  masked     image_masks["right_wrist_0_rgb"] = False (tokens present but unattended)
and reports the change of the action velocity, the text loss and the flow loss, the prefix token count,
and the gradient of the loss with respect to each image. Prints one JSON object.
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from model_data import FullCorpus, JointTokenizer, read_json
from policy import bootstrap_openpi

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rows_n = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 4
if "--rows" in sys.argv:
    args.remove(str(rows_n))
contract, dataset = read_json(args[0]), Path(args[1])
checkpoint = Path(args[2]) if len(args) > 2 else None
bootstrap_openpi({"openpi_root": contract["model"]["openpi_root"]})

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from joint_model import config_from_contract, initialize_from_base, load_full_checkpoint  # noqa: E402

tokenizer = JointTokenizer(contract["model"]["tokenizer"])
corpus = FullCorpus(dataset, contract, tokenizer)
if checkpoint is None:
    capacities, _ = corpus.measure_capacities()
    tokenizer.capacities = capacities
    model, _ = initialize_from_base(config_from_contract(contract, capacities),
                                    contract["model"]["base_checkpoint"], int(contract["training"]["seed"]))
else:
    model, metadata = load_full_checkpoint(checkpoint)
    capacities = metadata["capacities"]
    tokenizer.capacities = capacities
# rows from different episodes / times so that "swapped" is a genuinely different picture
indices = np.linspace(40, len(corpus.rows) - 40, rows_n).astype(np.int64)
batch = jax.tree.map(jnp.asarray, corpus.batch_indices(indices))
observation = batch["observation"]
names = tuple(observation.images)
raw, overlay = observation.images["right_wrist_0_rgb"], observation.images["base_0_rgb"]
report = {"weights": str(checkpoint or contract["model"]["base_checkpoint"]), "rows": [corpus.rows[int(i)]["sample_id"] for i in indices],
          "image_keys_in_model_order": list(names), "image_shapes": {k: list(v.shape) for k, v in observation.images.items()},
          "image_masks": {k: np.asarray(v).tolist() for k, v in observation.image_masks.items()},
          "raw_vs_overlay_mean_abs_pixel_diff": float(jnp.mean(jnp.abs(raw - overlay))),
          "capacities": capacities}

graph, state = nnx.split(model)
rng = jax.random.PRNGKey(7)


def with_raw(image=None, mask=None):
    images = dict(observation.images)
    masks = dict(observation.image_masks)
    if image is not None:
        images["right_wrist_0_rgb"] = image
    if mask is not None:
        masks["right_wrist_0_rgb"] = mask
    return dataclasses.replace(observation, images=images, image_masks=masks)


@jax.jit
def run(state, observation):
    m = nnx.merge(graph, state)
    prefix, valid = m.prefix(observation, batch["prev_latents"])
    total, metrics = m.loss(rng, dict(batch, observation=observation))
    noisy = jax.random.normal(rng, batch["actions"].shape)
    time = jnp.full((noisy.shape[0],), 0.5, jnp.float32)
    velocity = m.teacher_velocity(observation, noisy, time, batch["target_ids"], batch["target_mask"],
                                  batch["prev_latents"], batch["z_star"])
    return prefix.shape[1], jnp.sum(valid, axis=1), velocity.astype(jnp.float32), metrics


def summary(result):
    _, valid, velocity, metrics = result
    return np.asarray(velocity), {k: float(metrics[k]) for k in ("loss", "flow32", "flow7", "text_ce", "align", "verb")}, np.asarray(valid)


base = run(state, observation)
base_v, base_m, base_valid = summary(base)
again_v, _, _ = summary(run(state, observation))
prompt_len = observation.tokenized_prompt.shape[1]
prompt_valid = np.asarray(jnp.sum(observation.tokenized_prompt_mask, axis=1))
report.update({"prefix_tokens_total": int(base[0]), "prompt_capacity": int(prompt_len),
               "image_tokens_total": int(base[0]) - int(prompt_len),
               "valid_prefix_tokens_per_row": base_valid.tolist(), "valid_prompt_tokens_per_row": prompt_valid.tolist(),
               "baseline": base_m, "repeat_same_input_velocity_max_abs_diff": float(np.max(np.abs(again_v - base_v)))})
assert int(base[0]) == 3 * 256 + prompt_len, "prefix is not 3 x 256 image tokens + prompt"
assert np.array_equal(base_valid, 768 + prompt_valid), "not every third-image token is valid"
variants = {"zeros": with_raw(image=jnp.zeros_like(raw)), "swapped": with_raw(image=jnp.roll(raw, 1, axis=0)),
            "as_overlay": with_raw(image=overlay), "masked": with_raw(mask=jnp.zeros_like(observation.image_masks["right_wrist_0_rgb"]))}
report["interventions"] = {}
for name, changed in variants.items():
    v, m, valid = summary(run(state, changed))
    report["interventions"][name] = {
        "velocity_max_abs_diff": float(np.max(np.abs(v - base_v))), "velocity_mean_abs_diff": float(np.mean(np.abs(v - base_v))),
        "velocity7_mean_abs_diff": float(np.mean(np.abs(v[..., :7] - base_v[..., :7]))),
        "baseline_velocity_mean_abs": float(np.mean(np.abs(base_v))),
        "valid_prefix_tokens_per_row": valid.tolist(),
        **{f"d_{k}": m[k] - base_m[k] for k in ("flow32", "flow7", "text_ce")}, **m}


@jax.jit
def image_gradients(state, images):
    m = nnx.merge(graph, state)
    return jax.grad(lambda im: m.loss(rng, dict(batch, observation=dataclasses.replace(observation, images=im)))[0])(images)


gradients = image_gradients(state, dict(observation.images))
report["loss_gradient_l2_per_image"] = {k: float(jnp.sqrt(jnp.sum(jnp.square(v.astype(jnp.float32))))) for k, v in gradients.items()}
used = all(report["interventions"][k]["velocity_max_abs_diff"] > 100 * max(report["repeat_same_input_velocity_max_abs_diff"], 1e-6)
           for k in variants) and report["loss_gradient_l2_per_image"]["right_wrist_0_rgb"] > 0
report["third_image_used"] = bool(used)
print("THIRD_IMAGE_REPORT " + json.dumps(report))
if not used:
    raise SystemExit("third image does not influence the outputs")
