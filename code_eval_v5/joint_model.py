"""Joint pi0.5 flow and three-level text supervision, without upstream edits.

One shared prefix pass supplies differentiable KV caches to independent text
and action branches. Action shapes and positions never depend on teacher text.
Only target hidden states reach the existing tied vocabulary decoder.
All model parameters are trainable float32 masters; upstream transformer and
vision computation retain the configured bfloat16 execution dtype.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import gemma as _gemma
from openpi.models import model as model_lib
from openpi.models import pi0, pi0_config
from openpi.shared import nnx_utils

TRAINABLE_FILTER = nnx.Param


def block_mask_positions(prefix_valid, teacher_valid, action_valid):
    """Explicit forked attention and branch-local positions, including holes."""
    batch, p = prefix_valid.shape
    t, a = teacher_valid.shape[1], action_valid.shape[1]
    mask = jnp.zeros((batch, p+t+a, p+t+a), dtype=jnp.bool_)
    pp = prefix_valid[:, :, None] & prefix_valid[:, None, :]
    tp = teacher_valid[:, :, None] & prefix_valid[:, None, :]
    tt = (teacher_valid[:, :, None] & teacher_valid[:, None, :]
          & (jnp.arange(t)[:, None] >= jnp.arange(t)[None, :])[None])
    ap = action_valid[:, :, None] & prefix_valid[:, None, :]
    aa = action_valid[:, :, None] & action_valid[:, None, :]
    mask = mask.at[:, :p, :p].set(pp)
    mask = mask.at[:, p:p+t, :p].set(tp)
    mask = mask.at[:, p:p+t, p:p+t].set(tt)
    mask = mask.at[:, p+t:, :p].set(ap)
    mask = mask.at[:, p+t:, p+t:].set(aa)
    prefix_positions = jnp.cumsum(prefix_valid, axis=1, dtype=jnp.int32) - 1
    count = jnp.sum(prefix_valid, axis=1, dtype=jnp.int32)[:, None]
    text_positions = count + jnp.cumsum(teacher_valid, axis=1, dtype=jnp.int32) - 1
    action_positions = count + jnp.cumsum(action_valid, axis=1, dtype=jnp.int32) - 1
    return mask, jnp.concatenate((prefix_positions, text_positions, action_positions), axis=1)


def masked_flow_loss(prediction, target, action_mask):
    """Return detached native32 diagnostics and the supervised physical7 loss."""
    squared = jnp.square(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    count = jnp.maximum(jnp.sum(action_mask), 1)
    valid = action_mask[..., None]
    full = jnp.sum(jnp.where(valid, squared, 0.0)) / (count * prediction.shape[-1])
    physical = jnp.sum(jnp.where(valid, squared[..., :7], 0.0)) / (count * 7)
    # f32 variant: the native32 loss is live (official openpi supervises all 32 dims, padded targets = noise)
    return full, physical


def tied_decode(linen_module, hidden):
    return linen_module.embedder.decode(hidden)


@dataclasses.dataclass(frozen=True)
class JointConfig(pi0_config.Pi0Config):
    target_capacity: int = 32
    text_loss_weight: float = 0.1
    # Experiment group: A text Physical Words; B latent; C latent + verb supervision.
    memory_variant: str = "A"
    lat_token_id: int = 7
    latent_dim: int = 64
    sem_dim: int = 16
    max_entities: int = 3
    verb_count: int = 15
    align_loss_weight: float = 0.0
    verb_loss_weight: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.memory_variant not in ("A", "B", "C"):
            raise ValueError(f"Unknown memory variant {self.memory_variant!r}")
        if self.memory_variant == "A" and (self.align_loss_weight or self.verb_loss_weight):
            raise ValueError("Group A has no latent or verb losses")
        if self.memory_variant == "B" and self.verb_loss_weight:
            raise ValueError("Group B must have no verb supervision")
        if not self.pi05 or not self.discrete_state_input:
            raise ValueError("Joint model requires real pi0.5 native discrete state")
        if (self.paligemma_variant, self.action_expert_variant) != ("gemma_2b", "gemma_300m"):
            raise ValueError("Full-parameter joint model requires gemma_2b and gemma_300m")
        if self.dtype != "bfloat16":
            raise ValueError("Full-parameter joint model requires bfloat16 transformer computation")

    def create(self, rng):
        return JointPi05(self, rngs=nnx.Rngs(rng))


def config_from_contract(contract, capacities):
    config = contract["model"]
    return JointConfig(pi05=True, dtype="bfloat16", discrete_state_input=True,
                       action_horizon=config["action_horizon"], action_dim=config["action_dim"],
                       max_token_len=capacities["condition"], target_capacity=capacities["target"],
                       paligemma_variant=config["paligemma_variant"],
                       action_expert_variant=config["action_expert_variant"],
                       text_loss_weight=contract["training"]["text_loss_weight"],
                       memory_variant=config["memory_variant"],
                       align_loss_weight=contract["training"]["align_loss_weight"],
                       verb_loss_weight=contract["training"]["verb_loss_weight"])


# Parameters that do not exist in the official base checkpoint. Only these may be
# freshly initialized; every base parameter is still restored strictly.
NEW_HEADS = ("latent_out", "latent_in", "verb_head")


def _make_heads(config, rngs):
    width = _gemma.get_config(config.paligemma_variant).width
    heads = {}
    if config.memory_variant in ("B", "C"):
        heads["latent_out"] = nnx.Linear(width, config.latent_dim, rngs=rngs)
        heads["latent_in"] = nnx.Linear(config.latent_dim, width, rngs=rngs)
    if config.memory_variant == "C":
        heads["verb_head"] = nnx.Linear(config.sem_dim, config.verb_count, rngs=rngs)
    return heads


class JointPi05(pi0.Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.target_capacity = config.target_capacity
        self.text_loss_weight = config.text_loss_weight
        self.memory_variant = config.memory_variant
        self.lat_token_id = config.lat_token_id
        self.latent_dim = config.latent_dim
        self.sem_dim = config.sem_dim
        self.max_entities = config.max_entities
        self.align_loss_weight = config.align_loss_weight
        self.verb_loss_weight = config.verb_loss_weight
        for name, head in _make_heads(config, rngs).items():
            setattr(self, name, head)

    @property
    def uses_latents(self):
        return self.memory_variant in ("B", "C")

    def _add_latents(self, embedded, ids, valid, latents):
        """Add projected latents to the embeddings of <LAT> tokens, in order of occurrence."""
        is_lat = (ids == self.lat_token_id) & valid
        index = jnp.clip(jnp.cumsum(is_lat, axis=1) - 1, 0, self.max_entities - 1)
        chosen = jnp.take_along_axis(latents.astype(jnp.float32), index[..., None], axis=1)
        return embedded + (is_lat[..., None] * self.latent_in(chosen)).astype(embedded.dtype)

    def _project_latent(self, hidden):
        """Raw latent of a hidden state, and its norm-8 version (the scale of z*)."""
        raw = self.latent_out(hidden.astype(jnp.float32))
        return raw, raw / (jnp.linalg.norm(raw, axis=-1, keepdims=True) + 1e-6) * 8.0

    def prefix(self, observation, prev_latents=None):
        # Images have already undergone the SAME nonaugmenting transform in
        # training and policy. Calling upstream preprocessing would reintroduce
        # semantic-breaking training augmentation.
        # v2: embed_prefix embeds obs.images in dict order, so require exactly the three openpi
        # slots in canonical order (overlay front, wrist, clean front); 3 x 256 image tokens + prompt.
        if tuple(observation.images) != model_lib.IMAGE_KEYS or tuple(observation.image_masks) != model_lib.IMAGE_KEYS:
            raise ValueError(f"v2 needs images {model_lib.IMAGE_KEYS} in this order, got {tuple(observation.images)}")
        tokens, valid, _ = self.embed_prefix(observation)
        if tokens.shape[1] != 3 * 256 + observation.tokenized_prompt.shape[1]:
            raise ValueError(f"Unexpected prefix length {tokens.shape[1]} for three 256-token images")
        if self.uses_latents:
            # embed_prefix concatenates images first, so the prompt is the tail.
            count = observation.tokenized_prompt.shape[1]
            prompt = self._add_latents(tokens[:, -count:], observation.tokenized_prompt,
                                       observation.tokenized_prompt_mask, prev_latents)
            tokens = jnp.concatenate((tokens[:, :-count], prompt), axis=1)
        return tokens, valid

    def joint_forward(self, observation, noisy_actions, time, target_ids, target_mask,
                      prev_latents=None, teacher_latents=None):
        prefix, prefix_valid = self.prefix(observation, prev_latents)
        prefix_mask = prefix_valid[:, :, None] & prefix_valid[:, None, :]
        prefix_positions = jnp.cumsum(prefix_valid, axis=1, dtype=jnp.int32) - 1
        (prefix_hidden, _), cache = self.PaliGemma.llm(
            [prefix, None], mask=prefix_mask, positions=prefix_positions,
        )
        count = jnp.sum(prefix_valid, axis=1, dtype=jnp.int32)[:, None]

        suffix, action_valid, _, condition = self.embed_suffix(observation, noisy_actions, time)
        action_mask = jnp.concatenate((
            action_valid[:, :, None] & prefix_valid[:, None, :],
            action_valid[:, :, None] & action_valid[:, None, :],
        ), axis=-1)
        action_positions = count + jnp.cumsum(action_valid, axis=1, dtype=jnp.int32) - 1
        (_, action), _ = self.PaliGemma.llm(
            [None, suffix], mask=action_mask, positions=action_positions,
            kv_cache=cache, adarms_cond=[None, condition],
        )
        velocity = self.action_out_proj(action[:, -self.action_horizon:])

        teacher_ids = jnp.where(target_mask, target_ids, 0)
        teacher = self.PaliGemma.llm(teacher_ids, method="embed")
        if self.uses_latents:
            # Teacher forcing: later text reads z*, stopped from the target side.
            teacher = self._add_latents(teacher, teacher_ids, target_mask,
                                        jax.lax.stop_gradient(teacher_latents))
        indices = jnp.arange(target_mask.shape[1])
        causal = indices[:, None] >= indices[None, :]
        teacher_mask = jnp.concatenate((
            target_mask[:, :, None] & prefix_valid[:, None, :],
            target_mask[:, :, None] & target_mask[:, None, :] & causal[None],
        ), axis=-1)
        teacher_positions = count + jnp.cumsum(target_mask, axis=1, dtype=jnp.int32) - 1
        (language, _), _ = self.PaliGemma.llm(
            [teacher, None], mask=teacher_mask, positions=teacher_positions, kv_cache=cache,
        )
        # Physical last-valid index, not sum(valid)-1 (which fails for holes).
        last = jnp.max(jnp.where(prefix_valid, jnp.arange(prefix.shape[1])[None], -1), axis=1)
        first = prefix_hidden[jnp.arange(prefix_hidden.shape[0]), last][:, None]
        predictors = jnp.concatenate((first, language[:, :-1]), axis=1)
        return velocity, predictors

    def target_cross_entropy(self, hidden, target_ids, target_mask):
        # Decode only target predictors in one batched projection. A conditional
        # token scan can retain a vocabulary-sized decoder residual per token.
        labels = jnp.where(target_mask, target_ids, 0)
        logits = self.PaliGemma.llm(hidden, method=tied_decode).astype(jnp.float32)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.sum(jnp.where(target_mask, losses, 0.0)) / jnp.maximum(jnp.sum(target_mask), 1)

    def loss(self, rng, batch):
        noise_rng, time_rng = jax.random.split(rng)
        # Values outside the recording cannot affect the noisy action block or
        # valid predictions. Keep their noise tokens but never their labels.
        actions = jnp.where(batch["action_mask"][..., None], batch["actions"], 0.0)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1.0, (actions.shape[0],)) * 0.999 + 0.001
        noisy = time[:, None, None] * noise + (1.0-time[:, None, None]) * actions
        velocity, hidden = self.joint_forward(batch["observation"], noisy, time,
                                              batch["target_ids"], batch["target_mask"],
                                              batch.get("prev_latents"), batch.get("z_star"))
        flow, physical = masked_flow_loss(velocity, noise-actions, batch["action_mask"])
        # target_loss_mask drops only the Action line of unreliable rows; those tokens
        # remain teacher input in target_mask, so later lines still read them.
        text = self.target_cross_entropy(hidden, batch["target_ids"], batch["target_loss_mask"])
        align = verb = jnp.asarray(0.0, jnp.float32)
        if self.uses_latents:
            align, verb = self.latent_losses(hidden, batch)
        # f32 variant: (32/7)*mean32 = flow7 + (25/7)*mean over padded dims; real dims keep their weight vs text
        total = ((32.0 / 7.0) * flow + self.text_loss_weight * text
                 + self.align_loss_weight * align + self.verb_loss_weight * verb)
        return total, {"loss": total, "flow32": jax.lax.stop_gradient(flow), "flow7": physical, "text_ce": text,
                       "align": align, "verb": verb,
                       "target_tokens": jnp.sum(batch["target_mask"]),
                       "loss_tokens": jnp.sum(batch["target_loss_mask"]),
                       "valid_action_steps": jnp.sum(batch["action_mask"])}

    def latent_losses(self, hidden, batch):
        """L_align = 1-cos(z, z*), and multi-label verb BCE on z_sem only (group C).

        predictors[i] predicts token i, so z comes from the state that predicted
        <LAT>: causal, and the same value the free-running generator can compute.
        """
        is_lat = (batch["target_ids"] == self.lat_token_id) & batch["target_mask"]
        order = jnp.cumsum(is_lat, axis=1) - 1
        entity = jnp.arange(self.max_entities)
        match = is_lat[:, :, None] & (order[:, :, None] == entity[None, None])
        position = jnp.argmax(match, axis=1)
        present = jnp.any(match, axis=1)
        selected = jnp.take_along_axis(hidden, position[..., None], axis=1)
        raw, z = self._project_latent(selected)
        target = jax.lax.stop_gradient(batch["z_star"]).astype(jnp.float32)
        cosine = jnp.sum(z * target, axis=-1) / 64.0  # both have norm 8
        mask = batch["align_mask"] & present
        align = jnp.sum(jnp.where(mask, 1.0 - cosine, 0.0)) / jnp.maximum(jnp.sum(mask), 1)
        verb = jnp.asarray(0.0, jnp.float32)
        if self.memory_variant == "C":
            logits = self.verb_head(raw[..., :self.sem_dim])
            losses = optax.sigmoid_binary_cross_entropy(logits, batch["verb_targets"].astype(jnp.float32))
            verb_mask = batch["verb_mask"] & present
            verb = (jnp.sum(jnp.where(verb_mask[..., None], losses, 0.0))
                    / jnp.maximum(jnp.sum(verb_mask) * logits.shape[-1], 1))
        return align, verb

    def teacher_velocity(self, observation, noisy_actions, time, target_ids, target_mask,
                         prev_latents=None, teacher_latents=None):
        return self.joint_forward(observation, noisy_actions, time, target_ids, target_mask,
                                  prev_latents, teacher_latents)[0]

    def sample_from_prefix(self, rng, observation, prefix, prefix_valid, num_steps=10):
        batch_size = observation.state.shape[0]
        prefix_mask = prefix_valid[:, :, None] & prefix_valid[:, None, :]
        prefix_positions = jnp.cumsum(prefix_valid, axis=1, dtype=jnp.int32) - 1
        _, cache = self.PaliGemma.llm([prefix, None], mask=prefix_mask, positions=prefix_positions)
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        dt = -1.0 / num_steps

        def step(i, noisy):
            time = jnp.full((batch_size,), 1.0 + i*dt, jnp.float32)
            suffix, valid, _, condition = self.embed_suffix(observation, noisy, time)
            mask = jnp.concatenate((valid[:, :, None] & prefix_valid[:, None, :],
                                    valid[:, :, None] & valid[:, None, :]), axis=-1)
            positions = jnp.sum(prefix_valid, axis=1, dtype=jnp.int32)[:, None] + jnp.cumsum(valid, axis=1) - 1
            (_, hidden), _ = self.PaliGemma.llm([None, suffix], mask=mask, positions=positions,
                                               kv_cache=cache, adarms_cond=[None, condition])
            return noisy + dt * self.action_out_proj(hidden[:, -self.action_horizon:])

        return jax.lax.fori_loop(0, num_steps, step, noise)

    def next_text_logits(self, prefix, prefix_valid, generated, generated_count, generated_latents=None):
        """Next-token logits, and the latent to store if that token is <LAT> (None for group A)."""
        # Production/verification reference until cached greedy parity is proved.
        valid = jnp.arange(generated.shape[1])[None] < generated_count
        ids = jnp.where(valid, generated, 0)
        teacher = self.PaliGemma.llm(ids, method="embed")
        if self.uses_latents:
            teacher = self._add_latents(teacher, ids, jnp.broadcast_to(valid, ids.shape), generated_latents)
        empty_actions = jnp.zeros((prefix.shape[0], 0), jnp.bool_)
        mask, positions = block_mask_positions(prefix_valid, valid, empty_actions)
        (hidden, _), _ = self.PaliGemma.llm([jnp.concatenate((prefix, teacher), axis=1), None],
                                           positions=positions, mask=mask)
        last_prefix = jnp.max(jnp.where(prefix_valid, jnp.arange(prefix.shape[1])[None], -1), axis=1)
        last = jnp.where(generated_count > 0, prefix.shape[1] + generated_count-1, last_prefix)
        selected = hidden[jnp.arange(hidden.shape[0]), last]
        logits = self.PaliGemma.llm(selected, method=tied_decode).astype(jnp.float32)
        return logits, self._project_latent(selected)[1] if self.uses_latents else None


def parameter_component(path):
    """Disjoint parameter groups shared by manifests and actual update norms."""
    parts = tuple(str(part) for part in path)
    if parts[:2] == ("PaliGemma", "img"):
        return "vision"
    if parts[:2] == ("PaliGemma", "llm"):
        return "action" if any(part.endswith("_1") for part in parts[2:]) else "vlm"
    return "other"


def parameter_report(model):
    """Count every trainable Param without copying or casting its array data."""
    flat = traverse_util.flatten_dict(nnx.state(model, TRAINABLE_FILTER).to_pure_dict())
    if not flat:
        raise ValueError("Checkpoint has no model parameters")
    counts = dict.fromkeys(("vlm", "action", "vision", "other"), 0)
    leaves = dict.fromkeys(counts, 0)
    for path, value in flat.items():
        if value.dtype != jnp.float32:
            raise ValueError(f"Full-parameter master weights must be float32: {path}: {value.dtype}")
        component = parameter_component(path)
        counts[component] += int(value.size)
        leaves[component] += 1
    count = sum(counts.values())
    return {"parameter_count": count, "trainable_parameter_count": count,
            "parameter_leaf_count": len(flat), "trainable_leaf_count": len(flat),
            "parameter_dtype": "float32", "parameter_count_by_component": counts,
            "trainable_leaf_count_by_component": leaves, "strict_extra_params": True}


def _load_strict_parameters(config, params, seed):
    # Shape evaluation creates no random parameter buffers. Replacing abstract
    # state reuses restored arrays: no separately initialized full model tree.
    abstract = nnx.eval_shape(config.create, jax.random.key(seed))
    graph, state = nnx.split(abstract)
    expected = traverse_util.flatten_dict(state.to_pure_dict())
    restored = traverse_util.flatten_dict(params)
    if set(restored) != set(expected):
        missing, extra = set(expected)-set(restored), set(restored)-set(expected)
        raise ValueError(f"Strict parameter restore failed: missing={sorted(missing)}, extra={sorted(extra)}")
    for path, value in restored.items():
        if value.shape != expected[path].shape:
            raise ValueError(f"Parameter shape mismatch at {path}: {value.shape} != {expected[path].shape}")
        if value.dtype != jnp.float32:
            raise ValueError(f"Restored master parameter must be float32: {path}: {value.dtype}")
    state.replace_by_pure_dict(params)
    model = nnx.merge(graph, state)
    model.eval()
    return model


def initialize_from_base(config, checkpoint, seed, *, restore_sharding=None):
    # Restore directly into float32 masters. In particular, never round the
    # original base through a reduced-precision intermediate before widening.
    params = model_lib.restore_params(checkpoint, dtype=jnp.float32, sharding=restore_sharding)
    base_leaves = len(traverse_util.flatten_dict(params))
    # Only the explicitly listed new heads are initialized here; any other missing
    # or extra base leaf is still rejected by the strict loader below.
    head_rngs = nnx.Rngs(jax.random.fold_in(jax.random.PRNGKey(seed), 2026))
    heads = _make_heads(config, head_rngs)
    if set(heads) & set(params):
        raise ValueError(f"Base checkpoint already contains new heads: {sorted(set(heads) & set(params))}")
    params = {**params, **{name: nnx.state(head, nnx.Param).to_pure_dict() for name, head in heads.items()}}
    model = _load_strict_parameters(config, params, seed)
    report = parameter_report(model)
    report.update({"base_checkpoint": str(checkpoint), "base_keys_restored": base_leaves,
                   "base_restore_dtype": "float32", "random_base_parameters": 0,
                   "new_head_leaves": report["parameter_leaf_count"] - base_leaves,
                   "new_heads": sorted(heads)})
    return model, report


def load_full_checkpoint(checkpoint_dir, *, restore_sharding=None):
    import json
    from model_data import sha256_file

    root = Path(checkpoint_dir)
    if not (root / "COMPLETE.json").is_file():
        raise ValueError(f"Not a completed joint checkpoint: {root}")
    marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    if sha256_file(root / "metadata.json") != marker["metadata_sha256"]:
        raise ValueError("Completed checkpoint metadata checksum mismatch")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    config = JointConfig(**metadata["model_config"])
    # Saved training masters must already be float32; do not silently repair a
    # lower-precision checkpoint, omit extra leaves or fill missing parameters.
    params = model_lib.restore_params(root / "params", dtype=None, sharding=restore_sharding)
    model = _load_strict_parameters(config, params, metadata["training"]["seed"])
    report = parameter_report(model)
    if report != metadata["parameters"]:
        raise ValueError("Restored parameter manifest differs from the saved full inference model")
    return model, metadata


def real_weight_diagnostics(model, batch, seed=732):
    """Parent-invoked full-master leakage/tail-gradient checks; never a mock."""
    parameters = parameter_report(model)
    latents = (batch.get("prev_latents"), batch.get("z_star"))
    velocity = nnx_utils.module_jit(model.teacher_velocity)
    run = lambda *args: velocity(*args, *latents)
    rng = jax.random.PRNGKey(seed)
    noisy = jax.random.normal(rng, batch["actions"].shape)
    time = jnp.full((1,), 0.53, jnp.float32)
    observation, ids, valid = batch["observation"], batch["target_ids"], batch["target_mask"]
    baseline = np.asarray(run(observation, noisy, time, ids, valid), np.float32)
    if model.uses_latents:
        # The current round's latents, like its text, must never reach the action branch.
        changed = np.asarray(velocity(observation, noisy, time, ids, valid, latents[0], latents[1] + 3.0), np.float32)
        if float(np.max(np.abs(changed - baseline))) > 1e-6:
            raise AssertionError("Real-weight action isolation failed for current-round latents")
    interventions = {
        "teacher_content": ((ids + 19) % 257152, valid),
        "teacher_padding": (jnp.where(valid, ids, 93), valid),
        "teacher_valid_length": (ids, jnp.arange(ids.shape[1])[None] < 1),
        "teacher_capacity": (jnp.pad(ids, ((0, 0), (0, 32)), constant_values=31),
                             jnp.pad(valid, ((0, 0), (0, 32)), constant_values=False)),
    }
    errors = {}
    for name, (changed_ids, changed_valid) in interventions.items():
        result = np.asarray(run(observation, noisy, time, changed_ids, changed_valid), np.float32)
        error = float(np.max(np.abs(result-baseline)))
        errors[name] = error
        # Teacher interventions leave action-branch inputs and shapes unchanged.
        tolerance = 1e-6
        if not np.isfinite(result).all() or error > tolerance:
            raise AssertionError(f"Real-weight action isolation failed for {name}: {error} > {tolerance}")
    changed_observation = dataclasses.replace(
        observation, tokenized_prompt=jnp.where(observation.tokenized_prompt_mask, observation.tokenized_prompt, 99))
    changed = np.asarray(run(changed_observation, noisy, time, ids, valid), np.float32)
    errors["condition_padding"] = float(np.max(np.abs(changed-baseline)))
    if not np.isfinite(changed).all() or errors["condition_padding"] > 1e-6:
        raise AssertionError("Masked condition token values alter actions")
    del run

    graph, state = nnx.split(model)

    @jax.jit
    def padding_gradient(state, actions, data):
        restored = nnx.merge(graph, state)

        def loss(value):
            return restored.loss(rng, dict(data, actions=value))

        return jax.value_and_grad(loss, has_aux=True)(actions)

    (value, losses), gradient = padding_gradient(state, batch["actions"], batch)
    gradient = np.asarray(gradient)
    absent = ~np.asarray(batch["action_mask"])
    if not absent.any():
        raise ValueError("Padding diagnostic requires a genuine partial tail sample")
    absent_max = float(np.max(np.abs(gradient[absent])))
    if (not all(np.isfinite(float(metric)) for metric in losses.values())
            or not np.isfinite(gradient).all() or absent_max != 0.0):
        raise AssertionError(f"Nonfinite loss/gradient or absent-future gradient={absent_max}")
    # Check direct output supervision, not gradients through noisy model inputs:
    # shared features can still couple unused input dimensions to valid outputs.
    output_gradient = jax.grad(
        lambda output: masked_flow_loss(output, jnp.zeros_like(output), batch["action_mask"])[1]
    )(jnp.ones_like(batch["actions"]))
    padded_dimension_gradient = float(jnp.max(jnp.abs(output_gradient[..., 7:])))
    if padded_dimension_gradient != 0.0:
        raise AssertionError("Unused output dimensions contribute direct action-loss gradients")

    @jax.jit
    def text_padding_gradient(state, observation, noisy, time, ids, valid):
        restored = nnx.merge(graph, state)
        # Reserve one diagnostic padding slot even when this particular target
        # exactly fills the measured corpus capacity.
        valid = valid.at[:, -1].set(False)
        _, hidden = restored.joint_forward(observation, noisy, time, ids, valid, *latents)
        ce, hidden_gradient = jax.value_and_grad(
            lambda h: restored.target_cross_entropy(h, ids, valid))(hidden)
        changed_hidden = jnp.where(valid[..., None], hidden, jnp.full_like(hidden, 7.0))
        changed_ids = jnp.where(valid, ids, 93)
        changed_ce = restored.target_cross_entropy(changed_hidden, changed_ids, valid)
        padded_gradient = jnp.max(jnp.where(valid[..., None], 0.0, jnp.abs(hidden_gradient)))
        return ce, changed_ce, padded_gradient, jnp.all(jnp.isfinite(hidden_gradient))

    text_ce, changed_text_ce, text_padding_max, text_finite = text_padding_gradient(
        state, observation, noisy, time, jnp.asarray(ids), jnp.asarray(valid))
    text_error = abs(float(text_ce)-float(changed_text_ce))
    if (not bool(text_finite) or not np.isfinite(float(text_ce))
            or text_error > 1e-6 or float(text_padding_max) != 0.0):
        raise AssertionError("Actual tied-decoder CE or gradient depends on padded target positions")
    return {"status": "passed", "actual_weights": True, "parameters": parameters,
            "action_max_abs_error": errors,
            "absent_action_gradient_max": absent_max,
            "unused_output_direct_gradient_max": padded_dimension_gradient,
            "text_padding_gradient_max": float(text_padding_max),
            "text_padding_ce_error": text_error,
            "diagnostic_loss": float(value), "diagnostic_flow32": float(losses["flow32"]),
            "diagnostic_flow7": float(losses["flow7"]), "diagnostic_text_ce": float(losses["text_ce"])}
