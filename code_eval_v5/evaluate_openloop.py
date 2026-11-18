"""Held-out checkpoint action/text evaluation, never a closed-loop success test.

API: evaluate_checkpoint(checkpoint_dir, dataset_root, output_dir, step).
CLI: evaluate_openloop.py --checkpoint CHECKPOINT --dataset DATASET --output OUT --step N

Default evaluation visits every held-out dense frame in both history modes.
Teacher history uses the labelled previous memory and z*(t-20). Generated history
uses the Ledger Driver's rendered memory and the generated latents of round t-20.
Both use INITIAL_MEMORY and zero latents for t<20. Independent episode / t%20 lanes
batch genuine recurrence without ever using t-1 or oracle repair. Predicted actions
are never executed, and model done never stops the expert stream. An invalid
generated memory terminates its lane as the driver would: later rows of that lane
are recorded as memory_terminated, fail every text metric, have no action to score,
and are a model outcome rather than an evaluation error.
--smoke-max-rows is explicitly non-final, not full validation. --batch-size
defaults to 16; --generated-batch-size can independently set recurrent batches.
Full-forward greedy decoding runs its token loop on the GPU, returning selected
IDs and finite flags only once; host EOS and memory-capacity rules are unchanged.

Continuous action metrics are null if any required component is unscorable;
finite-only diagnostics and all missing/nonfinite denominators are also saved.
Invalid/missing counts fail all-row accuracy but have no invented numeric MAE.
Raw control coordinates 0:3/3:6 are compared without unit conversion: these are
stored controller translation/rotation units, not assumed metres or radians.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback
import uuid

import numpy as np
from PIL import Image

from memory_format import LATENT_DIM, step_memory
from model_data import INITIAL_MEMORY, FullCorpus, make_observation, read_json, save_json, sha256_file
from policy import JointMemoryPolicy

EVALUATION_SEED = 531
MODES = ("teacher_history", "generated_history")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _json_actions(value):
    if value is None:
        return None
    return [[float(x) if np.isfinite(x) else None for x in row] for row in value]


def _certified_text(result, previous_text, variant, total):
    """Policy validity retains EOS checking; independently re-run the Ledger Driver."""
    text = result.get("memory_text")
    parsed = step_memory(previous_text, text, variant, expected_total=total)
    valid = bool(result.get("memory_valid") is True and parsed["memory_valid"])
    done = bool(valid and parsed["sequence_complete"])
    if result.get("sequence_complete") is not done:
        valid, done = False, False
    # A malformed explicit completion claim must not disappear from false-done
    # statistics just because the strict parser rejects the rest of its text.
    claim = isinstance(text, str) and any(re.match(r"Task:[ \t]*Complete\b", line) for line in text.split("\n"))
    return {"format_valid": valid, "certified_count": parsed["certified_count"] if valid else None,
            "sequence_complete": done, "raw_done_claim": bool(claim),
            "next_memory_text": parsed["next_memory_text"] if valid else None}


def _layer_scores(text, row, variant):
    """Exact per-line agreement with the label; absent lines are wrong, never skipped."""
    target = row["memory_target_text"].split("\n")
    lines = text.split("\n") if isinstance(text, str) else []
    same = [len(lines) == 3 and lines[i] == target[i] for i in range(3)]
    return {"action_exact": same[0] if _action_scored(row, variant) else None,
            "subgoal_delta_correct": same[1], "task_correct": same[2]}


def _action_scored(row, variant):
    # Group B has no verbs to get right; unreliable replay rows have no trustworthy label.
    return variant in ("A", "C") and bool(row["memory_target"]["action_reliable"])


def _latent_cosines(generated, row, corpus):
    """Cosines of generated z with z* for entities visible in the window; empty for group A."""
    if corpus.variant == "A":
        return []
    stars, valid = corpus.z_star(row), corpus.z_valid(row)
    result = []
    for z, star, ok in zip(np.asarray(generated, np.float64), np.asarray(stars, np.float64), np.asarray(valid), strict=True):
        norms = float(np.linalg.norm(z) * np.linalg.norm(star))
        if ok and np.isfinite(z).all() and norms > 0:
            result.append(float(np.dot(z, star) / norms))
    return result


class _Metrics:
    def __init__(self):
        self.counts = Counter()
        self.action_expected = 0
        self.action_scored = np.zeros(7, np.int64)
        self.action_nonfinite = np.zeros(7, np.int64)
        self.absolute_error = np.zeros(7, np.float64)
        self.normalized_squared_error = np.zeros(7, np.float64)
        self.gripper_correct = 0
        self.count_absolute_error = 0
        self.latent_cosine_sum = 0.0
        self.latent_cosine_count = 0

    def add(self, record, expert, predicted, normalizer):
        self.counts["samples"] += 1
        self.counts[record["status"]] += 1
        self.counts["inference_attempted"] += int(record["inference_attempted"])
        self.counts["inference_returned"] += int(record["inference_returned"])
        self.counts["actual_done"] += int(record["target_sequence_complete"])
        self.counts["actual_undone"] += int(not record["target_sequence_complete"])
        # A terminated lane produced no action; it is a text failure, not missing action data.
        if record["status"] != "memory_terminated":
            self.action_expected += len(expert)
        if record["action_exact"] is not None:
            self.counts["action_exact_scored"] += 1
            self.counts["action_exact_correct"] += int(record["action_exact"])
        self.counts["subgoal_delta_correct"] += int(record["subgoal_delta_correct"])
        self.counts["task_correct"] += int(record["task_correct"])
        self.latent_cosine_sum += sum(record["latent_cosines"])
        self.latent_cosine_count += len(record["latent_cosines"])
        if record["memory_text"] is not None:
            self.counts["text_outputs"] += 1
            self.counts["format_valid"] += int(record["format_valid"])
            self.counts["invalid_text_outputs"] += int(not record["format_valid"])
        else:
            self.counts["missing_text_outputs"] += 1
        predicted_count = record["certified_count"]
        if predicted_count is None:
            self.counts["count_unscorable"] += 1
        else:
            self.counts["count_scored"] += 1
            error = abs(predicted_count - record["target_count"])
            self.count_absolute_error += error
            self.counts["count_correct"] += int(error == 0)
        if record["target_sequence_complete"]:
            self.counts["true_done"] += int(record["sequence_complete"])
        else:
            self.counts["false_done"] += int(record["raw_done_claim"])
            self.counts["false_done_valid"] += int(record["sequence_complete"])
            self.counts["actual_undone_missing_text"] += int(record["memory_text"] is None)
            self.counts["actual_undone_invalid_text"] += int(record["memory_text"] is not None and not record["format_valid"])
        if predicted is None:
            self.counts["action_missing_or_malformed_samples"] += 1
            return
        values = predicted[:len(expert)].astype(np.float64)
        target = expert.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            errors = np.abs(values - target)
            scale = normalizer.stats["actions"]["denominator"].astype(np.float64)
            squared = np.square((values - target) * (2.0 / scale))
        finite = np.isfinite(values) & np.isfinite(errors) & np.isfinite(squared)
        self.action_scored += finite.sum(axis=0)
        self.action_nonfinite += (~finite).sum(axis=0)
        self.absolute_error += np.where(finite, errors, 0.0).sum(axis=0)
        self.normalized_squared_error += np.where(finite, squared, 0.0).sum(axis=0)
        self.gripper_correct += int(np.sum(finite[:, 6] & (np.sign(values[:, 6]) == np.sign(target[:, 6]))))
        self.counts["action_finite_samples"] += int(finite.all())
        self.counts["action_nonfinite_samples"] += int(not finite.all())

    def summary(self):
        n = self.counts["samples"]
        scored = self.action_scored
        missing = self.action_expected - scored
        trans_n, rot_n, all_n = int(scored[:3].sum()), int(scored[3:6].sum()), int(scored.sum())
        trans = _ratio(float(self.absolute_error[:3].sum()), trans_n)
        rot = _ratio(float(self.absolute_error[3:6].sum()), rot_n)
        mse = _ratio(float(self.normalized_squared_error.sum()), all_n)
        action = {
            "translation_mae": trans if not missing[:3].any() else None,
            "rotation_mae": rot if not missing[3:6].any() else None,
            "normalized_mse7": mse if not missing.any() else None,
            "translation_mae_finite_only": trans,
            "rotation_mae_finite_only": rot,
            "normalized_mse7_finite_only": mse,
            "gripper_sign_accuracy": _ratio(self.gripper_correct, self.action_expected),
            "gripper_sign_accuracy_finite_only": _ratio(self.gripper_correct, int(scored[6])),
            "gripper_sign_correct": self.gripper_correct,
            "expected_timesteps": self.action_expected,
            "expected_components7": self.action_expected * 7,
            "scored_components7": all_n,
            "missing_components7": int(missing.sum()),
            "expected_translation_components": self.action_expected * 3,
            "scored_translation_components": trans_n,
            "expected_rotation_components": self.action_expected * 3,
            "scored_rotation_components": rot_n,
            "expected_gripper_timesteps": self.action_expected,
            "scored_gripper_timesteps": int(scored[6]),
            "scored_components_by_dimension": scored.tolist(),
            "missing_components_by_dimension": missing.tolist(),
            "nonfinite_components_by_dimension": self.action_nonfinite.tolist(),
            "finite_samples": self.counts["action_finite_samples"],
            "nonfinite_samples": self.counts["action_nonfinite_samples"],
            "missing_or_malformed_samples": self.counts["action_missing_or_malformed_samples"],
            "absolute_error_sum_by_dimension": self.absolute_error.tolist(),
            "normalized_squared_error_sum_by_dimension": self.normalized_squared_error.tolist(),
            "mae_by_dimension_finite_only": [_ratio(float(self.absolute_error[i]), int(scored[i])) for i in range(7)],
            "normalized_mse_by_dimension_finite_only": [_ratio(float(self.normalized_squared_error[i]), int(scored[i])) for i in range(7)],
        }
        text = {
            "count_accuracy": _ratio(self.counts["count_correct"], n),
            "count_accuracy_numerator": self.counts["count_correct"],
            "count_accuracy_denominator": n,
            "count_mae": _ratio(self.count_absolute_error, self.counts["count_scored"]),
            "count_mae_valid_explicit": _ratio(self.count_absolute_error, self.counts["count_scored"]),
            "count_mae_valid_explicit_denominator": self.counts["count_scored"],
            "count_mae_denominator": self.counts["count_scored"],
            "count_absolute_error_sum": self.count_absolute_error,
            "count_mae_unscorable_samples": self.counts["count_unscorable"],
            "format_valid_rate": _ratio(self.counts["format_valid"], n),
            "format_valid_numerator": self.counts["format_valid"],
            "format_valid_denominator": n,
            "false_done_rate": _ratio(self.counts["false_done"], self.counts["actual_undone"]),
            "false_done_numerator": self.counts["false_done"],
            "false_done_denominator": self.counts["actual_undone"],
            "false_done_valid_format_numerator": self.counts["false_done_valid"],
            "done_recall": _ratio(self.counts["true_done"], self.counts["actual_done"]),
            "done_recall_numerator": self.counts["true_done"],
            "done_recall_denominator": self.counts["actual_done"],
            "invalid_text_outputs": self.counts["invalid_text_outputs"],
            "missing_text_outputs": self.counts["missing_text_outputs"],
            "actual_undone_missing_text_outputs": self.counts["actual_undone_missing_text"],
            "actual_undone_invalid_text_outputs": self.counts["actual_undone_invalid_text"],
            "action_exact_accuracy": _ratio(self.counts["action_exact_correct"], self.counts["action_exact_scored"]),
            "action_exact_numerator": self.counts["action_exact_correct"],
            "action_exact_denominator": self.counts["action_exact_scored"],
            "subgoal_delta_accuracy": _ratio(self.counts["subgoal_delta_correct"], n),
            "subgoal_delta_numerator": self.counts["subgoal_delta_correct"],
            "task_accuracy": _ratio(self.counts["task_correct"], n),
            "task_numerator": self.counts["task_correct"],
            "layer_accuracy_denominator": n,
            "latent_cosine_mean": _ratio(self.latent_cosine_sum, self.latent_cosine_count),
            "latent_cosine_count": self.latent_cosine_count,
            "memory_terminated_rows": self.counts["memory_terminated"],
        }
        coverage = {"rows_recorded": n, "inference_attempted": self.counts["inference_attempted"],
                    "inference_returned": self.counts["inference_returned"],
                    "inference_coverage": _ratio(self.counts["inference_returned"], n),
                    "status_counts": {key: self.counts[key] for key in
                                      ("ok", "action_error", "conditioning_error", "inference_error", "history_unavailable",
                                       "memory_terminated", "smoke_budget_excluded")}}
        return {"action": action, "text": text, "coverage": coverage}


def _load_corpus(checkpoint, root, policy):
    manifest_path = root / "split_manifest.json"
    manifest = read_json(manifest_path)
    train_split, val_split = manifest["splits"]["train"], manifest["splits"]["val"]
    contract = deepcopy(policy.metadata["contract"])
    contract["dataset"]["index"] = val_split["index"]
    corpus = FullCorpus(root, contract, policy.tokenizer)
    if corpus.variant != policy.variant:
        raise ValueError(f"Corpus variant {corpus.variant} differs from checkpoint variant {policy.variant}")
    if corpus.fingerprint["norm_stats_sha256"] != policy.metadata["dataset_fingerprint"]["norm_stats_sha256"]:
        raise ValueError("Validation dataset normalization is not the checkpoint's train-only normalization")
    # All inference and error scaling use the saved asset, not refitted validation statistics.
    corpus.normalizer = policy.normalizer
    train_path = corpus.path(train_split["index"])
    train_hash = sha256_file(train_path)
    if train_hash != policy.metadata["dataset_fingerprint"]["index_sha256"]:
        raise ValueError("Current training split differs from checkpoint training data")
    if sha256_file(checkpoint / "norm_stats.json") != corpus.fingerprint["norm_stats_sha256"]:
        raise ValueError("Checkpoint and dataset normalization assets differ")
    train_episodes, train_arrays, train_samples = set(), set(), set()
    with train_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            train_episodes.add((int(row["task_id"]), str(row["episode_id"])))
            train_arrays.add(str(corpus.path(row["episode_npz"])))
            train_samples.add(row["sample_id"])
    groups = defaultdict(list)
    for index, row in enumerate(corpus.rows):
        key = (int(row["task_id"]), str(row["episode_id"]))
        if key in train_episodes or str(corpus.path(row["episode_npz"])) in train_arrays or row["sample_id"] in train_samples:
            raise ValueError(f"Training/validation leakage: {row['sample_id']}")
        groups[key].append(index)
    observed = Counter()
    by_task = defaultdict(Counter)
    totals = {int(task["task_id"]): int(task["required_goals"]) for task in contract["source"]["tasks"]}
    for key, indices in groups.items():
        indices.sort(key=lambda index: int(corpus.rows[index]["observation_index"]))
        _, actions = corpus.episodes[corpus.rows[indices[0]]["episode_npz"]]
        if [int(corpus.rows[index]["observation_index"]) for index in indices] != list(range(len(actions))):
            raise ValueError(f"Validation episode is not dense at every frame: {key}")
        observed["actions"] += len(actions)
        by_task[str(key[0])]["actions"] += len(actions)
        observed["episodes"] += 1
        by_task[str(key[0])]["episodes"] += 1
        for index in indices:
            row = corpus.rows[index]
            frame = int(row["observation_index"])
            if int(row["action_start"]) != frame or int(row["valid_action_steps"]) != min(20, len(actions) - frame):
                raise ValueError(f"Validation future-action tail differs: {row['sample_id']}")
            target = row["memory_target"]
            total = totals[key[0]]
            complete = target["task_text"] == "Complete"
            if (int(target["total_goals"]) != total or not 0 <= int(target["confirmed_goals"]) <= total
                    or target["task_text"] not in ("Incomplete", "Complete")
                    or (complete and int(target["confirmed_goals"]) != total)):
                raise ValueError(f"Invalid certified target: {row['sample_id']}")
            # The label must itself pass the driver from its own previous memory, so the
            # previous_memory -> target chain is rechecked here, not only in FullCorpus.
            parsed = step_memory(row["previous_memory_text"], row["memory_target_text"], policy.variant,
                                 expected_total=total)
            if (not parsed["memory_valid"] or parsed["certified_count"] != int(target["confirmed_goals"])
                    or parsed["sequence_complete"] != complete):
                raise ValueError(f"Target count/text mismatch: {row['sample_id']}")
            if policy.metadata["instruction_totals"].get(row["instruction"]) != total:
                raise ValueError(f"Held-out instruction is absent or inconsistent in saved tokenizer/config: {row['sample_id']}")
            observed["queries"] += 1
            observed["valid_action_steps"] += int(row["valid_action_steps"])
            by_task[str(key[0])]["queries"] += 1
            by_task[str(key[0])]["valid_action_steps"] += int(row["valid_action_steps"])
    for field in ("episodes", "actions", "queries"):
        if observed[field] != int(val_split[field]):
            raise ValueError(f"Validation manifest {field} differs: {observed[field]} != {val_split[field]}")
    if set(by_task) != set(val_split["tasks"]):
        raise ValueError("Validation task membership differs from split manifest")
    for task, counts in by_task.items():
        for field in ("episodes", "actions", "queries"):
            if counts[field] != int(val_split["tasks"][task][field]):
                raise ValueError(f"Validation task {task} manifest {field} differs")
    evidence = {"split_manifest_sha256": sha256_file(manifest_path),
                "training_index_sha256": train_hash, "validation_index_sha256": corpus.fingerprint["index_sha256"],
                "normalization_sha256": corpus.fingerprint["norm_stats_sha256"],
                "checkpoint_tokenizer_sha256": sha256_file(checkpoint / "tokenizer.model"),
                "train_validation_episode_overlap": 0, "train_validation_npz_overlap": 0,
                "train_validation_sample_overlap": 0, "train_normalization_matches_checkpoint": True,
                "validation_counts": dict(observed), "validation_tasks": {key: dict(value) for key, value in by_task.items()},
                "normalization_scope": "saved training-only statistics; never refit on held-out rows",
                "dense_frame_coverage": "every original frame 0..N-1; action windows t:t+20 overlap and tail-mask",
                "history_recurrence": "labelled memory and z* of t-20 in teacher mode; driver-rendered memory and generated latents of t-20 in generated mode; INITIAL_MEMORY and zero latents for first20",
                "generated_history_uses_validation_targets_only_for_scoring": True}
    return corpus, sorted(groups.items()), evidence


class _BatchedInference:
    """Batch the saved model, preserving policy greedy/EOS/memory-bound rules.

    The full-forward token loop stays on the GPU; selected IDs and finite flags
    cross to the host once. Parameters remain explicit JIT arguments, not
    captured constants or per-row model replicas.
    """
    def __init__(self, policy):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        self.policy = policy
        graph, self.state = nnx.split(policy.model)
        use_latents = policy.variant != "A"
        lat_token, max_entities = policy.lat_token_id, policy.max_entities
        vocab = policy.tokenizer.vocab_size
        eos, bos = policy.tokenizer.sp.eos_id(), policy.tokenizer.sp.bos_id()

        def sample(state, keys, observation, prefix, valid):
            model = nnx.merge(graph, state)

            def one(key, row, row_prefix, row_valid):
                singleton = jax.tree.map(lambda value: value[None], row)
                return model.sample_from_prefix(key, singleton, row_prefix[None], row_valid[None])[0]

            # Each row keeps exactly its own folded/split RNG, independently of
            # batching, padding, other failed rows, checkpoint, or history mode.
            return jax.vmap(one)(keys, observation, prefix, valid)

        def choose(logits):
            finite = jnp.all(jnp.isfinite(logits), axis=-1)
            scores = logits[:, :vocab]
            for special in (0, bos):
                if special != eos:
                    scores = scores.at[:, special].set(-jnp.inf)
            tokens = jnp.argmax(scores, axis=-1).astype(jnp.int32)
            return tokens, finite

        def next_token(state, prefix, valid, generated, count, latents, latent_counts, finished):
            model = nnx.merge(graph, state)
            logits, z = model.next_text_logits(prefix, valid, generated, count, latents if use_latents else None)
            tokens, finite = choose(logits)
            if use_latents:
                # Finished rows keep their latents; surplus slots share the last index,
                # and the driver rejects that grammar anyway.
                written = (tokens == lat_token) & ~finished
                rows = jnp.arange(tokens.shape[0])
                slot = jnp.minimum(latent_counts, max_entities - 1)
                value = jnp.where(written[:, None], z.astype(latents.dtype), latents[rows, slot])
                latents = latents.at[rows, slot].set(value)
                latent_counts = latent_counts + written.astype(latent_counts.dtype)
            # Finished rows are ignored by the host; their independent padding
            # cannot affect another row's attention, greedy tokens, or metrics.
            return generated.at[:, count].set(tokens), tokens, finite, latents, latent_counts

        def generate_text(state, prefix, valid):
            capacity = policy.metadata["capacities"]["target"]
            batch_size = valid.shape[0]
            generated = jnp.zeros((batch_size, capacity), jnp.int32)
            choices = jnp.zeros((capacity, batch_size), jnp.int32)
            finite_flags = jnp.ones((capacity, batch_size), jnp.bool_)
            done = jnp.zeros((batch_size,), jnp.bool_)
            latents = jnp.zeros((batch_size, max_entities, LATENT_DIM), jnp.float32)
            latent_counts = jnp.zeros((batch_size,), jnp.int32)

            def condition(carry):
                position, _, _, _, finished, _, _ = carry
                return (position < capacity) & ~jnp.all(finished)

            def advance(carry):
                position, generated, choices, finite_flags, finished, latents, latent_counts = carry
                generated, chosen, finite, latents, latent_counts = next_token(
                    state, prefix, valid, generated, position, latents, latent_counts, finished)
                return (position + 1, generated, choices.at[position].set(chosen),
                        finite_flags.at[position].set(finite),
                        finished | (chosen == eos) | ~finite, latents, latent_counts)

            position, _, choices, finite_flags, _, latents, _ = jax.lax.while_loop(
                condition, advance, (jnp.array(0, jnp.int32), generated, choices, finite_flags, done,
                                     latents, latent_counts),
            )
            return choices, finite_flags, position, latents

        self.sample = jax.jit(sample)
        self.generate_text = jax.jit(generate_text)
        seed = jax.random.PRNGKey(EVALUATION_SEED)
        self.row_keys = jax.jit(jax.vmap(lambda index: jax.random.split(jax.random.fold_in(seed, index))[1]))

    def infer(self, observations, previous_latents, indices, instructions, previous_texts, batch_size):
        import jax
        import jax.numpy as jnp

        policy = self.policy
        size = len(observations)
        # Fixed shapes avoid recompiling every final or partially failed batch.
        # Padding is observational only and never appears in evaluation counts.
        padded = observations + [observations[-1]] * (batch_size - size)
        observation = jax.tree.map(lambda *values: np.concatenate(values, axis=0), *padded)
        latent_input = None
        if policy.variant != "A":
            latent_input = np.stack(previous_latents + [previous_latents[-1]] * (batch_size - size)).astype(np.float32)
        prefix, valid = policy._prefix(observation, latent_input)
        row_indices = indices + [indices[-1]] * (batch_size - size)
        keys = self.row_keys(jnp.asarray(row_indices, jnp.uint32))
        native = np.asarray(self.sample(self.state, keys, observation, prefix, valid), np.float32)
        if native.shape != (batch_size, 20, 32):
            raise ValueError(f"Expected batched native actions[{batch_size},20,32], got {native.shape}")
        stats = policy.normalizer.stats["actions"]
        # Same saved inverse transform as policy.inverse_actions, vectorized;
        # preserve nonfinite components for honest finite/missing denominators.
        with np.errstate(over="ignore", invalid="ignore"):
            actions = (native[:size, :, :7] + 1.0) / 2.0 * stats["denominator"] + stats["lower"]
        memory_capacity = policy.metadata["capacities"]["previous_memory"]
        choices, finite_flags, executed, generated_latents = self.generate_text(self.state, prefix, valid)
        host_choices, host_flags, host_latents = jax.device_get((choices, finite_flags, generated_latents))
        executed = int(executed)
        active = np.ones(size, bool)
        tokens = [[] for _ in range(size)]
        texts = [""] * size
        eos = np.zeros(size, bool)
        reasons = ["target_capacity"] * size
        failures = [None] * size
        for position in range(executed):
            host_chosen, host_finite = host_choices[position], host_flags[position]
            for row in np.flatnonzero(active):
                try:
                    if not host_finite[row]:
                        raise FloatingPointError("Nonfinite Gemma text logits")
                    token = int(host_chosen[row])
                    if token == policy.tokenizer.sp.eos_id():
                        eos[row], active[row], reasons[row] = True, False, "eos"
                        continue
                    candidate = tokens[row] + [token]
                    decoded = policy.tokenizer.sp.decode(candidate)
                    if len(policy.tokenizer.memory_ids(decoded)) > memory_capacity:
                        active[row], reasons[row] = False, "previous_memory_capacity"
                        continue
                    tokens[row], texts[row] = candidate, decoded
                except Exception as error:
                    active[row], failures[row] = False, error
            if not active.any():
                break
        results = []
        for row, instruction in enumerate(instructions):
            if failures[row] is not None:
                results.append(failures[row])
                continue
            expected_total = policy.metadata["instruction_totals"].get(instruction)
            parsed = step_memory(previous_texts[row], texts[row], policy.variant,
                                 eos_terminated=bool(eos[row]), expected_total=expected_total)
            if expected_total is None:
                parsed["sequence_complete"] = False
            results.append({"actions": actions[row], "memory_text": texts[row], **parsed,
                            "eos_terminated": bool(eos[row]), "generation_stop": reasons[row],
                            "generated_tokens": len(tokens[row]), "latents": np.array(host_latents[row])})
        return results


def _row_batches(groups, mode, batch_size):
    if mode == "teacher_history":
        batch = []
        for key, indices in groups:
            for index in indices:
                batch.append((key, index))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
        return
    # A batch contains at most one row from each (episode, t%20) lane.
    # Successors become ready only after the yielded batch has been processed.
    pending = deque((key, indices, lane) for key, indices in groups
                    for lane in range(min(20, len(indices))))
    while pending:
        batch = [pending.popleft() for _ in range(min(batch_size, len(pending)))]
        yield [(key, indices[position]) for key, indices, position in batch]
        for key, indices, position in batch:
            if position + 20 < len(indices):
                pending.append((key, indices, position + 20))


def _error_record(error, phase, sample_id):
    return {"type": type(error).__name__, "message": str(error), "phase": phase, "sample_id": sample_id}


def _evaluate_mode(policy, inference, corpus, groups, output, mode, smoke_max_rows, batch_size):
    overall = _Metrics()
    tasks, episodes = defaultdict(_Metrics), defaultdict(_Metrics)
    histories = {}
    selected = None
    if smoke_max_rows is not None:
        # Both modes use the same explicitly budgeted chronological rows, with
        # every t-20 ancestor included; scheduling never changes smoke scope.
        selected = set()
        for _, indices in groups:
            selected.update(indices[:max(0, smoke_max_rows - len(selected))])
            if len(selected) == smoke_max_rows:
                break
    prediction_path = output / f"{mode}_predictions.jsonl"
    temporary = prediction_path.with_name(prediction_path.name + "." + uuid.uuid4().hex + ".tmp")
    batch_count = padded_rows = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for batch in _row_batches(groups, mode, batch_size):
            entries, prepared, observations = [], [], []
            for key, index in batch:
                task_id, episode_id = key
                row = corpus.rows[index]
                target = row["memory_target"]
                count, start = int(row["valid_action_steps"]), int(row["action_start"])
                frame = int(row["observation_index"])
                states, actions = corpus.episodes[row["episode_npz"]]
                expert = actions[start:start + count]
                record = {"sample_id": row["sample_id"], "task_id": task_id, "episode_id": episode_id,
                          "observation_index": frame, "mode": mode, "history_frame": frame - 20 if frame >= 20 else None,
                          "valid_action_steps": count, "status": "ok", "error": None,
                          "inference_attempted": False, "inference_returned": False,
                          "previous_memory_text": None, "memory_text": None,
                          "target_count": int(target["confirmed_goals"]),
                          "target_sequence_complete": target["task_text"] == "Complete",
                          "format_valid": False, "certified_count": None,
                          "sequence_complete": False, "raw_done_claim": False, "next_memory_text": None,
                          **_layer_scores(None, row, corpus.variant), "latent_cosines": [],
                          "eos_terminated": None, "generation_stop": None, "generated_tokens": None,
                          "actions": None, "expert_actions": _json_actions(expert)}
                entry = {"key": key, "index": index, "row": row, "record": record, "expert": expert,
                         "predicted": None, "latents": None}
                entries.append(entry)
                terminated = False
                if mode == "teacher_history":
                    history, history_latents, history_error = (
                        row["previous_memory_text"], corpus.previous_latents(row), None)
                elif frame < 20:
                    history, history_latents, history_error = INITIAL_MEMORY, policy.empty_latents(), None
                else:
                    history, history_latents, history_error, terminated = histories.pop((*key, frame - 20))
                if selected is not None and index not in selected:
                    record["status"] = "smoke_budget_excluded"
                    continue
                if terminated:
                    record["status"] = "memory_terminated"
                    record["error"] = history_error
                    continue
                if history is None:
                    record["status"] = "history_unavailable"
                    record["error"] = history_error
                    continue
                record["previous_memory_text"] = history
                try:
                    policy.tokenizer.condition(row["instruction"], history,
                                               policy.normalizer.normalize("state", states[frame]))
                except Exception as error:
                    record["status"] = "conditioning_error"
                    record["error"] = _error_record(error, "conditioning", row["sample_id"])
                    continue
                try:
                    with Image.open(corpus.path(row["image_path"])) as image:
                        main = np.asarray(image.convert("RGB"))
                    with Image.open(corpus.path(row["wrist_image_path"])) as image:
                        wrist = np.asarray(image.convert("RGB"))
                    with Image.open(corpus.path(row["raw_image_path"])) as image:
                        raw = np.asarray(image.convert("RGB"))
                    observation = make_observation(main, wrist, raw, states[frame], row["instruction"], history,
                                                   policy.tokenizer, policy.normalizer)
                except Exception as error:
                    record["status"] = "inference_error"
                    record["error"] = _error_record(error, "observation_preparation", row["sample_id"])
                    continue
                record["inference_attempted"] = True
                entry["history"], entry["history_latents"] = history, history_latents
                prepared.append(entry)
                observations.append(observation)
            if prepared:
                batch_count += 1
                padded_rows += batch_size - len(prepared)
                try:
                    results = inference.infer(observations, [entry["history_latents"] for entry in prepared],
                                              [entry["index"] for entry in prepared],
                                              [entry["row"]["instruction"] for entry in prepared],
                                              [entry["history"] for entry in prepared], batch_size)
                    if len(results) != len(prepared):
                        raise ValueError("Batched inference returned an incorrect row count")
                except Exception as error:
                    results = [error] * len(prepared)
                for entry, result in zip(prepared, results, strict=True):
                    record, row = entry["record"], entry["row"]
                    if isinstance(result, Exception):
                        record["status"] = "inference_error"
                        record["error"] = _error_record(result, "batched_inference", row["sample_id"])
                        continue
                    record["inference_returned"] = True
                    record["memory_text"] = result["memory_text"]
                    record.update(_certified_text(result, entry["history"], policy.variant,
                                                  int(row["memory_target"]["total_goals"])))
                    record.update(_layer_scores(result["memory_text"], row, policy.variant))
                    record["latent_cosines"] = _latent_cosines(result["latents"], row, corpus)
                    entry["latents"] = result["latents"]
                    for field in ("eos_terminated", "generation_stop", "generated_tokens"):
                        record[field] = result[field]
                    try:
                        predicted = np.asarray(result["actions"], np.float64)
                        if predicted.shape != (20, 7):
                            raise ValueError(f"Expected actions[20,7], got {predicted.shape}")
                        entry["predicted"] = predicted
                        record["actions"] = _json_actions(predicted)
                        record["nonfinite_action_components"] = int(np.sum(~np.isfinite(predicted[:record["valid_action_steps"]])))
                        if record["nonfinite_action_components"]:
                            raise FloatingPointError("Nonfinite required physical7 action components")
                    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as error:
                        record["status"] = "action_error"
                        record["error"] = _error_record(error, "action_scoring", row["sample_id"])
            for entry in entries:
                record, key = entry["record"], entry["key"]
                if mode == "generated_history":
                    # The driver decides the next history. Invalid grammar or an illegal ledger
                    # edit terminates the lane and is carried forward; done does not stop it.
                    if record["next_memory_text"] is not None:
                        histories[(*key, record["observation_index"])] = (
                            record["next_memory_text"], entry["latents"], None, False)
                    elif record["status"] == "memory_terminated":
                        histories[(*key, record["observation_index"])] = (None, None, record["error"], True)
                    elif record["inference_returned"]:
                        histories[(*key, record["observation_index"])] = (None, None, {
                            "type": "invalid_model_memory", "message": "Ledger Driver rejected the generated memory",
                            "phase": "ledger_driver", "sample_id": record["sample_id"]}, True)
                    else:
                        histories[(*key, record["observation_index"])] = (None, None, record["error"], False)
                for metrics in (overall, tasks[str(key[0])], episodes[key]):
                    metrics.add(record, entry["expert"], entry["predicted"], policy.normalizer)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    temporary.replace(prediction_path)
    episode_records = [{"task_id": key[0], "episode_id": key[1], "expected_rows": len(indices),
                        **episodes[key].summary()} for key, indices in groups]
    episode_path = output / f"{mode}_episodes.json"
    save_json(episode_path, episode_records)
    result = overall.summary()
    result["coverage"].update({"expected_rows": len(corpus.rows), "expected_episodes": len(groups),
                                "recorded_episodes": len(episode_records),
                                "episodes_with_unavailable_history": sum(item["coverage"]["status_counts"]["history_unavailable"] > 0
                                                                         for item in episode_records),
                                "all_rows_accounted_for": result["coverage"]["rows_recorded"] == len(corpus.rows),
                                # A row ended by the driver is decided by the model, not left uninferred.
                                "all_rows_inferred": overall.counts["inference_returned"] + overall.counts["memory_terminated"] == len(corpus.rows),
                                "rows_inference_returned": overall.counts["inference_returned"],
                                "batch_size": batch_size, "inference_batches": batch_count,
                                "padding_rows_not_scored": padded_rows})
    result["by_task"] = {key: value.summary() for key, value in sorted(tasks.items())}
    result["artifacts"] = {"predictions": str(prediction_path), "episodes": str(episode_path)}
    result["history"] = ("labelled memory and z* of t-20, INITIAL_MEMORY and zero latents for t<20; offline teacher-history diagnostic"
                         if mode == "teacher_history" else
                         "driver-rendered memory and generated latents of t-20, INITIAL_MEMORY and zero latents for t<20; independent t%20 lanes; invalid memory terminates its lane; no oracle repair or done stopping")
    return result


def _wandb_metrics(modes, step):
    metrics = {"validation/optimizer_step": step}
    for mode, report in modes.items():
        for section in ("action", "text", "coverage"):
            for key, value in report[section].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"validation/{mode}/{section}/{key}"] = value
    return metrics


def evaluate_checkpoint(checkpoint_dir, dataset_root, output_dir, step, *, smoke_max_rows=None,
                        batch_size=16, generated_batch_size=None, modes=MODES):
    """Load a saved policy and evaluate the selected independent history modes.

    A positive smoke_max_rows bounds inference per mode, not reporting: excluded
    rows remain explicit in coverage and all-row denominators. Such a report is
    always marked non-final. Fatal setup failures are recorded and re-raised;
    row-level failures remain in the report and never disappear from metrics.
    """
    checkpoint = Path(checkpoint_dir).resolve()
    root = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative optimizer update count")
    if smoke_max_rows is not None and (isinstance(smoke_max_rows, bool) or not isinstance(smoke_max_rows, int) or smoke_max_rows < 1):
        raise ValueError("smoke_max_rows must be a positive integer or None")
    if generated_batch_size is None:
        generated_batch_size = batch_size
    for name, value in (("batch_size", batch_size), ("generated_batch_size", generated_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Evaluation modes must be distinct recognized history modes")
    report = {"schema_version": 1, "checkpoint": str(checkpoint), "dataset_root": str(root),
              "optimizer_step": step, "evaluation_seed": EVALUATION_SEED,
              "scope": "held-out demonstration open-loop controls and independently generated text; not simulator success",
              "primary_action_mode": "teacher_history", "smoke_max_rows": smoke_max_rows,
              "batch_sizes": {"teacher_history": batch_size, "generated_history": generated_batch_size},
              "inference_backend": "batched saved prefix/flow/full-forward greedy text; vocabulary argmax on device; row-keyed flow RNG",
              "non_final_smoke": smoke_max_rows is not None, "status": "running", "modes": {},
              "started_at": datetime.now(timezone.utc).isoformat(),
              "metric_definitions": {
                  "translation_mae": "mean absolute error of raw controller dimensions 0:3 over valid tail-masked timesteps; native stored control units",
                  "rotation_mae": "mean absolute error of raw controller dimensions 3:6 over valid tail-masked timesteps; native stored control units",
                  "normalized_mse7": "mean squared difference of saved train-quantile-normalized sampled physical7 actions and expert actions; not flow training loss",
                  "gripper_sign_accuracy": "np.sign(predicted[6]) == np.sign(expert[6]); zero distinct; missing/nonfinite predictions incorrect; all required timesteps denominator",
                  "continuous_action_failures": "primary MAE/MSE null if required components unscorable; finite_only diagnostics always expose scored/expected/missing/nonfinite counts",
                  "memory_terminated": "generated-history rows after the Ledger Driver rejected a memory in that lane; every text metric fails, no action is scored, and they are excluded from action denominators; a model outcome, not an evaluation error",
                  "count_accuracy": "driver-valid, EOS-terminated generated memory whose ledger tick count equals the target tick count; every validation row denominator; invalid/missing/terminated incorrect",
                  "count_mae": "absolute tick-count error on driver-valid memories only; denominator and unscorable count explicit; no fabricated counts",
                  "count_mae_valid_explicit": "same statistic as count_mae, with an explicitly named valid-output denominator",
                  "format_valid_rate": "policy EOS validity AND an independent Ledger Driver step (Action grammar for this variant, legal edit, expected item total); all validation rows denominator",
                  "false_done_rate": "explicit Task:Complete claim, including malformed claims and whitespace after the colon, on rows whose label is Incomplete / all such rows; missing/invalid coverage separately reported",
                  "done_recall": "driver-valid Complete on rows whose label is Complete / all such rows; invalid/missing fail",
                  "action_exact_accuracy": "generated Action line exactly equals the label; reliable-replay rows of groups A and C only; missing/invalid outputs incorrect; null for group B",
                  "subgoal_delta_accuracy": "generated Subgoal edit line exactly equals the label; every validation row denominator",
                  "task_accuracy": "generated Task line exactly equals the label; every validation row denominator",
                  "latent_cosine_mean": "mean cosine of generated z with z*, per entity visible in the window, over rows with returned text; null for group A",
                  "no_validation_leakage": "disjoint train/val episode, NPZ and sample identities checked; saved training tokenizer/norm/config used; targets never fed in generated-history mode",
                  "teacher_history_limit": "teacher-history consumes the labelled memory and z* of t-20, INITIAL_MEMORY and zero latents for t<20, never t or t-1; not autonomous rollout",
              }}
    summary_path = output / "summary.json"
    save_json(summary_path, report)
    try:
        marker = read_json(checkpoint / "COMPLETE.json")
        if int(marker["optimizer_step"]) != step:
            raise ValueError(f"Requested step {step} differs from checkpoint {marker['optimizer_step']}")
        policy = JointMemoryPolicy(checkpoint)
        corpus, groups, evidence = _load_corpus(checkpoint, root, policy)
        report["dataset_evidence"] = evidence
        inference = _BatchedInference(policy)
        for mode in modes:
            report["modes"][mode] = _evaluate_mode(policy, inference, corpus, groups, output, mode,
                                                    smoke_max_rows, report["batch_sizes"][mode])
            save_json(summary_path, report)
        errors = any(mode["action"]["missing_components7"] or
                     any(mode["coverage"]["status_counts"][name] for name in
                         ("action_error", "conditioning_error", "inference_error", "history_unavailable"))
                     for mode in report["modes"].values())
        report["status"] = "non_final_smoke" if smoke_max_rows is not None else "completed_with_failures" if errors else "completed"
        report["full_validation"] = (smoke_max_rows is None and set(report["modes"]) == set(MODES)
                                     and all(mode["coverage"]["all_rows_accounted_for"] for mode in report["modes"].values()))
        report["wandb_metrics"] = _wandb_metrics(report["modes"], step)
        if smoke_max_rows is not None:
            report["wandb_metrics"] = {key.replace("validation/", "validation_smoke/", 1): value for key, value in report["wandb_metrics"].items()}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(summary_path, report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["full_validation"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(summary_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16, help="Teacher-history inference batch size (default 16; e.g. 64 if memory permits)")
    parser.add_argument("--generated-batch-size", type=int, help="Generated-history lane batch size; defaults to --batch-size")
    parser.add_argument("--smoke-max-rows", type=int, help="NON-FINAL smoke only: maximum chronological dense rows eligible per mode")
    parser.add_argument("--mode", choices=MODES, help="Run one independent history mode for parallel validation")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.dataset, args.output, args.step, smoke_max_rows=args.smoke_max_rows,
                                 batch_size=args.batch_size, generated_batch_size=args.generated_batch_size,
                                 modes=(args.mode,) if args.mode else MODES)
    print(json.dumps({"summary": str(args.output.resolve() / "summary.json"), "status": report["status"],
                      "optimizer_step": report["optimizer_step"], "full_validation": report["full_validation"]}, allow_nan=False))
    if report["status"] == "completed_with_failures":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
