"""Four-GPU, batch-64 full-parameter pi0.5 training, without accumulation.

Normal CLI: --contract contract.json --dataset dataset --output training [--resume CKPT]
Exactly 10,000 optimizer updates; all 2,000-step checkpoints are retained. A CPU
supervisor owns live W&B and launches one resumable training segment followed by
independent open-loop validation. Exiting each GPU worker releases its allocator
before evaluation; no training/evaluation model copies coexist on a GPU.

--no-wandb skips W&B for any run; training.jsonl and validation.jsonl still hold every metric.
--smoke-steps N (up to 500) runs a separate, explicitly non-final real four-GPU batch-64
smoke and proves exact saved parameter/Adam/RNG/sampler reload. Add
--smoke-validation-rows N to request a small NON-FINAL validation budget;
without that flag validation still covers every held-out row. Every update logs
input/update/wall timings and each GPU's allocator-reported peak memory.
--verify-checkpoint CKPT checks every saved leaf against live-state save hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np

from model_data import FullCorpus, JointTokenizer, read_json, save_json, sha256_file
from policy import bootstrap_openpi

PER_DEVICE_BATCH = int(os.environ.get("PI05_PER_DEVICE_BATCH", 16))  # 32 x 4 GPUs = global batch 128
DEVICE_COUNT = int(os.environ.get("PI05_DEVICES", 4))
GLOBAL_BATCH = PER_DEVICE_BATCH * DEVICE_COUNT
# Short comparison runs set both; the learning-rate schedule then decays over the shorter run.
TOTAL_UPDATES = int(os.environ.get("PI05_TOTAL_UPDATES", 10_000))
CHECKPOINT_INTERVAL = int(os.environ.get("PI05_CHECKPOINT_INTERVAL", 2_000))
MILESTONES = tuple(range(CHECKPOINT_INTERVAL, TOTAL_UPDATES + 1, CHECKPOINT_INTERVAL))
COUNTERS = ("epoch", "position", "optimizer_step", "batches_seen", "samples_seen")
EVENT_PREFIX = "PI05_EVENT "
SOURCE_FILES = ("joint_model.py", "model_data.py", "train.py", "policy.py",
                "evaluate_openloop.py", "memory_format.py", "ledger_v2.py")
MAX_SMOKE_STEPS = 500  # Long enough to read loss and gradient magnitudes before fixing weights.
MEMORY_VARIANTS = {"A": {"latents": False, "verbs": False},
                   "B": {"latents": True, "verbs": False},
                   "C": {"latents": True, "verbs": True}}
SAMPLER_ALGORITHM = "numpy.default_rng(SeedSequence([seed,epoch])).permutation(N)"


def append_log(path, event):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def _verify_contract(contract):
    training, model = contract["training"], contract["model"]
    required = {"global_batch_size": GLOBAL_BATCH, "per_device_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation": 1, "num_devices": DEVICE_COUNT,
                "optimizer_steps": TOTAL_UPDATES, "checkpoint_interval": CHECKPOINT_INTERVAL,
                "validation_interval": CHECKPOINT_INTERVAL, "ema": False,
                "text_loss_weight": 0.1, "action_loss_metric": "flow7"}
    for key, value in required.items():
        if training.get(key) != value:
            raise ValueError(f"training.{key} must be {value!r}, got {training.get(key)!r}")
    variant = MEMORY_VARIANTS.get(model.get("memory_variant"))
    if variant is None:
        raise ValueError(f"model.memory_variant must be one of {sorted(MEMORY_VARIANTS)}, got {model.get('memory_variant')!r}")
    align, verb = training.get("align_loss_weight"), training.get("verb_loss_weight")
    if not isinstance(align, (int, float)) or not isinstance(verb, (int, float)) or align < 0 or verb < 0:
        raise ValueError("training.align_loss_weight and verb_loss_weight must be nonnegative numbers")
    if (align > 0) != variant["latents"] or (verb > 0) != variant["verbs"]:
        raise ValueError(f"Loss weights disagree with memory variant {model['memory_variant']}: align={align}, verb={verb}")
    if bool(contract["dataset"].get("latents_dir")) != variant["latents"]:
        raise ValueError(f"dataset.latents_dir disagrees with memory variant {model['memory_variant']}")
    if (model["action_dim"], model["physical_action_dim"], model["action_horizon"]) != (32, 7, 20):
        raise ValueError("Expected native32/physical7/horizon20")
    if (model["paligemma_variant"], model["action_expert_variant"]) != ("gemma_2b", "gemma_300m"):
        raise ValueError("Full-parameter training requires the original model variants")
    if any(contract["execution"][key] != 20 for key in
           ("prediction_horizon", "executed_steps", "memory_update_interval")):
        raise ValueError("Dense supervision retains the 20-step execution/memory protocol")
    if not 0 < int(training["warmup_optimizer_steps"]) < TOTAL_UPDATES:
        raise ValueError("Warmup must lie strictly inside the 10,000-update schedule")
    from model_data import IMAGE_SLOTS
    if "raw_image_path" not in contract["dataset"]["row_fields"]:
        raise ValueError("v2 contract must list dataset.row_fields raw_image_path (clean front image)")
    if model.get("image_slots") != IMAGE_SLOTS:
        raise ValueError(f"model.image_slots must be {IMAGE_SLOTS}, got {model.get('image_slots')!r}")
    if Path(contract["dataset"]["index"]).name != "train.jsonl":
        raise ValueError("Training must select the episode-holdout train.jsonl index")


def checkpoint_payload(opt_state, rng, counters):
    return {"optimizer": opt_state, "rng": rng,
            **{key: np.asarray(counters[key], np.int32) for key in COUNTERS}}


def completed_checkpoints(output):
    candidates = [path for path in Path(output).glob("checkpoint_*")
                  if (path / "COMPLETE.json").is_file()]
    return sorted(candidates, key=lambda path: (
        int(read_json(path / "COMPLETE.json")["optimizer_step"]), path.name))


def _tree_digests(tree):
    """Exact all-leaf proof; transfer at most one global leaf to host at a time."""
    import jax

    result = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(jax.device_get(value))
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        result[jax.tree_util.keystr(path)] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
        del array
    return result


def save_checkpoint(output, state, payload, metadata, corpus, reason):
    import jax
    import orbax.checkpoint as ocp

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    counters = {key: int(np.asarray(payload[key])) for key in COUNTERS}
    _check_sampler(counters, len(corpus.rows))
    _check_optimizer_step(payload["optimizer"], counters["optimizer_step"])
    step = counters["optimizer_step"]
    if (reason == "interrupted" and step in MILESTONES
            and metadata["mode"] == "full_10000_update_training"
            and not (output / f"checkpoint_{step:06d}").exists()):
        reason = "milestone"
    if reason == "milestone":
        if step not in MILESTONES:
            raise ValueError(f"Not a retained milestone update: {step}")
        name = f"checkpoint_{step:06d}"
    else:
        name = f"checkpoint_{reason}_u{step:06d}_{uuid.uuid4().hex[:12]}"
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"Never overwrite a retained checkpoint: {destination}")
    params = {"params": state.to_pure_dict()}
    parameter_bytes = sum(int(value.size * value.dtype.itemsize) for value in jax.tree.leaves(params))
    training_bytes = sum(int(value.size * value.dtype.itemsize) for value in jax.tree.leaves(payload))
    required = int((parameter_bytes + training_bytes) * 1.15) + 512 * 1024**2
    if shutil.disk_usage(output).free < required:
        raise OSError(f"Atomic save needs {required} free bytes; no retained checkpoint will be deleted")
    temporary = output / (".incomplete-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        # Hash the live arrays, not checkpoint files, so verification tests the
        # complete save/restore path rather than merely re-reading file bytes.
        proof = {"params": _tree_digests(params), "train_state": _tree_digests(payload)}
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(temporary / "params", args=ocp.args.PyTreeSave(params))
            checkpointer.save(temporary / "train_state", args=ocp.args.PyTreeSave(payload))
        saved = {**metadata, "sampler": {
            **counters, "reason": reason, "sample_count": len(corpus.rows),
            "seed": int(metadata["training"]["seed"]), "algorithm": SAMPLER_ALGORITHM,
            "numpy_version": np.__version__, "tail": "-1 fillers; both losses exclude fillers"}}
        save_json(temporary / "metadata.json", saved)
        save_json(temporary / "reload_proof.json", proof)
        shutil.copyfile(corpus.root / "norm_stats.json", temporary / "norm_stats.json")
        shutil.copyfile(corpus.root / "split_manifest.json", temporary / "split_manifest.json")
        shutil.copyfile(corpus.tokenizer.path, temporary / "tokenizer.model")
        source_directory = temporary / "source"
        source_directory.mkdir()
        for name, expected in metadata["source_sha256"].items():
            snapshot = source_directory / name
            shutil.copyfile(Path(__file__).with_name(name), snapshot)
            if sha256_file(snapshot) != expected:
                raise RuntimeError(f"Source changed after training initialization: {name}")
        for name, expected in (("norm_stats.json", metadata["normalization_sha256"]),
                               ("split_manifest.json", metadata["split_manifest_sha256"]),
                               ("tokenizer.model", metadata["tokenizer"]["sha256"])):
            if sha256_file(temporary / name) != expected:
                raise RuntimeError(f"Checkpoint dependency changed during training: {name}")
        save_json(temporary / "COMPLETE.json", {
            **counters, "reason": reason, "created_utc": datetime.now(timezone.utc).isoformat(),
            "full_inference_parameter_bytes": parameter_bytes,
            "training_state_bytes": training_bytes, "gradient_accumulator_bytes": 0,
            "checkpoint_payload_bytes": parameter_bytes + training_bytes,
            "atomic_save_required_free_bytes": required,
            "training_state_contains_base_weights": False,
            "metadata_sha256": sha256_file(temporary / "metadata.json"),
            "reload_proof_sha256": sha256_file(temporary / "reload_proof.json")})
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    save_json(output / "latest.json", {"checkpoint": destination.name, **counters})
    return destination


def make_mesh():
    import jax
    from jax.sharding import Mesh

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if jax.process_count() != 1 or len(devices) != DEVICE_COUNT:
        raise RuntimeError(f"Expose exactly four GPUs in one process, got {jax.devices()}")
    return Mesh(np.asarray(devices, dtype=object), ("fsdp",))


def state_shardings(tree, mesh):
    """Shard large masters/moments; replicate only small or indivisible leaves."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    def choose(value):
        axes = [axis for axis, size in enumerate(value.shape) if size >= DEVICE_COUNT and size % DEVICE_COUNT == 0]
        spec = [None] * len(value.shape)
        if axes and math.prod(value.shape) * np.dtype(value.dtype).itemsize >= 1024**2:
            axis = max(axes, key=lambda index: value.shape[index])
            spec[axis] = "fsdp"
        return NamedSharding(mesh, P(*spec))

    return jax.tree.map(choose, tree)


def batch_shardings(batch, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    def choose(value):
        if not value.shape or value.shape[0] != GLOBAL_BATCH:
            raise ValueError(f"Every batch leaf must have leading global batch64, got {value.shape}")
        return NamedSharding(mesh, P("fsdp", *([None] * (value.ndim - 1))))

    return jax.tree.map(choose, batch)


def restore_training_state(checkpoint, template, shardings):
    import jax
    import orbax.checkpoint as ocp

    restore_args = jax.tree.map(
        lambda value, sharding: ocp.ArrayRestoreArgs(
            restore_type=jax.Array, sharding=sharding, global_shape=value.shape, dtype=value.dtype),
        template, shardings)
    with ocp.PyTreeCheckpointer() as checkpointer:
        return checkpointer.restore(Path(checkpoint) / "train_state", args=ocp.args.PyTreeRestore(
            item=template, restore_args=restore_args))


def make_optimizer(training):
    import optax

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=training["learning_rate"] / (training["warmup_optimizer_steps"] + 1),
        peak_value=training["learning_rate"], warmup_steps=training["warmup_optimizer_steps"],
        decay_steps=TOTAL_UPDATES, end_value=training["final_learning_rate"])
    optimizer = optax.chain(
        optax.clip_by_global_norm(training["gradient_clip_norm"]),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, mu_dtype=np.float32,
                    weight_decay=training["weight_decay"]))
    return optimizer, schedule


def make_training_function(graph, optimizer, parameter_shardings, optimizer_shardings, data_shardings, mesh):
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import optax
    from joint_model import TRAINABLE_FILTER, parameter_component

    replicated = NamedSharding(mesh, P())

    def update(state, opt_state, rng, batch):
        parameters = state.filter(TRAINABLE_FILTER)
        other = state.filter(nnx.Not(TRAINABLE_FILTER))
        valid_samples = batch["sample_mask"] > 0
        batch = {**batch, "action_mask": batch["action_mask"] & valid_samples[:, None],
                 "target_mask": batch["target_mask"] & valid_samples[:, None]}
        # Filler rows are already all-zero; re-mask defensively like the fields above.
        for name in ("target_loss_mask", "align_mask", "verb_mask"):
            if name in batch:
                batch[name] = batch[name] & valid_samples[:, None]

        def objective(trainable):
            model = nnx.merge(graph, trainable, other)
            # These are GLOBAL arrays. Both token/timestep numerators and their
            # denominators reduce over all four partitions, not per-card means.
            return model.loss(rng, batch)

        (_, metrics), gradients = jax.value_and_grad(
            jax.checkpoint(objective, prevent_cse=False), has_aux=True)(parameters)
        updates, next_opt = optimizer.update(gradients, opt_state, parameters)
        next_parameters = optax.apply_updates(parameters, updates)
        previous = traverse_util.flatten_dict(parameters.to_pure_dict())
        following = traverse_util.flatten_dict(next_parameters.to_pure_dict())
        squared = {name: jnp.asarray(0.0, jnp.float32) for name in ("vlm", "action", "vision", "other")}
        for path, value in following.items():
            squared[parameter_component(path)] += jnp.sum(jnp.square(value - previous[path]))
        metrics = {**metrics, "gradient_norm": optax.global_norm(gradients),
                   "parameter_update_l2": jnp.sqrt(sum(squared.values())),
                   "valid_samples": jnp.sum(valid_samples),
                   **{f"{name}_update_l2": jnp.sqrt(value) for name, value in squared.items()}}
        model = nnx.merge(graph, state)
        nnx.update(model, next_parameters)
        return nnx.state(model), next_opt, metrics

    return jax.jit(update, in_shardings=(parameter_shardings, optimizer_shardings, replicated, data_shardings),
                   out_shardings=(parameter_shardings, optimizer_shardings, replicated), donate_argnums=(0, 1))


def _host_metrics(metrics):
    result = {key: float(np.asarray(value)) for key, value in metrics.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Nonfinite training metrics: {result}")
    return result


def _check_sampler(counters, sample_count):
    updates_per_epoch = math.ceil(sample_count / GLOBAL_BATCH)
    if sample_count <= 0:
        raise ValueError("Exact sampler continuation requires a nonempty training split")
    epoch, batches_in_epoch = divmod(counters["optimizer_step"], updates_per_epoch)
    expected = {"epoch": epoch, "position": batches_in_epoch * GLOBAL_BATCH,
                "optimizer_step": counters["optimizer_step"], "batches_seen": counters["optimizer_step"],
                "samples_seen": epoch * sample_count + batches_in_epoch * GLOBAL_BATCH}
    if counters != expected or not 0 <= counters["optimizer_step"] <= TOTAL_UPDATES:
        raise ValueError(f"Sampler is not an exact full-epoch continuation: {counters}, expected {expected}")


def _check_optimizer_step(opt_state, step):
    """Both Adam bias correction and the LR schedule must resume this update."""
    import jax

    counts = [int(np.asarray(value)) for path, value in jax.tree_util.tree_flatten_with_path(opt_state)[0]
              if path and getattr(path[-1], "name", None) == "count"]
    if len(counts) != 2 or any(count != step for count in counts):
        raise ValueError(f"Adam/schedule counters {counts} disagree with optimizer_step={step}")


def _device_memory_metrics(mesh):
    """Allocator peaks include initialization/compilation; never claim nvidia-smi usage."""
    devices = []
    fields = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "bytes_reserved", "peak_bytes_reserved")
    for device in mesh.devices.flat:
        stats = device.memory_stats() or {}
        devices.append({"device_id": device.id, **{
            name: int(stats[name]) if stats.get(name) is not None else None for name in fields}})
    peaks = [item["peak_bytes_in_use"] for item in devices if item["peak_bytes_in_use"] is not None]
    return {"gpu_memory": devices, "gpu_peak_memory_bytes": max(peaks) if peaks else None,
            "gpu_peak_memory_devices_reported": len(peaks)}


def train_segment(contract, dataset, output, smoke_steps, resume, segment_end):
    import flax.nnx as nnx
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
    from joint_model import (TRAINABLE_FILTER, config_from_contract, initialize_from_base,
                             load_full_checkpoint, parameter_report)

    mesh = make_mesh()
    replicated = NamedSharding(mesh, P())
    host_restore = SingleDeviceSharding(jax.devices("cpu")[0])
    training = contract["training"]
    seed = int(training["seed"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (Path(dataset) / "split_manifest.json").is_file():
        raise FileNotFoundError("Episode holdout split_manifest.json is mandatory")
    tokenizer = JointTokenizer(contract["model"]["tokenizer"])
    corpus = FullCorpus(dataset, contract, tokenizer)
    if corpus.split_name != "train" or corpus.split_manifest["splits"]["train"]["queries"] != corpus.split_manifest["splits"]["train"]["actions"]:
        raise ValueError("Training must use every dense row of the episode-holdout training split")
    capacities, lengths = corpus.measure_capacities()
    tokenizer.capacities = capacities
    config = config_from_contract(contract, capacities)
    optimizer, schedule = make_optimizer(training)
    metadata = {
        "schema_version": 4, "contract": contract, "training": training,
        "openpi_root": contract["model"]["openpi_root"], "model_config": dataclasses.asdict(config),
        "capacities": capacities, "measured_lengths": lengths, "tokenizer": tokenizer.metadata(),
        "dataset_fingerprint": corpus.fingerprint, "coverage": corpus.coverage,
        "split_manifest_sha256": sha256_file(corpus.root / "split_manifest.json"),
        "normalization_sha256": sha256_file(corpus.root / "norm_stats.json"),
        "data_source": corpus.split_manifest["source"],
        "normalization": corpus.split_manifest["normalization"],
        "sampling": {"training_queries": "every t in range(episode_length)",
                     "previous_memory": "target[t-20] if t>=20, otherwise initial memory",
                     "action_targets": "actions[t:min(t+20,N)], masked absent future steps",
                     "executed_steps": 20, "memory_update_interval": 20},
        "instruction_totals": {row["instruction"]: int(row["memory_target"]["total_goals"]) for row in corpus.rows},
        "optimizer": {"name": "AdamW", "b1": 0.9, "b2": 0.95, "eps": 1e-8,
                      "weight_decay": training["weight_decay"], "gradient_clip_norm": training["gradient_clip_norm"],
                      "schedule": "warmup_cosine", "total_optimizer_steps": TOTAL_UPDATES,
                      "trainable_filter": "nnx.Param", "master_dtype": "float32", "moment_dtype": "float32",
                      "gradient_accumulation": 1},
        "mode": "real_batch64_smoke_nonfinal" if smoke_steps else "full_10000_update_training",
        "training_method": "full_parameter_finetuning", "compute_dtype": str(config.dtype),
        "distributed": {"strategy": "GSPMD_FSDP", "mesh_axis": "fsdp", "devices": DEVICE_COUNT,
                        "global_batch": GLOBAL_BATCH, "per_device_batch": PER_DEVICE_BATCH,
                        "gradient_accumulation": 1, "loss_reduction": "global masked token/timestep means",
                        "parameter_sharding": "largest divisible dimension for leaves >=1MiB; otherwise replicated",
                        "optimizer_sharding": "same shape rule as masters; no replicated full Adam tree",
                        "rematerialization": "jax.checkpoint pure loss, plus native model rematerialization"},
        "update_component_classification": {
            "vision": "PaliGemma/img", "vlm": "PaliGemma/llm excluding *_1 action expert paths",
            "action": "PaliGemma/llm containing *_1 path components", "other": "remaining projections and MLPs"},
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        "versions": {name: package_metadata.version(name) for name in
                     ("jax", "jaxlib", "numpy", "flax", "optax", "orbax-checkpoint")},
        "rng": {"implementation": str(jax.config.jax_default_prng_impl),
                "split": "next_rng,loss_rng=jax.random.split(rng) once per optimizer update"},
        "devices": [{"id": device.id, "kind": device.device_kind, "platform": device.platform}
                    for device in mesh.devices.flat]}
    if resume is None:
        if os.environ.get("FINETUNE_FROM"):
            # Fine-tune: all parameters from a finished run, fresh optimizer and counters.
            model, _ = load_full_checkpoint(os.environ["FINETUNE_FROM"], restore_sharding=host_restore)
            save_json(output / "base_restore.json", {"finetune_from": os.environ["FINETUNE_FROM"]})
        else:
            model, initialization = initialize_from_base(config, contract["model"]["base_checkpoint"], seed,
                                                         restore_sharding=host_restore)
            save_json(output / "base_restore.json", initialization)
    else:
        model, saved = load_full_checkpoint(resume, restore_sharding=host_restore)
        for key in ("model_config", "tokenizer", "dataset_fingerprint", "split_manifest_sha256",
                    "normalization_sha256", "normalization", "data_source", "sampling",
                    "instruction_totals", "training", "optimizer", "source_sha256", "versions", "rng", "mode"):
            if metadata[key] != saved[key]:
                raise ValueError(f"Exact resume prerequisite changed: {key}")
        sampler = saved["sampler"]
        if (sampler["numpy_version"] != np.__version__ or sampler["seed"] != seed
                or sampler["algorithm"] != SAMPLER_ALGORITHM or sampler["sample_count"] != len(corpus.rows)):
            raise ValueError("Exact shuffle resume requires unchanged sampler seed/algorithm/version/size")
    metadata["parameters"] = parameter_report(model)
    graph, state = nnx.split(model)
    del model
    parameter_shardings = state_shardings(state, mesh)
    state = jax.device_put(state, parameter_shardings)
    parameter_bytes = sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(state))
    partitioned_bytes = sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(state)
                            if not value.sharding.is_fully_replicated)
    metadata["distributed"].update({"parameter_bytes": int(parameter_bytes),
                                     "partitioned_parameter_bytes": int(partitioned_bytes)})
    if partitioned_bytes <= 0:
        raise RuntimeError("No actual FSDP parameter partitioning was established")
    trainable = state.filter(TRAINABLE_FILTER)
    opt_template = jax.eval_shape(optimizer.init, trainable)
    optimizer_shardings = state_shardings(opt_template, mesh)
    rng = jax.device_put(jax.random.PRNGKey(seed), replicated)
    counters = dict.fromkeys(COUNTERS, 0)
    with mesh:
        if resume is None:
            opt_state = jax.jit(optimizer.init, in_shardings=(parameter_shardings.filter(TRAINABLE_FILTER),),
                                out_shardings=optimizer_shardings)(trainable)
        else:
            template = checkpoint_payload(opt_template, rng, counters)
            restore_shardings = {"optimizer": optimizer_shardings, "rng": replicated,
                                 **dict.fromkeys(COUNTERS, replicated)}
            restored = restore_training_state(resume, template, restore_shardings)
            opt_state, rng = restored["optimizer"], restored["rng"]
            counters = {key: int(np.asarray(restored[key])) for key in COUNTERS}
            if counters != {key: int(saved["sampler"][key]) for key in COUNTERS}:
                raise ValueError("Saved sampler metadata disagrees with serialized state")
            del restored
    jax.block_until_ready((state, opt_state, rng))
    optimizer_bytes = sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(opt_state))
    partitioned_optimizer_bytes = sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(opt_state)
                                      if not value.sharding.is_fully_replicated)
    if partitioned_optimizer_bytes <= 0:
        raise RuntimeError("Adam moments were not actually partitioned over four devices")
    metadata["distributed"].update({"optimizer_bytes": int(optimizer_bytes),
                                     "partitioned_optimizer_bytes": int(partitioned_optimizer_bytes),
                                     "gradient_accumulator_bytes": 0,
                                     "restore_placement": "CPU then per-leaf FSDP; never full GPU replication"})
    _check_optimizer_step(opt_state, counters["optimizer_step"])
    del trainable, opt_template
    _check_sampler(counters, len(corpus.rows))
    if not counters["optimizer_step"] < segment_end <= (smoke_steps or TOTAL_UPDATES):
        raise ValueError(f"Invalid segment boundary {segment_end} after {counters['optimizer_step']}")
    if not smoke_steps and segment_end not in MILESTONES:
        raise ValueError("Production segments must end at a retained 2,000-update milestone")
    save_json(output / "run_metadata.json", metadata)
    emit({"event": "initialized", "optimizer_step": counters["optimizer_step"],
          "distributed": metadata["distributed"], "parameters": metadata["parameters"],
          **_device_memory_metrics(mesh)})
    stop_requested = {"signal": None}

    def request_stop(signum, frame):
        del frame
        stop_requested["signal"] = signum

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    committed = True
    last_checkpoint = resume
    update = None
    epoch_order = None
    order_epoch = None
    started = time.monotonic()
    try:
        with mesh:
            while counters["optimizer_step"] < segment_end:
                if stop_requested["signal"] is not None:
                    raise InterruptedError(f"Stop requested by signal {stop_requested['signal']}")
                step_started = time.monotonic()
                epoch, position = counters["epoch"], counters["position"]
                if order_epoch != epoch:
                    epoch_order = corpus.epoch_order(seed, epoch)
                    order_epoch = epoch
                    if not np.array_equal(np.sort(epoch_order), np.arange(len(corpus.rows))):
                        raise AssertionError("Epoch permutation duplicates or omits training rows")
                selected = epoch_order[position:position + GLOBAL_BATCH]
                valid_count = len(selected)
                indices = np.full(GLOBAL_BATCH, -1, np.int64)
                indices[:valid_count] = selected
                host_batch = corpus.batch_indices(indices)
                if not np.array_equal(host_batch["sample_mask"], indices >= 0):
                    raise AssertionError("Batch filler mask disagrees with the deterministic sampler")
                if any(np.any(host_batch[name][valid_count:])
                       for name in ("action_mask", "target_mask", "target_loss_mask", "align_mask", "verb_mask")
                       if name in host_batch):
                    raise AssertionError("Filler rows must have zero action/text/latent/verb loss weight")
                shardings = batch_shardings(host_batch, mesh)
                batch = jax.device_put(host_batch, shardings)
                jax.block_until_ready(batch)
                input_seconds = time.monotonic() - step_started
                compilation_included = update is None
                if update is None:
                    update = make_training_function(graph, optimizer, parameter_shardings,
                                                    optimizer_shardings, shardings, mesh)
                next_rng, loss_rng = jax.random.split(rng)
                committed = False
                update_started = time.monotonic()
                next_state, next_opt, metrics = update(state, opt_state, loss_rng, batch)
                jax.block_until_ready((next_state, next_opt, metrics))
                update_seconds = time.monotonic() - update_started
                metrics = _host_metrics(metrics)
                expected_counts = {"valid_samples": valid_count,
                                   "valid_action_steps": int(np.sum(host_batch["action_mask"])),
                                   "target_tokens": int(np.sum(host_batch["target_mask"]))}
                if any(metrics[key] != count for key, count in expected_counts.items()):
                    raise AssertionError(f"Loss denominators are not global real-row counts: {metrics}, {expected_counts}")
                expected_loss = ((32.0 / 7.0) * metrics["flow32"] + training["text_loss_weight"] * metrics["text_ce"]
                                 + training["align_loss_weight"] * metrics["align"]
                                 + training["verb_loss_weight"] * metrics["verb"])
                # Metrics are float32 on device, so the sum can differ by rounding.
                if not math.isclose(metrics["loss"], expected_loss, rel_tol=1e-5, abs_tol=1e-5):
                    raise AssertionError(f"Objective is not flow7 + weighted text/align/verb: {metrics['loss']} != {expected_loss}")
                for component, count in metadata["parameters"]["parameter_count_by_component"].items():
                    if count and metrics[f"{component}_update_l2"] <= 0:
                        raise FloatingPointError(f"Optimizer did not actually change {component} parameters")
                at_epoch_end = position + valid_count == len(corpus.rows)
                next_counters = {"epoch": epoch + int(at_epoch_end),
                                 "position": 0 if at_epoch_end else position + valid_count,
                                 "optimizer_step": counters["optimizer_step"] + 1,
                                 "batches_seen": counters["batches_seen"] + 1,
                                 "samples_seen": counters["samples_seen"] + valid_count}
                state, opt_state, rng, counters = next_state, next_opt, next_rng, next_counters
                committed = True
                _check_sampler(counters, len(corpus.rows))
                step_seconds = time.monotonic() - step_started
                event = {"event": "optimizer_update", **counters, **metrics,
                         "global_batch_size": GLOBAL_BATCH, "per_device_batch_size": PER_DEVICE_BATCH,
                         "gradient_accumulation": 1, "masked_fillers": GLOBAL_BATCH - valid_count,
                         "valid_samples_per_device": [int(np.sum(indices[i:i + PER_DEVICE_BATCH] >= 0))
                                                      for i in range(0, GLOBAL_BATCH, PER_DEVICE_BATCH)],
                         "learning_rate": float(schedule(counters["optimizer_step"] - 1)),
                         "input_seconds": input_seconds, "update_seconds": update_seconds,
                         "step_seconds": step_seconds, "compilation_included": compilation_included,
                         "real_samples_per_second": valid_count / step_seconds,
                         "elapsed_seconds": time.monotonic() - started, **_device_memory_metrics(mesh)}
                append_log(output / "training.jsonl", event)
                emit(event)
                if at_epoch_end:
                    save_json(output / f"coverage_epoch_{epoch + 1:06d}.json", {
                        "epoch": epoch + 1, **corpus.coverage, "no_drop_last": True,
                        "visited_sample_ids": [corpus.rows[int(index)]["sample_id"] for index in epoch_order],
                        "visited_episode_ids": sorted({(str(row["task_id"]), str(row["episode_id"]))
                                                       for row in corpus.rows}),
                        "optimizer_steps_this_epoch": math.ceil(len(corpus.rows) / GLOBAL_BATCH),
                        "cumulative_optimizer_steps": counters["optimizer_step"]})
                del batch, host_batch
            reason = "smoke" if smoke_steps else "milestone"
            last_checkpoint = save_checkpoint(output, state, checkpoint_payload(opt_state, rng, counters),
                                              metadata, corpus, reason)
            result = {"checkpoint": str(last_checkpoint), "output": str(output), "counters": counters,
                      "smoke_steps": smoke_steps, "final_training": not smoke_steps and segment_end == TOTAL_UPDATES,
                      "elapsed_seconds": time.monotonic() - started, **_device_memory_metrics(mesh)}
            save_json(output / "segment_result.json", result)
            emit({"event": "checkpoint_saved", "optimizer_step": counters["optimizer_step"],
                  "checkpoint": str(last_checkpoint)})
            return result
    except BaseException:
        if committed:
            try:
                with mesh:
                    save_checkpoint(output, state, checkpoint_payload(opt_state, rng, counters),
                                    metadata, corpus, "interrupted")
            except BaseException as error:
                print(f"Interruption save failed; retaining all earlier checkpoints: {error}", file=sys.stderr, flush=True)
        else:
            print(f"Donated buffers are uncommitted; exact resume starts at {last_checkpoint}", file=sys.stderr, flush=True)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def verify_checkpoint(checkpoint, output):
    import flax.nnx as nnx
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
    from joint_model import TRAINABLE_FILTER, load_full_checkpoint

    checkpoint = Path(checkpoint).resolve()
    marker = read_json(checkpoint / "COMPLETE.json")
    if sha256_file(checkpoint / "reload_proof.json") != marker["reload_proof_sha256"]:
        raise ValueError("Checkpoint reload-proof checksum mismatch")
    expected = read_json(checkpoint / "reload_proof.json")
    mesh = make_mesh()
    replicated = NamedSharding(mesh, P())
    model, metadata = load_full_checkpoint(checkpoint, restore_sharding=SingleDeviceSharding(jax.devices("cpu")[0]))
    _, state = nnx.split(model)
    del model
    state = jax.device_put(state, state_shardings(state, mesh))
    optimizer, _ = make_optimizer(metadata["training"])
    opt_template = jax.eval_shape(optimizer.init, state.filter(TRAINABLE_FILTER))
    template = checkpoint_payload(opt_template, jax.random.PRNGKey(0), dict.fromkeys(COUNTERS, 0))
    shardings = {"optimizer": state_shardings(opt_template, mesh), "rng": replicated,
                 **dict.fromkeys(COUNTERS, replicated)}
    with mesh:
        restored = restore_training_state(checkpoint, template, shardings)
    actual = {"params": _tree_digests({"params": state.to_pure_dict()}), "train_state": _tree_digests(restored)}
    for group in expected:
        if actual[group] != expected[group]:
            different = sorted(key for key in set(actual[group]) | set(expected[group])
                               if actual[group].get(key) != expected[group].get(key))
            raise AssertionError(f"Exact {group} checkpoint reload mismatch: {different[:20]}")
    counters = {key: int(np.asarray(restored[key])) for key in COUNTERS}
    _check_sampler(counters, metadata["sampler"]["sample_count"])
    _check_optimizer_step(restored["optimizer"], counters["optimizer_step"])
    if counters != {key: int(metadata["sampler"][key]) for key in COUNTERS}:
        raise AssertionError("Reloaded RNG/sampler/optimizer proof disagrees with metadata")
    report = {"status": "passed", "checkpoint": str(checkpoint), "optimizer_step": counters["optimizer_step"],
              "comparison": "all parameter, Adam, RNG and sampler leaf shapes/dtypes/SHA256 exactly equal",
              "parameter_leaves": len(actual["params"]), "training_state_leaves": len(actual["train_state"]),
              "parameters": metadata["parameters"],
              "final_training": metadata["mode"] == "full_10000_update_training" and counters["optimizer_step"] == TOTAL_UPDATES}
    destination = Path(output) / f"reload_verification_{checkpoint.name}.json"
    save_json(destination, report)
    emit({"event": "reload_verified", "optimizer_step": counters["optimizer_step"], "report": str(destination)})
    return report


def _run_process(command, *, env=None, on_event=None):
    """Stream worker metrics immediately; never run training in this supervisor."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, text=True,
                               encoding="utf-8", errors="replace", bufsize=1, env=env)
    stopped = {"signal": None}

    def forward_signal(signum, frame):
        del frame
        stopped["signal"] = signum
        if process.poll() is None:
            process.send_signal(signum)

    handlers = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if on_event is not None and line.startswith(EVENT_PREFIX):
                on_event(json.loads(line[len(EVENT_PREFIX):]))
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        if stopped["signal"] is not None:
            raise InterruptedError(f"Supervisor stopped by signal {stopped['signal']}; saved checkpoints are retained")
    finally:
        process.stdout.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait()


def _wandb_run(contract, output, smoke_steps, disabled):
    if disabled:
        return None
    import wandb

    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("Live training requires WANDB_MODE=online; offline/disabled is not live logging")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Online W&B requires WANDB_API_KEY supplied only in the supervisor runtime environment")
    # wandb.login(key=...) can write .netrc. Let online init consume the runtime
    # environment instead; never persist, print, or put this secret in config.
    config = contract["training"].get("wandb", {})
    if config.get("enabled", True) is not True or config.get("mode", "online") != "online":
        raise RuntimeError("The contract must enable live online W&B")
    identity_path = Path(output) / "wandb_run.json"
    identity = read_json(identity_path) if identity_path.is_file() else {"id": wandb.util.generate_id()}
    run = wandb.init(project=config.get("project", "pi05-retrain-10k"), entity=config.get("entity"),
                     id=identity["id"], resume="allow", mode="online", dir=str(output),
                     name=config.get("name", Path(output).name),
                     config={"training": contract["training"], "nonfinal_smoke": bool(smoke_steps),
                             "global_batch_size": GLOBAL_BATCH, "per_device_batch_size": PER_DEVICE_BATCH,
                             "devices": DEVICE_COUNT, "gradient_accumulation": 1})
    if run is None or run.settings.mode != "online":
        raise RuntimeError("W&B did not start an online run")
    # Use the actual optimizer counter as the chart x-axis, including validation
    # at the SAME checkpoint step and deterministic replay after interruption.
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("validation/*", step_metric="optimizer_step")
    run.define_metric("validation_smoke/*", step_metric="optimizer_step")
    save_json(identity_path, {"id": run.id, "url": run.url, "project": run.project, "mode": "online"})
    return run


def supervise(args, contract, output):
    output.mkdir(parents=True, exist_ok=True)
    existing = completed_checkpoints(output)
    resume = args.resume.resolve() if args.resume else (existing[-1] if existing else None)
    if resume is not None:
        if not (resume / "COMPLETE.json").is_file():
            raise ValueError(f"Resume is not a completed checkpoint: {resume}")
        if existing and int(read_json(resume / "COMPLETE.json")["optimizer_step"]) < int(
                read_json(existing[-1] / "COMPLETE.json")["optimizer_step"]):
            raise ValueError("Refusing to rewind an output directory with later retained checkpoints")
        marker = read_json(resume / "COMPLETE.json")
        if sha256_file(resume / "metadata.json") != marker["metadata_sha256"]:
            raise ValueError("Resume metadata differs from its completion marker")
        saved = read_json(resume / "metadata.json")
        expected_mode = "real_batch64_smoke_nonfinal" if args.smoke_steps else "full_10000_update_training"
        if saved["mode"] != expected_mode or saved["contract"] != contract:
            raise ValueError("Resume must use the unchanged contract and training/smoke mode")
        expected_sources = {name: sha256_file(Path(__file__).with_name(name)) for name in SOURCE_FILES}
        if saved["source_sha256"] != expected_sources:
            raise ValueError("Exact resume requires unchanged model/data/trainer/evaluator sources")
        for name, expected in (("split_manifest.json", saved["split_manifest_sha256"]),
                               ("norm_stats.json", saved["normalization_sha256"]),
                               (contract["dataset"]["index"], saved["dataset_fingerprint"]["index_sha256"])):
            if sha256_file(args.dataset / name) != expected:
                raise ValueError(f"Exact resume dataset dependency changed: {name}")
        if not args.smoke_steps:
            for step in MILESTONES:
                if step <= int(marker["optimizer_step"]):
                    retained = output / f"checkpoint_{step:06d}" / "COMPLETE.json"
                    if not retained.is_file() or int(read_json(retained)["optimizer_step"]) != step:
                        raise ValueError(f"Resume requires retained milestone {step} in this output; do not modify an old run")
    run = _wandb_run(contract, output, args.smoke_steps, args.no_wandb)
    common = [sys.executable, "-u", str(Path(__file__).resolve()), "--contract", str(args.contract.resolve()),
              "--dataset", str(args.dataset.resolve()), "--output", str(output)]
    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.pop("WANDB_API_KEY", None)

    def log_event(event):
        if run is not None and event.get("event") == "optimizer_update":
            numeric = {f"train/{key}": value for key, value in event.items()
                       if isinstance(value, (int, float)) and key != "optimizer_step"}
            for item in event.get("gpu_memory", []):
                numeric.update({f"train/gpu_{item['device_id']}/{key}": value
                                for key, value in item.items() if key != "device_id" and value is not None})
            run.log({"optimizer_step": event["optimizer_step"], **numeric})
        elif run is not None and event.get("event") == "initialized":
            run.config.update({"actual_distributed": event["distributed"], "parameters": event["parameters"]},
                              allow_val_change=True)

    def evaluate(checkpoint):
        checkpoint = Path(checkpoint)
        step = int(read_json(checkpoint / "COMPLETE.json")["optimizer_step"])
        destination = output / "validation" / checkpoint.name
        marker_path = destination / "VALIDATED.json"
        expected_marker = {"step": step, "checkpoint_metadata_sha256": sha256_file(checkpoint / "metadata.json"),
                           "evaluator_sha256": sha256_file(Path(__file__).with_name("evaluate_openloop.py")),
                           "final_validation": not bool(args.smoke_steps),
                           "smoke_max_rows": args.smoke_validation_rows,
                           "batch_size": args.validation_batch_size,
                           "generated_batch_size": args.generated_validation_batch_size or args.validation_batch_size}
        marker = read_json(marker_path) if marker_path.is_file() else None
        summary_path = destination / "summary.json"
        valid_existing = (marker is not None and all(marker.get(key) == value for key, value in expected_marker.items())
                          and summary_path.is_file() and marker.get("summary_sha256") == sha256_file(summary_path))
        if not valid_existing:
            evaluation_env = dict(child_env)
            visible = evaluation_env.get("CUDA_VISIBLE_DEVICES")
            devices = visible.split(",") if visible else ["0", "1"]
            if len(devices) < 2:
                raise RuntimeError("Parallel validation requires two visible GPU devices")
            commands = []
            for mode, device in zip(("teacher_history", "generated_history"), devices[:2], strict=True):
                mode_env = dict(evaluation_env)
                mode_env["CUDA_VISIBLE_DEVICES"] = device
                command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate_openloop.py")),
                           "--checkpoint", str(checkpoint), "--dataset", str(args.dataset.resolve()),
                           "--output", str(destination / mode), "--step", str(step), "--mode", mode,
                           "--batch-size", str(args.validation_batch_size),
                           "--generated-batch-size", str(args.generated_validation_batch_size or args.validation_batch_size)]
                if args.smoke_validation_rows is not None:
                    command += ["--smoke-max-rows", str(args.smoke_validation_rows)]
                commands.append((command, mode_env))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(subprocess.run, command, env=mode_env, check=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                           for command, mode_env in commands]
                for future in futures:
                    try:
                        completed = future.result()
                    except subprocess.CalledProcessError as error:
                        print(error.stdout or "", file=sys.stderr, flush=True)
                        raise
                    print(completed.stdout, flush=True)
            teacher = read_json(destination / "teacher_history" / "summary.json")
            generated = read_json(destination / "generated_history" / "summary.json")
            if (set(teacher["modes"]) != {"teacher_history"} or set(generated["modes"]) != {"generated_history"}
                    or any(report["optimizer_step"] != step or report["checkpoint"] != str(checkpoint.resolve())
                           or report["smoke_max_rows"] != args.smoke_validation_rows
                           for report in (teacher, generated))):
                raise RuntimeError("Independent validation reports disagree on checkpoint or evaluation scope")
            modes = {**teacher["modes"], **generated["modes"]}
            if (teacher["dataset_evidence"] != generated["dataset_evidence"]
                    or teacher["batch_sizes"] != generated["batch_sizes"]
                    or teacher["metric_definitions"] != generated["metric_definitions"]):
                raise RuntimeError("Independent validation modes have inconsistent dataset or metric definitions")
            from evaluate_openloop import _wandb_metrics
            errors = any(mode["action"]["missing_components7"]
                         or any(mode["coverage"]["status_counts"][name] for name in
                                ("action_error", "conditioning_error", "inference_error", "history_unavailable"))
                         for mode in modes.values())
            bounded = args.smoke_validation_rows is not None
            summary = {**teacher, "modes": modes, "started_at": min(teacher["started_at"], generated["started_at"]),
                       "finished_at": max(teacher["finished_at"], generated["finished_at"]),
                       "status": "non_final_smoke" if bounded else "completed_with_failures" if errors else "completed",
                       # An invalid generated ledger ends its lane (memory_terminated). That is a
                       # measured model failure, so every row must be accounted for, not inferred.
                       "full_validation": not bounded and all(mode["coverage"]["all_rows_accounted_for"] for mode in modes.values()),
                       "wandb_metrics": _wandb_metrics(modes, step)}
            if bounded:
                summary["wandb_metrics"] = {
                    key.replace("validation/", "validation_smoke/", 1): value
                    for key, value in summary["wandb_metrics"].items()}
            save_json(summary_path, summary)
        summary = read_json(summary_path)
        bounded = args.smoke_validation_rows is not None
        if (summary.get("optimizer_step") != step or summary.get("smoke_max_rows") != args.smoke_validation_rows
                or summary.get("status") != ("non_final_smoke" if bounded else "completed")
                or summary.get("non_final_smoke") != bounded
                or (not bounded and summary.get("full_validation") is not True)):
            raise RuntimeError(f"Checkpoint evaluation did not satisfy the declared coverage: {summary_path}")
        if not valid_existing:
            save_json(marker_path, {**expected_marker, "summary_sha256": sha256_file(summary_path)})
        if run is not None:
            metrics = summary["wandb_metrics"]
            if args.smoke_steps:
                metrics = {key.replace("validation/", "validation_smoke/", 1): value for key, value in metrics.items()}
            run.log({"optimizer_step": step, **metrics})
        append_log(output / "validation.jsonl", {"optimizer_step": step, "checkpoint": str(checkpoint),
                                                 "summary": str(summary_path), **expected_marker})
        emit({"event": "checkpoint_validated", "optimizer_step": step,
              "summary": str(summary_path), "full_validation": summary["full_validation"],
              "nonfinal_training_smoke": bool(args.smoke_steps)})

    exit_code = 1
    try:
        # A crash after saving but before validation must not skip that milestone.
        for checkpoint in existing:
            marker = read_json(checkpoint / "COMPLETE.json")
            if marker.get("reason") == "milestone":
                evaluate(checkpoint)
        current_step = int(read_json(resume / "COMPLETE.json")["optimizer_step"]) if resume else 0
        target = args.smoke_steps or TOTAL_UPDATES
        if current_step > target:
            raise ValueError("Resume is beyond the requested optimizer-step target")
        result = None
        while current_step < target:
            end = target if args.smoke_steps else min(target, (current_step // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL)
            command = common + ["--worker", "--segment-end", str(end)]
            if args.smoke_steps:
                command += ["--smoke-steps", str(args.smoke_steps)]
            if resume:
                command += ["--resume", str(resume)]
            _run_process(command, env=child_env, on_event=log_event)
            result = read_json(output / "segment_result.json")
            if result["counters"]["optimizer_step"] != end:
                raise AssertionError("Worker did not complete the requested exact segment")
            resume = Path(result["checkpoint"])
            current_step = end
            if args.smoke_steps:
                _run_process(common + ["--verify-checkpoint", str(resume)], env=child_env, on_event=log_event)
            evaluate(resume)
        if not args.smoke_steps:
            for step in MILESTONES:
                checkpoint = output / f"checkpoint_{step:06d}"
                if not (checkpoint / "COMPLETE.json").is_file():
                    raise AssertionError(f"Missing retained milestone {step}; an external resume must bring all earlier milestones")
                if not (output / "validation" / checkpoint.name / "VALIDATED.json").is_file():
                    raise AssertionError(f"Missing independent checkpoint validation at {step}")
        result = {"status": "completed", "checkpoint": str(resume), "optimizer_step": current_step,
                  "target_optimizer_steps": target, "final_training": not bool(args.smoke_steps),
                  "global_batch_size": GLOBAL_BATCH, "per_device_batch_size": PER_DEVICE_BATCH,
                  "devices": DEVICE_COUNT, "gradient_accumulation": 1,
                  "retained_milestones": list(MILESTONES) if not args.smoke_steps else [],
                  "validation": (f"non-final smoke, explicit {args.smoke_validation_rows}-row budget"
                                 if args.smoke_validation_rows is not None
                                 else "all held-out episodes and rows, both history modes"),
                  "nonfinal_training_smoke": bool(args.smoke_steps),
                  "wandb_url": run.url if run else None}
        save_json(output / "training_result.json", result)
        emit({"event": "run_completed", **result})
        exit_code = 0
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Completed checkpoint; otherwise resume latest in output")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help=f"1-{MAX_SMOKE_STEPS} real batch64 updates, separate non-final output")
    parser.add_argument("--smoke-validation-rows", type=int,
                        help="Explicit NON-FINAL smoke subset; omission evaluates all held-out rows")
    parser.add_argument("--validation-batch-size", type=int, default=16,
                        help="Single-GPU checkpoint evaluator batch size (default: 16)")
    parser.add_argument("--generated-validation-batch-size", type=int,
                        help="Generated-history lane batch size; defaults to --validation-batch-size")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip W&B; every metric is still written to training.jsonl and validation.jsonl")
    parser.add_argument("--verify-checkpoint", type=Path, help="Verify exact all-leaf parameter/Adam/RNG/sampler reload")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--segment-end", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.smoke_steps <= MAX_SMOKE_STEPS:
        parser.error(f"Smoke steps must be 0 or 1-{MAX_SMOKE_STEPS}")
    if args.smoke_validation_rows is not None and (not args.smoke_steps or args.smoke_validation_rows < 1):
        parser.error("A positive --smoke-validation-rows requires explicitly non-final --smoke-steps")
    if args.validation_batch_size < 1 or (args.generated_validation_batch_size is not None and args.generated_validation_batch_size < 1):
        parser.error("Validation batch sizes must be positive")
    if args.smoke_steps and args.resume:
        parser.error("Smoke runs start from base weights in separate directories")
    contract = read_json(args.contract)
    _verify_contract(contract)
    if args.worker or args.verify_checkpoint:
        bootstrap_openpi({"openpi_root": contract["model"]["openpi_root"]})
        if args.verify_checkpoint:
            verify_checkpoint(args.verify_checkpoint, args.output)
        else:
            if args.segment_end is None:
                parser.error("Internal training worker requires --segment-end")
            train_segment(contract, args.dataset, args.output, args.smoke_steps, args.resume, args.segment_end)
        return
    destination = args.output.resolve()
    if args.smoke_steps:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        destination = destination / "smoke" / stamp
    supervise(args, contract, destination)


if __name__ == "__main__":
    main()
