"""Deployment of the full-parameter trained joint pi0.5 model, groups A/B/C.

CLI: policy.py --serve-json CHECKPOINT
     policy.py --diagnose CHECKPOINT --dataset DATASET
Only observational fields (overlay front, wrist and clean front images, state), the previous memory
text and the previous round's latent vectors enter infer(). Requests need raw_png_b64 (clean front). Each generated memory is checked and applied by the Ledger
Driver (memory_format.step_memory); this is a model-output check, not a substitute
for independent rollout scoring. An invalid memory gives next_memory_text=None and the
caller must stop with invalid_model_memory: nothing is repaired from simulator truth.
Checkpoint loading strictly retains all float32 masters; computation uses the
saved bfloat16 transformer configuration without a reduced-precision weight copy.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from memory_format import LATENT_DIM, step_memory
from model_data import (FullCorpus, JointTokenizer, QuantileNormalizer, make_observation,
                        read_json, save_json, sha256_file)


def bootstrap_openpi(metadata):
    root = Path(metadata["openpi_root"])
    if not (root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"Saved OpenPI checkout is unavailable: {root}")
    sys.path.insert(0, str(root / "src"))
    client = root / "packages" / "openpi-client" / "src"
    if client.is_dir():
        sys.path.insert(0, str(client))


class JointMemoryPolicy:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        saved = read_json(self.checkpoint_dir / "metadata.json")
        bootstrap_openpi(saved)
        import jax
        from joint_model import load_full_checkpoint
        from openpi.shared import nnx_utils

        # The common strict loader enforces full model variants and float32
        # masters; policy must not recast or substitute any saved parameters.
        self.model, self.metadata = load_full_checkpoint(self.checkpoint_dir)
        if sha256_file(self.checkpoint_dir / "norm_stats.json") != self.metadata["dataset_fingerprint"]["norm_stats_sha256"]:
            raise ValueError("Checkpoint normalization asset checksum mismatch")
        if sha256_file(self.checkpoint_dir / "tokenizer.model") != self.metadata["tokenizer"]["sha256"]:
            raise ValueError("Checkpoint tokenizer asset checksum mismatch")
        self.normalizer = QuantileNormalizer(self.checkpoint_dir / "norm_stats.json")
        self.tokenizer = JointTokenizer(self.checkpoint_dir / "tokenizer.model", self.metadata["capacities"])
        if self.tokenizer.metadata() != self.metadata["tokenizer"]:
            raise ValueError("Checkpoint tokenizer/format metadata mismatch")
        self.variant = self.metadata["model_config"]["memory_variant"]
        self.max_entities = int(self.model.max_entities)
        self.lat_token_id = int(self.model.lat_token_id)
        if self.lat_token_id != int(self.tokenizer.lat_id):
            raise ValueError("Model and tokenizer disagree on the latent slot token")
        self._prefix = nnx_utils.module_jit(self.model.prefix)
        self._sample = nnx_utils.module_jit(self.model.sample_from_prefix)
        self._text_logits = nnx_utils.module_jit(self.model.next_text_logits)
        self._rng = jax.random.PRNGKey(self.metadata["training"]["seed"])

    def empty_latents(self):
        return np.zeros((self.max_entities, LATENT_DIM), np.float32)

    def previous_latent_batch(self, previous_latents):
        """Group A has no latent input. Otherwise require float32[K,64], zeros if absent."""
        if self.variant == "A":
            return None
        if previous_latents is None:
            return self.empty_latents()[None]
        value = np.asarray(previous_latents, np.float32)
        if value.shape != (self.max_entities, LATENT_DIM) or not np.isfinite(value).all():
            raise ValueError(f"previous_latents must be finite float32[{self.max_entities},{LATENT_DIM}]")
        return value[None]

    def infer(self, main_rgb, wrist_rgb, raw_rgb, state, instruction, previous_memory_text, previous_latents=None):
        import jax
        import jax.numpy as jnp

        observation = make_observation(main_rgb, wrist_rgb, raw_rgb, state, instruction, previous_memory_text,
                                       self.tokenizer, self.normalizer)
        prefix, valid = self._prefix(observation, self.previous_latent_batch(previous_latents))
        self._rng, action_rng = jax.random.split(self._rng)
        native = np.asarray(self._sample(action_rng, observation, prefix, valid), np.float32)[0]
        if native.shape != (20, 32) or not np.isfinite(native).all():
            raise FloatingPointError("Nonfinite or incorrectly shaped flow output")
        actions = self.normalizer.inverse_actions(native[:, :7])
        capacity = self.metadata["capacities"]["target"]
        generated = jnp.zeros((1, capacity), jnp.int32)
        latents = self.empty_latents()
        latent_count = 0
        tokens = []
        eos = False
        for index in range(capacity):
            generated_latents = None if self.variant == "A" else jnp.asarray(latents)[None]
            logits, z = self._text_logits(prefix, valid, generated, jnp.asarray(index, jnp.int32), generated_latents)
            logits = np.asarray(logits)[0]
            if not np.isfinite(logits).all():
                raise FloatingPointError("Nonfinite Gemma text logits")
            # The model vocabulary includes image/control slots; use only real
            # SentencePiece text pieces. BOS/padding are not generated content.
            scores = logits[:self.tokenizer.vocab_size].copy()
            for special in (0, self.tokenizer.sp.bos_id()):
                if special != self.tokenizer.sp.eos_id():
                    scores[special] = -np.inf
            token = int(np.argmax(scores))
            if token == self.tokenizer.sp.eos_id():
                eos = True
                break
            candidate = tokens + [token]
            decoded = self.tokenizer.sp.decode(candidate)
            # Bound the decoded/re-encoded recurrence, not just raw output IDs.
            # Stop without inventing a valid completion if normalization expands
            # beyond the reserved previous-memory token budget.
            if len(self.tokenizer.memory_ids(decoded)) > self.metadata["capacities"]["previous_memory"]:
                break
            tokens = candidate
            generated = generated.at[0, index].set(token)
            if token == self.lat_token_id and z is not None:
                # Surplus slots share the last index; the driver rejects that grammar anyway.
                latents[min(latent_count, self.max_entities - 1)] = np.asarray(z, np.float32)[0]
                latent_count += 1
        text = self.tokenizer.sp.decode(tokens)
        expected_total = self.metadata["instruction_totals"].get(instruction)
        parsed = step_memory(previous_memory_text, text, self.variant, eos_terminated=eos,
                             expected_total=expected_total)
        if parsed.get("memory_valid") is not True:  # lenient evaluation: accept a checklist of another length
            relaxed = step_memory(previous_memory_text, text, self.variant, eos_terminated=True, expected_total=None)
            if relaxed.get("memory_valid") is True:
                parsed = relaxed
                parsed["lenient"] = "accepted_other_length"
        if expected_total is None:
            parsed["sequence_complete"] = False
        return {"actions": actions, "memory_text": text, "latents": latents, **parsed}


def inference_probe(model, batch, seed=531):
    """Fixed-noise actions and genuine first-token logits for strict reload."""
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils

    prefix_fn = nnx_utils.module_jit(model.prefix)
    action_fn = nnx_utils.module_jit(model.sample_from_prefix)
    logits_fn = nnx_utils.module_jit(model.next_text_logits)
    observation = batch["observation"]
    prefix, valid = prefix_fn(observation, batch.get("prev_latents"))
    actions = np.asarray(action_fn(jax.random.PRNGKey(seed), observation, prefix, valid))
    logits, _ = logits_fn(prefix, valid, jnp.zeros((1, model.target_capacity), jnp.int32), jnp.asarray(0))
    logits = np.asarray(logits)
    if not np.isfinite(actions).all() or not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite strict reload probe")
    return {"actions": actions, "first_token_logits": logits}


def infer_dataset_row(policy, corpus, index):
    row = corpus.rows[index]
    state = corpus.episodes[row["episode_npz"]][0][row["observation_index"]]
    with Image.open(corpus.path(row["image_path"])) as image:
        main = np.asarray(image.convert("RGB"))
    with Image.open(corpus.path(row["wrist_image_path"])) as image:
        wrist = np.asarray(image.convert("RGB"))
    with Image.open(corpus.path(row["raw_image_path"])) as image:
        raw = np.asarray(image.convert("RGB"))
    # This is an explicitly labelled offline supervised-history diagnostic, not
    # a rollout. Simulation passes its own generated previous memory to infer.
    result = policy.infer(main, wrist, raw, state, row["instruction"], row["previous_memory_text"],
                          corpus.previous_latents(row))
    return {**result, "actions": result["actions"].tolist(), "latents": result["latents"].tolist(),
            "sample_id": row["sample_id"],
            "scope": "offline checkpoint inference; supervised preceding-query history, not closed-loop"}


def _serve(path):
    # Preserve stdout as a strict JSON-lines transport, including during imports
    # and weight loading. Libraries and diagnostics are redirected to stderr.
    transport = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        policy = JointMemoryPolicy(path)
    required = {"main_png_b64", "wrist_png_b64", "raw_png_b64", "state", "instruction", "previous_memory_text"}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not required <= set(request) <= required | {"previous_latents"}:
                raise ValueError(f"Policy request requires exactly {sorted(required)} and optional previous_latents; oracle fields are forbidden")
            images = []
            for key in ("main_png_b64", "wrist_png_b64", "raw_png_b64"):
                with Image.open(io.BytesIO(base64.b64decode(request[key], validate=True))) as image:
                    images.append(np.asarray(image.convert("RGB")))
            with contextlib.redirect_stdout(sys.stderr):
                result = policy.infer(*images, np.asarray(request["state"], np.float32),
                                      request["instruction"], request["previous_memory_text"],
                                      request.get("previous_latents"))
            result["actions"] = result["actions"].tolist()
            result["latents"] = result["latents"].tolist()
        except Exception as error:
            print(f"Policy request failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            result = {"error": f"{type(error).__name__}: {error}"}
        transport.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        transport.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve-json", type=Path)
    mode.add_argument("--diagnose", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.serve_json is not None:
        _serve(args.serve_json)
        return
    if args.dataset is None:
        parser.error("--diagnose requires --dataset")
    policy = JointMemoryPolicy(args.diagnose)
    from joint_model import real_weight_diagnostics

    corpus = FullCorpus(args.dataset, policy.metadata["contract"], policy.tokenizer)
    if corpus.fingerprint != policy.metadata["dataset_fingerprint"]:
        raise ValueError("Diagnostic corpus differs from the saved training corpus")
    index = next(i for i, row in enumerate(corpus.rows) if row["valid_action_steps"] < 20)
    diagnostic = real_weight_diagnostics(policy.model, corpus.batch(index))
    diagnostic["strict_checkpoint_reload"] = True
    diagnostic["inference"] = infer_dataset_row(policy, corpus, index)
    destination = args.report or (args.diagnose / "real_weight_diagnostics.json")
    save_json(destination, diagnostic)
    print(json.dumps({"report": str(destination), "status": "passed"}), file=sys.stderr)


if __name__ == "__main__":
    main()
