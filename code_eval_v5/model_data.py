"""Full-corpus inputs shared by joint training and deployment.

The only prompt normalization is OpenPI's native instruction cleanup. State is
quantile normalized in eight physical dimensions, digitized with the native
257-edge rule, and only then padded for the model. Dense queries cover every
frame; previous memory is target[t-20], or initial memory for the first 20.
Memory has independent token capacity; no image augmentation or text truncation
is performed. Rows are schema-checked without retaining large annotation evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ledger_v2 import Ledger, apply_delta, parse_delta
from memory_format import (INITIAL_MEMORY, LAT, LATENT_DIM, MAX_ENTITIES, VERBS, render_memory_text,
                           render_previous, verb_targets)

PROMPT_VERSION = "native-state8-ledger-previous-memory-abc-v2-rawfront3"
# v2: three genuine images. The key order is the openpi IMAGE_KEYS order and also the sorted order
# that jax pytree flattening gives a dict, so the prefix is always [overlay front, wrist, clean front].
IMAGE_SLOTS = {"base_0_rgb": "front image with trajectory overlay (image_path)",
               "left_wrist_0_rgb": "wrist image (wrist_image_path)",
               "right_wrist_0_rgb": "clean front image without overlay (raw_image_path)"}
TRACKER_IMAGES = Path(os.environ["TRACKER_IMAGES"]) if os.environ.get("TRACKER_IMAGES") else None
DENSE_QUERY_COUNTS = {6: 56379, 8: 43039}
MODEL_ROW_FIELDS = (
    "sample_id", "task_id", "episode_id", "observation_index", "history_start", "history_end",
    "instruction", "previous_memory_text", "memory_target_text", "image_path", "wrist_image_path",
    "raw_image_path", "episode_npz", "action_start", "valid_action_steps",
)
MEMORY_TARGET_FIELDS = (
    "action_text", "subgoal_delta", "task_text", "confirmed_goals", "total_goals", "action_reliable",
)


def compact_index_row(row, contract):
    """Check source schema without retaining per-frame tracks or goal evidence."""
    absent = (set(contract["dataset"]["row_fields"]) | set(MODEL_ROW_FIELDS) | {"memory_target"}) - row.keys()
    if absent:
        raise ValueError(f"Missing row fields: {sorted(absent)}")
    target = row["memory_target"]
    absent = (set(contract["dataset"]["memory_target_fields"]) | set(MEMORY_TARGET_FIELDS)) - target.keys()
    if absent:
        raise ValueError(f"Missing memory target fields: {sorted(absent)}")
    rendered = (f"Action:{target['action_text']}\nSubgoal:{target['subgoal_delta']}\nTask:{target['task_text']}")
    if rendered != row["memory_target_text"]:
        raise ValueError(f"Memory object/text mismatch: {row['sample_id']}")
    if target["action_reliable"] is not row.get("action_label_reliable"):
        raise ValueError(f"Reliability flag mismatch: {row['sample_id']}")
    if "previous_query_index" in row:
        t = row["observation_index"]
        expected = t - 20 if t >= 20 else None
        if row["previous_query_index"] != expected:
            raise ValueError(f"Incorrect previous_query_index: {row['sample_id']}")
    # v2: the clean front image is a mandatory third input; it must be its own file, never the overlay.
    if (not isinstance(row["raw_image_path"], str) or not row["raw_image_path"]
            or row["raw_image_path"] in (row["image_path"], row["wrist_image_path"])):
        raise ValueError(f"raw_image_path must name a separate clean front image: {row['sample_id']}")
    result = {field: row[field] for field in MODEL_ROW_FIELDS}
    result["memory_target"] = {field: target[field] for field in MEMORY_TARGET_FIELDS}
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def round32(value):
    return ((int(value) + 31) // 32) * 32


class QuantileNormalizer:
    """Quantiles with observed-range fallback for collapsed training intervals."""
    def __init__(self, path):
        self.raw = read_json(path)
        if set(self.raw) != {"state", "actions"}:
            raise ValueError("norm_stats.json must have exactly state/actions root keys")
        self.stats = {}
        required = ("mean", "std", "q01", "q99", "min", "max")
        for key, dimension in (("state", 8), ("actions", 7)):
            entry = self.raw[key]
            missing = set(required).difference(entry)
            if missing:
                raise ValueError(f"Missing {key} normalization statistics {sorted(missing)}; refit the full training corpus before training or deployment")
            arrays = {name: np.asarray(entry[name], np.float32) for name in required}
            if any(x.shape != (dimension,) or not np.isfinite(x).all() for x in arrays.values()):
                raise ValueError(f"Invalid {key} normalization statistics")
            if (np.any(arrays["min"] > arrays["q01"]) or np.any(arrays["q01"] > arrays["q99"])
                    or np.any(arrays["q99"] > arrays["max"])):
                raise ValueError(f"Invalid {key} normalization bounds: expected min <= q01 <= q99 <= max")
            collapsed = arrays["q01"] == arrays["q99"]
            arrays["lower"] = np.where(collapsed, arrays["min"], arrays["q01"])
            arrays["upper"] = np.where(collapsed, arrays["max"], arrays["q99"])
            arrays["denominator"] = arrays["upper"] - arrays["lower"] + 1e-6
            self.stats[key] = arrays

    def normalize(self, key, value):
        stats = self.stats[key]
        value = np.asarray(value, np.float32)
        if value.shape[-1] != stats["q01"].size or not np.isfinite(value).all():
            raise ValueError(f"Invalid physical {key} input: {value.shape}")
        return (value - stats["lower"]) / stats["denominator"] * 2.0 - 1.0

    def inverse_actions(self, value):
        value = np.asarray(value, np.float32)
        if value.shape != (20, 7) or not np.isfinite(value).all():
            raise ValueError("Expected finite normalized actions[20,7]")
        stats = self.stats["actions"]
        return (value + 1.0) / 2.0 * stats["denominator"] + stats["lower"]


class JointTokenizer:
    def __init__(self, path, capacities=None):
        import sentencepiece as spm

        self.path = Path(path)
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.capacities = capacities
        self.suffix_ids = self.sp.encode("\nAction: ")
        self.vocab_size = self.sp.get_piece_size()
        self.lat_id = self.sp.piece_to_id(LAT)
        self.newline_id = self.sp.piece_to_id("\n")
        if self.sp.encode("a" + LAT + "b").count(self.lat_id) != 1 or self.sp.encode("a\nb").count(self.newline_id) != 1:
            raise ValueError("Tokenizer must encode the latent slot and newline as single pieces")
        if not (0 <= self.sp.bos_id() < self.vocab_size and 0 <= self.sp.eos_id() < self.vocab_size):
            raise ValueError("Tokenizer needs BOS and EOS")

    def base_ids(self, instruction, normalized_state):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Original instruction is required")
        state = np.asarray(normalized_state, np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Native discrete input requires actual finite state8")
        cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
        discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\nPrevious memory:\n"
        return self.sp.encode(text, add_bos=True)

    def memory_ids(self, text):
        if not isinstance(text, str):
            raise ValueError("Memory must be text, not an oracle object")
        return self.sp.encode(text)

    def target_ids(self, text):
        if not isinstance(text, str) or not text.isascii() or "\r" in text:
            raise ValueError("Target memory must be ASCII text with LF section separators")
        lines = text.split("\n")
        if len(lines) != 3 or any(not line.startswith(label) or not line[len(label):].strip()
                                  or line[len(label):len(label) + 1].isspace()
                                  for line, label in zip(lines, ("Action:", "Subgoal:", "Task:"), strict=True)):
            raise ValueError("Target must contain exactly the three nonempty memory sections without whitespace after colons")
        return self.memory_ids(text) + [self.sp.eos_id()]

    @staticmethod
    def pad(ids, capacity):
        if len(ids) > capacity:
            raise ValueError(f"Token overflow ({len(ids)} > {capacity}); truncation is forbidden")
        result = np.zeros(capacity, np.int32)
        valid = np.arange(capacity) < len(ids)
        result[:len(ids)] = ids
        return result, valid

    def condition(self, instruction, previous_memory_text, normalized_state):
        memory = self.memory_ids(previous_memory_text)
        if len(memory) > self.capacities["previous_memory"]:
            raise ValueError("Previous memory exceeds the saved generation/conditioning bound")
        ids = self.base_ids(instruction, normalized_state) + memory + self.suffix_ids
        return self.pad(ids, self.capacities["condition"])

    def target(self, text):
        return self.pad(self.target_ids(text), self.capacities["target"])

    def action_line_length(self, ids):
        """Tokens of the Action line, excluding its newline."""
        ids = list(ids)
        if self.newline_id not in ids:
            raise ValueError("Target has no section newline")
        return ids.index(self.newline_id)

    def metadata(self):
        return {"sha256": sha256_file(self.path), "vocabulary_size": self.vocab_size,
                "bos_id": self.sp.bos_id(), "eos_id": self.sp.eos_id(), "pad_id": 0,
                "prompt_version": PROMPT_VERSION, "capacities": self.capacities,
                "state_dimension_before_tokenization": 8,
                "state_bins": "np.digitize(state,np.linspace(-1,1,257)[:-1])-1; no clipping",
                "format": "encode(Task: native-cleaned instruction, State: bins;\\nPrevious memory:\\n,BOS) + encode(previous_memory) + encode(\\nAction: )",
                "normalization": "training q01/q99; collapsed intervals use training min/max; epsilon=1e-6; no clipping",
                "image_transform": "RGB uint8[224,224,3] -> float32 /255*2-1; no augmentation",
                "image_slots": IMAGE_SLOTS}


def image_array(image):
    value = np.asarray(image)
    if value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise ValueError(f"Images must already be genuine RGB uint8[224,224,3], got {value.shape}/{value.dtype}")
    return value.astype(np.float32) / 255.0 * 2.0 - 1.0


def make_observation(main_rgb, wrist_rgb, raw_rgb, state, instruction, previous_memory_text, tokenizer, normalizer):
    from openpi.models.model import Observation

    normalized = normalizer.normalize("state", state)
    token_ids, token_mask = tokenizer.condition(instruction, previous_memory_text, normalized)
    # v2: the third openpi slot carries the clean (no overlay) front image, unmasked. There is no
    # fallback: a missing clean image is an error, never a copy of the overlay image.
    if raw_rgb is None:
        raise ValueError("The clean front image (right_wrist_0_rgb slot) is required")
    images = {"base_0_rgb": image_array(main_rgb)[None], "left_wrist_0_rgb": image_array(wrist_rgb)[None],
              "right_wrist_0_rgb": image_array(raw_rgb)[None]}
    if tuple(images) != tuple(IMAGE_SLOTS):
        raise AssertionError("Image slot order changed")
    return Observation(
        images=images, image_masks={name: np.ones(1, bool) for name in images},
        state=np.pad(normalized, (0, 24))[None],
        tokenized_prompt=token_ids[None], tokenized_prompt_mask=token_mask[None],
    )


class FullCorpus:
    def __init__(self, root, contract, tokenizer):
        self.root = Path(root).resolve()
        self.index_path = self.path(contract["dataset"]["index"])
        self.variant = contract["model"]["memory_variant"]
        self.latents_dir = contract["dataset"].get("latents_dir")
        if (self.variant != "A") != bool(self.latents_dir):
            raise ValueError("Groups B and C require dataset.latents_dir; group A must not have one")
        self.latents = {}
        self.rows = []
        index_digest = hashlib.sha256()
        with self.index_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Blank index row {line_number}")
                index_digest.update(line)
                self.rows.append(compact_index_row(json.loads(line), contract))
        index_sha256 = index_digest.hexdigest()
        if not self.rows:
            raise ValueError("Empty full-corpus index")
        self.normalizer = QuantileNormalizer(self.root / "norm_stats.json")
        self.tokenizer = tokenizer
        self.episodes = {}
        groups = defaultdict(list)
        sample_ids = set()
        checked_files = set()
        for i, row in enumerate(self.rows):
            if row["sample_id"] in sample_ids:
                raise ValueError(f"Duplicate sample_id {row['sample_id']}")
            sample_ids.add(row["sample_id"])
            groups[(int(row["task_id"]), str(row["episode_id"]))].append((i, row))
            for field in ("episode_npz", "image_path", "wrist_image_path", "raw_image_path"):
                if row[field] not in checked_files:
                    path = self.path(row[field])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    checked_files.add(row[field])
        counts = Counter()
        frame_counts = Counter()
        query_counts = Counter()
        npz_owners = {}
        for key, members in groups.items():
            names = {row["episode_npz"] for _, row in members}
            if len(names) != 1:
                raise ValueError(f"Multiple NPZs for {key}")
            relative = names.pop()
            if relative in npz_owners:
                raise ValueError(f"Different episodes share an NPZ: {key} and {npz_owners[relative]}")
            npz_owners[relative] = key
            with np.load(self.path(relative), allow_pickle=False) as archive:
                states = np.asarray(archive["states"], np.float32)
                actions = np.asarray(archive["actions"], np.float32)
            n = len(actions)
            if actions.shape != (n, 7) or states.shape != (n, 8) or n == 0:
                raise ValueError(f"Bad dense arrays in {relative}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Nonfinite physical data in {relative}")
            self.episodes[relative] = (states, actions)
            ordered = sorted(members, key=lambda pair: pair[1]["observation_index"])
            if [row["observation_index"] for _, row in ordered] != list(range(n)):
                raise ValueError(f"Missing/duplicate dense observation indices in {key}")
            instruction = ordered[0][1]["instruction"]
            ledgers = []
            for _, row in ordered:
                t = row["observation_index"]
                # The stored previous memory must equal the Ledger Driver's own result
                # at t-20 (A format), so training input and deployment cannot diverge.
                ledger_before = Ledger() if t < 20 else ledgers[t - 20]
                previous = (INITIAL_MEMORY if t < 20
                            else render_previous(ordered[t - 20][1]["memory_target"]["action_text"], ledger_before))
                if (row["action_start"] != t or row["valid_action_steps"] != min(20, n-t)
                        or row["history_start"] != max(0, t-20) or row["history_end"] != t):
                    raise ValueError(f"Incorrect history/tail boundary: {row['sample_id']}")
                if row["instruction"] != instruction or row["previous_memory_text"] != previous:
                    raise ValueError(f"Instruction change or incorrect target[t-20] memory: {row['sample_id']}")
                lines = row["memory_target_text"].split("\n")
                ledgers.append(apply_delta(ledger_before, parse_delta("\n".join(lines[1:])), t))
                if sum(item.done for item in ledgers[-1].items) != row["memory_target"]["confirmed_goals"]:
                    raise ValueError(f"Ledger count mismatch: {row['sample_id']}")
                row["previous_memory_text"] = render_memory_text(previous, self.variant)
                row["memory_target_text"] = render_memory_text(row["memory_target_text"], self.variant)
            if self.variant != "A":
                stem = Path(relative).stem
                with np.load(Path(self.latents_dir) / f"{stem}.npz", allow_pickle=False) as archive:
                    z, z_valid = np.asarray(archive["z"], np.float32), np.asarray(archive["valid"], bool)
                if z.shape[0] != n or z.shape[2] != LATENT_DIM or z.shape[1] > MAX_ENTITIES:
                    raise ValueError(f"Bad latent table for {stem}: {z.shape}")
                self.latents[relative] = (z, z_valid)
            counts[key[0]] += 1
            frame_counts[key[0]] += n
            query_counts[key[0]] += len(ordered)
        manifest_path = self.root / "split_manifest.json"
        self.split_manifest = read_json(manifest_path) if manifest_path.is_file() else None
        self.split_name = None
        if self.split_manifest is not None:
            manifest = self.split_manifest
            if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
                raise ValueError("Dataset split manifest is incomplete or unsupported")
            matches = [(name, split) for name, split in manifest["splits"].items()
                       if self.path(split["index"]) == self.index_path]
            if len(matches) != 1:
                raise ValueError("Dataset index must identify exactly one manifest split")
            self.split_name, expected = matches[0]
            if index_sha256 != expected["index_sha256"]:
                raise ValueError("Dataset index differs from its split manifest")
            if sha256_file(self.root / "norm_stats.json") != manifest["normalization"]["sha256"]:
                raise ValueError("Normalization differs from the training-only split statistics")
            if manifest["normalization"]["fit_split"] != "train":
                raise ValueError("Normalization must be fitted on training episodes only")
            if manifest["integrity"].get("episode_disjoint") is not True:
                raise ValueError("Training and validation must be episode-disjoint")
            if set(manifest["splits"]) != {"train", "val"}:
                raise ValueError("Split manifest must contain exactly train and val")
            all_memberships = [(int(record["task_id"]), str(record["episode_id"]))
                               for record in manifest["episodes"]]
            if len(all_memberships) != len(set(all_memberships)):
                raise ValueError("An episode appears more than once in the split manifest")
            records = {(int(record["task_id"]), str(record["episode_id"])): record
                       for record in manifest["episodes"]}
            task_contracts = {int(task["task_id"]): task for task in contract["source"]["tasks"]}
            if {key[0] for key in records} != set(task_contracts):
                raise ValueError("Full manifest task membership differs from the released corpus")
            if len({record["episode_npz"] for record in records.values()}) != len(records):
                raise ValueError("Different manifest episodes share an NPZ")
            for key, record in records.items():
                if (record["split"] not in manifest["splits"] or record["actions"] <= 0
                        or record["queries"] != record["actions"]):
                    raise ValueError(f"Invalid dense manifest episode: {key}")
            for tid, task in task_contracts.items():
                selected = [record for key, record in records.items() if key[0] == tid]
                if (len(selected) != task["metadata_episode_count"]
                        or sum(record["actions"] for record in selected) != task["metadata_frame_count"]
                        or sum(record["queries"] for record in selected) != task["metadata_frame_count"]):
                    raise ValueError(f"Full manifest omits released episodes or dense frames for task {tid}")
            for name, split in manifest["splits"].items():
                selected = [record for record in records.values() if record["split"] == name]
                if {int(tid) for tid in split["tasks"]} != set(task_contracts):
                    raise ValueError(f"Manifest {name} task membership differs from the contract")
                if (len(selected), sum(record["actions"] for record in selected),
                        sum(record["queries"] for record in selected)) != (
                        split["episodes"], split["actions"], split["queries"]):
                    raise ValueError(f"Manifest {name} totals differ from its episode records")
                for tid in task_contracts:
                    members = [record for record in selected if int(record["task_id"]) == tid]
                    summary = split["tasks"][str(tid)]
                    episode_ids = [str(record["episode_id"]) for record in members]
                    if (not members or sorted(episode_ids) != sorted(summary["episode_ids"])
                            or (len(members), sum(record["actions"] for record in members),
                                sum(record["queries"] for record in members)) != (
                                summary["episodes"], summary["actions"], summary["queries"])):
                        raise ValueError(f"Manifest {name} task {tid} membership/counts are inconsistent")
            source = manifest["source"]
            if (source["episodes"], source["actions"], source["queries"]) != (
                    len(records), sum(record["actions"] for record in records.values()),
                    sum(record["queries"] for record in records.values())):
                raise ValueError("Full manifest source totals differ from its episode records")
            training_membership = {key for key, record in records.items() if record["split"] == "train"}
            norm_membership = [(int(record["task_id"]), str(record["episode_id"]))
                               for record in manifest["normalization"]["episodes"]]
            if (set(norm_membership) != training_membership or len(norm_membership) != len(training_membership)
                    or manifest["normalization"]["dense_rows"] != manifest["splits"]["train"]["actions"]):
                raise ValueError("Normalization membership/counts must contain training episodes exactly once")
            expected_memberships = {(int(record["task_id"]), str(record["episode_id"]))
                                    for record in manifest["episodes"] if record["split"] == self.split_name}
            if set(groups) != expected_memberships:
                raise ValueError("Dataset episode membership differs from its manifest split")
            for key, members in groups.items():
                record = records[key]
                relative = members[0][1]["episode_npz"]
                if (record["episode_npz"] != relative or record["queries"] != len(members)
                        or record["actions"] != len(self.episodes[relative][1])):
                    raise ValueError(f"Dataset episode arrays/counts differ from the manifest: {key}")
            expected_tasks = {int(tid) for tid in expected["tasks"]}
            if expected_tasks != {int(task["task_id"]) for task in contract["source"]["tasks"]}:
                raise ValueError("Manifest task membership differs from the contract")
            if set(counts) != expected_tasks:
                raise ValueError("Dataset task membership differs from its manifest split")
            for tid in sorted(expected_tasks):
                task = expected["tasks"][str(tid)]
                actual = (counts[tid], frame_counts[tid], query_counts[tid])
                if actual != (task["episodes"], task["actions"], task["queries"]):
                    raise ValueError(f"Incomplete {self.split_name} task {tid}: {actual}")
                if {episode for task_id, episode in groups if task_id == tid} != set(task["episode_ids"]):
                    raise ValueError(f"Incorrect episode membership for {self.split_name} task {tid}")
            if (len(groups), sum(frame_counts.values()), len(self.rows)) != (
                    expected["episodes"], expected["actions"], expected["queries"]):
                raise ValueError("Dataset totals differ from its manifest split")
        else:
            expected_tasks = {int(task["task_id"]) for task in contract["source"]["tasks"]}
            if set(counts) != expected_tasks:
                raise ValueError("Dataset task membership differs from the full released corpus")
            for task in contract["source"]["tasks"]:
                tid = int(task["task_id"])
                if counts[tid] != task["metadata_episode_count"] or frame_counts[tid] != task["metadata_frame_count"]:
                    raise ValueError(f"Incomplete corpus task {tid}: {counts[tid]} episodes/{frame_counts[tid]} frames")
            if dict(query_counts) != DENSE_QUERY_COUNTS:
                raise ValueError(f"Expected 99418 dense full-corpus queries, got {dict(query_counts)}")
        self.coverage = {"samples": len(self.rows), "episodes": len(groups),
                         "episodes_by_task": dict(counts), "frames_by_task": dict(frame_counts),
                         "queries_by_task": dict(query_counts), "all_source_actions_covered": True,
                         "query_stride": 1, "memory_lag": 20, "action_horizon": 20}
        self.fingerprint = {"index_sha256": index_sha256,
                            "norm_stats_sha256": sha256_file(self.root / "norm_stats.json")}
        failure_index = contract["dataset"].get("failure_index")
        if failure_index and self.split_name == "train":
            self._add_failures(failure_index, contract)
        if self.split_manifest is not None:
            self.coverage["split"] = self.split_name
            self.coverage["all_source_actions_covered"] = False
            self.coverage["all_split_actions_covered"] = True
            self.fingerprint["split_manifest_sha256"] = sha256_file(manifest_path)

    def _add_failures(self, failure_index, contract):
        """Append injected-failure rows (training episodes only). They supervise memory, never actions."""
        path = self.path(failure_index)
        rows = []
        with path.open("rb") as stream:
            for line in stream:
                row = json.loads(line)
                compact = compact_index_row(row, contract)
                if compact["valid_action_steps"] != 0:
                    raise ValueError(f"Failure row must not supervise actions: {compact['sample_id']}")
                compact["previous_memory_text"] = render_memory_text(compact["previous_memory_text"], self.variant)
                compact["memory_target_text"] = render_memory_text(compact["memory_target_text"], self.variant)
                rows.append(compact)
        known = {row["sample_id"] for row in self.rows}
        for relative in sorted({row["episode_npz"] for row in rows}):
            with np.load(self.path(relative), allow_pickle=False) as archive:
                self.episodes[relative] = (np.asarray(archive["states"], np.float32), np.asarray(archive["actions"], np.float32))
            if self.variant != "A":
                with np.load(Path(self.latents_dir) / f"{Path(relative).stem}.npz", allow_pickle=False) as archive:
                    self.latents[relative] = (np.asarray(archive["z"], np.float32), np.asarray(archive["valid"], bool))
        for row in rows:
            if row["sample_id"] in known:
                raise ValueError(f"Duplicate failure sample {row['sample_id']}")
            for field in ("image_path", "wrist_image_path", "raw_image_path"):
                if not self.path(row[field]).is_file():
                    raise FileNotFoundError(row[field])
        self.rows.extend(rows)
        self.coverage["failure_rows"] = len(rows)
        self.coverage["failure_episodes"] = len({row["episode_npz"] for row in rows})
        self.fingerprint["failure_index_sha256"] = sha256_file(path)

    def path(self, relative):
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError(f"Dataset path escapes root: {relative}")
        return result

    def measure_capacities(self):
        maximum = {"condition_observed": 0, "fixed_condition": 0, "previous_observed": 0, "target_observed": 0}
        for row in self.rows:
            states, _ = self.episodes[row["episode_npz"]]
            state = self.normalizer.normalize("state", states[row["observation_index"]])
            base = len(self.tokenizer.base_ids(row["instruction"], state)) + len(self.tokenizer.suffix_ids)
            previous = len(self.tokenizer.memory_ids(row["previous_memory_text"]))
            target = len(self.tokenizer.target_ids(row["memory_target_text"]))
            maximum["condition_observed"] = max(maximum["condition_observed"], base + previous)
            maximum["fixed_condition"] = max(maximum["fixed_condition"], base)
            maximum["previous_observed"] = max(maximum["previous_observed"], previous)
            maximum["target_observed"] = max(maximum["target_observed"], target)
        target_capacity = round32(maximum["target_observed"])
        previous_capacity = round32(max(target_capacity, maximum["previous_observed"]))
        # Reserve the entire generated-memory bound, not only ground-truth
        # previous lengths. Extra 32 tokens allow unseen native state-bin widths.
        capacities = {"condition": round32(maximum["fixed_condition"] + previous_capacity + 32),
                      "target": target_capacity, "previous_memory": previous_capacity}
        return capacities, maximum

    def batch(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("Corpus row index must be an integer")
        if not 0 <= int(index) < len(self.rows):
            raise IndexError(f"Corpus row index outside dataset: {index}")
        row = self.rows[int(index)]
        states, source_actions = self.episodes[row["episode_npz"]]
        main_path = self.path(row["image_path"])
        if TRACKER_IMAGES and row["memory_target"]["action_reliable"]:
            # Tracker-drawn overlay; rows whose replay left the recording keep the stored image.
            candidate = TRACKER_IMAGES / row["image_path"]
            if candidate.is_file():
                main_path = candidate
        with Image.open(main_path) as image:
            main = np.asarray(image.convert("RGB"))
        with Image.open(self.path(row["wrist_image_path"])) as image:
            wrist = np.asarray(image.convert("RGB"))
        with Image.open(self.path(row["raw_image_path"])) as image:
            raw = np.asarray(image.convert("RGB"))
        observation = make_observation(main, wrist, raw, states[row["observation_index"]], row["instruction"],
                                       row["previous_memory_text"], self.tokenizer, self.normalizer)
        count, start = int(row["valid_action_steps"]), int(row["action_start"])
        actions = np.zeros((1, 20, 32), np.float32)
        actions[0, :count, :7] = self.normalizer.normalize("actions", source_actions[start:start+count])
        ids, valid = self.tokenizer.target(row["memory_target_text"])
        loss_mask = valid.copy()
        reliable = bool(row["memory_target"]["action_reliable"])
        # Group B writes no verbs, so its Action line cannot be wrong; A and C mask it.
        if not reliable and self.variant != "B":
            loss_mask[:self.tokenizer.action_line_length(ids[valid])] = False
        result = {"observation": observation, "actions": actions,
                  "action_mask": (np.arange(20) < count)[None],
                  "target_ids": ids[None], "target_mask": valid[None], "target_loss_mask": loss_mask[None],
                  "sample_mask": np.ones(1, bool)}
        if self.variant != "A":
            result["prev_latents"] = self.previous_latents(row)[None]
            result["z_star"] = self.z_star(row)[None]
            # z* comes from recorded tracks, so it stays valid on unreliable rows.
            result["align_mask"] = self.z_valid(row)[None]
        if self.variant == "C":
            verbs = np.zeros((MAX_ENTITIES, len(VERBS)), np.float32)
            parsed = verb_targets(row["memory_target"]["action_text"])
            verbs[:len(parsed)] = parsed
            result["verb_targets"] = verbs[None]
            result["verb_mask"] = ((np.arange(MAX_ENTITIES) < len(parsed)) & reliable)[None]
        return result

    def _pad_entities(self, value):
        result = np.zeros((MAX_ENTITIES, *value.shape[1:]), value.dtype)
        result[:len(value)] = value
        return result

    def z_star(self, row):
        """z* of this window, float32[MAX_ENTITIES,64]; zeros for absent or never-visible bowls."""
        return self._pad_entities(self.latents[row["episode_npz"]][0][row["observation_index"]])

    def z_valid(self, row):
        return self._pad_entities(self.latents[row["episode_npz"]][1][row["observation_index"]])

    def previous_latents(self, row):
        """z*(t-20) as teacher previous latents; zeros in the first window and for group A."""
        t = int(row["observation_index"])
        if self.variant == "A" or t < 20:
            return np.zeros((MAX_ENTITIES, LATENT_DIM), np.float32)
        return self._pad_entities(self.latents[row["episode_npz"]][0][t - 20])

    def batch_indices(self, indices):
        """Stack real rows and zero-mask -1 tail slots without duplicating samples.

        This is data assembly, not a microbatch/gradient-accumulation loop. All
        real examples are passed to one model call by the global-batch trainer.
        """
        import jax

        indices = np.asarray(indices)
        if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu":
            raise ValueError("Batch indices must be a nonempty one-dimensional integer sequence")
        if np.any(indices < -1) or np.any(indices >= len(self.rows)):
            raise IndexError("Only dataset row indices and -1 filler slots are allowed")
        real = indices >= 0
        rows = [self.batch(int(index)) if valid else None
                for index, valid in zip(indices, real, strict=True)]
        template = next((row for row in rows if row is not None), None)
        if template is None:
            # An all-padding shard still has a valid static structure and zero
            # loss weight; this seed row is never returned as a real example.
            template = self.batch(0)
        filler = jax.tree.map(np.zeros_like, template)
        result = jax.tree.map(lambda *parts: np.concatenate(parts, axis=0),
                              *(row if row is not None else filler for row in rows))
        if result["sample_mask"].dtype != np.bool_ or not np.array_equal(result["sample_mask"], real):
            raise AssertionError("Batch assembly changed real/filler membership")
        if (np.any(result["action_mask"][~real]) or np.any(result["target_mask"][~real])
                or np.any(result["target_loss_mask"][~real])
                or ("align_mask" in result and np.any(result["align_mask"][~real]))
                or ("verb_mask" in result and np.any(result["verb_mask"][~real]))):
            raise AssertionError("Padded rows must not supervise actions, text or latents")
        return result

    def epoch_order(self, seed, epoch):
        return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)])).permutation(len(self.rows))
