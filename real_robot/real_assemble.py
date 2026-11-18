"""Step 3: split, normalization, manifest and training contract for the real-robot dataset built by real_build.py.

  real_assemble.py --out DATASET --contract-in contracts/contract_C_v3.json --contract-out DATASET/contract_C_real.json
                   [--val 5] [--steps 20000 --interval 5000 --devices 4 --per-device-batch 32]
                   [--openpi-root P --base-checkpoint P --tokenizer P --python P]
Writes DATASET/train.jsonl, val.jsonl, norm_stats.json, split_manifest.json and the contract.
The last --val episodes are validation; normalization is fitted on the training episodes only.
Run code/rma_latents.py DATASET DATASET/latents afterwards (scheme-C latent targets); the contract already points there.
The numbers in the contract's training block must equal the PI05_* environment variables train.py is started with.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_spec as S  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--contract-in", type=Path, required=True)
p.add_argument("--contract-out", type=Path, required=True)
p.add_argument("--val", type=int, default=5)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--interval", type=int, default=5000)
p.add_argument("--devices", type=int, default=4)
p.add_argument("--per-device-batch", type=int, default=32)
for name in ("openpi-root", "base-checkpoint", "tokenizer", "python"):
    p.add_argument(f"--{name}")
args = p.parse_args()
dataset = args.out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


names = sorted((path.stem for path in (dataset / "rows").glob("*.jsonl")), key=lambda s: int(s.split("seed")[1]))
if len(names) <= args.val:
    raise SystemExit("not enough episodes")
split = {name: ("val" if i >= len(names) - args.val else "train") for i, name in enumerate(names)}
counts = {name: sum(1 for _ in open(dataset / "rows" / f"{name}.jsonl")) for name in names}
for part in ("train", "val"):
    with open(dataset / f"{part}.jsonl.tmp", "w") as out:
        for name in names:
            if split[name] == part:
                out.write(open(dataset / "rows" / f"{name}.jsonl").read())
    os.replace(dataset / f"{part}.jsonl.tmp", dataset / f"{part}.jsonl")

states, actions = [], []
for name in names:
    if split[name] == "train":
        z = np.load(dataset / "episodes" / f"{name}.npz")
        states.append(z["states"]), actions.append(z["actions"])
stats = {}
for key, values in (("state", np.concatenate(states)), ("actions", np.concatenate(actions))):
    stats[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                  "q01": np.quantile(values, 0.01, axis=0).tolist(), "q99": np.quantile(values, 0.99, axis=0).tolist(),
                  "min": values.min(0).tolist(), "max": values.max(0).tolist()}
(dataset / "norm_stats.json").write_text(json.dumps(stats, indent=1))

tid = S.TASK_ID
records = [{"task_id": tid, "episode_id": n, "split": split[n], "actions": counts[n], "queries": counts[n],
            "episode_npz": f"episodes/{n}.npz"} for n in names]
splits = {}
for part in ("train", "val"):
    chosen = [r for r in records if r["split"] == part]
    total = sum(r["actions"] for r in chosen)
    splits[part] = {"index": f"{part}.jsonl", "index_sha256": sha(dataset / f"{part}.jsonl"), "episodes": len(chosen),
                    "actions": total, "queries": total,
                    "tasks": {str(tid): {"episodes": len(chosen), "actions": total, "queries": total,
                                         "episode_ids": [r["episode_id"] for r in chosen]}}}
frames = sum(r["actions"] for r in records)
manifest = {"schema_version": 1, "complete": True, "splits": splits,
            "normalization": {"sha256": sha(dataset / "norm_stats.json"), "fit_split": "train",
                              "episodes": [{"task_id": tid, "episode_id": r["episode_id"]} for r in records if r["split"] == "train"],
                              "dense_rows": splits["train"]["actions"]},
            "integrity": {"episode_disjoint": True}, "episodes": records,
            "source": {"episodes": len(records), "actions": frames, "queries": frames},
            "label_rule": "real_build.py: checklist items cut from the right gripper (close/open x3), causal ticks, "
                          "scheme-B accumulated Action line; see real_robot/README.md",
            "real": {"lerobot_repo": "wwccww/yx_bowls", "top_crop_xy_side": list(S.TOP_CROP), "wrist_crop_xy_side": list(S.WRIST_CROP),
                     "state8": "right_joint_0..5, right_gripper, right_gripper", "action7": "right_joint_0..5, right_gripper (absolute targets)",
                     "objects": list(S.NAMES), "items": list(S.ITEMS), "fps": S.FPS}}
(dataset / "split_manifest.json").write_text(json.dumps(manifest, indent=1))

contract = json.loads(args.contract_in.read_text(encoding="utf-8"))
contract["project"] = "pi05_memory_C_real_yx_bowls"
contract["source"]["repository"] = "wwccww/yx_bowls"
contract["source"]["split_policy"] = f"last {args.val} episodes are validation"
contract["source"]["tasks"] = [{"task_id": tid, "name": "yx_bowls_swap", "filename": "yx_bowls",
                                "metadata_episode_count": len(records), "metadata_frame_count": frames,
                                "required_goals": len(S.ITEMS)}]
contract["dataset"]["latents_dir"] = str((dataset / "latents" / "C").resolve())
contract["dataset"]["episode_arrays"] = {
    "states": "float32[N,8]: right_joint_0..5 (rad), right_gripper, right_gripper",
    "actions": "float32[N,7]: absolute targets right_joint_0..5 (rad), right_gripper; no delta conversion",
    "centres": "float32[N,K,2] bowl centres in the cropped 256-px top image, 0 where not seen",
    "visibility": "bool[N,K]", "action_text": "per-window Action line (U str)"}
contract["memory"]["action_scope"] = ("One clause per bowl (blue = bowl on the right plate at the start, orange = left). "
                                      "Directions are in the cropped top image: left/right = image x, forward/backward = image up/down. "
                                      "picked/placed come from the recorded right gripper.")
contract["memory"]["minimum_visible_pixels_for_motion_label"] = S.MIN_PIXELS
t = contract["training"]
t.update(optimizer_steps=args.steps, checkpoint_interval=args.interval, validation_interval=args.interval,
         num_devices=args.devices, per_device_batch_size=args.per_device_batch,
         global_batch_size=args.devices * args.per_device_batch, warmup_optimizer_steps=min(1000, args.steps // 10))
t["wandb"] = {"enabled": False}
for key in ("coverage", "checkpoints"):
    t.pop(key, None)
for key, value in (("openpi_root", args.openpi_root), ("base_checkpoint", args.base_checkpoint),
                   ("tokenizer", args.tokenizer), ("python", args.python)):
    if value:
        contract["model"][key] = value
args.contract_out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("ASSEMBLE_DONE", {q: (splits[q]["episodes"], splits[q]["actions"]) for q in splits}, "contract", args.contract_out)
print(f"launch with: PI05_DEVICES={args.devices} PI05_PER_DEVICE_BATCH={args.per_device_batch} PI05_TOTAL_UPDATES={args.steps} "
      f"PI05_CHECKPOINT_INTERVAL={args.interval}")
