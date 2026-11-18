"""Save RGB and simulator segmentation of expert replays from TRAINING initial states, optionally with an
injected failure, for offline tuning of the RGB tracker. Held-out states are never touched.

Usage: tracker_replay.py OUT_ROOT KIND TASK COUNT      KIND = normal | empty_hand | drop
Writes OUT_ROOT/KIND/tNN_<episode>.npz with rgb, seg, bowl_sids, task_id, fired_frame.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from closedloop_inject import install_injection  # noqa: E402
from closedloop_ledger import install  # noqa: E402

OLD = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
sys.path.insert(0, str(OLD))
import simulate  # noqa: E402

site = os.environ["SIMULATOR_TORCH_SITE"]
sys.path.insert(0, site)
import torch  # noqa: E402,F401
sys.path.remove(site)

out_root, kind, task_id, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
HOME = Path.home()
DATASET = HOME / "pi05-abc/dataset"
INPUTS = HOME / "pi05-ledger-A/inputs"
contract = json.loads((CODE / "contract_A.json").read_text())
spec = next({"task_id": t["task_id"], "name": t["name"], "required_goals": t["required_goals"]}
            for t in contract["source"]["tasks"] if t["task_id"] == task_id)
manifest = json.loads((DATASET / "split_manifest.orig.json").read_text())
train = {f"t{int(e['task_id']):02d}_{e['episode_id']}" for e in manifest["episodes"] if e["split"] == "train"}
index = {e["name"]: e for e in json.loads((INPUTS / "index.json").read_text())["episodes"]}
lost = set(json.loads((HOME / "pi05-ledger-A/labels_A2/report.json").read_text())["summary"]["episodes_with_lost_replay"])
names = sorted((n for n in train - lost if n.startswith(f"t{task_id:02d}")), key=lambda n: int(n.split("_")[-1]))
# Spread over the training set. A different offset per kind, so kinds do not share initial states.
offset = {"normal": 0, "empty_hand": 1, "drop": 2}[kind] + int(os.environ.get("REPLAY_OFFSET", "0"))
step = max(len(names) // count, 1)
chosen = [names[(i * step + offset) % len(names)] for i in range(count)]

source = simulate.configure_official_source(OLD / "vendor/libero-mem", Path(tempfile.mkdtemp()), DATASET)
install(simulate)
log_path = Path(tempfile.mkdtemp()) / "inject.jsonl"
base_environment = simulate.OfficialEnvironment
folder = out_root / kind
folder.mkdir(parents=True, exist_ok=True)
for number, name in enumerate(chosen):
    simulate.OfficialEnvironment = base_environment
    if kind != "normal":
        install_injection(simulate, kind, str(log_path))
        simulate.OfficialEnvironment.created = number
    meta = index[name]
    z = np.load(INPUTS / f"{name}.npz")
    env = simulate.OfficialEnvironment(source, spec, z["initial_state"], meta["instruction"])
    rgb, seg = [], []
    try:
        for frame, action in enumerate(z["actions"].astype(np.float64)):
            main, _, segmentation, _ = env.observe()
            rgb.append(main.copy())
            seg.append(segmentation.copy())
            env.score_observation(frame)  # the official goal counter only advances here, and injection reads it
            env.step(action)
        main, _, segmentation, _ = env.observe()
        rgb.append(main.copy())
        seg.append(segmentation.copy())
        sids = sorted((int(s) for s, row in env.registry.items() if row["category"] == "bowl"))
    finally:
        env.close()
    fired = -1
    if kind != "normal" and log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        hit = [r for r in rows if r["episode_number"] == number and r["event"] == "fired"]
        fired = hit[-1]["frame"] if hit else -1
    np.savez_compressed(folder / f"{name}.npz", rgb=np.asarray(rgb, np.uint8), seg=np.asarray(seg, np.uint8),
                        bowl_sids=np.asarray(sids), task_id=task_id, fired_frame=fired)
    print(kind, name, "frames", len(rgb), "fired", fired, flush=True)
print("REPLAY_DONE", flush=True)
