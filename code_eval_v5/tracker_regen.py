"""Redraw the main training images with the RGB tracker's masks instead of simulator segmentation.

Labels, states, actions and wrist images stay as they are; only the trajectory overlay changes. The raw
HDF5 has no camera frames, so frames come from replaying the same controls as the dataset was built from.
Failure episodes repeat gen_failures.py exactly (same call order), and fired_at is checked against the rows.

Writes OUT/<image_path of each row> and OUT/log/<key>.npz with predicted and true bowl centres, so that the
amount of tracking error seen in training can be measured. Truth is never used to draw anything unless
REGEN_TRUTH=1, the control run that must reproduce the stored images.

Usage: tracker_regen.py OUT SHARD NSHARDS [--limit N] [--only normal|failure]
"""
import json
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from closedloop_inject import install_injection  # noqa: E402
from closedloop_ledger import install  # noqa: E402
from closedloop_t import TrackerClient, centre  # noqa: E402

OLD = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
sys.path.insert(0, str(OLD))
import simulate  # noqa: E402

site = os.environ["SIMULATOR_TORCH_SITE"]
sys.path.insert(0, site)
import torch  # noqa: E402,F401
sys.path.remove(site)

out, shard, shards = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
truth_control = bool(os.environ.get("REGEN_TRUTH"))
HOME = Path.home()
DATASET = HOME / "pi05-abc/dataset"
INPUTS = HOME / "pi05-ledger-A/inputs"
contract = json.loads((CODE / "contract_A.json").read_text())
specs = {t["task_id"]: {"task_id": t["task_id"], "name": t["name"], "required_goals": t["required_goals"]}
         for t in contract["source"]["tasks"]}
manifest = json.loads((DATASET / "split_manifest.orig.json").read_text())
train = {f"t{int(e['task_id']):02d}_{e['episode_id']}" for e in manifest["episodes"] if e["split"] == "train"}
index = {e["name"]: e for e in json.loads((INPUTS / "index.json").read_text())["episodes"]}
lost = set(json.loads((HOME / "pi05-ledger-A/labels_A2/report.json").read_text())["summary"]["episodes_with_lost_replay"])
names = sorted(train - lost, key=lambda n: (n[:3], int(n.split("_")[-1])))

jobs = []
if only != "failure":
    jobs += [(name, "normal", 0) for name in names]
if only != "normal":
    # Same enumeration as gen_failures.py, so each episode gets the same injected goal.
    rows_dir = DATASET / "failures/rows"
    for prefix in ("t06", "t08"):
        for number, name in enumerate(n for n in names if n.startswith(prefix)):
            kind = "empty_hand" if prefix == "t06" or number % 2 == 0 else "drop"
            if (rows_dir / f"{name}_{kind}.jsonl").exists():
                jobs.append((name, kind, number))
jobs = [job for i, job in enumerate(jobs) if i % shards == shard]
if limit:
    jobs = jobs[:limit]
print("jobs", len(jobs), flush=True)

source = simulate.configure_official_source(OLD / "vendor/libero-mem", Path(tempfile.mkdtemp()), DATASET)
install(simulate)
base_environment = simulate.OfficialEnvironment
log_path = Path(tempfile.mkdtemp()) / "inject.jsonl"
(out / "log").mkdir(parents=True, exist_ok=True)
client = None if truth_control else TrackerClient()
renderer_js = CODE / contract["visual"]["renderer_asset"]

with simulate.CanonicalRenderer(renderer_js) as renderer:
    for name, kind, number in jobs:
        key = name if kind == "normal" else f"{name}_{kind}"
        if (out / "log" / f"{key}.npz").exists():
            continue
        meta = index[name]
        z = np.load(INPUTS / f"{name}.npz")
        actions = z["actions"].astype(np.float64)
        n = len(actions)
        if kind == "normal":
            wanted = set(range(n))
            prefix, fired_expected = f"images/{name}", None
        else:
            rows = [json.loads(line) for line in (DATASET / "failures/rows" / f"{key}.jsonl").open()]
            wanted = {r["observation_index"] for r in rows}
            prefix, fired_expected = f"failures/images/{key}", rows[0]["failure"]["fired_at"]
        simulate.OfficialEnvironment = base_environment
        if kind != "normal":
            install_injection(simulate, kind, str(log_path))
            simulate.OfficialEnvironment.created = number
        env = simulate.OfficialEnvironment(source, specs[meta["task_id"]], z["initial_state"], meta["instruction"])
        try:
            sids = sorted((s for s, row in env.registry.items() if row["category"] == "bowl"), key=int)
            history = deque(maxlen=simulate.CADENCE + 1)
            pred, true, init_report = [], [], None
            folder = out / prefix
            folder.mkdir(parents=True, exist_ok=True)

            def capture(frame):
                global init_report
                main, _, segmentation, _ = env.observe()
                if truth_control:
                    drawn = segmentation
                else:
                    if frame == 0:
                        init_report, labels = client.reset(main, len(sids))
                    else:
                        labels = client.step(main)
                    drawn = np.zeros(segmentation.shape, np.uint8)
                    for i, sid in enumerate(sids, 1):
                        drawn[labels == i] = int(sid)
                pred.append([centre(drawn == int(s)) or [np.nan, np.nan] for s in sids])
                true.append([centre(segmentation == int(s)) or [np.nan, np.nan] for s in sids])
                entities = simulate.measure_entities(drawn, env.registry,
                                                     minimum_visible_pixels=contract["memory"]["minimum_visible_pixels_for_motion_label"])
                history.append({"frame": frame, "entities": entities})
                if frame:  # gen_failures.py scores from frame 1; injection reads this counter
                    env.score_observation(frame)
                if frame in wanted:
                    tracks = simulate.make_tracks(list(history), env.registry)
                    png, _ = renderer.render(main, frame, max(0, frame - simulate.CADENCE), env.registry, tracks)
                    (folder / f"{frame:06d}.png").write_bytes(png)

            fired_at = None
            capture(0)
            for t in range(n):
                env.step(actions[t])
                if t + 1 < n:
                    capture(t + 1)
                    if kind != "normal" and fired_at is None and env.fired:  # same moment as gen_failures.py
                        fired_at = env.t
            if kind != "normal" and fired_at != fired_expected:
                print("FIRE_MISMATCH", key, fired_at, fired_expected, flush=True)
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
                continue
            np.savez_compressed(out / "log" / f"{key}.npz", pred=np.asarray(pred, np.float32), true=np.asarray(true, np.float32),
                                sids=np.asarray([int(s) for s in sids]), kind=kind, fired=-1 if fired_at is None else fired_at)
            print("DONE", key, "frames", n, "images", len(wanted),
                  "bowls", None if init_report is None else init_report.get("bowls_found"), flush=True)
        finally:
            env.close()
print("REGEN_DONE", shard, flush=True)
