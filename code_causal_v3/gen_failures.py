"""Failure demonstrations: replay expert controls of TRAINING episodes with one injected failure.

Injection is the same as closedloop_inject (T6 empty_hand; T8 alternating empty_hand / drop). Images
come from the same renderer as closed-loop evaluation, labels from the same label_A3 code as the normal
data, now on what really happened in this replay, so a failed step is simply not ticked.
Rows kept: [fire, fire+120] and the last 40 frames (the end-of-sequence "Complete" temptation). All
of them have valid_action_steps=0: the arm carries on with a now-wrong plan, so these rows train the
memory only and never the action branch. Held-out and validation episodes are never used.

Writes under dataset/failures: episodes/<name>.npz, images/<name>/, rows/<name>.jsonl.
Usage: gen_failures.py SHARD NSHARDS [--limit N]
"""
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from closedloop_inject import chosen_goal, install_injection  # noqa: E402
from closedloop_ledger import install  # noqa: E402

OLD = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
sys.path.insert(0, str(OLD))
import simulate  # noqa: E402

site = os.environ["SIMULATOR_TORCH_SITE"]
sys.path.insert(0, site)
import torch  # noqa: E402,F401
sys.path.remove(site)
sys.path.insert(0, str(Path.home() / "pi05-ledger-A"))
from label_A3 import label_episode  # noqa: E402

import tempfile  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
HOME = Path.home()
DATASET = HOME / "pi05-abc/dataset"
FAIL_DIR = os.environ.get("FAIL_DIR", "failures")
TAG = os.environ.get("FAIL_TAG", "")
OUT = DATASET / FAIL_DIR
INPUTS = HOME / "pi05-ledger-A/inputs"
AFTER, TAIL = 120, 40
contract = json.loads((CODE / "contract_A.json").read_text())
specs = {t["task_id"]: {"task_id": t["task_id"], "name": t["name"], "required_goals": t["required_goals"]}
         for t in contract["source"]["tasks"]}
manifest = json.loads((DATASET / "split_manifest.orig.json").read_text())
train = {f"t{int(e['task_id']):02d}_{e['episode_id']}" for e in manifest["episodes"] if e["split"] == "train"}
index = {e["name"]: e for e in json.loads((INPUTS / "index.json").read_text())["episodes"]}
report = json.loads((HOME / "pi05-ledger-A/labels_A2/report.json").read_text())["summary"]
lost = set(report["episodes_with_lost_replay"])
names = sorted(train - lost, key=lambda n: (n[:3], int(n.split("_")[-1])))
jobs = []
for task_id, prefix in ((6, "t06"), (8, "t08")):
    for number, name in enumerate(n for n in names if n.startswith(prefix)):
        if os.environ.get("FAIL_MODE") == "balance":
            # The first pass gave T8 drops only at goals 2 and 4, and T8 empty hands only at 1 and 3.
            if task_id == 8:
                jobs.append((name, "drop", number + (number % 2)))
                jobs.append((name, "empty_hand", number + 1 - (number % 2)))
            continue
        kind = "empty_hand" if task_id == 6 or number % 2 == 0 else "drop"
        jobs.append((name, kind, number))
jobs = [j for i, j in enumerate(jobs) if i % shards == shard]
if limit:
    jobs = jobs[:limit]
print("episodes for this shard", len(jobs), flush=True)

source = simulate.configure_official_source(OLD / "vendor/libero-mem", Path(tempfile.mkdtemp()), DATASET)
install(simulate)
base_environment = simulate.OfficialEnvironment
log_path = Path(tempfile.mkdtemp()) / "inject.jsonl"
for sub in ("episodes", "images", "rows"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
renderer_js = CODE / contract["visual"]["renderer_asset"]
counts = {"written": 0, "not_fired": 0, "diverged": 0}

with simulate.CanonicalRenderer(renderer_js) as renderer:
    for name, kind, number in jobs:
        meta = index[name]
        z = np.load(INPUTS / f"{name}.npz")
        task_id = meta["task_id"]
        simulate.OfficialEnvironment = base_environment
        install_injection(simulate, kind, str(log_path))
        simulate.OfficialEnvironment.created = number
        env = simulate.OfficialEnvironment(source, specs[task_id], z["initial_state"], meta["instruction"])
        try:
            base, sim = env.base, env.env.sim
            model = sim.model
            registry = meta["registry"]
            sids = [str(s) for s in z["sids"]]
            bowls = sorted([v for v in registry.values() if v["category"] == "bowl"], key=lambda v: v["segmentation_id"])
            plates = sorted([v for v in registry.values() if v["category"] == "plate"], key=lambda v: v["segmentation_id"])
            geom_of = {sid: set(entry["geom_ids"]) for sid, entry in env.geom_mapping.items()}
            bowl_geoms = [geom_of[str(b["segmentation_id"])] for b in bowls]
            plate_geoms = [geom_of[str(p["segmentation_id"])] for p in plates]
            plate_number = [int(p["physical_name"].split("_")[-1]) for p in plates]
            grip_geoms = env.grippers
            bowl_body = [base.obj_body_id[b["physical_name"]] for b in bowls]
            actions = z["actions"].astype(np.float64)
            n = len(actions)
            history = deque(maxlen=simulate.CADENCE + 1)
            rec = {k: [] for k in ("bowl_pos", "bowl_grip", "bowl_support", "conf", "states", "centres", "visibility", "areas")}
            frames = []

            def capture(frame):
                main, wrist, segmentation, state = env.observe()
                entities = simulate.measure_entities(segmentation, env.registry,
                                                     minimum_visible_pixels=contract["memory"]["minimum_visible_pixels_for_motion_label"])
                history.append({"frame": frame, "entities": entities})
                d = sim.data
                grip = np.zeros(len(bowls), bool)
                support = np.full(len(bowls), -1, np.int16)
                for c in d.contact[:d.ncon]:
                    g1, g2 = int(c.geom1), int(c.geom2)
                    for b, gs in enumerate(bowl_geoms):
                        other = g2 if g1 in gs else g1 if g2 in gs else None
                        if other is None or other in gs:
                            continue
                        if other in grip_geoms:
                            grip[b] = True
                            continue
                        hit = next((plate_number[p] for p, pg in enumerate(plate_geoms) if other in pg), None)
                        if hit is None:
                            hit = next((100 + ob for ob, og in enumerate(bowl_geoms) if other in og), None)
                        hit = 0 if hit is None else hit
                        rank = lambda s: 3 if 1 <= s < 100 else 2 if s >= 100 else 1 if s == 0 else 0
                        if rank(hit) > rank(int(support[b])):
                            support[b] = hit
                rec["bowl_pos"].append(np.array([d.body_xpos[i] for i in bowl_body]))
                rec["bowl_grip"].append(grip)
                rec["bowl_support"].append(support)
                rec["conf"].append(env.score_observation(frame)["confirmed_goals"] if frame else 0)
                rec["states"].append(np.asarray(state, np.float32))
                centres, visible, areas = [], [], []
                for sid in sids:
                    entity = entities.get(sid) or {}
                    ok = bool(entity.get("reliable")) and entity.get("center") is not None
                    centres.append(entity.get("center") if ok else [0.0, 0.0])
                    visible.append(ok)
                    areas.append(entity.get("area", 0))
                rec["centres"].append(centres)
                rec["visibility"].append(visible)
                rec["areas"].append(areas)
                return main, wrist

            pending = []
            main, wrist = capture(0)
            pending.append((0, main, wrist, list(history)))
            renders = {}
            for t in range(n):
                env.step(actions[t])
                if t + 1 < n:
                    main, wrist = capture(t + 1)
                    fired_at = env.t if env.fired and "fired_at" not in renders else None
                    if fired_at is not None:
                        renders["fired_at"] = fired_at
                    start = renders.get("fired_at")
                    frame = t + 1
                    # The end of the episode is only known from n; everything is causal otherwise.
                    if (start is not None and frame <= start + AFTER) or frame >= n - TAIL:
                        if start is not None:
                            history_start = max(0, frame - simulate.CADENCE)
                            tracks = simulate.make_tracks(list(history), env.registry)
                            png, _ = renderer.render(main, frame, history_start, env.registry, tracks)
                            folder = OUT / "images" / f"{name}_{kind}{TAG}"
                            folder.mkdir(exist_ok=True)
                            (folder / f"{frame:06d}.png").write_bytes(png)
                            (folder / f"{frame:06d}_wrist.png").write_bytes(simulate.png_bytes(simulate.resize_rgb(wrist, size=224)))
                            frames.append(frame)
            fired_at = renders.get("fired_at")
            if fired_at is None:
                counts["not_fired"] += 1
                print("NOT_FIRED", name, kind, flush=True)
                continue
            recorded = z["confirmed_goals"].astype(int)
            if int(recorded[fired_at]) != env.goal - 1:
                counts["diverged"] += 1
                print("DIVERGED", name, kind, "recorded", int(recorded[fired_at]), "goal", env.goal, flush=True)
                import shutil
                shutil.rmtree(OUT / "images" / f"{name}_{kind}{TAG}", ignore_errors=True)
                continue
            conf = np.asarray(rec["conf"], int)
            conf = np.maximum.accumulate(conf)
            phys = {"bowl_sids": np.array([b["segmentation_id"] for b in bowls]), "bowl_pos": np.asarray(rec["bowl_pos"]),
                    "bowl_support": np.asarray(rec["bowl_support"]), "bowl_grip": np.asarray(rec["bowl_grip"]),
                    "centre_err": np.zeros((n, len(bowls))), "eef_err": np.zeros(n)}
            states = np.asarray(rec["states"], np.float32)
            inp = {"sids": z["sids"], "visibility": np.asarray(rec["visibility"], bool), "confirmed_goals": conf, "states": states}
            rows, stats, info = label_episode(meta, phys, inp)
            episode_key = f"{name}_{kind}{TAG}"
            np.savez_compressed(OUT / "episodes" / f"{episode_key}.npz", states=states, actions=actions.astype(np.float32),
                                centres=np.asarray(rec["centres"], np.float32), visibility=np.asarray(rec["visibility"], bool),
                                sids=z["sids"], confirmed_goals=conf, fired_at=fired_at, goal=env.goal,
                                action_text=np.array([r["memory_target"]["action_text"] for r in rows]),
                                action_reliable=np.array([r["memory_target"]["action_reliable"] for r in rows]))
            with (OUT / "rows" / f"{episode_key}.jsonl").open("w") as stream:
                for frame in frames:
                    row = rows[frame]
                    stream.write(json.dumps({
                        "sample_id": f"{episode_key}_{frame:06d}", "task_id": task_id, "episode_id": episode_key,
                        "observation_index": frame, "history_start": max(0, frame - 20), "history_end": frame,
                        "instruction": meta["instruction"], "previous_memory_text": row["previous_memory_text"],
                        "memory_target_text": row["memory_target_text"], "memory_target": row["memory_target"],
                        "action_label_reliable": row["memory_target"]["action_reliable"],
                        "image_path": f"{FAIL_DIR}/images/{episode_key}/{frame:06d}.png",
                        "wrist_image_path": f"{FAIL_DIR}/images/{episode_key}/{frame:06d}_wrist.png",
                        "episode_npz": f"{FAIL_DIR}/episodes/{episode_key}.npz", "action_start": frame, "valid_action_steps": 0,
                        "failure": {"kind": kind, "goal": env.goal, "fired_at": fired_at}}) + "\n")
            counts["written"] += 1
            print("WRITTEN", episode_key, "goal", env.goal, "fired", fired_at, "rows", len(frames), "final env", int(conf[-1]), "of", len(meta["required_sequence"]),
                  "final task", rows[-1]["memory_target"]["task_text"], dict(stats) if counts["written"] <= 2 else "", flush=True)
        finally:
            env.close()
print("SHARD_DONE", shard, counts, flush=True)
