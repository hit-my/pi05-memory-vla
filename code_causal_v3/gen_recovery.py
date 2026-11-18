"""Failure-and-recovery demonstrations on TRAINING episodes, with no state teleporting.

Expert controls are replayed up to the start of goal k. A scripted Cartesian controller then makes a
physically real failure, and picks the bowl up from where it really is and places it properly:
  empty_hand : closes above the rim, lifts, carries air to the target, opens, then recovers
  drop       : grasps and lifts, opens in mid-carry, then recovers at once
  near_miss  : grasps and places 4-6 cm off the plate centre, opens, then recovers
  empty_abort: closes above the rim and lifts, then recovers at once from the closed empty hand
Labels use the same label_A3 code on what really happened, so the failed attempt is not ticked and
the recovery is. Action supervision: the failed attempt never (valid_action_steps=0), the recovery yes.
Writes under dataset/recovery. Usage: gen_recovery.py SHARD NSHARDS [--limit N]
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
OLD = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
sys.path.insert(0, str(OLD))
import simulate  # noqa: E402

site = os.environ["SIMULATOR_TORCH_SITE"]
sys.path.insert(0, site)
import torch  # noqa: E402,F401
sys.path.remove(site)
sys.path.insert(0, str(Path.home() / "pi05-ledger-A"))
from label_A3 import label_episode  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
HOME = Path.home()
DATASET = HOME / "pi05-abc/dataset"
OUT = DATASET / "recovery"
INPUTS = HOME / "pi05-ledger-A/inputs"
PHYSICS = HOME / "pi05-ledger-A/physics"
contract = json.loads((CODE / "contract_A.json").read_text())
specs = {t["task_id"]: {"task_id": t["task_id"], "name": t["name"], "required_goals": t["required_goals"]}
         for t in contract["source"]["tasks"]}
manifest = json.loads((DATASET / "split_manifest.orig.json").read_text())
train = {f"t{int(e['task_id']):02d}_{e['episode_id']}" for e in manifest["episodes"] if e["split"] == "train"}
index = {e["name"]: e for e in json.loads((INPUTS / "index.json").read_text())["episodes"]}
lost = set(json.loads((HOME / "pi05-ledger-A/labels_A2/report.json").read_text())["summary"]["episodes_with_lost_replay"])
names = sorted(train - lost, key=lambda n: (n[:3], int(n.split("_")[-1])))
KINDS = ("empty_hand", "drop", "near_miss", "empty_abort")
jobs = []
for prefix, task_id in (("t06", 6), ("t08", 8)):
    for number, name in enumerate(n for n in names if n.startswith(prefix)):
        goal = number % 5 + 2 if task_id == 6 else number % 4 + 1
        jobs.append((name, KINDS[number % len(KINDS)], goal, number))
jobs = [j for i, j in enumerate(jobs) if i % shards == shard]
if limit:
    jobs = jobs[:limit]
print("episodes for this shard", len(jobs), flush=True)
for sub in ("episodes", "images", "rows"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
source = simulate.configure_official_source(OLD / "vendor/libero-mem", Path(tempfile.mkdtemp()), DATASET)
renderer_js = CODE / contract["visual"]["renderer_asset"]
VMAX, KP, DUMMY = 0.55, 12.0, simulate.DUMMY_ACTION
counts = {"written": 0, "recovered": 0, "not_recovered": 0, "skipped": 0}


class Done(Exception):
    pass


with simulate.CanonicalRenderer(renderer_js) as renderer:
    for name, kind, goal_k, number in jobs:
        meta = index[name]
        z = np.load(INPUTS / f"{name}.npz")
        expert_phys = np.load(PHYSICS / f"{name}.npz")
        task_id = meta["task_id"]
        registry = meta["registry"]
        by_phys = {v["physical_name"]: k for k, v in registry.items()}
        goal = meta["required_sequence"][goal_k - 1]
        recorded = z["confirmed_goals"].astype(int)
        start_frame = 0 if goal_k == 1 else int(np.argmax(recorded >= goal_k - 1)) + 22
        bowls = sorted([v for v in registry.values() if v["category"] == "bowl"], key=lambda v: v["segmentation_id"])
        plates = sorted([v for v in registry.values() if v["category"] == "plate"], key=lambda v: v["segmentation_id"])
        b_index = [str(s) for s in expert_phys["bowl_sids"]].index(by_phys[goal["physical_bowl"]])
        # Expert grasp: hand position relative to the bowl at the first sustained gripper contact of this goal.
        grip = expert_phys["bowl_grip"][:, b_index]
        contact = [f for f in range(start_frame, len(grip) - 5) if grip[f:f + 5].all()]
        if not contact or int(recorded[min(start_frame, len(recorded) - 1)]) != goal_k - 1:
            counts["skipped"] += 1
            print("SKIP", name, kind, "no expert grasp or diverged", flush=True)
            continue
        offset = expert_phys["eef_pos"][contact[0]] - expert_phys["bowl_pos"][contact[0], b_index]
        env = simulate.OfficialEnvironment(source, specs[task_id], z["initial_state"], meta["instruction"])
        key = f"{name}_{kind}"
        try:
            base = env.env
            while not hasattr(base, "obj_body_id") and hasattr(base, "env"):
                base = base.env
            sim = env.env.sim
            sids = [str(s) for s in z["sids"]]
            geom_of = {sid: set(entry["geom_ids"]) for sid, entry in env.geom_mapping.items()}
            bowl_geoms = [geom_of[str(b["segmentation_id"])] for b in bowls]
            plate_geoms = [geom_of[str(p["segmentation_id"])] for p in plates]
            plate_number = [int(p["physical_name"].split("_")[-1]) for p in plates]
            names_geom = [sim.model.geom_id2name(i) or "" for i in range(sim.model.ngeom)]
            grip_geoms = {i for i, n in enumerate(names_geom) if n.startswith("gripper0_") or "finger" in n}
            bowl_body = [base.obj_body_id[b["physical_name"]] for b in bowls]
            bowl_id = base.obj_body_id[goal["physical_bowl"]]
            plate_id = base.obj_body_id[goal["physical_plate"]]
            rec = {k: [] for k in ("bowl_pos", "bowl_grip", "bowl_support", "conf", "states", "centres", "visibility")}
            history = deque(maxlen=simulate.CADENCE + 1)
            actions, frames_images, phase_of = [], {}, {}
            phase = ["expert"]
            folder = OUT / "images" / key
            folder.mkdir(exist_ok=True)

            def capture(frame):
                main, wrist, segmentation, state = env.observe()
                entities = simulate.measure_entities(segmentation, env.registry,
                                                     minimum_visible_pixels=contract["memory"]["minimum_visible_pixels_for_motion_label"])
                history.append({"frame": frame, "entities": entities})
                d = sim.data
                grip_now = np.zeros(len(bowls), bool)
                support = np.full(len(bowls), -1, np.int16)
                for c in d.contact[:d.ncon]:
                    g1, g2 = int(c.geom1), int(c.geom2)
                    for b, gs in enumerate(bowl_geoms):
                        other = g2 if g1 in gs else g1 if g2 in gs else None
                        if other is None or other in gs:
                            continue
                        if other in grip_geoms:
                            grip_now[b] = True
                            continue
                        hit = next((plate_number[p] for p, pg in enumerate(plate_geoms) if other in pg), None)
                        if hit is None:
                            hit = next((100 + ob for ob, og in enumerate(bowl_geoms) if other in og), None)
                        hit = 0 if hit is None else hit
                        rank = lambda s: 3 if 1 <= s < 100 else 2 if s >= 100 else 1 if s == 0 else 0
                        if rank(hit) > rank(int(support[b])):
                            support[b] = hit
                rec["bowl_pos"].append(np.array([d.body_xpos[i] for i in bowl_body]))
                rec["bowl_grip"].append(grip_now)
                rec["bowl_support"].append(support)
                rec["conf"].append(env.score_observation(frame)["confirmed_goals"] if frame else 0)
                rec["states"].append(np.asarray(state, np.float32))
                centres, visible = [], []
                for sid in sids:
                    entity = entities.get(sid) or {}
                    ok = bool(entity.get("reliable")) and entity.get("center") is not None
                    centres.append(entity.get("center") if ok else [0.0, 0.0])
                    visible.append(ok)
                rec["centres"].append(centres)
                rec["visibility"].append(visible)
                phase_of[frame] = phase[0]
                if phase[0] != "expert":
                    tracks = simulate.make_tracks(list(history), env.registry)
                    png, _ = renderer.render(main, frame, max(0, frame - simulate.CADENCE), env.registry, tracks)
                    (folder / f"{frame:06d}.png").write_bytes(png)
                    (folder / f"{frame:06d}_wrist.png").write_bytes(simulate.png_bytes(simulate.resize_rgb(wrist, size=224)))

            def act(action):
                actions.append(np.asarray(action, np.float64))
                env.step(action)
                capture(len(actions))

            def eef():
                return np.asarray(env.obs["robot0_eef_pos"], np.float64)

            def width():
                q = env.obs["robot0_gripper_qpos"]
                return float(q[0] - q[1])

            def go(target, gripper, tol=0.006, max_steps=120):
                for _ in range(max_steps):
                    error = np.asarray(target) - eef()
                    if np.linalg.norm(error) < tol:
                        return
                    action = DUMMY.copy()
                    action[:3] = np.clip(KP * error, -VMAX, VMAX)
                    action[6] = gripper
                    act(action)

            def hold(gripper, steps):
                for _ in range(steps):
                    action = DUMMY.copy()
                    action[6] = gripper
                    act(action)

            OPEN, CLOSE = -1.0, 1.0
            ABOVE = 0.13

            def bowl_xyz():
                return np.array(sim.data.body_xpos[bowl_id], dtype=np.float64)

            def plate_xyz():
                return np.array(sim.data.body_xpos[plate_id], dtype=np.float64)

            def pick(z_extra=0.0):
                position = bowl_xyz()
                go(position + offset + [0, 0, ABOVE], OPEN)
                position = bowl_xyz()
                go(position + offset + [0, 0, z_extra], OPEN, tol=0.004)
                hold(CLOSE, 12)
                go(eef() + [0, 0, ABOVE], CLOSE)

            def place(shift=np.zeros(3)):
                plate = plate_xyz() + shift
                go(plate + offset * [1, 1, 0] + [0, 0, ABOVE + offset[2] + 0.01], CLOSE)
                go(plate + offset + [0, 0, 0.014], CLOSE, tol=0.005)
                hold(OPEN, 12)
                go(eef() + [0, 0, ABOVE], OPEN)

            capture(0)
            for t in range(start_frame):
                act(z["actions"][t].astype(np.float64))
            if rec["conf"][-1] != goal_k - 1:
                counts["skipped"] += 1
                print("SKIP", key, "replay count", rec["conf"][-1], "at start", flush=True)
                continue
            before = bowl_xyz()
            phase[0] = "failure"
            fail_start = len(actions)
            rng = np.random.default_rng(number)
            if kind == "empty_hand":
                pick(z_extra=0.05)
                place()
            elif kind == "empty_abort":
                # Closed on air and lifted: the policy learns to reopen and retry instead of carrying on.
                pick(z_extra=0.05)
            elif kind == "drop":
                pick()
                midpoint = (eef() + plate_xyz() + offset * [1, 1, 0] + [0, 0, ABOVE]) / 2
                aside = rng.normal(size=2)
                aside = 0.09 * aside / np.linalg.norm(aside)
                if np.linalg.norm(midpoint[:2] - eef()[:2]) < 0.06:
                    midpoint[:2] = eef()[:2] + aside
                go(midpoint, CLOSE)
                hold(OPEN, 12)
                go(eef() + [0, 0, 0.05], OPEN, max_steps=30)
            else:
                angle = rng.uniform(0, 2 * np.pi)
                radius = rng.uniform(0.04, 0.06)
                pick()
                place(np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0]))
            fail_end = len(actions)
            env_after_failure = rec["conf"][-1]
            rest = bowl_xyz()
            moved_cm = float(np.linalg.norm(rest[:2] - before[:2]) * 100)
            target_cm = float(np.linalg.norm(rest[:2] - plate_xyz()[:2]) * 100)
            phase[0] = "recovery"
            pick()
            place()
            hold(OPEN, 6)
            recovery_end = len(actions)
            actions_array = np.asarray(actions, np.float32)
            n = len(rec["conf"])
            conf = np.maximum.accumulate(np.asarray(rec["conf"], int))
            recovered = int(conf[-1]) >= goal_k
            failed_properly = env_after_failure == goal_k - 1
            # Pad the action list to one per frame; the pad is outside every supervised window.
            actions_full = np.vstack((actions_array, np.asarray(DUMMY, np.float32)[None]))
            phys = {"bowl_sids": np.array([b["segmentation_id"] for b in bowls]), "bowl_pos": np.asarray(rec["bowl_pos"]),
                    "bowl_support": np.asarray(rec["bowl_support"]), "bowl_grip": np.asarray(rec["bowl_grip"]),
                    "centre_err": np.zeros((n, len(bowls))), "eef_err": np.zeros(n)}
            states = np.asarray(rec["states"], np.float32)
            inp = {"sids": z["sids"], "visibility": np.asarray(rec["visibility"], bool), "confirmed_goals": conf, "states": states}
            rows, stats, info = label_episode(meta, phys, inp)
            np.savez_compressed(OUT / "episodes" / f"{key}.npz", states=states, actions=actions_full,
                                centres=np.asarray(rec["centres"], np.float32), visibility=np.asarray(rec["visibility"], bool),
                                sids=z["sids"], confirmed_goals=conf, fired_at=fail_start, goal=goal_k,
                                fail_end=fail_end, recovery_end=recovery_end,
                                action_text=np.array([r["memory_target"]["action_text"] for r in rows]),
                                action_reliable=np.array([r["memory_target"]["action_reliable"] for r in rows]))
            good = recovered and failed_properly
            with (OUT / "rows" / f"{key}.jsonl").open("w") as stream:
                for frame in range(fail_start + 1, n):
                    row = rows[frame]
                    in_recovery = good and frame >= fail_end
                    valid = min(20, recovery_end - frame) if in_recovery else 0
                    stream.write(json.dumps({
                        "sample_id": f"{key}_{frame:06d}", "task_id": task_id, "episode_id": key,
                        "observation_index": frame, "history_start": max(0, frame - 20), "history_end": frame,
                        "instruction": meta["instruction"], "previous_memory_text": row["previous_memory_text"],
                        "memory_target_text": row["memory_target_text"], "memory_target": row["memory_target"],
                        "action_label_reliable": row["memory_target"]["action_reliable"],
                        "image_path": f"recovery/images/{key}/{frame:06d}.png",
                        "wrist_image_path": f"recovery/images/{key}/{frame:06d}_wrist.png",
                        "episode_npz": f"recovery/episodes/{key}.npz", "action_start": frame, "valid_action_steps": max(valid, 0),
                        "failure": {"kind": kind, "goal": goal_k, "fail_start": fail_start, "fail_end": fail_end,
                                    "phase": "recovery" if frame >= fail_end else "failure"}}) + "\n")
            counts["written"] += 1
            counts["recovered" if good else "not_recovered"] += 1
            print("WRITTEN", key, "goal", goal_k, "start", start_frame, "fail", fail_start, fail_end, "end", recovery_end,
                  "env_after_failure", env_after_failure, "final_env", int(conf[-1]), "bowl moved cm", round(moved_cm, 1), "from target cm", round(target_cm, 1),
                  "recovered", recovered, "final tick", rows[-1]["memory_target"]["confirmed_goals"], flush=True)
        finally:
            env.close()
print("SHARD_DONE", shard, counts, flush=True)
