"""Build A-format training rows for the RoboMemArena pilot from the subtask HDF5s.

Per episode (task, seed): concatenate the subtask files in order, draw every main image with the RGB tracker
(rma_perception), and write labels:
  Subgoal checklist = the subtask sequence; item k is ticked at the last frame of subtask k.
  Task Complete only once every item is ticked (the final frame).
  Action line, per tracked object, over the 20-frame window: an object is held from a gripper-close command
  inside a subtask that names it until the next open command. A held object takes the end-effector's motion
  in the robot base frame (up/down/forward/backward/left/right, two halves of the window, 1.5 cm threshold)
  plus picked/placed events; an object that is not held is still, or occluded when the tracker sees it in
  fewer than half of the window frames. Nothing comes from simulator state: proprioception, commanded
  gripper and the subtask annotation only.

Usage: rma_build.py RAW OUT SHARD NSHARDS
Writes OUT/images/<ep>/NNNNNN.png and _wrist.png, OUT/episodes/<ep>.npz, OUT/rows/<ep>.jsonl, OUT/log/<ep>.json
"""
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import rma_spec  # noqa: E402
from ledger_v2 import delta_text, ledger_from_count  # noqa: E402
from memory_format import INITIAL_MEMORY, render_previous  # noqa: E402
from rma_perception import Perception  # noqa: E402

raw, out, shard, shards = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
AXES = ((0, "forward", "backward"), (1, "left", "right"), (2, "up", "down"))
THRESHOLD = 0.015


LEROBOT = os.environ.get("SOURCE") == "lerobot"  # RAW is then rma_lerobot.py output (front/, wrist/, meta/)


def lerobot_episodes():
    only = {int(x) for x in os.environ.get("TASKS", "").split(",") if x}
    jobs = []
    for path in sorted((raw / "meta").glob("t*_seed*.npz")):
        task_id, seed = int(path.stem[1:3]), int(path.stem.split("seed")[1])
        if (not only or task_id in only) and (raw / "front" / f"{path.stem}.npy").exists():
            jobs.append((task_id, seed, path.stem))
    # interleave tasks so every shard gets a mix
    jobs.sort(key=lambda j: (j[1], j[0]))
    return jobs


def lerobot_load(stem):
    m = np.load(raw / "meta" / f"{stem}.npz")
    rgb = np.load(raw / "front" / f"{stem}.npy")
    wrist = np.load(raw / "wrist" / f"{stem}.npy")
    sub = m["subtask"]
    n = min(len(rgb), len(wrist), len(sub))
    sub = sub[:n]
    ends = [t for t in range(n - 1) if sub[t + 1] != sub[t]] + [n - 1]
    texts = [str(x) for x in m["subtask_texts"]]
    assert len(texts) == len(ends), (stem, len(texts), len(ends))
    return rgb[:n], wrist[:n], m["states"][:n].astype(np.float32), m["actions"][:n].astype(np.float32), ends, texts


def episodes():
    if LEROBOT:
        return lerobot_episodes()
    jobs = []
    only = {int(x) for x in os.environ.get("TASKS", "").split(",") if x}
    for task_id, spec in sorted(rma_spec.TASKS.items()):
        if only and task_id not in only:
            continue
        folder = raw / spec["folder"]
        if not folder.is_dir():
            continue
        by_seed = {}
        for path in folder.glob("*.hdf5"):
            text, order, seed = rma_spec.parse_subtask(path.name)
            by_seed.setdefault(seed, []).append((order, text, path))
        n_sub = max(len(v) for v in by_seed.values())
        complete = [(seed, parts) for seed, parts in sorted(by_seed.items()) if len(parts) == n_sub]
        for seed, parts in complete[:int(os.environ.get("MAX_SEEDS", "26"))]:
            jobs.append((task_id, seed, sorted(parts)))
    return jobs


def load(parts):
    rgb, wrist, states, actions, ends, texts = [], [], [], [], [], []
    for order, text, path in parts:
        with h5py.File(path, "r") as f:
            g = f["data/demo_0"]
            a, w = g["obs/agentview_rgb"][:], g["obs/eye_in_hand_rgb"][:]
            s = np.concatenate((g["obs/ee_states"][:], g["obs/gripper_states"][:]), axis=1)
            act = g["actions"][:]
        if rgb and np.array_equal(rgb[-1][-1], a[0]):  # boundary frame stored twice
            a, w, s, act = a[1:], w[1:], s[1:], act[1:]
        rgb.append(a), wrist.append(w), states.append(s), actions.append(act)
        ends.append(sum(len(x) for x in rgb) - 1)
        texts.append(text)
    return (np.concatenate(rgb), np.concatenate(wrist), np.concatenate(states).astype(np.float32),
            np.concatenate(actions).astype(np.float32), ends, texts)


def held_table(task_id, actions, ends, texts, n_objects):
    """held[t, k]; events[t] = list of (k, 'picked'|'placed')."""
    held = np.zeros((len(actions), n_objects), bool)
    events = {}
    closed = actions[:, 6] > 0
    current = None
    subtask = np.searchsorted(np.asarray(ends), np.arange(len(actions)))
    for t in range(len(actions)):
        if t and closed[t] and not closed[t - 1]:
            k = rma_spec.subtask_object(task_id, texts[min(subtask[t], len(texts) - 1)])
            if k is not None and current is None:
                current = k
                events.setdefault(t, []).append((k, "picked"))
        if t and not closed[t] and closed[t - 1] and current is not None:
            events.setdefault(t, []).append((current, "placed"))
            current = None
        if current is not None:
            held[t, current] = True
    return held, events


def directions(pos):
    words = []
    for half in (pos[: len(pos) // 2 + 1], pos[len(pos) // 2:]):
        d = half[-1] - half[0]
        axis = int(np.argmax(np.abs(d)))
        if abs(d[axis]) >= THRESHOLD:
            word = AXES[axis][1] if d[axis] > 0 else AXES[axis][2]
            if not words or words[-1] != word:
                words.append(word)
    return words


def action_text(t, states, held, events, visible, names):
    start = max(0, t - 20)
    clauses = []
    for k, name in enumerate(names):
        evs = [e for f in range(start + 1, t + 1) for (kk, e) in events.get(f, []) if kk == k]
        moved = directions(states[start:t + 1, :3]) if held[start:t + 1, k].any() else []
        if moved:
            body = " -> ".join(moved) + "".join(f" ; {e}" for e in evs)
        elif evs:
            body = " ; ".join(evs)
        elif visible[start:t + 1, k].mean() < 0.5:
            body = "occluded"
        else:
            body = "still"
        clauses.append(f"{name}: {body}")
    return " | ".join(clauses)


def main():
    jobs = [j for i, j in enumerate(episodes()) if i % shards == shard]
    print("episodes", len(jobs), flush=True)
    for sub in ("images", "episodes", "rows", "log"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    perception = None
    for task_id, seed, parts in jobs:
        ep = f"t{task_id:02d}_seed{seed}"
        if (out / "rows" / f"{ep}.jsonl").exists():
            continue
        started = time.time()
        rgb, wrist, states, actions, ends, texts = lerobot_load(parts) if LEROBOT else load(parts)
        n = len(rgb)
        if perception is None:
            perception = Perception(task_id)
        else:
            perception.task_id, perception.spec = task_id, rma_spec.TASKS[task_id]
            perception.registry = rma_spec.registry(task_id)
        names = rma_spec.names(task_id)
        folder = out / "images" / ep
        folder.mkdir(parents=True, exist_ok=True)
        centres = np.zeros((n, len(names), 2), np.float32)
        visible = np.zeros((n, len(names)), bool)
        for t in range(n):
            overlay = perception.reset(rgb[t], states[t]) if t == 0 else perception.step(rgb[t], states[t])
            (folder / f"{t:06d}.png").write_bytes(perception.png)
            Image.fromarray(wrist[t]).resize((224, 224), Image.BILINEAR).save(folder / f"{t:06d}_wrist.png")
            for k, c in enumerate(perception.centres):
                if c is not None:
                    centres[t, k], visible[t, k] = c, True
        held, events = held_table(task_id, actions, ends, texts, len(names))
        confirmed = np.array([sum(1 for e in ends if e <= t) for t in range(n)])
        items = texts
        acts = [action_text(t, states, held, events, visible, names) for t in range(n)]
        rows = []
        for t in range(n):
            previous = INITIAL_MEMORY if t < 20 else render_previous(acts[t - 20], ledger_from_count(items, int(confirmed[t - 20])))
            delta = delta_text(None if t < 20 else int(confirmed[t - 20]), int(confirmed[t]), items)
            task_text = delta.split("\nTask:")[1]
            target = {"action_text": acts[t], "subgoal_delta": delta.split("\n")[0][len("Subgoal:"):], "task_text": task_text,
                      "confirmed_goals": int(confirmed[t]), "total_goals": len(items), "action_reliable": True}
            rows.append({"sample_id": f"{ep}_{t:06d}", "task_id": task_id, "episode_id": ep, "observation_index": t,
                         "history_start": max(0, t - 20), "history_end": t, "instruction": rma_spec.TASKS[task_id]["prompt"],
                         "previous_memory_text": previous, "previous_query_index": t - 20 if t >= 20 else None,
                         "memory_update_interval": 20,
                         "memory_target_text": f"Action:{acts[t]}\n{delta}", "memory_target": target,
                         "image_path": f"images/{ep}/{t:06d}.png", "wrist_image_path": f"images/{ep}/{t:06d}_wrist.png",
                         "episode_npz": f"episodes/{ep}.npz", "action_start": t, "valid_action_steps": min(20, n - t),
                         "action_label_reliable": True})
        np.savez_compressed(out / "episodes" / f"{ep}.npz", states=states, actions=actions, centres=centres,
                            visibility=visible, action_text=np.array(acts), confirmed_goals=confirmed, ends=np.array(ends))
        with open(out / "rows" / f"{ep}.jsonl", "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        (out / "log" / f"{ep}.json").write_text(json.dumps({"frames": n, "items": items, "tracker": perception.report,
                                                            "visible": visible.mean(axis=0).round(3).tolist(),
                                                            "seconds": round(time.time() - started, 1)}))
        print("DONE", ep, n, "visible", visible.mean(axis=0).round(2).tolist(), "s", round(time.time() - started), flush=True)
    print("BUILD_DONE", shard, flush=True)


if __name__ == "__main__":
    main()
