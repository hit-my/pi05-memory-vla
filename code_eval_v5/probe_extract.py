"""Can each representation tell the outcome apart at the moment the model ticks a goal?

One window per decision, from closed-loop rollouts:
  good       : ledger ticks goal k and the env confirms it within 20 frames (normal 10k runs)
  near_miss  : ledger ticks k, env never does, bowl rests 3-8 cm off the plate centre (6k/8k/10k, by replay)
  empty_hand : false tick after injected empty hand        drop : false tick after injected drop
Representations of the bowl in the 21-frame window ending at the tick:
  raw (the encoder's 63-d input), zB, zC (standardized target encoders), features (11 physical numbers).
Output: probe.npz for probe_fit.py. This is a diagnostic only; these windows are never trained on.
"""
import json
import pickle
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from target_encoder import Encoder, PIXEL_SCALE, WINDOW  # noqa: E402

HOME = Path.home()
CL = Path("xxxx/pi05-abc/closedloop")
INJ = Path("xxxx/pi05-abc/inject/full")
LEDGER_A = HOME / "pi05-ledger-A"
index = json.loads((LEDGER_A / "inputs/index.json").read_text())["episodes"]
meta = {}
for episode in index:
    meta.setdefault(episode["task_id"], episode)
camera = json.loads((LEDGER_A / "probe/camera_probe.json").read_text())
GOALS = {6: [("akita_black_bowl_1", "plate_1")] * 7,
         8: [("akita_black_bowl_1", "plate_4"), ("akita_black_bowl_2", "plate_1"),
             ("akita_black_bowl_3", "plate_2"), ("akita_black_bowl_1", "plate_3")]}


def project(points, task_id):
    cam = camera[str(task_id)]
    K, E = np.asarray(cam["K"]), np.asarray(cam["cam_to_world"])
    w2c = np.linalg.inv(E)
    pc = w2c[:3, :3] @ points.T + w2c[:3, 3:4]
    uv = (K @ pc)[:2] / pc[2:3]
    return np.stack([uv[0] - 0.5, uv[1] - 0.5], -1)


cache = {}


def load_track(folder):
    if folder not in cache:
        track = {}
        for line in (folder / "observations.jsonl").read_text().splitlines():
            row = json.loads(line)
            track[row["frame"]] = {s: np.asarray(e["center"], float) for s, e in row["entities"].items()
                                   if e.get("reliable") and e.get("center") is not None}
        cache[folder] = (track, np.load(folder / "rollout.npz")["states"].astype(float))
    return cache[folder]


def window(folder, task_id, goal, t):
    track, states = load_track(folder)
    reg = meta[task_id]["registry"]
    by_phys = {v["physical_name"]: k for k, v in reg.items()}
    sid = by_phys[GOALS[task_id][goal - 1][0]]
    plates = [k for k, v in reg.items() if v["category"] == "plate"]
    frames = np.arange(t - WINDOW + 1, t + 1)
    vis = np.array([f >= 0 and f in track and sid in track[f] for f in frames])
    raw_pos = np.array([track[f][sid] if v else [0.0, 0.0] for f, v in zip(frames, vis)])
    if not vis.any():
        return None
    relative = raw_pos - raw_pos[np.argmax(vis)]
    x = np.concatenate((np.where(vis[:, None], relative, 0.0) / PIXEL_SCALE, vis[:, None].astype(float)), axis=1).astype(np.float32)
    seen = raw_pos[vis]
    centres = [track[0][p] for p in plates if p in track.get(0, {})]
    nearest = min(np.linalg.norm(seen[-1] - c) for c in centres) if centres else 0.0
    lo = max(0, t - WINDOW + 1)
    width = states[lo:t + 1, 6] - states[lo:t + 1, 7]
    eef = project(states[:t + 1, :3], task_id)
    eef_window = eef[frames[vis].astype(int)]
    bowl_steps, eef_steps = np.diff(seen, axis=0), np.diff(eef_window, axis=0)
    together = 0.0
    if len(bowl_steps):
        denominator = np.linalg.norm(bowl_steps) * np.linalg.norm(eef_steps)
        together = float(np.sum(bowl_steps * eef_steps) / denominator) if denominator > 1e-6 else 0.0
    speed = float(np.mean(np.linalg.norm(bowl_steps[-5:], axis=1))) if len(bowl_steps) else 0.0
    features = [(seen[-1] - seen[0])[0] / 50, (seen[-1] - seen[0])[1] / 50,
                float(np.sum(np.linalg.norm(bowl_steps, axis=1))) / 50 if len(bowl_steps) else 0.0, speed,
                nearest / 50, float(np.linalg.norm(seen[-1] - eef_window[-1])) / 50, together,
                float(np.linalg.norm(seen[-1] - seen[0])) / max(float(np.linalg.norm(eef_window[-1] - eef_window[0])), 1.0),
                float(width[-1]) * 10, float(width.mean()) * 10, float(vis.mean())]
    return x, np.asarray(features, np.float32)


samples = []  # (class, group, demo, task, x, features)


VERB_WORDS = ("up", "down", "forward", "backward", "left", "right", "still", "occluded", "picked", "placed")


def spoken_verbs(folder, task_id, goal, t):
    """What the model's own Action text said about this bowl at the tick (zeros for group B, which has no verbs)."""
    colour = {"akita_black_bowl_1": "blue", "akita_black_bowl_2": "orange", "akita_black_bowl_3": "purple"}[GOALS[task_id][goal - 1][0]]
    for line in (folder / "queries.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "response" and row["frame"] == t:
            action = (row["response"].get("generated_text") or "").split("\n")[0]
            for clause in action[len("Action:"):].split(" | "):
                if clause.startswith(colour):
                    words = clause.replace("<unused0>", "").split(":", 1)[1].replace("->", " ").replace(";", " ").split()
                    return [float(w in words) for w in VERB_WORDS]
    return [0.0] * len(VERB_WORDS)


def add(kind, group, task_id, demo, folder, goal, t):
    built = window(folder, task_id, goal, t)
    if built is not None:
        samples.append((kind, group, demo, task_id, *built, spoken_verbs(folder, task_id, goal, t)))


# Good placements and near misses.
for group in "ABC":
    for task_id, task in ((6, "t06"), (8, "t08")):
        run = json.loads((CL / "full_10k" / group / task / "run.json").read_text())
        for episode in run["episodes"]:
            folder = CL / "full_10k" / group / task / f"{task}_{episode['episode_id']}"
            replies = [json.loads(l) for l in (folder / "queries.jsonl").read_text().splitlines()]
            replies = [r for r in replies if r.get("event") == "response"]
            scores = {}
            for line in (folder / "oracle_scores.jsonl").read_text().splitlines():
                row = json.loads(line)
                scores[row["frame"]] = row["confirmed_goals"]
            for goal in range(1, len(GOALS[task_id]) + 1):
                tick = next((r["frame"] for r in replies if (r["response"].get("certified_count") or 0) >= goal), None)
                first = next((f for f in sorted(scores) if scores[f] >= goal), None)
                if tick is not None and first is not None and first <= tick + 20 and abs(first - tick) <= 40:
                    add("good", group, task_id, episode["episode_id"], folder, goal, tick)
for run_name in ("full_6k", "full_8k", "full_10k"):
    for path in (CL / run_name / "anatomy").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "error" in row or row["tick"] is None or not 3 <= row["rest_xy"] < 8 or row["rest_dz"] > 3:
                continue
            group, task, demo = row["run"].split()
            task_id = int(task[1:])
            add("near_miss", group, task_id, demo, CL / run_name / group / task / f"{task}_{demo}", row["goal"], row["tick"])
# Injected failures that the model ticked anyway.
rows = json.loads((INJ / "summary.json").read_text())["rows"]
for row in rows:
    if row.get("valid") and row.get("false_tick"):
        task_id = int(row["task"][1:])
        add(row["kind"], row["group"], task_id, row["demo"], INJ / row["kind"] / row["group"] / row["task"] / f"{row['task']}_{row['demo']}", row["goal"], row["tick"])

x = np.stack([s[4] for s in samples])
out = {"kinds": np.array([s[0] for s in samples]), "groups": np.array([s[1] for s in samples]),
       "demos": np.array([f"t{s[3]:02d}_{s[2]}" for s in samples]), "tasks": np.array([s[3] for s in samples]),
       "raw": x.reshape(len(x), -1), "features": np.stack([s[5] for s in samples]),
       "verbs": np.asarray([s[6] for s in samples], np.float32)}
for name in "BC":
    with open(HOME / f"pi05-abc/latents_probe_params_{name}.pkl", "rb") as stream:
        params = pickle.load(stream)
    encoder = Encoder(nnx.Rngs(0))
    graph, state = nnx.split(encoder)
    state.replace_by_pure_dict(params)
    encoder = nnx.merge(graph, state)
    stats = json.loads((HOME / f"pi05-abc/latents_probe/{name}/standardize.json").read_text())
    z = np.asarray(encoder(jnp.asarray(x)))
    scaled = (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8) * 8.0 - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    out[f"z{name}"] = scaled / (np.linalg.norm(scaled, axis=1, keepdims=True) + 1e-8) * 8.0
np.savez_compressed(Path("xxxx/pi05-abc/probe.npz"), **out)
kinds, counts = np.unique(out["kinds"], return_counts=True)
print("PROBE_SAVED", dict(zip(kinds.tolist(), counts.tolist())), flush=True)
