"""Per-frame outcome word for each bowl, from observable signals: held, on-plate, off-plate (or none if unseen).

Written as OUT/<episode>.npz: states[N,K] of int8 (0 none, 1 held, 2 on-plate, 3 off-plate). Checked against
simulator contact and support on normal episodes, where the replay physics exists.
Usage: state_words.py OUT
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HOME = Path.home()
INPUTS = HOME / "pi05-ledger-A/inputs"
DATASET = HOME / "pi05-abc/dataset"
PHYSICS = HOME / "pi05-ledger-A/physics"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
index = {e["name"]: e for e in json.loads((INPUTS / "index.json").read_text())["episodes"]}
camera = json.loads((HOME / "pi05-ledger-A/probe/camera_probe.json").read_text())
PLATE_PX, HAND_PX, CLOSED = 30.0, 35.0, 0.03


def project(points, task_id):
    cam = camera[str(task_id)]
    K, E = np.asarray(cam["K"]), np.asarray(cam["cam_to_world"])
    w2c = np.linalg.inv(E)
    pc = w2c[:3, :3] @ points.T + w2c[:3, 3:4]
    uv = (K @ pc)[:2] / pc[2:3]
    return np.stack([uv[0] - 0.5, uv[1] - 0.5], -1)


def episode_states(z, meta):
    reg = meta["registry"]
    sids = [str(s) for s in z["sids"]]
    bowls = sorted([s for s in sids if reg[s]["category"] == "bowl"], key=lambda s: reg[s]["segmentation_id"])
    plates = [s for s in sids if reg[s]["category"] == "plate"]
    centres, visible = z["centres"].astype(float), z["visibility"].astype(bool)
    states_8 = z["states"].astype(float)
    eef = project(states_8[:, :3], meta["task_id"])
    width = states_8[:, 6] - states_8[:, 7]
    plate_at = []
    for p in plates:
        j = sids.index(p)
        if visible[:, j].any():
            plate_at.append(centres[int(np.argmax(visible[:, j])), j])
    plate_at = np.asarray(plate_at)
    table = np.zeros((len(centres), len(bowls)), np.int8)
    for k, sid in enumerate(bowls):
        j = sids.index(sid)
        for t in np.flatnonzero(visible[:, j]):
            point = centres[t, j]
            held = width[t] < CLOSED and np.linalg.norm(point - eef[t]) < HAND_PX and t >= 1
            near = np.min(np.linalg.norm(plate_at - point, axis=1)) <= PLATE_PX
            # A bowl resting on a plate under a closed hand reads as on-plate until it actually lifts.
            if held and near and t >= 5 and np.linalg.norm(point - centres[max(0, t - 5), j]) < 3:
                held = False
            table[t, k] = 1 if held else 2 if near else 3
    return table


paths = [(p.stem, p, index[p.stem]) for p in sorted(INPUTS.glob("t0*.npz"))]
for directory in ("failures", "failures2"):
    paths += [(p.stem, p, index[re.sub(r"_(empty_hand|drop)\d*$", "", p.stem)]) for p in sorted((DATASET / directory / "episodes").glob("*.npz"))]
agree = Counter()
for number, (name, path, meta) in enumerate(paths):
    with np.load(path) as z:
        table = episode_states(z, meta)
    np.savez_compressed(out / f"{name}.npz", states=table)
    physics_path = PHYSICS / f"{name}.npz"
    if physics_path.is_file():
        p = np.load(physics_path)
        truth = np.where(p["bowl_grip"] & (p["bowl_support"] == -1), 1, np.where((p["bowl_support"] >= 1) & (p["bowl_support"] < 100), 2, 3))
        mask = table > 0
        for pred, real in zip(table[mask], truth[mask]):
            agree[(int(pred), int(real))] += 1
    if number % 100 == 0:
        print(number, "/", len(paths), flush=True)
total = sum(agree.values())
print("agreement with physics (predicted, truth):", {str(k): v for k, v in sorted(agree.items())})
print("accuracy", round(sum(v for k, v in agree.items() if k[0] == k[1]) / total, 4))
for word in (1, 2, 3):
    tp = agree[(word, word)]
    print(word, "precision", round(tp / max(sum(v for k, v in agree.items() if k[0] == word), 1), 3), "recall", round(tp / max(sum(v for k, v in agree.items() if k[1] == word), 1), 3))
print("STATE_WORDS_DONE", len(paths), flush=True)
