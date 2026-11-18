"""Shared helpers for the tracker-variant comparison (offline, demos only, RGB + proprioception; no simulator state)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path("/data/lgd/rma/trackv")
LR = Path("/data/lgd/rma/lr_eps")
DS = Path("/data/lgd/rma/full/dataset_causal/episodes")
CAM = json.load(open("/pfs/pfs-7jnepv/lgd/pi05-abc/rma/camera26.json"))
WORST = [17, 11, 14, 12, 13, 9, 8, 18, 25, 26, 23, 3, 19, 1]
FIRST = [17, 11, 8, 25, 19, 1]
MIN_PIXELS = 20


def episodes():
    import os
    f = ROOT / os.environ.get("TV_EPS", "eps.json")
    if f.exists():
        return json.loads(f.read_text())
    strict = json.load(open("/data/lgd/rma/trackq/strict.json"))
    eps = []
    for task in WORST:
        paths = sorted(DS.glob(f"t{task:02d}_seed*.npz"))
        E = strict[str(task)]
        i = max(range(len(E)), key=lambda i: E[i]["A"] + E[i]["B"] + E[i]["Dg"])
        eps.append(paths[i].stem)
    for task in FIRST:
        stem = sorted(DS.glob(f"t{task:02d}_seed*.npz"))[0].stem
        if stem not in eps:
            eps.append(stem)
    f.write_text(json.dumps(eps))
    return eps


def load(ep):
    rgb = np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")
    m = np.load(LR / "meta" / f"{ep}.npz")
    n = min(len(rgb), len(m["states"]))
    return np.ascontiguousarray(rgb[:n]), m["states"][:n].astype(np.float64), m["actions"][:n].astype(np.float64)


def task_of(ep):
    return int(ep[1:3])


def eef_pixels(ep, states):
    P = np.asarray(CAM[str(task_of(ep))]["matrix"])
    h = np.c_[states[:, :3], np.ones(len(states))] @ P.T
    return h[:, :2] / h[:, 2:3]


def project(ep, points):
    P = np.asarray(CAM[str(task_of(ep))]["matrix"])
    h = np.c_[points, np.ones(len(points))] @ P.T
    return h[:, :2] / h[:, 2:3]


def centres_from_masks(masks):
    """masks bool [N, K, H, W] -> centres [N, K, 2] (nan where not visible), visibility [N, K]"""
    n, k = masks.shape[:2]
    centres = np.full((n, k, 2), np.nan)
    vis = np.zeros((n, k), bool)
    ys, xs = np.mgrid[0:masks.shape[2], 0:masks.shape[3]]
    for t in range(n):
        for j in range(k):
            m = masks[t, j]
            c = int(m.sum())
            if c >= MIN_PIXELS:
                centres[t, j] = (xs[m].mean(), ys[m].mean())
                vis[t, j] = True
    return centres, vis


def save_masks(method, ep, masks, **extra):
    out = ROOT / "out" / method
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{ep}.npz", masks=np.packbits(masks, axis=-1), shape=np.asarray(masks.shape), **extra)


def load_masks(method, ep):
    z = np.load(ROOT / "out" / method / f"{ep}.npz")
    shape = tuple(z["shape"])
    return np.unpackbits(z["masks"], axis=-1)[..., :shape[-1]].astype(bool).reshape(shape), z
