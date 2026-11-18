"""Per-frame measurements for the v5 rules = tv_fuse_v5.frame_features5 (v4 measurements + mask bounding box), cached.
Mask source per episode: v5/track_split/<ep>.npz (track5b.py, split check) > v5/track/<ep>.npz (track5.py) > all/final_fix (v4 re-track) > all/final.
feat5.py SHARD NSHARDS [task,task..] [force]"""
import sys
from pathlib import Path

import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "v5"))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
from tv_common_v4 import eef_pixels, load  # noqa: E402
import tv_fuse_v5  # noqa: E402

OUT = HERE / "v5/feat5"
FLOOR = {18: 8, 19: 8}


def src(ep):
    for d in ("v5/track_split", "v5/track", "all/final_fix", "all/final"):
        p = HERE / d / f"{ep}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(ep)


def build(ep):
    p = src(ep)
    z = np.load(p)
    masks = np.unpackbits(z["masks"], axis=-1)[..., :256].astype(bool)
    rgb, states, actions = load(ep)
    n = len(masks)
    assert n == len(rgb), (ep, n, len(rgb))
    e = eef_pixels(ep, states)
    floor = FLOOR.get(int(ep[1:3]), 20)  # tasks 18/19: masks down to 8 px are measured (the rule threshold min_area is applied by the rules)
    rows = [tv_fuse_v5.frame_features5(masks[t], rgb[t], e[t], tv_fuse_v5.capsule(ep, states[t]), floor, blobs=int(ep[1:3]) in FLOOR) for t in range(n)]
    F = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    F.update(e=e, closed=np.r_[False, actions[:-1, 6] > 0], width=states[:, 6], src=str(p.parent.relative_to(HERE)))
    return F


if __name__ == "__main__":
    shard, shards = int(sys.argv[1]), int(sys.argv[2])
    tasks = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 and sys.argv[3] != "all" else None
    force = len(sys.argv) > 4
    OUT.mkdir(parents=True, exist_ok=True)
    eps = sorted(p.stem for p in (HERE / "all/final").glob("t*_seed*.npz") if "tmp" not in p.name)
    done = 0
    for i, ep in enumerate(eps):
        if i % shards != shard or (tasks and int(ep[1:3]) not in tasks):
            continue
        if (OUT / f"{ep}.npz").exists() and not force:
            continue
        F = build(ep)
        np.savez(OUT / f"{ep}.tmp.npz", **F)
        (OUT / f"{ep}.tmp.npz").rename(OUT / f"{ep}.npz")
        done += 1
    print("FEAT5_DONE", shard, done, flush=True)
