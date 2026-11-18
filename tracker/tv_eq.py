"""Online FusedTracks (rma_fuse.py, fed with the dataset's tracker centroids) vs the batch tracks of tv_all.py (all/fuse)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rma_fuse import FusedTracks  # noqa: E402
from tv_common import CAM, DS, LR, ROOT, task_of  # noqa: E402

files = sorted((LR / "meta").glob("t*_seed*.npz"))[::int(sys.argv[1]) if len(sys.argv) > 1 else 40]
worst, mism, total = 0.0, 0, 0
for path in files:
    ep = path.stem
    m, z, b = np.load(path), np.load(DS / f"{ep}.npz"), np.load(ROOT / "all/fuse" / f"{ep}.npz")
    n = min(len(z["centres"]), len(m["states"]))
    rgb = np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")
    c, v = z["centres"][:n].astype(float), z["visibility"][:n].astype(bool)
    k = v.shape[1]
    tr = FusedTracks(CAM[str(task_of(ep))]["matrix"])
    oc, ov = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    for t in range(n):
        cur = [c[t, j].tolist() if v[t, j] else None for j in range(k)]
        cc, vv = tr.reset(cur, rgb[0], m["states"][0]) if t == 0 else tr.step(cur, rgb[t], m["states"][t], m["actions"][t - 1, 6] > 0)
        for j in range(k):
            if cc[j] is not None:
                oc[t, j] = cc[j]
            ov[t, j] = vv[j]
    d = float(np.nanmax(np.abs(np.nan_to_num(oc[1:]) - np.nan_to_num(b["centres"][1:n].astype(float)))))
    worst = max(worst, d)
    mism += int((ov[1:] != b["visibility"][1:n]).sum())
    total += ov[1:].size
print(f"{len(files)} episodes: worst position difference {worst:.4f} px, visibility mismatches {mism} of {total}")
