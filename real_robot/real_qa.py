"""Quality check of the bowl tracks against what the task physically requires (no model, no GPU).

  real_qa.py RAW
For every episode: object 0 starts right of object 1; only object 0 moves in the first pick-place (to the spare
plate); at the end object 0 stands where object 1 started and object 1 where object 0 started (the swap).
Prints the failing episodes and visibility statistics; exit code 1 if any episode fails.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_spec as S  # noqa: E402

raw = Path(sys.argv[1])
names = sorted((p.stem for p in (raw / "tracks").glob("t*_seed*.npz")), key=lambda s: int(s.split("seed")[1]))
bad, vis, repairs = [], [], {"collision": 0, "reacquired": 0, "split": 0}
for ep in names:
    z, m = np.load(raw / "tracks" / f"{ep}.npz"), np.load(raw / "meta" / f"{ep}.npz")
    c, ev, n = z["centres"], [int(e) for e in m["events"]], len(z["centres"])

    def pos(k, a, b):
        seg = c[a:b, k]
        seg = seg[np.isfinite(seg).all(-1)]
        return np.median(seg, 0) if len(seg) else np.array([np.nan, np.nan])

    start = [pos(k, 0, max(5, ev[0] - 10)) for k in (0, 1)]
    after_first = [pos(k, ev[1] + 8, ev[2] - 3) for k in (0, 1)]
    end = [pos(k, ev[5] + 8, n) for k in (0, 1)]
    moved0, moved1 = np.linalg.norm(after_first[0] - start[0]), np.linalg.norm(after_first[1] - start[1])
    swap0, swap1 = np.linalg.norm(end[0] - start[1]), np.linalg.norm(end[1] - start[0])
    ok = start[0][0] > start[1][0] and moved0 > 25 and moved1 < 12 and swap0 < 20 and swap1 < 20
    vis.append(z["visibility"].mean(0))
    for _, _, kind in json.loads(str(z["repairs"])) if "repairs" in z.files else []:
        repairs[kind] += 1
    if not ok:
        bad.append((ep, [round(float(v), 1) for v in (moved0, moved1, swap0, swap1)]))
vis = np.array(vis)
print("episodes", len(names), "failing", len(bad))
for b in bad:
    print("  FAIL", b, "(moved0, moved1, end0-to-start1, end1-to-start0 in px)")
print("visible fraction per object: mean", vis.mean(0).round(3).tolist(), "min", vis.min(0).round(2).tolist())
print("tracker repairs:", repairs)
sys.exit(1 if bad else 0)
