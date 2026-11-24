"""Recompute OUT/tracks from the stored raw masks with the current reporting rules of real_tracker.py (CPU, seconds).

  real_rules.py RAW [first [last]]
Reads RAW/masks/<ep>.npz (written by real_track.py), RAW/front/<ep>.npy and RAW/meta/<ep>.npz; rewrites RAW/tracks/<ep>.npz.
The result is what real_track.py would write with the same rules: both call ObjectTracker._report frame by frame.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402
from real_tracker import ObjectTracker  # noqa: E402


def main():
    raw = Path(sys.argv[1])
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    last = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 6
    done = 0
    for path in sorted((raw / "masks").glob("t*_seed*.npz"), key=lambda p: int(p.stem.split("seed")[1])):
        ep = path.stem
        if not first <= int(ep.split("seed")[1]) <= last:
            continue
        z = np.load(path)
        front = np.load(raw / "front" / f"{ep}.npy", mmap_mode="r")
        closed = np.load(raw / "meta" / f"{ep}.npz")["closed"]
        unpack = lambda a: np.unpackbits(a, axis=-1).astype(bool)
        n, k = len(z["masks"]), len(S.NAMES)
        tracker = ObjectTracker.offline()
        tracker.start(unpack(z["first"]), np.asarray(front[0]))
        centres, area = np.full((n, k, 2), np.nan, np.float32), np.zeros((n, k), np.int32)
        for t in range(n):
            centres[t] = tracker.replay(unpack(z["masks"][t]), bool(closed[t - 1]) if t else False, front[t], first=t == 0)
            area[t] = tracker.area
        old = np.load(raw / "tracks" / f"{ep}.npz")
        visible = np.isfinite(centres).all(-1)
        np.savez_compressed(raw / "tracks" / f"{ep}.npz", centres=centres, visibility=visible, retired=np.zeros((n, k), bool),
                            area=area, det=str(old["det"]), repairs=json.dumps(tracker.events))
        done += 1
        print("RULES", ep, n, "visible", visible.mean(0).round(2).tolist(), tracker.events, flush=True)
    print("RULES_DONE", done)


if __name__ == "__main__":
    main()