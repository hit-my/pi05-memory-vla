"""real_qa.py RAW: object tracks against what the task physically requires (no model, no GPU).
Objects 0-2 = far / middle / near cup, 3-5 = block 0 / 1 / 2. For every episode:
  * start order: cups top to bottom, blocks bottom to top;
  * an object that is NOT being carried does not move: outside its own grasp (plus SETTLE frames after the release) every
    reported centre is within STAY px of where it stands, and the path it draws there is shorter than PATH px in total
    (STAY allows the one correction of its standing centre when the gripper moves off it; PATH bounds everything it draws);
  * cup i moves more than MOVE px during grasp i and is back within BACK px of its start afterwards;
  * the block equal to the count moves during grasp 4 and ends more than MOVE px from its start; the other two never move.
Prints failing episodes and statistics; exit code 1 if any episode fails."""
import json, sys
from pathlib import Path
import numpy as np
MOVE, STAY, BACK, PATH, SETTLE = 20.0, 15.0, 30.0, 30.0, 46  # SETTLE: the tracker lets a released object settle for up to 45 frames
raw = Path(sys.argv[1]); names = sorted((p.stem for p in (raw / "tracks").glob("t*_seed*.npz")), key=lambda s: int(s.split("seed")[1]))
bad, vis, paths = [], [], []
for ep in names:
    z, m = np.load(raw / "tracks" / f"{ep}.npz"), np.load(raw / "meta" / f"{ep}.npz"); c, n = z["centres"].astype(float), len(z["centres"])
    ev, count = [int(e) for e in m["events"]], int(m["count"]); seen = np.isfinite(c).all(-1); why = []
    grasp = {0: 0, 1: 1, 2: 2, 3 + count: 3}  # object -> the grasp that carries it
    def med(k, a, b):
        t = np.arange(max(0, a), min(n, b)); t = t[seen[t, k]] if len(t) else t
        return np.median(c[t, k], 0) if len(t) else np.array([np.nan, np.nan])
    start = [med(k, 0, max(5, ev[0] - 10)) for k in range(6)]
    if not (start[0][1] < start[1][1] < start[2][1]) or not (start[3][1] > start[4][1] > start[5][1]): why.append("start order")
    for k in range(6):
        g = grasp.get(k)
        periods = [(0, n)] if g is None else [(0, ev[2 * g] - 3), (ev[2 * g + 1] + SETTLE, n)]
        for a, b in periods:
            t = np.arange(max(0, a), max(0, min(n, b))); t = t[seen[t, k]] if len(t) else t
            if len(t) < 3: continue
            rest = np.median(c[t, k], 0); far = t[np.hypot(*(c[t, k] - rest).T) > STAY]; length = float(np.hypot(*np.diff(c[t, k], axis=0).T).sum()); paths.append(length)
            if len(far): why.append(f"object {k} is {np.hypot(*(c[far, k] - rest).T).max():.0f}px off while standing (frames {int(far[0])}..{int(far[-1])}, {len(far)})")
            elif length > PATH: why.append(f"object {k} draws {length:.0f}px while standing in {a}..{b}")
        if g is not None:
            t = np.arange(ev[2 * g], ev[2 * g + 1]); t = t[seen[t, k]]
            moved = float(np.hypot(*(c[t, k] - start[k]).T).max()) if len(t) else float("nan")
            if not moved > MOVE: why.append(f"object {k} did not move in its own grasp ({moved:.0f}px)")
            after = med(k, min(ev[2 * g + 1] + SETTLE, n - 6), n)
            if not np.isfinite(after).all():  # the episode ends with the gripper still over it: where it was last seen
                t = np.arange(ev[2 * g], n); t = t[seen[t, k]]; after = c[t[-1], k] if len(t) else after
            d = float(np.hypot(*(after - start[k])))
            if k < 3 and not d < BACK: why.append(f"cup {k} ends {d:.0f}px from its start")
            if k >= 3 and not d > MOVE: why.append(f"block {k - 3} ends only {d:.0f}px from its start")
    vis.append(seen.mean(0))
    if why: bad.append((ep, why))
print("episodes", len(names), "failing", len(bad), "visibility mean", np.mean(vis, 0).round(3).tolist(), "min", np.min(vis, 0).round(3).tolist(),
      "path drawn while standing: median", round(float(np.median(paths)), 1), "max", round(float(np.max(paths)), 1))
steps = [p for p in paths if p > 0]
print("standing periods", len(paths), "with any drawn path", len(steps), "of those: median", round(float(np.median(steps)), 1) if steps else 0, "px, over 15 px:", sum(p > 15 for p in steps))
for ep, why in bad: print("  FAIL", ep, "; ".join(why[:4])[:330])
sys.exit(1 if bad else 0)