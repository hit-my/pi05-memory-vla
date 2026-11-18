"""summarize_test_v5.py DIR : per-task table of test_perception_v5.py results (DIR/<ep>.json)"""
import collections
import json
import sys
from pathlib import Path

rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("t*_seed*.json"))]
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
print("task eps frames | masks: frames differ %  | centres: equal eps, frames differ %, max px | visibility frames differ | retired eq | png identical % | tracker ms/frame")
tot = collections.Counter()
for task in sorted(by):
    R = by[task]
    f = sum(r["frames"] for r in R)
    md = sum(r["masks_frames_differ"] for r in R)
    cd = sum(r["centres_frames_differ"] for r in R)
    vd = sum(r["visibility_frames_differ"] for r in R)
    pe = sum(r["png_equal"] for r in R)
    tot.update(frames=f, md=md, cd=cd, vd=vd, pe=pe, eps=len(R), ceq=sum(r["centres_equal"] for r in R), req=sum(r["retired_equal"] for r in R))
    print(f"{task:4d} {len(R):3d} {f:6d} | {100 * md / f:6.1f} | {sum(r['centres_equal'] for r in R)}/{len(R)} {100 * cd / f:6.1f} {max(r['centres_max_px'] for r in R):6.3f} | "
          f"{vd:5d} | {sum(r['retired_equal'] for r in R)}/{len(R)} | {100 * pe / f:6.1f} | {sum(r['ms_per_frame']['tracker'] * r['frames'] for r in R) / f:6.1f}")
origins = json.loads((Path(sys.argv[1]) / "origins.json").read_text()) if (Path(sys.argv[1]) / "origins.json").exists() else {}
if origins:
    print("\nper mask origin (where the stored masks were computed):")
    groups = collections.defaultdict(list)
    for r in rows:
        o = origins.get(r["ep"], "?")
        groups["local" if o.startswith("local") and "final_fix" not in o else o.split("+")[0] if "final_fix" not in o else o].append(r)
    for o, R in sorted(groups.items()):
        f = sum(r["frames"] for r in R)
        print(f"  {o:28s} eps {len(R):3d} frames {f:6d} | masks differ {100 * sum(r['masks_frames_differ'] for r in R) / f:5.1f}% | centres equal eps "
              f"{sum(r['centres_equal'] for r in R)}/{len(R)}, frames differ {100 * sum(r['centres_frames_differ'] for r in R) / f:5.1f}%, max "
              f"{max(r['centres_max_px'] for r in R):.3f} px | visibility frames differ {sum(r['visibility_frames_differ'] for r in R)} | "
              f"png identical {100 * sum(r['png_equal'] for r in R) / f:5.1f}% | eps: {' '.join(r['ep'] for r in R if not r['centres_equal'])[:300]}")
f = tot["frames"]
print(f" ALL {tot['eps']:3d} {f:6d} | {100 * tot['md'] / f:6.1f} | {tot['ceq']}/{tot['eps']} {100 * tot['cd'] / f:6.1f} {max(r['centres_max_px'] for r in rows):6.3f} | "
      f"{tot['vd']:5d} | {tot['req']}/{tot['eps']} | {100 * tot['pe'] / f:6.1f}")
