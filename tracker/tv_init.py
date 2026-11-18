"""Frame-0 localisation of each named object: fixed click (current) vs Grounding DINO text detection.
Truth per episode = the gripper pixel at the object's first grasp (clicks26.json, refined blob centre when available).
Usage: tv_init.py PER_TASK  -> init/init.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
from rma_tracker import NamedTracker  # noqa: E402
from tracker_core import Options  # noqa: E402
from tv_common import LR, ROOT  # noqa: E402

per_task = int(sys.argv[1])
clicks = json.load(open("/pfs/pfs-7jnepv/lgd/pi05-abc/rma/clicks26.json"))["episodes"]
tracker = NamedTracker("/data/lgd/pi05-abc/tracker/assets", Options(selection="detector", max_mem_frames=10))
rows = []
by_task = {}
for ep in sorted(clicks):
    by_task.setdefault(int(ep[1:3]), []).append(ep)
for task, eps in sorted(by_task.items()):
    for ep in eps[::max(1, len(eps) // per_task)][:per_task]:
        truth = [None if p is None else (p[2:] if len(p) == 4 else p[:2]) for p in clicks[ep]]
        rgb = np.ascontiguousarray(np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")[0])
        objects = [{"prompts": list(prompts), "max_area": rma_spec.MAX_AREA} for _, _, prompts in rma_spec.TASKS[task]["objects"]]
        found = tracker._best_boxes(rgb, objects)
        for k, t in enumerate(truth):
            if t is None:
                continue
            fixed = rma_spec.CLICKS[task][k]
            box = found[k]["box"]
            dino = None if box is None else [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            rows.append(dict(ep=ep, task=task, obj=k, truth=t, fixed=float(np.hypot(fixed[0] - t[0], fixed[1] - t[1])),
                             dino=None if dino is None else float(np.hypot(dino[0] - t[0], dino[1] - t[1])), score=found[k]["score"]))
    part = [r for r in rows if r["task"] == task]
    print(f"task {task:2d} n={len(part):3d}  fixed click within 12px {100 * np.mean([r['fixed'] < 12 for r in part]):5.1f}%   "
          f"DINO within 12px {100 * np.mean([r['dino'] is not None and r['dino'] < 12 for r in part]):5.1f}%  (no box {100 * np.mean([r['dino'] is None for r in part]):4.1f}%)", flush=True)
    (ROOT / "init").mkdir(exist_ok=True)
    (ROOT / "init" / "init.json").write_text(json.dumps(rows))
print(f"ALL fixed {100 * np.mean([r['fixed'] < 12 for r in rows]):.1f}%  DINO {100 * np.mean([r['dino'] is not None and r['dino'] < 12 for r in rows]):.1f}%")
