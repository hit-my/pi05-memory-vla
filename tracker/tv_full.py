"""Re-track every demonstration with the chosen method: first-frame masks as today (clicks -> SAM2.1 -> CUTIE frame 0),
then the SAM3 tracker frame by frame, then "seen only + gripper veto".
Output xxxx/rma/trackv/all/sam3veto/<ep>.npz: centres/visibility (after veto), raw_centres/raw_visibility (SAM3 as is),
area [N,K], events. Usage: tv_full.py SHARD NSHARDS [reverse]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
from rma_sam3_stream import Sam3Stream  # noqa: E402
from tv_common import LR, ROOT, centres_from_masks, load, load_masks, task_of  # noqa: E402
from tv_fuse import observe_only  # noqa: E402
from tv_render import robot_masks  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
out = ROOT / "all" / "sam3veto"
out.mkdir(parents=True, exist_ok=True)
eps = sorted(p.stem for p in (LR / "meta").glob("t*_seed*.npz"))
order = [e for i, e in enumerate(eps) if i % shards == shard]
if len(sys.argv) > 3:
    order = order[::-1]
stream = Sam3Stream()
named = None
for ep in order:
    if (out / f"{ep}.npz").exists():
        continue
    start = time.time()
    rgb, states, actions = load(ep)
    n = len(rgb)
    task = task_of(ep)
    k = len(rma_spec.TASKS[task]["objects"])
    if (ROOT / "out/cutie" / f"{ep}.npz").exists():
        first = load_masks("cutie", ep)[0][0]
    else:
        if named is None:
            from rma_tracker import NamedTracker
            from tracker_core import Options
            named = NamedTracker("xxxx/pi05-abc/tracker/assets", Options(selection="detector", max_mem_frames=10))
        _, labels = named.initialize_named(rgb[0].copy(), rma_spec.detector_objects(task))
        first = np.stack([labels == j + 1 for j in range(k)])
    masks = np.zeros((n, k, 256, 256), bool)
    ms = stream.reset(rgb[0], [first[j] for j in range(k)])
    for j in range(k):
        masks[0, j] = ms[j]
    for t in range(1, n):
        ms = stream.step(rgb[t])
        for j in range(k):
            masks[t, j] = ms[j]
    raw_c, raw_v = centres_from_masks(masks)
    c, v, events = observe_only(ep, masks, robot_masks(ep, states), rgb, states, actions)
    np.savez_compressed(out / f"{ep}.npz.tmp.npz", centres=c.astype(np.float32), visibility=v, raw_centres=raw_c.astype(np.float32), raw_visibility=raw_v,
                        area=masks.reshape(n, k, -1).sum(-1).astype(np.int32), first=np.packbits(first, axis=-1), events=json.dumps(events))
    (out / f"{ep}.npz.tmp.npz").rename(out / f"{ep}.npz")
    print(f"{ep} frames {n} objects {k} drawn {100 * v.mean():.0f}% (raw {100 * raw_v.mean():.0f}%) {(time.time() - start) / 60:.1f} min", flush=True)
print("FULL_DONE", shard)
