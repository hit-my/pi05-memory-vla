"""Cloud worker for the final tracker (same computation as tv_full_b.py): polls inbox/<ep>.ready, tracks, writes
all/final/<ep>.npz, then deletes that episode's frames. Usage: tv_cloud.py SHARD NSHARDS"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
import rma_spec  # noqa: E402
from rma_sam3_stream import Sam3Stream  # noqa: E402
from tv_common import LR, ROOT, centres_from_masks, load, task_of  # noqa: E402
from tv_fuse import observe_only  # noqa: E402
from tv_render import robot_masks  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
inbox, out = ROOT / "inbox", ROOT / "all" / "final"
out.mkdir(parents=True, exist_ok=True)
stream = Sam3Stream()
print("READY", shard, flush=True)
while not (ROOT / "STOP").exists():
    todo = sorted(p.stem for p in inbox.glob("*.ready") if zlib.crc32(p.stem.encode()) % shards == shard)
    if not todo:
        time.sleep(8)
        continue
    ep = todo[0]
    start = time.time()
    try:
        if not (out / f"{ep}.npz").exists():
            rgb, states, actions = load(ep)
            n = len(rgb)
            k = len(rma_spec.TASKS[task_of(ep)]["objects"])
            lab = np.load(ROOT / "initB" / f"{ep}.npy")
            first = np.stack([lab == j + 1 for j in range(k)])
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
            np.savez_compressed(out / f"{ep}.npz.tmp.npz", centres=c.astype(np.float32), visibility=v, raw_centres=raw_c.astype(np.float32),
                                raw_visibility=raw_v, area=masks.reshape(n, k, -1).sum(-1).astype(np.int32), first=np.packbits(first, axis=-1),
                                masks=np.packbits(masks, axis=-1), events=json.dumps(events))
            (out / f"{ep}.npz.tmp.npz").rename(out / f"{ep}.npz")
            print(f"{ep} frames {n} objects {k} drawn {100 * v.mean():.0f}% (raw {100 * raw_v.mean():.0f}%) {(time.time() - start) / 60:.1f} min", flush=True)
    except Exception as error:  # keep the worker alive; the episode stays for the lab node
        print("FAILED", ep, repr(error)[:300], flush=True)
        (inbox / f"{ep}.failed").write_text(repr(error))
    (inbox / f"{ep}.ready").unlink(missing_ok=True)
    (LR / "front" / f"{ep}.npy").unlink(missing_ok=True)
print("STOPPED", shard, flush=True)
