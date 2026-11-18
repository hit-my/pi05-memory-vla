"""Final v5 outputs for every demo -> all/final_v5/<ep>.npz
  centres / visibility      the per-task choice of constants_v5.json (rules + retirement where chosen)
  raw_centres / raw_visibility / area   SAM3 centroids of the masks used
  retired [N,K]             retirement flags fed to the rules (offline: from the subtask labels = checklist stand-in)
  events (json), config (json: cfg, retire, chosen, first_frame), mask_source (directory of the masks, relative to trackv/)
The rules run on the cached measurements v5/feat5 (test_veto_v5.py checks this path == masks path == online class).
v5full.py SHARD NSHARDS [task,task|all] [force]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "v5"))
import tv_fuse_v5  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
tasks = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 and sys.argv[3] != "all" else None
force = len(sys.argv) > 4
out = HERE / "all" / "final_v5"
out.mkdir(parents=True, exist_ok=True)
CONST = tv_fuse_v5.load_constants()
eps = sorted(p.stem for p in (HERE / "all/final").glob("t*_seed*.npz") if "tmp" not in p.name)
done = 0
for i, ep in enumerate(eps):
    task = int(ep[1:3])
    if i % shards != shard or (tasks and task not in tasks) or ((out / f"{ep}.npz").exists() and not force):
        continue
    F = dict(np.load(HERE / "v5/feat5" / f"{ep}.npz"))
    n = len(F["v"])
    tc = CONST["tasks"][str(task)]
    ret = tv_fuse_v5.retired_from_labels_v5(ep, n, tc["retire"])
    cfg_t = tv_fuse_v5.config_for(task)
    c, v, events = tv_fuse_v5.observe_from_features(task, F, retired=ret if ret.any() else None, masks=tv_fuse_v5.load_masks(ep) if tv_fuse_v5.needs_masks(cfg_t) else None)
    raw_v = F["v"] & (F["area"] >= 20)  # raw = the SAM3 centroid of masks with >= 20 px (same definition for every task)
    raw_c = np.where(raw_v[..., None], F["c"], np.nan)
    np.savez_compressed(out / f"{ep}.tmp.npz", centres=c.astype(np.float32), visibility=v, raw_centres=raw_c.astype(np.float32), raw_visibility=raw_v,
                        area=F["area"].astype(np.int32), retired=ret, events=json.dumps([(int(t), a, int(j)) for t, a, j in events]),
                        config=json.dumps(dict(cfg=tv_fuse_v5.config_for(task), retire=tc["retire"], chosen=tc["chosen"], first_frame=tc["first_frame"])),
                        mask_source=str(F["src"]))
    (out / f"{ep}.tmp.npz").rename(out / f"{ep}.npz")
    done += 1
print("V5FULL_DONE", shard, done, flush=True)
