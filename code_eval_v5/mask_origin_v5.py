"""mask_origin_v5.py ep [ep ...] -> JSON {ep: origin} : where the stored masks behind all/final_v5 were computed.
all/final was tracked partly on this node (NVIDIA B20Z) and partly on cloud GPUs (cloud_track.sh: A100, RTX Pro 6000 x2,
RTX 6000D); a cloud result was copied into all/final only when no local result existed, so the origin of all/final/<ep>.npz
is the cloud host whose all/cloud_<host>/<ep>.npz is byte-identical to it, else local. v5/track(_split) (tasks 18/19) and
all/final_fix (25/26, re-tracked objects) were computed locally; unchanged objects of final_fix are copies from all/final."""
import filecmp
import json
import sys
from pathlib import Path

TV = Path("xxxx/rma/trackv")
out = {}
for ep in sys.argv[1:]:
    if (TV / "v5/track_split" / f"{ep}.npz").exists() or (TV / "v5/track" / f"{ep}.npz").exists():
        out[ep] = "local(v5/track)"
        continue
    base = "local"
    for d in sorted(TV.glob("all/cloud_*")):
        p = d / f"{ep}.npz"
        if p.exists() and filecmp.cmp(p, TV / "all/final" / f"{ep}.npz", shallow=False):
            base = d.name
    out[ep] = base + ("+final_fix" if (TV / "all/final_fix" / f"{ep}.npz").exists() else "")
print(json.dumps(out))
