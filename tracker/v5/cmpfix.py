"""track5 vs track5b masks for the episodes where a track5b check fired; error/100 with the chosen rule set; by check type and half"""
import json
from multiprocessing import Pool
import numpy as np
import ev5, feat5, tv_fuse_v5

CFG = {18: {"mode": "raw", "min_area": 8, "_no_constants": True}, 19: {"hold_dup": True, "area_floor": 500, "_no_constants": True}}


def build(ep, d):
    orig = feat5.src
    feat5.src = lambda e: ev5.TV / d / f"{e}.npz"
    try:
        return feat5.build(ep)
    finally:
        feat5.src = orig


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    r = {}
    for d in ("v5/track", "v5/track_fix"):
        F = build(ep, d)
        c, v, _ = tv_fuse_v5.observe_from_features(int(ep[1:3]), F, cfg=CFG[int(ep[1:3])])
        s = ev5.score(g, c, v)
        r[d] = (s["wrong"] + s["dh_far"] + s["miss30"], s["n"])
    return ep, r


if __name__ == "__main__":
    eps = sorted(p.stem for p in (ev5.TV / "v5/track_fix").glob("t1?_seed*.npz") if "tmp" not in p.name)
    reason = {ep: json.load(open(ev5.TV / "v5/scan" / f"{ep}.json"))["reasons"][0][0] for ep in eps}
    with Pool(23) as p:
        R = dict(p.map(one, eps))
    tot = {}
    for ep in eps:
        if R[ep] is None:
            print(ep, "GT bad"); continue
        a, b = R[ep]["v5/track"], R[ep]["v5/track_fix"]
        h = ev5.half_of(ep)
        print(f"{ep} half {h} {reason[ep]:8s} errors track5 {a[0]:5d}  track5b {b[0]:5d}  (of {a[1]})")
        key = (h, reason[ep])
        t = tot.setdefault(key, [0, 0, 0]); t[0] += a[0]; t[1] += b[0]; t[2] += 1
    for k, v in sorted(tot.items()):
        print("half", k[0], k[1], "episodes", v[2], "errors track5", v[0], "track5b", v[1])
