"""early look at the re-tracked tasks 18/19 (selection half only): raw of the new tracks vs raw of the v4 tracks, per episode.
early1819.py [render N]"""
import json
import sys
from multiprocessing import Pool

import numpy as np

import ev5
import feat5
import tv_fuse_v5

VARS = {"raw": {"mode": "raw"}, "raw_min8": {"mode": "raw", "min_area": 8}, "raw_min8_blob": {"mode": "raw", "min_area": 8, "blob": "largest"},
        "v5_big_hd_min8_blob": {"area_floor": 500, "min_area": 8, "hold_dup": True, "blob": "largest"},
        "v5_big_min8_blob": {"area_floor": 500, "min_area": 8, "blob": "largest"}, "v5": {}, "v5_nodup": {"dup_frac": 0.0}, "v5_holdpre": {"hold_rest": "pre"}, "v5_big": {"area_floor": 500},
        "v5_big_min8": {"area_floor": 500, "min_area": 8}, "v5_big_holdpre_min8": {"area_floor": 500, "min_area": 8, "hold_rest": "pre"},
        "v5_big_hd": {"area_floor": 500, "hold_dup": True}, "v5_big_hd_min8": {"area_floor": 500, "min_area": 8, "hold_dup": True},
        "v5_big_hd_holdpre_min8": {"area_floor": 500, "min_area": 8, "hold_dup": True, "hold_rest": "pre"}}


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    F = feat5.build(ep)
    np.savez(feat5.OUT / f"{ep}.tmp.npz", **F); (feat5.OUT / f"{ep}.tmp.npz").rename(feat5.OUT / f"{ep}.npz")
    out = {}
    M = None
    for name, cfg in VARS.items():
        cfg = dict(cfg); cfg["_no_constants"] = True
        c, v, _ = tv_fuse_v5.observe_from_features(int(ep[1:3]), F, cfg=cfg, masks=M if tv_fuse_v5.needs_masks(cfg) else None)
        out[name] = ev5.score(g, c, v, per_frame=True)
        out[name]["per"] = [[int((out[name]["flags"][:, k] == x).sum()) for x in (1, 2, 3)] for k in range(g.K)]
        del out[name]["flags"]
    z = np.load(ev5.TV / "all/final_v4" / f"{ep}.npz")
    out["old_raw"] = ev5.score(g, z["raw_centres"], z["raw_visibility"])
    info = json.loads(str(np.load(ev5.TV / "v5/track" / f"{ep}.npz")["info"]))
    return ep, (out, info)


if __name__ == "__main__":
    eps = sorted(p.stem for p in (ev5.TV / "v5/track").glob("t1?_seed*.npz") if "tmp" not in p.name)
    eps = [e for e in eps if ev5.half_of(e) == "A"]
    with Pool(24) as p:
        R = [r for r in p.map(one, eps) if r[1] is not None]
    for t in (18, 19):
        rr = [(ep, r) for ep, r in R if int(ep[1:3]) == t]
        if not rr:
            continue
        print(f"--- task {t}: {len(rr)} selection-half episodes re-tracked")
        for name in ["old_raw"] + list(VARS):
            acc = {}
            for ep, (o, info) in rr:
                ev5.add(acc, o[name])
            print(ev5.line(name, acc), "|", " ".join(f"{ck}:{cv[1]}/{cv[2]}/{cv[3]}" for ck, cv in sorted(acc["cells"].items()) if cv[1] + cv[2] + cv[3]))
        SHOW = sys.argv[1] if len(sys.argv) > 1 else "raw"
        for ep, (o, info) in sorted(rr, key=lambda x: -(x[1][0][SHOW]["wrong"] + x[1][0][SHOW]["dh_far"] + x[1][0][SHOW]["miss30"])):
            s = o[SHOW]
            print(f"   {ep} {SHOW} err {s['wrong'] + s['dh_far'] + s['miss30']:5d} of {s['n']} per-object w/h/m {s['per']} | acquired {info['acquired']} p_first {[None if x is None else round(x, 2) for x in info['p_first']]} "
                  f"grasp_fix {info.get('grasp_fix')} dropped {info.get('dropped')}")
