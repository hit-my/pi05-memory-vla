"""w5.py EXP.json VARIANT OUTDIR tasks [N] [HALF] [ep:k,...]: worst (episode, object) of a variant on a half, strips (A = variant, B = raw) + events"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ev5
import render5
import tv_fuse_v5
from run5 import feat

exp = json.load(open(sys.argv[1]))
var = exp[sys.argv[2]]
out = Path(sys.argv[3])
tasks = [int(x) for x in sys.argv[4].split(",")]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 2
HALF = sys.argv[6] if len(sys.argv) > 6 else "A"
FORCE = sys.argv[7].split(",") if len(sys.argv) > 7 else []


def run(ep):
    F = feat(ep)
    n = len(F["v"])
    cfg = dict(var.get("cfg", {})); cfg["_no_constants"] = True
    ret = tv_fuse_v5.retired_from_labels_v5(ep, n, var.get("retire"))
    c, v, ev = tv_fuse_v5.observe_from_features(int(ep[1:3]), F, retired=ret if ret.any() else None, cfg=cfg, masks=tv_fuse_v5.load_masks(ep) if tv_fuse_v5.needs_masks(cfg) else None)
    return F, c, v, ev, ret


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    F, c, v, ev, ret = run(ep)
    s = ev5.score(g, c, v, per_frame=True)
    return ep, [[int((s["flags"][:, k] == x).sum()) for x in (1, 2, 3)] for k in range(g.K)]


def rend(a):
    ep, k = a
    F, c, v, ev, ret = run(ep)
    g = ev5.G(ep)
    fl = ev5.score(g, c, v, per_frame=True)["flags"][:, k]
    runs = []
    bad = np.flatnonzero(fl > 0)
    if len(bad):
        s = bad[0]; p = bad[0]
        for b in bad[1:]:
            if b - p > 3:
                runs.append((int(s), int(p), int(fl[s]))); s = b
            p = b
        runs.append((int(s), int(p), int(fl[s])))
    render5.strip(ep, k, c, v, F["c"], F["v"], out / f"{ep}_o{k}.jpg", title=sys.argv[2], g=g, e=F["e"], count=15)
    rt = [int(np.argmax(ret[:, j])) if ret[:, j].any() else -1 for j in range(ret.shape[1])]
    cl = np.flatnonzero(np.diff(F["closed"].astype(int)) != 0) + 1
    return f"{ep} o{k}: events {[(int(t), a, int(j)) for t, a, j in ev]} retired_at {rt} gripper toggles {cl.tolist()} error runs (start,end,type 1w/2h/3m) {runs[:14]}"


if __name__ == "__main__":
    out.mkdir(parents=True, exist_ok=True)
    eps = [e for t in tasks for e in ev5.task_eps(t, HALF)]
    with Pool(28) as p:
        jobs = []
        if FORCE:
            jobs = [(x.split(":")[0], int(x.split(":")[1])) for x in FORCE]
        else:
            res = [r for r in p.map(one, eps, chunksize=2) if r[1] is not None]
            for t in tasks:
                rr = sorted([(sum(c), ep, k, c) for ep, r in res if int(ep[1:3]) == t for k, c in enumerate(r)], reverse=True)
                print(f"task {t} top:", " ".join(f"{ep[4:]}/o{k}:{c[0]}w+{c[1]}h+{c[2]}m" for _, ep, k, c in rr[:10]), "| >20 errs:", sum(1 for x in rr if x[0] > 20), "of", len(rr))
                jobs += [(ep, k) for _, ep, k, c in rr[:N]]
        for line in p.map(rend, jobs):
            print(line)
