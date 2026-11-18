"""worst selection-half episodes per task for a key pair, per object, and render them. worst0.py KEYC KEYV OUTDIR task[,task] [n]"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ev5
import render5

kc, kv, out = sys.argv[1:4]
tasks = [int(x) for x in sys.argv[4].split(",")]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 2
D = sys.argv[6] if len(sys.argv) > 6 else str(ev5.TV / "all/final_v4")


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    z = np.load(f"{D}/{ep}.npz")
    s = ev5.score(g, z[kc], z[kv], per_frame=True)
    return ep, [[int((s["flags"][:, k] == c).sum()) for c in (1, 2, 3)] for k in range(g.K)]


def rend(a):
    ep, k = a
    z = np.load(f"{D}/{ep}.npz")
    render5.strip(ep, k, z[kc], z[kv], z["raw_centres"], z["raw_visibility"], Path(out) / f"{ep}_o{k}_{kc}.jpg", title=kc)


if __name__ == "__main__":
    Path(out).mkdir(parents=True, exist_ok=True)
    eps = [e for t in tasks for e in ev5.task_eps(t, "A")]
    with Pool(24) as p:
        res = [r for r in p.map(one, eps) if r[1] is not None]
        jobs = []
        for t in tasks:
            rr = [(sum(c), ep, k, c) for ep, r in res if int(ep[1:3]) == t for k, c in enumerate(r)]
            rr.sort(reverse=True)
            tot = sum(x[0] for x in rr)
            print(f"task {t} total errs {tot}; top:", " ".join(f"{ep[4:]}/o{k}:{c[0]}w+{c[1]}h+{c[2]}m" for _, ep, k, c in rr[:8]),
                  "| eps with >20 errs:", sum(1 for x in rr if x[0] > 20), "of", len(rr))
            jobs += [(ep, k) for _, ep, k, c in rr[:N]]
        p.map(rend, jobs)
