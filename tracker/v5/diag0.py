"""baseline breakdown: raw / v4 noretire / v4 retire / v4 handle per task, by phase cell. diag0.py [half]"""
import json
import sys
from multiprocessing import Pool

import numpy as np

import ev5

HALF = sys.argv[1] if len(sys.argv) > 1 else "A"
VARS = {"raw": ("raw_centres", "raw_visibility"), "v4nr": ("centres_noretire", "visibility_noretire"), "v4r": ("centres", "visibility"), "v4h": ("centres_handle", "visibility_handle")}


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    z = np.load(ev5.TV / "all/final_v4" / f"{ep}.npz")
    return ep, {name: ev5.score(g, z[a], z[b]) for name, (a, b) in VARS.items()}


if __name__ == "__main__":
    eps = [e for t in range(1, 27) for e in ev5.task_eps(t, HALF)]
    with Pool(24) as p:
        res = p.map(one, eps, chunksize=4)
    out = {}
    for t in range(1, 27):
        acc = {v: {} for v in VARS}
        for ep, r in res:
            if r is None or int(ep[1:3]) != t:
                continue
            for v in VARS:
                ev5.add(acc[v], r[v])
        print(f"--- task {t}")
        for v in VARS:
            cells = acc[v]["cells"]
            cs = " ".join(f"{ck}:{cv[0]//1000}k/{cv[1]}/{cv[2]}/{cv[3]}" for ck, cv in sorted(cells.items()) if cv[0])
            print(ev5.line(v, acc[v]), "|", cs)
        out[t] = {v: ev5.summary(acc[v]) for v in VARS}
    json.dump(out, open(ev5.TV / f"v5/diag0_{HALF}.json", "w"))
