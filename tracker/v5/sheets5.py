"""Contact sheets for the user: per task the median and the worst held-out episode by v5 error (errors per 100 valid object-frames of
the episode); zoomed strips of the object with most errors: GT box (green), v5 dot (red), raw SAM3 dot (yellow ring), gripper (cyan +).
sheets5.py OUTDIR [tasks]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ev5
import render5

OUT = Path(sys.argv[1])
TASKS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 27))
V5 = ev5.TV / "all/final_v5"


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    z = np.load(V5 / f"{ep}.npz")
    s = ev5.score(g, z["centres"], z["visibility"], per_frame=True)
    per = [int((s["flags"][:, k] > 0).sum()) for k in range(g.K)]
    return ep, dict(err=100.0 * (s["wrong"] + s["dh_far"] + s["miss30"]) / max(1, s["n"]), per=per)


def rend(a):
    ep, k, tag, err = a
    z = np.load(V5 / f"{ep}.npz")
    name = f"t{ep[1:3]}_{tag}_{ep[4:]}_obj{k}.jpg"
    render5.strip(ep, k, z["centres"], z["visibility"], z["raw_centres"], z["raw_visibility"], OUT / name,
                  title=f"task {int(ep[1:3])} {tag} held-out episode, v5 err {err:.2f}/100 | red = v5, yellow ring = raw SAM3", count=10)
    return name


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(26) as p:
        jobs, index = [], {}
        for t in TASKS:
            R = [(ep, r) for ep, r in p.map(one, ev5.task_eps(t, "B")) if r is not None]
            R.sort(key=lambda x: x[1]["err"])
            med, worst = R[len(R) // 2], R[-1]
            for tag, (ep, r) in (("median", med), ("worst", worst)):
                k = int(np.argmax(r["per"]))
                jobs.append((ep, k, tag, r["err"]))
            index[t] = dict(median=[med[0], round(med[1]["err"], 3)], worst=[worst[0], round(worst[1]["err"], 3)], episodes=len(R),
                            episodes_with_zero_errors=sum(1 for _, r in R if r["err"] == 0))
            print(t, index[t], flush=True)
        names = p.map(rend, jobs)
    json.dump(index, open(OUT / "index.json", "w"), indent=1)
    print(len(names), "sheets")
