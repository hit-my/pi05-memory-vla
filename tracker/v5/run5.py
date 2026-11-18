"""run5.py EXP.json HALF [tasks]   EXP.json: {variant: {"cfg": {...}, "retire": {...}, "tasks": [..] (optional)}}
prints per task one line per variant; saves exp/<EXP>_<HALF>.json. Features: v5/feat5/<ep>.npz if present else v4 feat."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ev5
import tv_fuse_v5

if __name__ == "__main__":
    exp = json.load(open(sys.argv[1]))
    HALF = sys.argv[2]
    TASKS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(1, 27))
FEAT5 = ev5.TV / "v5/feat5"


def feat(ep):
    p = FEAT5 / f"{ep}.npz"
    return dict(np.load(p)) if p.exists() else ev5.feat(ep)


def one(ep):
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    F = feat(ep)
    n = len(F["v"])
    task = int(ep[1:3])
    out = {}
    for name, var in exp.items():
        if "tasks" in var and task not in var["tasks"]:
            continue
        cfg = dict(var.get("cfg", {})); cfg["_no_constants"] = True
        ret = tv_fuse_v5.retired_from_labels_v5(ep, n, var.get("retire"))
        c, v, _ = tv_fuse_v5.observe_from_features(task, F, retired=ret if ret.any() else None, cfg=cfg)
        s = ev5.score(g, c, v)
        s["ep_err"] = s["wrong"] + s["dh_far"] + s["miss30"]
        out[name] = s
    return ep, out


if __name__ == "__main__":
    eps = [e for t in TASKS for e in ev5.task_eps(t, HALF)]
    with Pool(28) as p:
        res = p.map(one, eps, chunksize=2)
    table = {}
    for t in TASKS:
        print(f"--- task {t}")
        for name in exp:
            acc = {}
            worst = []
            for ep, r in res:
                if r is None or int(ep[1:3]) != t or name not in r:
                    continue
                ev5.add(acc, r[name])
                worst.append((r[name]["ep_err"], ep[4:]))
            if not acc:
                continue
            cells = acc["cells"]
            cs = " ".join(f"{ck}:{cv[1]}/{cv[2]}/{cv[3]}" for ck, cv in sorted(cells.items()) if cv[1] + cv[2] + cv[3])
            worst.sort(reverse=True)
            print(ev5.line(name, acc), "|", cs, "| worst", " ".join(f"{e}:{n}" for n, e in worst[:4]))
            table.setdefault(str(t), {})[name] = ev5.summary(acc)
    Path(ev5.TV / "v5/exp").mkdir(exist_ok=True)
    json.dump(table, open(ev5.TV / f"v5/exp/{Path(sys.argv[1]).stem}_{HALF}.json", "w"))
