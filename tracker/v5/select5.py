"""Per-task choice of the rule set on the selection half (A), check on the held-out half (B).
select5.py [tasks]  -> v5/exp/select_<tag>.json (all candidates' scores on A; the chosen one on A, B, all) and prints the table.
Candidates = rule variants x retirement triggers. Choice: lowest errors/100 on A; 'raw' is preferred when it is within 0.03 of the
best (fewest rules), then variants without retirement, then by name order."""
import json
import sys
from multiprocessing import Pool

import numpy as np

import ev5
import tv_fuse_v5
from run5 import feat

TASKS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [t for t in range(1, 27)]
TAG = sys.argv[2] if len(sys.argv) > 2 else "main"
CFG = {
    "raw": {"mode": "raw"},
    "raw_col20": {"mode": "raw", "raw_colour_tol": 20.0},
    "raw_col30": {"mode": "raw", "raw_colour_tol": 30.0},
    "raw_col45": {"mode": "raw", "raw_colour_tol": 45.0},
    "v5": {},
    "v5_nodup": {"dup_frac": 0.0},
    "v5_nopush": {"push": False},
    "v5_nodup_nopush": {"dup_frac": 0.0, "push": False},
    "v5_near18": {"near_px": 18.0},
    "v5_nodup_near18": {"dup_frac": 0.0, "near_px": 18.0},
    "v5_holdpre": {"hold_rest": "pre"},
    "v5_nodup_holdpre": {"dup_frac": 0.0, "hold_rest": "pre"},
    "v5_nodup_nopush_holdpre": {"dup_frac": 0.0, "push": False, "hold_rest": "pre"},
    "v5_nodup_hold": {"dup_frac": 0.0, "hold_rel": True},
}
# tasks 18/19 (re-tracked in v5): the rule sets combined with three modifiers
#   bigarea : the area rule only above 500 px (objects that start half hidden behind each other show 3x their first area later)
#   min8    : masks down to 8 px count (default 20)
#   blob    : the largest connected part of a mask stands for the mask
CFG["v5_holddup"] = {"hold_dup": True}
CFG["v5_holdpre_holddup"] = {"hold_rest": "pre", "hold_dup": True}
BASE19 = ("raw", "v5", "v5_nodup", "v5_holdpre", "v5_holddup", "v5_holdpre_holddup")
MODS = (("bigarea", {"area_floor": 500}), ("min8", {"min_area": 8}), ("blob", {"blob": "largest"}))
SMALL, _order = {}, []
for _mask in range(1, 8):
    _mods = [m for i, m in enumerate(MODS) if _mask >> i & 1]
    for _n in BASE19:
        if _n == "raw" and any(m[0] == "bigarea" for m in _mods):
            continue
        _cfg = dict(CFG[_n])
        for _, d in _mods:
            _cfg.update(d)
        _name = _n + "".join("_" + m[0] for m in _mods)
        SMALL[_name] = _cfg
        _order.append((len(_mods), BASE19.index(_n), _name))
CFG.update(SMALL)
PREF = ["raw", "v5", "v5_nodup", "v5_nopush", "v5_nodup_nopush", "v5_holdpre", "v5_nodup_holdpre", "v5_nodup_nopush_holdpre", "v5_near18", "v5_nodup_near18",
        "raw_col45", "raw_col30", "raw_col20", "v5_nodup_hold", "v5_holddup", "v5_holdpre_holddup"]
PREF += [name for _, _, name in sorted(_order)]
RET = {
    "none": {},
    "A": {"drawer": "close_end", "microwave": "place_open", "basket": "place_end"},
    "B": {"drawer": "close_move20", "microwave": "place_open", "basket": "place_end"},
    "C": {"microwave": "close_end"},
}
CONTAINER = {1: "basket", 2: "basket", 3: "basket", 4: "drawer", 5: "drawer", 11: "drawer", 12: "drawer", 13: "drawer", 14: "drawer", 17: "drawer",
             20: "microwave", 21: "microwave", 22: "microwave", 23: "microwave", 24: "microwave"}


def variants(task):
    out = []
    for c in CFG:
        if c in SMALL and task not in (18, 19):
            continue
        hold = c.endswith("hold")
        if hold and CONTAINER.get(task) != "microwave":
            continue
        for r in RET:
            if r != "none" and task not in CONTAINER:
                continue
            if r == "C" and CONTAINER.get(task) != "microwave":
                continue
            if hold and r != "C":
                continue
            if r == "B" and CONTAINER.get(task) != "drawer":
                continue
            out.append((c, r))
    return out


def one(a):
    ep, vs = a
    g = ev5.G(ep)
    if not g.ok:
        return ep, None
    F = feat(ep)
    n = len(F["v"])
    task = int(ep[1:3])
    out = {}
    M = [None]
    for c, r in vs:
        cfg = dict(CFG[c]); cfg["_no_constants"] = True
        ret = tv_fuse_v5.retired_from_labels_v5(ep, n, RET[r]) if RET[r] else None
        if tv_fuse_v5.needs_masks(cfg) and M[0] is None:
            M[0] = tv_fuse_v5.load_masks(ep)
        cc, vv, _ = tv_fuse_v5.observe_from_features(task, F, retired=ret if ret is not None and ret.any() else None, cfg=cfg,
                                                     masks=M[0] if tv_fuse_v5.needs_masks(cfg) else None)
        out[f"{c}|{r}"] = ev5.score(g, cc, vv)
    return ep, out


if __name__ == "__main__":
    res = {}
    with Pool(28) as p:
        jobs = [(e, variants(t)) for t in TASKS for e in ev5.task_eps(t, "A")]
        RA = dict(p.map(one, jobs, chunksize=2))
        chosen = {}
        for t in TASKS:
            sc = {}
            for c, r in variants(t):
                acc = {}
                for ep in ev5.task_eps(t, "A"):
                    if RA.get(ep):
                        ev5.add(acc, RA[ep][f"{c}|{r}"])
                sc[f"{c}|{r}"] = ev5.summary(acc)
            best = min(sc.values(), key=lambda s: s["err100"])["err100"]
            # the simplest candidate within 0.03 of the best: fewest switches first (order of PREF), then the earliest retirement variant
            good = [k for k in sc if sc[k]["err100"] <= best + 0.03]
            order = sorted(good, key=lambda k: (PREF.index(k.split("|")[0]), list(RET).index(k.split("|")[1])))
            chosen[t] = order[0]
            res[t] = dict(A_all=sc, chosen=order[0])
            top = sorted(sc, key=lambda k: sc[k]["err100"])[:6]
            print(f"task {t:2d} chosen {order[0]:28s} A {sc[order[0]]['err100']:.2f} | raw {sc['raw|none']['err100']:.2f} | top: " + " ".join(f"{k}={sc[k]['err100']:.2f}" for k in top), flush=True)
        jobs = [(e, [tuple(chosen[t].split("|"))]) for t in TASKS for e in ev5.task_eps(t, "B")]
        RB = dict(p.map(one, jobs, chunksize=2))
        jobsr = [(e, [("raw", "none")]) for t in TASKS for e in ev5.task_eps(t, "B")]
        RR = dict(p.map(one, jobsr, chunksize=2))
    for t in TASKS:
        acc, accr = {}, {}
        for ep in ev5.task_eps(t, "B"):
            if RB.get(ep):
                ev5.add(acc, RB[ep][chosen[t]]); ev5.add(accr, RR[ep]["raw|none"])
        res[t]["B"] = ev5.summary(acc); res[t]["B_raw"] = ev5.summary(accr)
        c, r = chosen[t].split("|")
        res[t]["cfg"] = CFG[c]; res[t]["retire"] = RET[r]
        s, sr = res[t]["B"], res[t]["B_raw"]
        print(f"task {t:2d} {chosen[t]:28s} HELD-OUT err {s['err100']:.2f} (raw {sr['err100']:.2f}) w10 {s['w10']:.2f} wrong {s['wrong']:.2f} hid {s['dhf']:.2f} miss {s['miss']:.2f}"
              f" | {'OK' if s['err100'] <= 1.0 and s['err100'] <= sr['err100'] + 0.1 and s['w10'] >= 97 else 'MISS'}")
    json.dump(res, open(ev5.TV / f"v5/exp/select_{TAG}.json", "w"), indent=1)
