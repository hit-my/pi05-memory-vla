"""v5 evaluation helpers: episode lists, split, GT scoring identical to gt/score_gt.py (+ the 'errors per 100' number).
GT is read HERE only (scoring); nothing in the tracker / rule code imports this module."""
import json
import sys
from pathlib import Path

import numpy as np

TV = Path("/data/lgd/rma/trackv")
GT = Path("/data/lgd/rma/gt")
sys.path.insert(0, str(TV)); sys.path.insert(0, str(TV / "v4")); sys.path.insert(0, str(TV / "code")); sys.path.insert(0, str(TV / "v5"))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
_QA = None


def qa():
    global _QA
    if _QA is None:
        _QA = json.load(open(GT / "qa.json"))
    return _QA


def all_eps():
    return sorted(p.stem for p in (TV / "all/final").glob("t*_seed*.npz") if "tmp" not in p.name)


def task_eps(task, half="AB"):
    """half: 'A' = selection (even positions of the sorted list), 'B' = held-out (odd positions), 'AB' = all"""
    eps = sorted(p.stem for p in (TV / "all/final").glob(f"t{task:02d}_seed*.npz") if "tmp" not in p.name)
    if half == "A":
        return eps[0::2]
    if half == "B":
        return eps[1::2]
    return eps


def half_of(ep):
    eps = task_eps(int(ep[1:3]))
    return "A" if eps.index(ep) % 2 == 0 else "B"


def phases(held):
    n = len(held)
    h = held.copy()
    idx = np.flatnonzero(held)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 6:
            h[a:b] = True
    run = 0; first = None
    for i in range(n):
        run = run + 1 if h[i] else 0
        if run >= 3:
            first = i - 2
            break
    ph = np.zeros(n, np.int8)
    if first is None:
        return ph
    ph[first:] = 2
    ph[first:][h[first:]] = 1
    return ph


class G:
    """ground truth of one episode (scoring only)"""

    def __init__(self, ep):
        q = qa().get(ep)
        self.ep = ep
        self.ok = q is not None and q["status"] not in ("bad", "error")
        g = np.load(GT / f"{ep}.npz")
        self.mc = g["mask_centre"].astype(np.float64); self.area = g["mask_area"]; self.bbox = g["bbox"].astype(np.float64)
        self.pc = g["proj_centre"].astype(np.float64); self.held = g["held"]; self.rd = g["robot_dist"]
        self.grip = g["grip_px"]
        self.n = len(self.mc)
        self.valid = np.ones(self.n, bool)
        for a, b in (q["bad_ranges"] if q else []):
            self.valid[a:b] = False
        self.K = self.mc.shape[1]
        self.ph = np.stack([phases(self.held[:, k]) for k in range(self.K)], 1)


KEYS = ("n", "vis8", "vis30", "drawn", "both", "w10", "on", "wrong", "dh", "dh_far", "miss8", "miss30", "hidden")


def score(g, C, V, per_frame=False):
    """returns dict of counts summed over objects (and optionally the per-frame error flags [n,K] int8:
    1 wrong, 2 drawn-hidden-far, 3 missing30)"""
    n = min(len(C), g.n)
    out = {k: 0 for k in KEYS}
    cells = {}
    flags = np.zeros((n, g.K), np.int8)
    for k in range(g.K):
        c, v = np.asarray(C[:n, k], np.float64), np.asarray(V[:n, k], bool)
        valid = g.valid[:n]
        drawn = v & np.isfinite(c).all(1) & valid
        a = g.area[:n, k]
        vis8, vis30 = (a >= 8) & valid, (a >= 30) & valid
        with np.errstate(invalid="ignore"):
            d = np.sqrt(((c - g.mc[:n, k]) ** 2).sum(1))
            b = g.bbox[:n, k]
            on = (c[:, 0] >= b[:, 0] - 2) & (c[:, 0] <= b[:, 2] + 2) & (c[:, 1] >= b[:, 1] - 2) & (c[:, 1] <= b[:, 3] + 2)
            dp = np.sqrt(((c - g.pc[:n, k]) ** 2).sum(1))
            near = g.rd[:n, k] <= 15
        both = vis8 & drawn
        wrong = both & (d > 10) & ~on
        dh = drawn & ~(a >= 8)
        dhf = dh & ~(dp <= 10)
        m8, m30 = vis8 & ~drawn, vis30 & ~drawn
        flags[wrong, k] = 1; flags[dhf, k] = 2; flags[m30, k] = 3
        vals = dict(n=valid.sum(), vis8=vis8.sum(), vis30=vis30.sum(), drawn=drawn.sum(), both=both.sum(), w10=(both & (d <= 10)).sum(), on=(both & on).sum(),
                    wrong=wrong.sum(), dh=dh.sum(), dh_far=dhf.sum(), miss8=m8.sum(), miss30=m30.sum(), hidden=(valid & ~(a >= 8)).sum())
        for key, x in vals.items():
            out[key] += int(x)
        ph = g.ph[:n, k]
        for p in range(3):
            for nr in (0, 1):
                m = (ph == p) & (near == bool(nr)) & valid
                acc = cells.setdefault(f"{'bca'[p]}{'fn'[nr]}", [0, 0, 0, 0])
                acc[0] += int(m.sum()); acc[1] += int((m & wrong).sum()); acc[2] += int((m & dhf).sum()); acc[3] += int((m & m30).sum())
    out["cells"] = cells
    if per_frame:
        out["flags"] = flags
    return out


def add(acc, s):
    for k in KEYS:
        acc[k] = acc.get(k, 0) + s[k]
    c = acc.setdefault("cells", {})
    for ck, cv in s["cells"].items():
        a = c.setdefault(ck, [0, 0, 0, 0])
        for i in range(4):
            a[i] += cv[i]
    return acc


def summary(acc):
    f = lambda a, b: 100.0 * a / b if b else float("nan")
    return dict(err100=f(acc["wrong"] + acc["dh_far"] + acc["miss30"], acc["n"]), w10=f(acc["w10"], acc["both"]), wrong=f(acc["wrong"], acc["n"]),
                dhf=f(acc["dh_far"], acc["n"]), miss=f(acc["miss30"], acc["n"]), wrong_sc=f(acc["wrong"], acc["both"]), miss_v=f(acc["miss30"], acc["vis30"]),
                dh_hid=f(acc["dh"], acc["hidden"]), n=acc["n"])


def line(name, acc):
    s = summary(acc)
    return f"{name:>14} err {s['err100']:6.2f} | w10 {s['w10']:6.2f} | wrong {s['wrong']:5.2f} hid {s['dhf']:5.2f} miss {s['miss']:5.2f} | n {s['n']}"


def feat(ep):
    from v4feat import feat_path
    return dict(np.load(feat_path(ep)))
