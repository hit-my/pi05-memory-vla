"""Tasks 18/19 frame-0 selection by appearance (v5).
OFFLINE (selection half only, uses GT to label): positives of object k = stored SAM3 track masks at frames 10..30 that GT confirms
are on object k (any of the tracked slots; the old first-frame choice was often on another tracked object), plus frame-0
detections that sit on object k; class 'other' = frame-0 detections on nothing tracked / merged blobs.
A softmax classifier on mask appearance (ff_lib.feat) is stored as JSON = the appearance prototypes.
RUN TIME (ff_select): frame-0 detections -> class probabilities -> one detection per object (assignment). No positions used.
ff_an2.py POOL(B|Z|BZ) TAG"""
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

import ev5
import ff_lib
import rma_spec

POOL = sys.argv[1] if len(sys.argv) > 1 else "BZ"
TAG = sys.argv[2] if len(sys.argv) > 2 else "a"
DET5 = str(ev5.TV / "v5/det5")
FR = (10, 15, 20, 25, 30)


def cands(ep):
    if POOL == "B":
        return ff_lib.candidates(ep)
    if POOL == "Z":
        return ff_lib.candidates(ep, DET5, only_extra=True)
    return ff_lib.candidates(ep, DET5)


def gt_label(c, a, bb, ar):
    """index of the tracked object a mask (centroid c, area a) sits on at frame 0 (GT boxes of frame 10, extended upwards by 6 px
    because the objects are still dropping in frame 0), -1 = none, -2 = on an object but wrong size (part / merged)"""
    hit = -1
    for k in range(len(bb)):
        b = bb[k]
        if np.isfinite(b).all() and b[0] - 2 <= c[0] <= b[2] + 2 and b[1] - 6 <= c[1] <= b[3] + 2:
            if 0.45 * ar[k] <= a <= 1.9 * ar[k] + 10:
                return k
            hit = -2
    return hit


def one(ep):
    g = np.load(ev5.GT / f"{ep}.npz")
    q = ev5.qa().get(ep, {})
    rgbs = np.load(f"xxxx/rma/lr_eps/front/{ep}.npy", mmap_mode="r")
    cand, sc = cands(ep)
    K = len(g["objects"])
    bb, ar = g["bbox"][10], g["mask_area"][10]
    rgb0 = np.ascontiguousarray(rgbs[0])
    X = np.array([ff_lib.feat(rgb0, m) for m in cand]) if len(cand) else np.zeros((0, 24), np.float32)
    lab = np.array([gt_label(ff_lib.centroid(m), m.sum(), bb, ar) for m in cand], int)
    # track-mask positives (selection only; GT-verified)
    TX, TY = [], []
    z = np.load(ev5.TV / "all/final" / f"{ep}.npz")
    for t in FR:
        ms = np.unpackbits(z["masks"][t], axis=-1)[..., :256].astype(bool)
        rgb = np.ascontiguousarray(rgbs[t])
        for j in range(K):
            a = int(ms[j].sum())
            if a < 20:
                continue
            c = ff_lib.centroid(ms[j])
            for k in range(K):
                if g["mask_area"][t, k] >= 20 and np.hypot(*(c - g["mask_centre"][t, k])) < 3 and 0.6 * g["mask_area"][t, k] <= a <= 1.6 * g["mask_area"][t, k] + 8 \
                        and g["robot_dist"][t, k] > 15:
                    TX.append(ff_lib.feat(rgb, ms[j])); TY.append(k)
    return dict(ep=ep, ok=bool(q.get("settled_ok", False)), X=X, lab=lab, score=sc, TX=np.array(TX).reshape(-1, 24), TY=np.array(TY, int),
                cen=np.array([ff_lib.centroid(m) for m in cand]).reshape(-1, 2))


def select(P, K, pmin=0.2):
    """P [M, K+1] class probabilities -> candidate index per object (-1 = none)"""
    sel = [-1] * K
    if len(P) == 0:
        return sel
    cost = -np.log(P[:, :K].T + 1e-9)
    for a, b in zip(*linear_sum_assignment(cost)):
        if P[b, a] >= pmin:
            sel[a] = int(b)
    return sel


if __name__ == "__main__":
    for task in (18, 19):
        K = len(rma_spec.TASKS[task]["objects"])
        names = [o[0] for o in rma_spec.TASKS[task]["objects"]]
        with Pool(24) as p:
            A = p.map(one, ev5.task_eps(task, "A")); B = p.map(one, ev5.task_eps(task, "B"))
        tr = [d for d in A if d["ok"]]
        X = np.concatenate([d["X"][d["lab"] != -2] for d in tr] + [d["TX"] for d in tr])
        y = np.concatenate([np.where(d["lab"][d["lab"] != -2] < 0, K, d["lab"][d["lab"] != -2]) for d in tr] + [d["TY"] for d in tr])
        # merged / part blobs are 'other' too
        Xm = np.concatenate([d["X"][d["lab"] == -2] for d in tr]); X = np.concatenate([X, Xm]); y = np.concatenate([y, np.full(len(Xm), K)])
        print(f"task {task} pool {POOL}: train rows {len(X)} class counts {np.bincount(y, minlength=K + 1).tolist()}; detectable (a right-size candidate on the object) A:",
              [int(sum((d['lab'] == k).any() for d in A if d['ok'])) for k in range(K)], "of", sum(d["ok"] for d in A),
              "B:", [int(sum((d['lab'] == k).any() for d in B if d['ok'])) for k in range(K)], "of", sum(d["ok"] for d in B))
        for C in (0.1, 1.0, 10.0):
            clf = ff_lib.Softmax(C=C).fit(X, y)
            for half, D in (("A", A), ("B", B)):
                right = np.zeros(K, int); none = np.zeros(K, int); n = 0; allok = 0
                bad = []
                for d in D:
                    if not d["ok"]:
                        continue
                    n += 1
                    P = clf.predict_proba(d["X"]) if len(d["X"]) else np.zeros((0, K + 1))
                    sel = select(P, K)
                    okk = [sel[k] >= 0 and d["lab"][sel[k]] == k for k in range(K)]
                    right += np.array(okk); none += np.array([s < 0 for s in sel])
                    allok += all(okk)
                    if not all(okk):
                        bad.append(d["ep"][4:] + ":" + "".join("." if o else ("n" if s < 0 else "x") for o, s in zip(okk, sel)))
                print(f"  C={C} half {half}: episodes {n}; right {dict(zip(names, right.tolist()))}; none {none.tolist()}; all right {allok}")
                if C == 1.0:
                    print("     bad:", " ".join(bad[:50]))
            if C == 1.0:
                json.dump(dict(clf=clf.to_json(), names=names, pool=POOL), open(ev5.TV / f"v5/ff_clf_{task}_{TAG}.json", "w"))
