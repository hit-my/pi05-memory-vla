"""Tasks 18/19 frame-0: label stored SAM3 candidates with GT (offline), learn an appearance classifier on the selection half,
select by classifier + assignment, report first-frame correctness on both halves.  ff_an.py [extra_det_dir]"""
import json
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

import ev5
import ff_lib
import rma_spec

EXTRA = sys.argv[1] if len(sys.argv) > 1 else None
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"


def inside(p, b, pad=2.0):
    return bool(np.isfinite(p).all() and np.isfinite(b).all() and b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad)


def one(ep):
    g = np.load(ev5.GT / f"{ep}.npz")
    q = ev5.qa().get(ep, {})
    rgb = np.load(f"xxxx/rma/lr_eps/front/{ep}.npy", mmap_mode="r")[0]
    cand, sc = ff_lib.candidates(ep, EXTRA)
    K = len(g["objects"])
    kidx = g["kidx"].tolist()
    ab, aa = g["all_bbox"][10], g["all_mask_area"][10]
    X, lab, cen, good = [], [], [], []
    for i in range(len(cand)):
        X.append(ff_lib.feat(rgb, cand[i]))
        c = ff_lib.centroid(cand[i]); bb = ff_lib.bbox(cand[i]); a = cand[i].sum()
        cen.append(c)
        l, best = -1, 0.0
        for j in range(ab.shape[0]):
            if not np.isfinite(ab[j]).all():
                continue
            iou = ff_lib.biou(bb, ab[j])
            if iou > best:
                best, l = iou, j
        strict = best >= 0.5 and 0.5 * aa[l] <= a <= 1.8 * aa[l] + 10
        # label: index into tracked objects (0..K-1), K = other object / background / merged blob
        lab.append(kidx.index(l) if (strict and l in kidx) else K)
        good.append(best)
    return dict(ep=ep, ok=bool(q.get("settled_ok", False)), X=np.array(X), lab=np.array(lab), cen=np.array(cen), score=sc, bb=g["bbox"][10], area=g["mask_area"][10],
                cand=np.packbits(cand, axis=-1))


def select(d, clf, clicks, K, w_click=0.0):
    """returns candidate index per object (or -1)"""
    if len(d["X"]) == 0:
        return [-1] * K
    P = clf.predict_proba(d["X"])  # [M, K+1] columns in clf.classes_ order
    cols = {int(c): i for i, c in enumerate(clf.classes_)}
    cost = np.zeros((K, len(d["X"])))
    for k in range(K):
        lp = np.log(P[:, cols[k]] + 1e-6)
        dist = np.hypot(*(d["cen"] - np.asarray(clicks[k], float)).T)
        cost[k] = -lp + w_click * dist
    sel = [-1] * K
    for a, b in zip(*linear_sum_assignment(cost)):
        if P[b, cols[a]] > 0.05:
            sel[a] = int(b)
    return sel


if __name__ == "__main__":
    out = {}
    for task in (18, 19):
        K = len(rma_spec.TASKS[task]["objects"])
        names = [o[0] for o in rma_spec.TASKS[task]["objects"]]
        with Pool(24) as p:
            A = p.map(one, ev5.task_eps(task, "A")); B = p.map(one, ev5.task_eps(task, "B"))
        tr = [d for d in A if d["ok"] and len(d["X"])]
        X = np.concatenate([d["X"] for d in tr]); y = np.concatenate([d["lab"] for d in tr])
        print(f"task {task}: train candidates {len(X)} label counts {np.bincount(y, minlength=K + 1).tolist()} (last = other); episodes with a strict candidate per object:",
              [int(sum((d['lab'] == k).any() for d in tr)) for k in range(K)], "of", len(tr))
        for C in (0.3, 1.0, 3.0):
            clf = ff_lib.Softmax(C=C).fit(X, y)
            for half, D in (("A", A), ("B", B)):
                right = np.zeros(K, int); strict = np.zeros(K, int); none = np.zeros(K, int); n = 0; allok = 0
                bad = []
                for d in D:
                    if not d["ok"]:
                        continue
                    n += 1
                    sel = select(d, clf, rma_spec.CLICKS[task], K)
                    okk = []
                    for k in range(K):
                        if sel[k] < 0:
                            none[k] += 1; okk.append(False); continue
                        r = inside(d["cen"][sel[k]], d["bb"][k]); right[k] += r; strict[k] += d["lab"][sel[k]] == k
                        okk.append(d["lab"][sel[k]] == k)
                    allok += all(okk)
                    if not all(okk):
                        bad.append(d["ep"][4:] + ":" + "".join("x" if not o else "." for o in okk))
                print(f"  C={C} half {half}: episodes {n}; centroid in GT box {dict(zip(names, right.tolist()))}; strict (bbox IoU>=.5, area ok) {dict(zip(names, strict.tolist()))}; "
                      f"none {none.tolist()}; all strict {allok}")
                if C == 1.0:
                    print("     bad:", " ".join(bad[:40]))
            if C == 1.0:
                json.dump(clf.to_json(), open(ev5.TV / f"v5/ff_clf_{task}_{TAG}.json", "w"))
