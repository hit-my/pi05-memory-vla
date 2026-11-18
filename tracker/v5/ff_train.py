"""Tasks 18/19: appearance classifier for the first-frame selection (v5), learned OFFLINE on the selection half.
Training rows: zoomed SAM3 detections (det5.py) on frame FT of the selection-half episodes, labelled with the GT boxes of the
same frame (object k if bbox IoU >= 0.5; 'other' if IoU < 0.2 with every tracked object; in between = dropped).
Evaluation: selection (ff_lib.select_objects) on both halves at frame FE; 'right' = the chosen detection has bbox IoU >= 0.4 with
the GT box of that object and its centroid is inside the box.
ff_train.py DET_TRAIN_DIR FT DET_EVAL_DIR FE TAG"""
import json
import sys
from multiprocessing import Pool

from pathlib import Path

import numpy as np

import ev5
import ff_lib
import rma_spec

DT, FT, DE, FE, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
JOINT = len(sys.argv) > 6
BETA = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0


def rows(ep, d, fr):
    g = np.load(ev5.GT / f"{ep}.npz")
    q = ev5.qa().get(ep, {})
    rgb = np.ascontiguousarray(np.load(f"/data/lgd/rma/lr_eps/front/{ep}.npy", mmap_mode="r")[fr])
    cand, sc = ff_lib.candidates(ep, d, only_extra=True)
    K = len(g["objects"])
    bb = g["bbox"][max(fr, 2)]
    X = np.array([ff_lib.feat(rgb, m) for m in cand]).reshape(-1, 24)
    iou = np.array([[ff_lib.biou(ff_lib.bbox(m), bb[k]) if np.isfinite(bb[k]).all() else 0.0 for k in range(K)] for m in cand]).reshape(-1, K)
    ar = g["mask_area"][max(fr, 2)]
    # "on object k": the detection lies (>= 80 %) inside the GT box of k, is at least a quarter of the object and not larger than it
    on = np.array([[bool(np.isfinite(bb[k]).all() and ff_lib.contain(ff_lib.bbox(m), bb[k]) >= 0.8 and 0.25 * ar[k] <= m.sum() <= 1.5 * ar[k] + 8) for k in range(K)] for m in cand]).reshape(-1, K)
    touch = np.array([[bool(np.isfinite(bb[k]).all() and ff_lib.contain(ff_lib.bbox(m), bb[k]) > 0.05) for k in range(K)] for m in cand]).reshape(-1, K)
    cen = np.array([ff_lib.centroid(m) for m in cand]).reshape(-1, 2)
    inside = np.array([[bool(np.isfinite(bb[k]).all() and bb[k][0] - 1 <= c[0] <= bb[k][2] + 1 and bb[k][1] - 1 <= c[1] <= bb[k][3] + 1) for k in range(K)] for c in cen]).reshape(-1, K)
    bad = any(a <= fr < b for a, b in q.get("bad_ranges", []))
    gc = g["mask_centre"][max(fr, 2)]
    pd = [float(np.hypot(*(gc[a] - gc[b]))) for a in range(K) for b in range(a + 1, K) if np.isfinite(gc[a]).all() and np.isfinite(gc[b]).all()]
    return dict(cen=cen, masks=np.packbits(cand, axis=-1), pd=pd, ep=ep, ok=bool(q.get("settled_ok", False)) and q.get("status") != "bad" and not bad, X=X, iou=iou, inside=inside, on=on, touch=touch, score=sc, area=g["mask_area"][max(fr, 2)])


def tr_one(ep):
    # DT may list several detection runs "dir:frame,dir:frame" (rows of all of them are used; only dirs that hold the episode)
    if ":" not in DT:
        return [rows(ep, DT, FT)]
    out = []
    for part in DT.split(","):
        d, fr = part.rsplit(":", 1)
        if (Path(d) / f"{ep}.npz").exists():
            out.append(rows(ep, d, int(fr)))
    return out


def ev_one(ep):
    return rows(ep, DE, FE)


if __name__ == "__main__":
    out = {}
    for task in (18, 19):
        K = len(rma_spec.TASKS[task]["objects"])
        names = [o[0] for o in rma_spec.TASKS[task]["objects"]]
        with Pool(24) as p:
            TR = [d for ds in p.map(tr_one, ev5.task_eps(task, "A")) for d in ds]
            EA = p.map(ev_one, ev5.task_eps(task, "A")); EB = p.map(ev_one, ev5.task_eps(task, "B"))
        X, y = [], []
        for d in TR:
            if not d["ok"] or not len(d["X"]):
                continue
            for i in range(len(d["X"])):
                ks = np.flatnonzero(d["on"][i])
                if len(ks):
                    X.append(d["X"][i]); y.append(int(ks[np.argmax(d["iou"][i, ks])]))
                else:  # merged blobs, parts of the cabinet, the robot, ...
                    X.append(d["X"][i]); y.append(K)
        X, y = np.array(X), np.array(y, int)
        pds = np.array([x for d in TR if d["ok"] for x in d["pd"]])
        DMAX = float(np.ceil(np.percentile(pds, 98))) if len(pds) else 25.0
        ATYP = [float(np.median([d["area"][k] for d in TR if d["ok"] and d["area"][k] >= 30])) for k in range(K)]
        print("typical area", ATYP)
        print("pairwise GT centre distance on A: median %.1f p98 %.1f max %.1f -> dmax %.0f" % (np.median(pds), np.percentile(pds, 98), pds.max(), DMAX))
        print(f"task {task}: train rows {len(X)} class counts {np.bincount(y, minlength=K + 1).tolist()}")
        for C in (0.3, 3.0):
            clf = ff_lib.Softmax(C=C).fit(X, y)
            for pmin in (0.5, 0.7, 0.8, 0.9):
                for half, D in (("A", EA), ("B", EB)):
                    right = np.zeros(K, int); none = np.zeros(K, int); wrong = np.zeros(K, int); vis = np.zeros(K, int); none_vis = np.zeros(K, int); n = 0; allok = 0
                    ratio = []
                    bad = []
                    for d in D:
                        if not d["ok"]:
                            continue
                        n += 1
                        P = clf.predict_proba(d["X"]) if len(d["X"]) else np.zeros((0, K + 1))
                        mk = np.unpackbits(d["masks"], axis=-1)[..., :256].astype(bool)
                        sel = ff_lib.select_joint(P, d["cen"], mk, K, pmin, DMAX, atyp=ATYP, beta=BETA) if JOINT else ff_lib.select_objects(P, K, pmin)
                        for k in range(K):
                            if sel[k] >= 0 and d["area"][k] >= 30:
                                ratio.append(float(mk[sel[k]].sum()) / d["area"][k])
                        okk = []
                        for k in range(K):
                            v = d["area"][k] >= 30
                            vis[k] += v
                            if sel[k] < 0:
                                none[k] += 1; none_vis[k] += v; okk.append(not v)
                            else:
                                r = d["on"][sel[k], k] and d["inside"][sel[k], k]
                                right[k] += r; wrong[k] += not r; okk.append(bool(r))
                        allok += all(okk)
                        if not all(okk):
                            bad.append(d["ep"][4:] + ":" + "".join("." if o else ("n" if s < 0 else "x") for o, s in zip(okk, sel)))
                    ex = np.zeros(K, int)
                    for d in D:
                        if d["ok"]:
                            ex += np.array([bool(len(d["on"]) and d["on"][:, k].any()) for k in range(K)])
                    rr = np.array(ratio)
                    print(f"  [exists {ex.tolist()}; chosen area / GT area: <0.4 in {int((rr < 0.4).sum())}, 0.4-1.5 in {int(((rr >= 0.4) & (rr <= 1.5)).sum())}, >1.5 in {int((rr > 1.5).sum())}]", end="")
                    print(f"  C={C} pmin={pmin} half {half}: eps {n}; right {right.tolist()} wrong {wrong.tolist()} none {none.tolist()} (none although visible>=30: {none_vis.tolist()}); visible {vis.tolist()}; all ok {allok}")
                    if C == 3.0 and pmin == 0.5:
                        print("     bad:", " ".join(bad[:60]))
            if C == 3.0:
                json.dump(dict(clf=clf.to_json(), names=names, dmax=DMAX, atyp=ATYP, beta=BETA, feat="ff_lib.feat (24-d)", trained_on=f"selection half, frame {FT}, zoomed detections"),
                          open(ev5.TV / f"v5/ff_clf_{task}_{TAG}.json", "w"))

