"""show the appearance selection on frame 0 (and the frame itself at t=0 and t=12): look_sel.py CLFTAG POOL ep ..."""
import json
import sys
import cv2
import numpy as np
import ev5
import ff_lib
sys.argv = [sys.argv[0]] + sys.argv[1:]
TAG, POOL = sys.argv[1], sys.argv[2]
import os
FR = int(os.environ.get("FR", "0"))
DET5 = str(ev5.TV / os.environ.get("DET", "v5/det5"))
Z = 6
tiles = []
for ep in sys.argv[3:]:
    task = int(ep[1:3])
    J = json.load(open(ev5.TV / f"v5/ff_clf_{task}_{TAG}.json"))
    clf = ff_lib.Softmax.from_json(J["clf"]); names = J["names"]; K = len(names)
    g = np.load(ev5.GT / f"{ep}.npz")
    rgbs = np.load(f"xxxx/rma/lr_eps/front/{ep}.npy", mmap_mode="r")
    rgb = np.ascontiguousarray(rgbs[FR])
    cand, sc = ff_lib.candidates(ep, DET5, only_extra=(POOL == "Z"))
    X = np.array([ff_lib.feat(rgb, m) for m in cand])
    P = clf.predict_proba(X)
    from scipy.optimize import linear_sum_assignment
    sel = [-1] * K
    sel = ff_lib.select_joint(P, np.array([ff_lib.centroid(m) for m in cand]), cand, K, 0.5, float(J.get("dmax", 25)), atyp=J.get("atyp"), beta=float(J.get("beta", 0)))
    bb = g["bbox"][10]
    cx, cy = np.nanmean(bb[:, [0, 2]]), np.nanmean(bb[:, [1, 3]])
    W = 48
    x0, y0 = int(np.clip(cx - W / 2, 0, 256 - W)), int(np.clip(cy - W / 2 - 4, 0, 256 - W))
    row = []
    for t in (FR, 12):
        img = cv2.cvtColor(np.ascontiguousarray(rgbs[t][y0:y0 + W, x0:x0 + W]), cv2.COLOR_RGB2BGR)
        row.append(cv2.resize(img, (W * Z, W * Z), interpolation=cv2.INTER_CUBIC))
    cols = [(0, 255, 0), (255, 128, 0), (255, 0, 255)]
    for k in range(K):
        img = row[0].copy(); P_other = P[:, K]
        b = g["bbox"][max(FR, 2)][k]
        if np.isfinite(b).all():
            cv2.rectangle(img, (int((b[0] - x0) * Z), int((b[1] - y0) * Z)), (int((b[2] - x0 + 1) * Z), int((b[3] - y0 + 1) * Z)), cols[k], 1)
        if sel[k] >= 0:
            mm = cv2.resize(cand[sel[k]][y0:y0 + W, x0:x0 + W].astype(np.uint8), (W * Z, W * Z), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 255, 255), 1)
            top = np.argsort(-P[:, k])[:3]
            cv2.putText(img, f"{names[k]} p={P[sel[k], k]:.2f} top3 {P[top, k].round(2).tolist()}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        row.append(img)
    while len(row) < 5:
        row.append(np.zeros_like(row[0]))
    cv2.putText(row[0], ep + " t=0", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(row[1], "t=12", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    tiles.append(np.concatenate(row, 1))
for i in range(0, len(tiles), 4):
    cv2.imwrite(str(ev5.TV / f"v5/look/sel_{TAG}_{i//4}.jpg"), np.concatenate(tiles[i:i + 4], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
