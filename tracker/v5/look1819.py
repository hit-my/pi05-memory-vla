"""frame-0 look at tasks 18/19: GT boxes of all sim objects (tracked = green + name), det candidates count, clicks"""
import sys
import cv2
import numpy as np
import ev5
import rma_spec

Z = 6
for task in (18, 19):
    tiles = []
    for ep in ev5.task_eps(task, "A")[:12]:
        g = np.load(ev5.GT / f"{ep}.npz")
        rgb = np.load(f"xxxx/rma/lr_eps/front/{ep}.npy", mmap_mode="r")[0]
        names = [str(x) for x in g["all_names"]]
        kidx = g["kidx"].tolist()
        ab = g["all_bbox"][10]
        x0, y0, W = 60, 70, 80
        if ep.endswith(("100", "102")):
            print(task, ep, "objects", [str(x) for x in g["objects"]], "sim", [str(x) for x in g["sim_names"]], "all", names)
            print("  bbox10", {names[i]: ab[i].round(0).tolist() for i in range(len(names))})
            print("  area10", {names[i]: int(g["all_mask_area"][10, i]) if "all_mask_area" in g.files else -1 for i in range(len(names))})
        img = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y0 + W, x0:x0 + W]), cv2.COLOR_RGB2BGR)
        clean = cv2.resize(img, (W * Z, W * Z), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (W * Z, W * Z), interpolation=cv2.INTER_NEAREST)
        P = lambda p: (int(round((p[0] - x0 + 0.5) * Z)), int(round((p[1] - y0 + 0.5) * Z)))
        for i, nm in enumerate(names):
            b = ab[i]
            if not np.isfinite(b).all():
                continue
            tr = i in kidx
            cv2.rectangle(img, P((b[0] - .5, b[1] - .5)), P((b[2] + .5, b[3] + .5)), (0, 255, 0) if tr else (255, 128, 0), 1)
            lab = (str(g["objects"][kidx.index(i)]) if tr else nm[:6])
            cv2.putText(img, lab, (P((b[0], b[1]))[0], P((b[0], b[1]))[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if tr else (255, 128, 0), 1, cv2.LINE_AA)
        for j, c in enumerate(rma_spec.CLICKS[task]):
            cv2.drawMarker(img, P(c), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(img, ep, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(np.concatenate([clean, img], 1))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    for r in range(0, len(rows), 2):
        cv2.imwrite(str(ev5.TV / f"v5/look/f0_t{task}_{r//2}.jpg"), np.concatenate(rows[r:r + 2], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
