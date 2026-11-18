"""candidate tiles for tasks 18/19: each det5 / initB candidate near the GT objects, outline on the zoomed crop. look_cand.py ep ep ..."""
import json
import sys
import cv2
import numpy as np
import ev5
import ff_lib

Z = 5
for ep in sys.argv[1:]:
    g = np.load(ev5.GT / f"{ep}.npz")
    rgb = np.load(f"/data/lgd/rma/lr_eps/front/{ep}.npy", mmap_mode="r")[0]
    bb = g["bbox"][10]
    names = [str(x) for x in g["objects"]]
    cx, cy = np.nanmean(bb[:, [0, 2]]), np.nanmean(bb[:, [1, 3]])
    W = 56
    x0, y0 = int(np.clip(cx - W / 2, 0, 256 - W)), int(np.clip(cy - W / 2, 0, 256 - W))
    tiles = []
    for tag, d in (("B", ev5.TV / "initB/det"), ("Z", ev5.TV / "v5/det5")):
        z = np.load(d / f"{ep}.npz")
        masks = np.unpackbits(z["masks"], axis=-1)[..., :256].astype(bool).reshape(-1, 256, 256)
        prompts = json.loads(str(z["prompts"])) if "prompts" in z.files else ["box", "bottle", "carton", "small object", "food package"]
        order = np.argsort(-z["scores"])
        for i in order:
            m = masks[i]
            a = int(m.sum())
            if a < 15 or a > 700:
                continue
            c = ff_lib.centroid(m)
            if not (x0 <= c[0] < x0 + W and y0 <= c[1] < y0 + W):
                continue
            img = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y0 + W, x0:x0 + W]), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (W * Z, W * Z), interpolation=cv2.INTER_CUBIC)
            mm = cv2.resize(m[y0:y0 + W, x0:x0 + W].astype(np.uint8), (W * Z, W * Z), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 255, 255), 1)
            for k in range(len(names)):
                b = bb[k]
                if np.isfinite(b).all():
                    col = [(0, 255, 0), (255, 128, 0), (255, 0, 255)][k]
                    cv2.rectangle(img, (int((b[0] - x0) * Z), int((b[1] - y0) * Z)), (int((b[2] - x0 + 1) * Z), int((b[3] - y0 + 1) * Z)), col, 1)
            cv2.putText(img, f"{tag} {prompts[int(z['prompt'][i])][:9]} {z['scores'][i]:.2f} a{a}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
    tiles = tiles[:42]
    per = 7
    rows = []
    for i in range(0, len(tiles), per):
        r = tiles[i:i + per]
        while len(r) < per:
            r.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(r, 1))
    bar = np.zeros((20, rows[0].shape[1], 3), np.uint8)
    cv2.putText(bar, f"{ep} GT boxes: " + " ".join(f"{n}={c}" for n, c in zip(names, ["green", "blue", "magenta"])), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(ev5.TV / f"v5/look/cand_{ep}.jpg"), np.concatenate([bar] + rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
