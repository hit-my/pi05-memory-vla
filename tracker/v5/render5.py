"""Zoomed strips: GT box (green; green x = projected centre when hidden), dot A (red, e.g. v5), dot B (yellow ring, e.g. raw),
corrected gripper pixel (cyan cross).  One strip per (episode, object): frames spread over the error frames of dot A.
usage as module: strip(ep, k, CA, VA, CB, VB, out, frames=None, title="")"""
import sys

import cv2
import numpy as np

import ev5

Z, W = 4, 96  # zoom, crop window (source px)


def pick_frames(flags, n, count=10):
    bad = np.flatnonzero(flags > 0)
    if len(bad) == 0:
        return np.linspace(0, n - 1, count).astype(int).tolist()
    # runs of errors: first frame of each run, then fill evenly
    starts = [bad[0]] + [b for a, b in zip(bad[:-1], bad[1:]) if b - a > 5]
    sel = set(int(s) for s in starts[:count // 2])
    rest = bad[np.linspace(0, len(bad) - 1, count).astype(int)]
    for r in rest:
        if len(sel) >= count:
            break
        sel.add(int(r))
    extra = np.linspace(0, n - 1, count).astype(int)
    for r in extra:
        if len(sel) >= count:
            break
        sel.add(int(r))
    return sorted(sel)


def strip(ep, k, CA, VA, CB, VB, out, frames=None, title="", g=None, e=None, count=10, flags=None, extra_pts=None):
    g = g or ev5.G(ep)
    rgb = np.load(f"/data/lgd/rma/lr_eps/front/{ep}.npy", mmap_mode="r")
    n = min(len(rgb), g.n, len(CA))
    if flags is None:
        flags = ev5.score(g, CA, VA, per_frame=True)["flags"][:, k]
    frames = frames or pick_frames(flags[:n], n, count)
    if e is None:
        from tv_common_v4 import eef_pixels
        m = np.load(f"/data/lgd/rma/lr_eps/meta/{ep}.npz")
        e = eef_pixels(ep, m["states"])
    tiles = []
    for t in frames:
        img = cv2.cvtColor(np.ascontiguousarray(rgb[t]), cv2.COLOR_RGB2BGR)
        # window centre: GT centre (or proj) and dot A
        pts = []
        c0 = g.mc[t, k] if np.isfinite(g.mc[t, k]).all() else g.pc[t, k]
        if np.isfinite(c0).all():
            pts.append(c0)
        if VA[t, k] and np.isfinite(CA[t, k]).all():
            pts.append(CA[t, k])
        ctr = np.mean(pts, 0) if pts else np.array([128, 128])
        x0 = int(np.clip(ctr[0] - W / 2, 0, 256 - W)); y0 = int(np.clip(ctr[1] - W / 2, 0, 256 - W))
        crop = cv2.resize(img[y0:y0 + W, x0:x0 + W], (W * Z, W * Z), interpolation=cv2.INTER_NEAREST)
        P = lambda p: (int(round((p[0] - x0 + 0.5) * Z)), int(round((p[1] - y0 + 0.5) * Z)))
        for j in range(g.K):
            b = g.bbox[t, j]
            col = (0, 255, 0) if j == k else (0, 120, 0)
            if g.area[t, j] >= 8 and np.isfinite(b).all():
                cv2.rectangle(crop, P((b[0] - 0.5, b[1] - 0.5)), P((b[2] + 0.5, b[3] + 0.5)), col, 1)
            elif np.isfinite(g.pc[t, j]).all():
                p = P(g.pc[t, j]); cv2.drawMarker(crop, p, col, cv2.MARKER_TILTED_CROSS, 10, 1)
        if VB is not None and VB[t, k] and np.isfinite(CB[t, k]).all():
            cv2.circle(crop, P(CB[t, k]), 9, (0, 255, 255), 1)
        if VA[t, k] and np.isfinite(CA[t, k]).all():
            cv2.circle(crop, P(CA[t, k]), 4, (0, 0, 255), -1)
        if np.isfinite(e[t]).all():
            cv2.drawMarker(crop, P(e[t]), (255, 255, 0), cv2.MARKER_CROSS, 14, 1)
        if extra_pts is not None:
            for p in extra_pts(t):
                cv2.drawMarker(crop, P(p), (255, 0, 255), cv2.MARKER_DIAMOND, 8, 1)
        tag = {0: "", 1: "WRONG", 2: "HID", 3: "MISS"}[int(flags[t])]
        cv2.putText(crop, f"{t} {tag} a{int(g.area[t, k])}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    per = 5
    rows = []
    for i in range(0, len(tiles), per):
        r = tiles[i:i + per]
        while len(r) < per:
            r.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(r, 1))
    sheet = np.concatenate(rows, 0)
    bar = np.zeros((22, sheet.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{ep} obj{k} {title}  green=GT box (x=hidden centre) red=A yellow ring=B cyan+=gripper", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.concatenate([bar, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    # render5.py OUTDIR DIR KEYC KEYV ep:k[:f1,f2,...] ...   (dot B = raw)
    from pathlib import Path
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    d, kc, kv = sys.argv[2:5]
    for spec in sys.argv[5:]:
        parts = spec.split(":")
        ep, k = parts[0], int(parts[1])
        fr = [int(x) for x in parts[2].split(",")] if len(parts) > 2 else None
        z = np.load(f"{d}/{ep}.npz")
        strip(ep, k, z[kc], z[kv], z["raw_centres"], z["raw_visibility"], out / f"{ep}_o{k}_{kc}.jpg", frames=fr, title=kc)
        print("wrote", ep, k)
