"""Side-by-side videos: SAM3 as is (left) vs SAM3 + gripper veto (right), for the episodes where the veto changes most.
On the right, a small red ring marks a point the veto removed in that frame. One episode per task, the 12 largest."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
import imageio_ffmpeg  # noqa: E402
from tv_common import eef_pixels, load  # noqa: E402

OUT = HERE / "vetocmp"
OUT.mkdir(exist_ok=True)
COL = [(255, 120, 0), (0, 140, 255), (200, 0, 200)]
S = 448
rows = []
for f in sorted((HERE / "all/final").glob("t*_seed*.npz")):
    if "tmp" in f.name:
        continue
    z = np.load(f)
    removed = z["raw_visibility"] & ~z["visibility"]
    rows.append((int(removed.sum()), f.stem, int(z["visibility"].size)))
best = {}
for n, ep, total in sorted(rows, reverse=True):
    best.setdefault(ep[:3], (n, ep, total))
picks = sorted(best.values(), reverse=True)[:12]
json.dump({"episodes_scanned": len(rows), "frames_removed_total": int(sum(r[0] for r in rows)), "object_frames_total": int(sum(r[2] for r in rows)),
           "picks": picks}, open(OUT / "summary.json", "w"))
print("scanned", len(rows), "removed share", sum(r[0] for r in rows) / sum(r[2] for r in rows), picks, flush=True)


def draw(frame, t, c, v, eef, closed, title, ghost=None):
    img = cv2.resize(frame[..., ::-1], (S, S), interpolation=cv2.INTER_LINEAR)
    k = S / 256
    for j in range(c.shape[1]):
        col = COL[j % 3]
        pts = [c[u, j] * k for u in range(max(0, t - 20), t + 1) if v[u, j]]
        if len(pts) > 1:
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        if v[t, j]:
            p = tuple(int(round(x)) for x in c[t, j] * k)
            cv2.circle(img, p, 6, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, 6, (255, 255, 255), 1, cv2.LINE_AA)
        elif ghost is not None and ghost[1][t, j]:
            p = tuple(int(round(x)) for x in ghost[0][t, j] * k)
            cv2.circle(img, p, 8, (0, 0, 255), 2, cv2.LINE_AA)
    p = tuple(int(round(x)) for x in eef[t] * k)
    cv2.drawMarker(img, p, (0, 0, 255) if closed[t] else (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    cv2.rectangle(img, (0, 0), (S, 20), (0, 0, 0), -1)
    cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for n, ep, total in picks:
    z = np.load(HERE / "all/final" / f"{ep}.npz")
    rgb, states, actions = load(ep)
    m = len(z["visibility"])
    eef = eef_pixels(ep, states)
    closed = actions[:, 6] > 0
    rc, rv, c, v = z["raw_centres"], z["raw_visibility"], z["centres"], z["visibility"]
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * S}x{S}", "-r", "20", "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(OUT / f"{ep}.mp4")], stdin=subprocess.PIPE)
    for t in range(m):
        proc.stdin.write(np.hstack([draw(rgb[t], t, rc, rv, eef, closed, f"SAM3 only   {ep}  f{t}"),
                                    draw(rgb[t], t, c, v, eef, closed, "SAM3 + veto   (red ring = point removed)", ghost=(rc, rv))]).tobytes())
    proc.stdin.close()
    proc.wait()
    print(ep, m, "removed", n, flush=True)
