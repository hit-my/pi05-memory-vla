"""Preview of the chosen overlay tracker on the unbiased 104 demos: left = current (CUTIE), right = SAM3 seen-only +
gripper veto. One mp4 and one contact sheet per episode. Usage: preview104.py SHARD NSHARDS"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["TV_EPS"] = "eps104.json"
HERE = Path("xxxx/rma/trackv")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
sys.path.append("xxxx/pi05-abc/tracker/assets/deps")
import imageio_ffmpeg  # noqa: E402
import tv_fuse  # noqa: E402
from tv_common import ROOT, centres_from_masks, eef_pixels, episodes, load, load_masks  # noqa: E402
from tv_render import hold_last, robot_masks  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
OUT = ROOT / "preview104b"
OUT.mkdir(exist_ok=True)
COL = [(255, 120, 0), (0, 140, 255), (200, 0, 200)]
S = 384


def draw(frame, t, c, v, eef, closed, title):
    img = cv2.resize(frame[..., ::-1], (S, S), interpolation=cv2.INTER_LINEAR)
    k = S / 256
    for j in range(c.shape[1]):
        col = COL[j % 3]
        pts = [c[u, j] * k for u in range(max(0, t - 20), t + 1) if v[u, j]]
        if len(pts) > 1:
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        if v[t, j]:
            p = tuple(int(round(x)) for x in c[t, j] * k)
            cv2.circle(img, p, 5, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, 5, (255, 255, 255), 1, cv2.LINE_AA)
    p = tuple(int(round(x)) for x in eef[t] * k)
    cv2.drawMarker(img, p, (0, 0, 255) if closed[t] else (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
    cv2.rectangle(img, (0, 0), (S, 18), (0, 0, 0), -1)
    cv2.putText(img, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for i, ep in enumerate(episodes()):
    if i % shards != shard or (OUT / f"{ep}.jpg").exists():
        continue
    if not (ROOT / "out/sam3b" / f"{ep}.npz").exists() or not (ROOT / "out/cutie" / f"{ep}.npz").exists():
        continue
    rgb, states, actions = load(ep)
    n = len(states)
    eef = eef_pixels(ep, states)
    closed = actions[:, 6] > 0
    old_c, old_v = centres_from_masks(load_masks("cutie", ep)[0][:n])
    new_c, new_v, _ = tv_fuse.observe_only(ep, load_masks("sam3b", ep)[0][:n], robot_masks(ep, states), rgb, states, actions)
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * S}x{S}", "-r", "20",
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", str(OUT / f"{ep}.mp4")], stdin=subprocess.PIPE)
    for t in range(n):
        proc.stdin.write(np.hstack([draw(rgb[t], t, old_c, old_v, eef, closed, f"current  f{t}"),
                                    draw(rgb[t], t, new_c, new_v, eef, closed, f"new: SAM3 seen-only + veto  {ep}")]).tobytes())
    proc.stdin.close()
    proc.wait()
    picks = [int(x) for x in np.linspace(n * 0.08, n - 1, 8)]
    sheet = np.vstack([np.hstack([cv2.resize(draw(rgb[t], t, c, v, eef, closed, f"{name} f{t}"), (224, 224)) for t in picks])
                       for name, c, v in (("current", old_c, old_v), ("new", new_c, new_v))])
    cv2.imwrite(str(OUT / f"{ep}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(ep, n, "drawn% current", round(100 * old_v.mean()), "new", round(100 * new_v.mean()), flush=True)
