"""Tracks for every variant, strict error counts, and side-by-side videos / contact sheets.
Variants: cutie (current), sub (cutie minus the projected gripper/wrist region), kin (sub + kinematic propagation while
grasped + static prior when released), sam2v, samurai, dam, long.
Usage: tv_render.py [ep ...]"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
from tv_common import ROOT, centres_from_masks, eef_pixels, episodes, load, load_masks, project  # noqa: E402

import os
ORDER = os.environ.get("TV_ORDER", "cutie,sub,kin,sam2v,samurai,dam,long").split(",")
VIEW = os.environ.get("TV_VIEW", "view")
from tv_fuse import fuse  # noqa: E402
TITLE = {"cutie": "A current (CUTIE)", "sub": "B1 CUTIE - robot region", "kin": "B2 B1 + grasp kinematics",
         "sam2v": "C SAM2.1 video", "samurai": "D SAMURAI", "dam": "E DAM4SAM", "long": "F SAM2Long", "f_cutie": "G1 CUTIE + kinematic fusion", "f_dam": "G2 DAM4SAM + kinematic fusion",
         "f_sam2v": "G3 SAM2.1 + kinematic fusion", "sam3": "H SAM3", "f_sam3": "G4 SAM3 + kinematic fusion", "dino": "I per-frame DINO + memory", "f_sam3det": "G5 SAM3 re-detect/5f + kinematics", "o_sam3": "J SAM3 seen only + gripper veto", "o_cutie": "J0 CUTIE seen only + gripper veto",
         "o_sam2v": "J1 SAM2.1 seen only + gripper veto"}
COLOURS = [(255, 120, 0), (0, 140, 255), (200, 0, 200)]  # BGR: blue, orange, purple
TRAIL, W = 20, 10


def robot_masks(ep, states):
    """Projected gripper + wrist capsule per frame (proprioception and the calibrated camera only)."""
    n = len(states)
    out = np.zeros((n, 256, 256), bool)
    for t in range(n):
        pos = states[t, :3]
        R, _ = cv2.Rodrigues(states[t, 3:6])
        z = R[:, 2]
        pts = project(ep, np.stack([pos + 0.03 * z, pos - 0.24 * z, pos + 0.05 * R[:, 0], pos + 0.05 * R[:, 1]]))
        radius = int(np.clip(max(np.hypot(*(pts[2] - eef_px(ep, pos))), np.hypot(*(pts[3] - eef_px(ep, pos)))), 6, 30))
        canvas = np.zeros((256, 256), np.uint8)
        a, b = pts[0], pts[1]
        if np.all(np.isfinite([a, b])) and np.all(np.abs([a, b]) < 2000):
            cv2.line(canvas, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 1, thickness=2 * radius)
        out[t] = canvas > 0
    return out


def eef_px(ep, pos):
    return project(ep, pos[None])[0]


def hold_last(centres, vis):
    shown = centres.copy()
    for j in range(centres.shape[1]):
        last = None
        for t in range(len(centres)):
            if vis[t, j]:
                last = centres[t, j]
            elif last is not None:
                shown[t, j] = last
    return shown


def kinematic(ep, masks, robot, rgb, states, actions):
    n, k = masks.shape[:2]
    e = eef_pixels(ep, states)
    closed = np.r_[False, actions[:-1, 6] > 0]  # command issued before this frame: causal
    c0, _ = centres_from_masks(masks[:1])
    p = c0[0].copy()
    ref = [np.median(rgb[0][masks[0, j]], axis=0) if masks[0, j].sum() else None for j in range(k)]
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    held, off = None, None
    ys, xs = np.mgrid[0:256, 0:256]
    for t in range(n):
        if t and closed[t] and not closed[t - 1] and held is None:
            d = [np.hypot(*(p[j] - e[t])) if np.all(np.isfinite(p[j])) else 1e9 for j in range(k)]
            j = int(np.argmin(d))
            if d[j] < 30:
                held, off = j, p[j] - e[t]
        if held is not None:
            p[held] = e[t] + off
            if not closed[t]:
                held = None
        for j in range(k):
            if not np.all(np.isfinite(p[j])):
                continue
            centres[t, j] = p[j]
            if j == held:
                vis[t, j] = True
                continue
            m = masks[t, j] & ~robot[t]
            x, y = int(np.clip(round(p[j][0]), 3, 252)), int(np.clip(round(p[j][1]), 3, 252))
            if m.sum() >= 20:
                c = np.array([xs[m].mean(), ys[m].mean()])
                if np.hypot(*(c - p[j])) < 12:
                    vis[t, j] = True
                    if np.hypot(*(c - e[t])) > 30:
                        p[j] = 0.8 * p[j] + 0.2 * c
                        ref[j] = np.median(rgb[t][m], axis=0)
                    continue
            if ref[j] is not None and not robot[t, y, x]:  # the tracker is elsewhere: is the object still seen at p?
                patch = np.median(rgb[t][y - 3:y + 4, x - 3:x + 4].reshape(-1, 3), axis=0)
                vis[t, j] = bool(np.abs(patch - ref[j]).mean() < 30)
    return centres, vis


def strict(centres, vis, eef, cmd):
    n, K = vis.shape
    c_all = np.nan_to_num(centres)
    idx = np.arange(W, n)
    opened = np.ones(len(idx), bool)
    for a in range(W + 16):
        opened &= ~cmd[np.maximum(idx - a, 0)]
    de = eef[idx] - eef[idx - W]
    dem = np.hypot(*de.T)
    e = dict(A=0, B=0, C=0, D=0, vis=float(vis.mean()))
    withs = []
    for k in range(K):
        c, v = c_all[:, k], vis[:, k]
        ok = v[idx] & v[idx - W]
        d = c[idx] - c[idx - W]
        dc = np.hypot(*d.T)
        w = ok & (dc > 6) & (dem > 6) & (np.hypot(*(d - de).T) < 0.5 * np.maximum(dc, dem)) & (np.hypot(*(c[idx] - eef[idx]).T) < 45)
        withs.append(w)
        e["A"] = max(e["A"], int((w & opened).sum()))
        e["D"] = max(e["D"], int((ok & opened & (dc > 6) & ~w).sum()))
        step = np.hypot(*(c[1:] - c[:-1]).T)
        e["C"] += int(((step > 25) & v[1:] & v[:-1]).sum())
    both = np.zeros(len(idx), bool)
    for i in range(K):
        for j in range(i + 1, K):
            both |= withs[i] & withs[j]
    e["B"] = int(both.sum())
    return e


def draw(frame, t, centres, vis, shown, eef, closed, title, size, robot=None):
    img = cv2.resize(frame[..., ::-1], (size, size), interpolation=cv2.INTER_LINEAR)
    s = size / 256
    if robot is not None:
        edge = cv2.resize(robot.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
    for j in range(centres.shape[1]):
        col = COLOURS[j % 3]
        pts = [shown[u, j] * s for u in range(max(0, t - TRAIL), t + 1) if np.all(np.isfinite(shown[u, j]))]
        if len(pts) > 1:
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        if np.all(np.isfinite(shown[t, j])):
            c = tuple(int(round(v)) for v in shown[t, j] * s)
            if vis[t, j]:
                cv2.circle(img, c, 5, col, -1, cv2.LINE_AA)
                cv2.circle(img, c, 5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 6, col, 1, cv2.LINE_AA)
    c = tuple(int(round(v)) for v in eef[t] * s)
    cv2.drawMarker(img, c, (0, 255, 0) if not closed[t] else (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
    cv2.rectangle(img, (0, 0), (size, 16), (0, 0, 0), -1)
    cv2.putText(img, title, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    eps = sys.argv[1:] or episodes()
    out = ROOT / VIEW
    out.mkdir(exist_ok=True)
    table = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
    for ep in eps:
        if not (ROOT / "out/cutie" / f"{ep}.npz").exists():
            continue
        rgb, states, actions = load(ep)
        n = len(rgb)
        eef = eef_pixels(ep, states)
        cmd = actions[:, 6] > 0
        robot = robot_masks(ep, states)
        base, _ = load_masks("cutie", ep)
        tracks = {}
        for m in ORDER:
            if m == "sub":
                tracks[m] = centres_from_masks(base & ~robot[:, None])
            elif m == "kin":
                tracks[m] = kinematic(ep, base, robot, rgb, states, actions)
            elif m == "f_sam3det" and (ROOT / "out/sam3det" / f"{ep}.npz").exists():
                from tv_fuse import load_dets
                tracks[m] = fuse(ep, base, robot, rgb, states, actions, dets=load_dets(ROOT / "out/sam3det" / f"{ep}.npz"))[:2]
            elif m.startswith("o_") and (ROOT / "out" / m[2:] / f"{ep}.npz").exists():
                from tv_fuse import observe_only
                tracks[m] = observe_only(ep, load_masks(m[2:], ep)[0], robot, rgb, states, actions)[:2]
            elif m.startswith("f_") and (ROOT / "out" / m[2:] / f"{ep}.npz").exists():
                tracks[m] = fuse(ep, load_masks(m[2:], ep)[0], robot, rgb, states, actions)[:2]
            elif (ROOT / "out" / m / f"{ep}.npz").exists():
                tracks[m] = centres_from_masks(load_masks(m, ep)[0])
        table[ep] = {m: strict(c, v, eef, cmd) for m, (c, v) in tracks.items()}
        table[ep]["frames"] = n
        shown = {m: hold_last(c, v) for m, (c, v) in tracks.items()}
        np.savez_compressed(out / f"{ep}_tracks.npz", **{f"{m}_c": c for m, (c, v) in tracks.items()}, **{f"{m}_v": v for m, (c, v) in tracks.items()})
        names = [m for m in ORDER if m in tracks]
        size = 320
        proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{4 * size}x{2 * size}",
                                 "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(out / f"{ep}.mp4")],
                                stdin=subprocess.PIPE)
        for t in range(n):
            panels = [draw(rgb[t], t, *tracks[m], shown[m], eef, cmd, TITLE[m], size, robot[t] if m in ("sub", "kin") or m[:2] in ("f_", "o_") else None) for m in names]
            raw = cv2.resize(rgb[t][..., ::-1], (size, size))
            cv2.putText(raw, f"{ep}  frame {t}/{n}  gripper {'CLOSED' if cmd[t] else 'open'}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            panels = (panels + [raw] + [np.zeros_like(raw)] * 8)[:8]
            proc.stdin.write(np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:])]).tobytes())
        proc.stdin.close()
        proc.wait()
        # contact sheet: 8 moments spread over the frames where the current tracker and the kinematic one disagree most
        gap = np.nan_to_num(np.hypot(*(shown[names[0]] - shown[names[-1]]).transpose(2, 0, 1))).max(axis=1)
        picks = []
        for seg in np.array_split(np.arange(n), 8):
            picks.append(int(seg[np.argmax(gap[seg])]))
        small = 200
        rows = [np.hstack([draw(rgb[t], t, *tracks[m], shown[m], eef, cmd, f"{TITLE[m]}  f{t}", small, robot[t] if m in ("sub", "kin") or m[:2] in ("f_", "o_") else None) for t in picks]) for m in names]
        cv2.imwrite(str(out / f"{ep}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        (out / "metrics.json").write_text(json.dumps(table))
        print(ep, n, {m: (table[ep][m]["A"], table[ep][m]["B"], table[ep][m]["C"], table[ep][m]["D"]) for m in names}, flush=True)
    print("\nframes with an unambiguous error, summed over episodes  (A stuck-to-arm | B two-on-arm | C jumps | D ghost motion | mean visibility)")
    for m in ORDER:
        rows = [v[m] for v in table.values() if m in v]
        if rows:
            print(f"{TITLE[m]:28s} n={len(rows):2d}  A {sum(r['A'] for r in rows):5d}  B {sum(r['B'] for r in rows):5d}  C {sum(r['C'] for r in rows):4d}  D {sum(r['D'] for r in rows):5d}  vis {np.mean([r['vis'] for r in rows]):.2f}")


if __name__ == "__main__":
    main()
