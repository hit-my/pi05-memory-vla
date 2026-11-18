"""Step 1: decode a LeRobot v2.1 episode, crop the cameras, and track the two bowls with SAM3 (tracker venv, 1 GPU).

  real_track.py LEROBOT_DIR OUT [first [last]]      (run real_extract.py first: it writes OUT/meta)
Writes, per episode <ep> = t27_seed<index>:
  OUT/front/<ep>.npy   uint8 [N,256,256,3]  cropped top camera (the tracker / overlay source frames)
  OUT/wrist/<ep>.npy   uint8 [N,256,256,3]  cropped right-wrist camera
  OUT/tracks/<ep>.npz  centres float32 [N,2,2] (x, y at 256 px; nan = not seen), visibility, retired, area, det,
                       repairs (frame, object, collision | reacquired: see real_tracker.py)
Cropping (real_spec.crop_resize) and tracking (real_tracker.BowlTracker) are the functions the deployment loop
calls, frame by frame and causally, so a live run reproduces these files.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402
from real_tracker import BowlTracker  # noqa: E402

W, H = 640, 480


def decode(path):
    """mp4 (AV1) -> uint8 [N,480,640,3], every stored frame once."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    last = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    for sub in ("front", "wrist", "tracks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tracker = BowlTracker()
    for index in range(first, last + 1):
        ep = S.episode_name(index)
        if (out / "tracks" / f"{ep}.npz").exists():
            continue
        n = len(np.load(out / "meta" / f"{ep}.npz")["states"])
        top = decode(src / f"videos/chunk-000/observation.images.top/episode_{index:06d}.mp4")
        right = decode(src / f"videos/chunk-000/observation.images.right_wrist/episode_{index:06d}.mp4")
        if not (len(top) == len(right) == n):
            raise SystemExit(f"{ep}: frames top {len(top)} wrist {len(right)} rows {n}")
        front = np.stack([S.crop_resize(f, S.TOP_CROP) for f in top])
        wrist = np.stack([S.crop_resize(f, S.WRIST_CROP) for f in right])
        k = len(S.NAMES)
        centres = np.full((n, k, 2), np.nan, np.float32)
        area = np.zeros((n, k), np.int32)
        centres[0], report = tracker.reset(front[0])
        area[0] = tracker.area
        for t in range(1, n):
            centres[t] = tracker.step(front[t])
            area[t] = tracker.area
        visible = np.isfinite(centres).all(-1)
        np.save(out / "front" / f"{ep}.npy", front)
        np.save(out / "wrist" / f"{ep}.npy", wrist)
        np.savez_compressed(out / "tracks" / f"{ep}.npz", centres=centres, visibility=visible,
                            retired=np.zeros((n, k), bool), area=area, det=json.dumps(report),
                            repairs=json.dumps(tracker.events))
        print("TRACKED", ep, n, "visible", visible.mean(0).round(2).tolist(), "repairs", tracker.events, report, flush=True)
    print("TRACK_DONE", flush=True)


if __name__ == "__main__":
    main()
