"""Step 1: read a yx_blocks hdf5 episode, crop the cameras, and track the three cups and the three numbered blocks with SAM3 (tracker venv, 1 GPU).

  real_track.py HDF5_DIR OUT [first [last]]      (run real_extract.py first: it writes OUT/meta; other episodes are skipped)
Writes, per kept episode <ep> = t28_seed<index>:
  OUT/front/<ep>.npy   uint8 [N,256,256,3]  cropped top camera (the tracker / overlay source frames)
  OUT/wrist/<ep>.npy   uint8 [N,256,256,3]  cropped right-wrist camera
  OUT/masks/<ep>.npz   raw SAM3 masks, bit-packed (first = first-frame detections, masks [N,6,256,32]); real_rules.py recomputes the
                       tracks from them in seconds when the reporting rules of real_tracker.py change
  OUT/tracks/<ep>.npz  centres float32 [N,6,2] (x, y at 256 px; nan = not seen), visibility, retired, area, det, repairs
Cropping (real_spec.crop_resize) and tracking (real_tracker.ObjectTracker) are the functions the deployment loop calls,
frame by frame and causally, so a live run reproduces these files.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402
from real_tracker import ObjectTracker  # noqa: E402

def decode(path, camera):
    """hdf5 episode -> uint8 [N,480,640,3] (the images are stored as JPEG bytes, zero padded)."""
    import io
    import h5py
    from PIL import Image
    with h5py.File(path, "r") as f:
        data = f[f"observations/images/{camera}"][:]
    return np.stack([np.asarray(Image.open(io.BytesIO(row.tobytes())).convert("RGB")) for row in data])

def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    last = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    for sub in ("front", "wrist", "tracks", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tracker = ObjectTracker()
    for index in range(first, last + 1):
        ep = S.episode_name(index)
        if (out / "tracks" / f"{ep}.npz").exists() or not (out / "meta" / f"{ep}.npz").exists():
            continue
        meta = np.load(out / "meta" / f"{ep}.npz")
        n, closed = len(meta["states"]), meta["closed"]
        top = decode(src / f"episode_init_{index}.hdf5", "top")
        right = decode(src / f"episode_init_{index}.hdf5", "right_wrist")
        if not (len(top) == len(right) == n):
            raise SystemExit(f"{ep}: frames top {len(top)} wrist {len(right)} rows {n}")
        front = np.stack([S.crop_resize(f, S.TOP_CROP) for f in top])
        wrist = np.stack([S.crop_resize(f, S.WRIST_CROP) for f in right])
        k = len(S.NAMES)
        centres = np.full((n, k, 2), np.nan, np.float32)
        area = np.zeros((n, k), np.int32)
        pack = lambda ms: np.packbits(np.stack(ms).astype(bool), axis=-1)
        centres[0], report = tracker.reset(front[0])
        raw = np.zeros((n,) + pack(tracker.raw).shape, np.uint8)
        raw[0] = pack(tracker.raw)
        area[0] = tracker.area
        for t in range(1, n):
            centres[t] = tracker.step(front[t], bool(closed[t - 1]))  # the command in force when frame t was taken
            area[t] = tracker.area
            raw[t] = pack(tracker.raw)
        visible = np.isfinite(centres).all(-1)
        np.save(out / "front" / f"{ep}.npy", front)
        np.save(out / "wrist" / f"{ep}.npy", wrist)
        np.savez_compressed(out / "masks" / f"{ep}.npz", first=pack(tracker.chosen), masks=raw)  # raw SAM3 masks: real_rules.py replays the rules
        np.savez_compressed(out / "tracks" / f"{ep}.npz", centres=centres, visibility=visible,
                            retired=np.zeros((n, k), bool), area=area, det=json.dumps(report),
                            repairs=json.dumps(tracker.events))
        print("TRACKED", ep, n, "visible", visible.mean(0).round(2).tolist(), "repairs", tracker.events, report, flush=True)
    print("TRACK_DONE", flush=True)


if __name__ == "__main__":
    main()
