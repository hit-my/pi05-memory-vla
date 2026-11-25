"""Does the training overlay (rma_build_v2r) equal what the evaluation side draws, given identical inputs?

  test_overlay_equiv_v2r.py DATASET TRACKS EP [EP ...]      (simulator venv, env as in pipeline_v2r.sh)
For each episode, every frame:
  1. eval path: rma_perception_v2.PerceptionV2 (the class rma_adapter_v2.gripper.patch plugs into the adapter) is run
     frame by frame through its own reset()/step() -> _frame_v2 code. Its tracker and its veto are replaced by stubs
     that return the stored track file's centres (the veto is what the track file already contains), so the DRAWING
     code under test is the evaluation code: history deque, make_tracks, CanonicalRenderer.render, PNG.
     Then the adapter's own resize() + png_bytes() are applied, as in rma_adapter_v2.infer_actions (main_png_b64).
  2. training path: the PNG stored in DATASET/images/<ep>/NNNNNN.png.
  Compared pixel for pixel (decoded RGB) and byte for byte (renderer PNG).
  Also: raw image on disk vs the adapter's resize(raw frame) (raw_png_b64), wrist likewise.
PerceptionV2 has no notion of retirement (its veto is v2), so from an object's retirement frame on the eval stub is
given "not seen" and the only possible difference is the cleared trail; those frames are counted separately.
Second part (--prev OLD_DATASET): frames whose 21-frame window has no drawn point at all in BOTH the old and the new
dataset must be byte-identical to the old dataset's image (same renderer, same Chromium, same base-image path).
"""
import sys

sys.dont_write_bytecode = True

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_v2r  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--prev=")]
prev = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--prev=")), None)
dataset, tracks_dir, eps = Path(args[0]), Path(args[1]), args[2:]
CODE = Path(os.environ.get("CODE", Path.home() / "pi05-abc/code_causal_v2")).resolve()
RAW = Path(os.environ.get("RAW", "xxxx/rma/lr_eps"))
sys.path.insert(0, str(CODE))
import rma_spec  # noqa: E402
import rma_perception_v2 as pv2  # noqa: E402
import rma_adapter_v2 as adapter  # noqa: E402


class StubTracker:
    def reset(self, rgb, task_id):
        return {"k": 0, "stub": True}, None, None

    def step(self, rgb):
        return None

    def close(self):
        pass


class StubVeto:
    """Returns the stored track (centres nan = not drawn), frame by frame, in OnlineVeto's interface."""

    def __init__(self, centres, drawn):
        self.c, self.v, self.t = centres, drawn, -1
        self.events = []

    def _frame(self):
        self.t += 1
        k = self.c.shape[1]
        self.raw_centres, self.raw_visible, self.area = self.c[self.t].astype(np.float64), self.v[self.t], np.zeros(k, int)
        return self.c[self.t].astype(np.float64), self.v[self.t]

    def reset(self, masks, rgb, state):
        return self._frame()

    def step(self, masks, rgb, state, last_gripper_cmd=None):
        return self._frame()


perception = None
summary = {}
for ep in eps:
    task = int(ep[1:3])
    rgb = np.load(RAW / "front" / f"{ep}.npy", mmap_mode="r")
    wrist = np.load(RAW / "wrist" / f"{ep}.npy", mmap_mode="r")
    with np.load(RAW / "meta" / f"{ep}.npz") as m:
        states, actions = m["states"].astype(np.float32), m["actions"].astype(np.float32)
    n = json.loads((dataset / "log" / f"{ep}.json").read_text())["frames"]
    k = len(rma_spec.names(task))
    centres, drawn, retired, _ = overlay_v2r.load_tracks(tracks_dir / f"{ep}.npz", n, k)
    if perception is None:
        perception = pv2.PerceptionV2.__new__(pv2.PerceptionV2)          # no tracker subprocess, no veto import side effects
        pv2.Perception.__init__(perception, task, tracker=StubTracker())   # starts the real CanonicalRenderer
        perception.timing = {}
    perception.task_id, perception.spec, perception.registry = task, rma_spec.TASKS[task], rma_spec.registry(task)
    perception.veto = StubVeto(centres, drawn)
    counts = {"frames": n, "png_bytes_equal": 0, "pixels_equal": 0, "pixel_diff_before_retirement": 0,
              "pixel_diff_after_retirement": 0, "max_abs_diff": 0, "raw_equal": 0, "wrist_equal": 0,
              "retired_from": [int(np.argmax(retired[:, j])) if retired[:, j].any() else None for j in range(k)]}
    for t in range(n):
        frame = np.ascontiguousarray(rgb[t])
        overlay = perception.reset(frame, states[t]) if t == 0 else perception.step(frame, states[t], last_gripper_cmd=float(actions[t - 1][6]))
        sent = np.asarray(Image.open(io.BytesIO(adapter.png_bytes(adapter.resize(overlay)))).convert("RGB"))  # main_png_b64
        stored_bytes = (dataset / "images" / ep / f"{t:06d}.png").read_bytes()
        stored = np.asarray(Image.open(io.BytesIO(stored_bytes)).convert("RGB"))
        counts["png_bytes_equal"] += stored_bytes == perception.png
        if np.array_equal(sent, stored):
            counts["pixels_equal"] += 1
        else:
            counts["max_abs_diff"] = max(counts["max_abs_diff"], int(np.abs(sent.astype(int) - stored.astype(int)).max()))
            counts["pixel_diff_after_retirement" if retired[t].any() else "pixel_diff_before_retirement"] += 1
        raw_sent = np.asarray(Image.open(io.BytesIO(adapter.png_bytes(adapter.resize(frame)))).convert("RGB"))      # raw_png_b64
        counts["raw_equal"] += np.array_equal(raw_sent, np.asarray(Image.open(dataset / "images" / ep / f"{t:06d}_raw.png").convert("RGB")))
        wrist_sent = adapter.resize(np.ascontiguousarray(wrist[t]))
        counts["wrist_equal"] += np.array_equal(wrist_sent, np.asarray(Image.open(dataset / "images" / ep / f"{t:06d}_wrist.png").convert("RGB")))
    if prev is not None:
        with np.load(prev / "episodes" / f"{ep}.npz") as zo:
            old_vis = zo["visibility"].astype(bool)
        blank = [t for t in range(n) if not old_vis[max(0, t - 20):t + 1].any() and not drawn[max(0, t - 20):t + 1].any()]
        same = sum((prev / "images" / ep / f"{t:06d}.png").read_bytes() == (dataset / "images" / ep / f"{t:06d}.png").read_bytes() for t in blank)
        counts["frames_without_any_point_in_old_and_new"] = len(blank)
        counts["of_those_byte_identical_to_previous_dataset"] = same
        wsame = sum(np.array_equal(np.asarray(Image.open(prev / "images" / ep / f"{t:06d}_wrist.png")),
                                   np.asarray(Image.open(dataset / "images" / ep / f"{t:06d}_wrist.png"))) for t in range(0, n, 25))
        counts["wrist_pixels_equal_to_previous_dataset"] = f"{wsame}/{len(range(0, n, 25))}"
    summary[ep] = counts
    print(ep, json.dumps(counts), flush=True)
perception.renderer.close()
print("EQUIV_DONE", flush=True)
