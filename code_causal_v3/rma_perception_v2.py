"""Trajectory image for RoboMemArena, v2: the finalised tracking pipeline, run online (causal, step by step).

  frame 0 : SAM3 "method B" first-frame masks (generic detections assigned to the fixed clicks, legacy click mask
            as fallback)                                                       } tracker subprocess rma_tracker_v2.py
  frame t : SAM3 tracker, frame by frame (rma_sam3_stream.Sam3Stream)          } (tracker venv, own torch)
  always  : "seen only + gripper veto" = tv_fuse.observe_only, step-wise       -> rma_online_veto.OnlineVeto (here)
  drawing : unchanged: make_tracks(history, registry) -> CanonicalRenderer.render, exactly as Perception._frame

Same public interface as rma_perception.Perception (it is a subclass):
  PerceptionV2(task_id).reset(rgb256, state8) / .step(rgb256, state8, last_gripper_cmd=None)
      -> overlay RGB uint8 [224,224,3] (the renderer's canvas; no resize by this class)
  .centres : list of [x, y] or None per object for the latest frame (None = not drawn: unseen or vetoed)
`last_gripper_cmd` is the gripper command (action[6]; > 0 = close) that was executed BEFORE this frame; None = open.
reset() is frame 0 and counts the gripper as open, as the offline code does (closed[0] = False).
Unlike Perception, `state` is required (the veto needs the end-effector pose and the finger opening, state[6]).

How the vetoed centres reach the renderer (least invasive): Perception._frame calls
measure_entities(label image) and puts the resulting dict {sid: {center, raw_center, area, reliable}} into
self.history; make_tracks() reads only `reliable` and `center` from it. PerceptionV2 builds the same dict directly
from the veto output (center = the accepted SAM3 mask centroid, reliable = accepted) and then runs the same lines
as Perception._frame (history deque of CADENCE + 1, make_tracks, renderer.render with the same arguments).
measure_entities itself is not used because it would (a) need a label image, which cannot represent the K
independent, possibly overlapping SAM3 masks, (b) re-measure the centroid after dropping components of <= 2 px,
which is not the centroid the offline pipeline stored (tv_common.centres_from_masks), and (c) know nothing of the
veto. Centres are rounded to float32, the precision of the stored offline results (all/final/<ep>.npz).
Extra attributes for logging/tests: .visible, .raw_centres, .raw_visible, .area, .masks, .first, .report,
.events (grasp / release / reacquired, as observe_only), .timing (seconds of the last call).
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from rma_online_veto import OnlineVeto  # noqa: E402
from rma_perception import CADENCE, SIZE, Perception, T, make_tracks  # noqa: E402

PACKED = SIZE * SIZE // 8  # bytes of one bit-packed mask


class TrackerProcessV2:
    """rma_tracker_v2.py in the tracker venv. Environment (all optional):
    TRACKER_PYTHON (default <tracker>/venv/bin/python), TRACKER_ROOT (legacy models, default <tracker>/assets),
    TRACKER_V2_PYTHONPATH (default <tracker>/gpu_extra:<tracker>/assets/deps), TRACKER_V2_THREADS (default 3: OMP/MKL
    thread caps), TRACKER_V2_GPU (CUDA_VISIBLE_DEVICES for the tracker only), TRACKER_V2_LEGACY (lazy | eager),
    TRACKER_V2_LOG (file for the tracker's stderr; default: inherit)."""

    def __init__(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = os.environ.get("TRACKER_V2_PYTHONPATH", f"{T}/gpu_extra:{T}/assets/deps")
        threads = os.environ.get("TRACKER_V2_THREADS", "3")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = threads
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TRACKER_THREADS", "2")
        if os.environ.get("TRACKER_V2_GPU"):
            env["CUDA_VISIBLE_DEVICES"] = os.environ["TRACKER_V2_GPU"]
        log = os.environ.get("TRACKER_V2_LOG")
        self.stderr = open(log, "ab") if log else None
        self.process = subprocess.Popen(
            [os.environ.get("TRACKER_PYTHON", f"{T}/venv/bin/python"), str(CODE / "rma_tracker_v2.py"),
             "--root", os.environ.get("TRACKER_ROOT", f"{T}/assets"), "--legacy", os.environ.get("TRACKER_V2_LEGACY", "lazy")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, env=env, cwd=str(CODE))
        ready = self.process.stdout.readline()
        if ready.strip() != b"READY":
            raise RuntimeError(f"Tracker v2 failed to start: {ready!r}")
        self.k = None

    def _send(self, header, rgb=None):
        self.process.stdin.write(header.encode() + b"\n" + (b"" if rgb is None else np.ascontiguousarray(rgb, np.uint8).tobytes()))
        self.process.stdin.flush()

    def _line(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Tracker v2 process ended ({self.process.poll()})")
        return json.loads(line)

    def _masks(self):
        data = self.process.stdout.read(self.k * PACKED)
        if len(data) != self.k * PACKED:
            raise RuntimeError(f"Tracker v2 process ended ({self.process.poll()})")
        packed = np.frombuffer(data, np.uint8).reshape(self.k, SIZE, SIZE // 8)
        return np.unpackbits(packed, axis=-1).astype(bool)

    def reset(self, rgb, task_id):
        """-> report, first-frame (method B) masks [K,256,256], SAM3 masks of frame 0 [K,256,256]"""
        self._send("RESET " + json.dumps({"task": int(task_id)}), rgb)
        report = self._line()
        self.k = int(report["k"])
        return report, self._masks(), self._masks()

    def step(self, rgb):
        self._send("STEP", rgb)
        return self._masks()

    def stat(self):
        self._send("STAT")
        return self._line()

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
        if self.stderr is not None:
            self.stderr.close()


class PerceptionV2(Perception):
    def __init__(self, task_id, tracker=None, renderer=None):
        super().__init__(task_id, tracker=tracker or TrackerProcessV2(), renderer=renderer)
        self.veto = OnlineVeto(task_id)
        self.timing = {}

    def _frame_v2(self, rgb, masks, centres, visible):
        """Perception._frame with the entity records taken from the veto instead of measure_entities(labels)."""
        veto = self.veto
        entities = {}
        for k, sid in enumerate(self.registry):  # registry keys are "1".."K" in object order
            seen = bool(visible[k])
            raw = [float(v) for v in np.float32(veto.raw_centres[k])] if veto.raw_visible[k] else None
            entities[sid] = {"center": [float(v) for v in np.float32(centres[k])] if seen else None,
                             "raw_center": raw, "area": int(veto.area[k]), "reliable": seen}
        self.history.append({"frame": self.frame, "entities": entities})
        self.centres = [entities[sid]["center"] for sid in self.registry]
        self.visible = [entities[sid]["reliable"] for sid in self.registry]
        self.raw_centres, self.raw_visible, self.area = [entities[sid]["raw_center"] for sid in self.registry], veto.raw_visible, veto.area
        self.masks, self.events = masks, veto.events
        tracks = make_tracks(list(self.history), self.registry)
        png, _ = self.renderer.render(np.ascontiguousarray(rgb, np.uint8), self.frame,
                                      max(0, self.frame - CADENCE), self.registry, tracks)
        self.png = png
        self.frame += 1
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    @staticmethod
    def _check(rgb, state):
        rgb = np.asarray(rgb)
        if rgb.shape != (SIZE, SIZE, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"PerceptionV2 needs a {SIZE}x{SIZE}x3 uint8 front image, got {rgb.shape} {rgb.dtype}")
        if state is None or len(state) < 7:
            raise ValueError("PerceptionV2 needs the 8-d state (ee_pos3, axis-angle3, finger qpos2)")
        return rgb

    def reset(self, rgb, state=None):
        rgb = self._check(rgb, state)
        self.frame = 0
        self.history = deque(maxlen=CADENCE + 1)
        t0 = time.perf_counter()
        self.report, self.first, masks = self.tracker.reset(rgb, self.task_id)
        t1 = time.perf_counter()
        centres, visible = self.veto.reset(masks, rgb, state)
        t2 = time.perf_counter()
        image = self._frame_v2(rgb, masks, centres, visible)
        self.timing = {"tracker": t1 - t0, "veto": t2 - t1, "render": time.perf_counter() - t2}
        return image

    def step(self, rgb, state=None, last_gripper_cmd=None):
        rgb = self._check(rgb, state)
        t0 = time.perf_counter()
        masks = self.tracker.step(rgb)
        t1 = time.perf_counter()
        centres, visible = self.veto.step(masks, rgb, state, last_gripper_cmd)
        t2 = time.perf_counter()
        image = self._frame_v2(rgb, masks, centres, visible)
        self.timing = {"tracker": t1 - t0, "veto": t2 - t1, "render": time.perf_counter() - t2}
        return image
