"""Trajectory image for RoboMemArena, v5: evaluation-time twin of the training overlays built from all/final_v5.

Per frame (causal):
  1. checklist signal  rma_checklist_v5.ChecklistSignalV5 (policy memory -> current item -> RetireSignalV5 retirement
                       flags + "pick up the <object>" object)  [tests: LabelSignal = the demo's subtask labels]
  2. masks             rma_tracker_v5.py subprocess (tracker venv): per-task first frame (method B / B2 / appearance 18-19),
                       SAM3 stream, for 18/19 the whole track5b procedure (it needs the gripper pixel, closed, pick)
  3. rules             rma_online_veto_v5.OnlineVetoV5 (per-task v5 configuration, retirement flags of step 1)
  4. drawing           overlay_v2r.OverlayState (unchanged copy of build_v2r/overlay_v2r.py: only finite centres of
                       non-retired objects are drawn, the trail of an object is cleared when it is retired, 21-frame
                       window) -> data_common.make_tracks -> CanonicalRenderer.render, as rma_build_v2r.py does.
                       Centres outside the image are not drawn (overlay_v2r.load_tracks does the same for the training
                       images); centres pass through float32 as the stored tracks do.
Interface:
  p = PerceptionV5(task_id)
  overlay = p.reset(rgb256, state8, checklist=None)                      frame 0 (gripper counted as open)
  overlay = p.step(rgb256, state8, last_gripper_cmd=None, checklist=None)  every later frame
  checklist = the policy's current memory text (or, with label_mode=True, the demo's current subtask text)
  -> overlay RGB uint8 [224,224,3]; also .png (renderer bytes), .centres (list [x,y] | None), .visible, .retired,
     .masks, .timing {tracker, veto, render, signal} seconds of the last call, .tracker_info, .signal_log
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import overlay_v2r  # noqa: E402  (unchanged copy of build_v2r/overlay_v2r.py)
from rma_checklist_v5 import ChecklistSignalV5, LabelSignal  # noqa: E402
from rma_online_veto_v5 import OnlineVetoV5  # noqa: E402
from rma_perception import SIZE, T, Perception, make_tracks  # noqa: E402

PACKED = SIZE * SIZE // 8


class TrackerProcessV5:
    """rma_tracker_v5.py in the tracker venv. Environment (optional): TRACKER_PYTHON, TRACKER_ROOT,
    TRACKER_V5_PYTHONPATH (default <tracker>/gpu_extra:<tracker>/assets/deps), TRACKER_V5_THREADS (default 3),
    TRACKER_V5_GPU (CUDA_VISIBLE_DEVICES for the tracker only), TRACKER_V5_LEGACY (lazy | eager), TRACKER_V5_LOG."""

    def __init__(self):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CHECKS")}
        env["PYTHONPATH"] = os.environ.get("TRACKER_V5_PYTHONPATH", f"{T}/gpu_extra:{T}/assets/deps")
        threads = os.environ.get("TRACKER_V5_THREADS", "3")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = threads
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TRACKER_THREADS", "2")
        if os.environ.get("TRACKER_V5_GPU"):
            env["CUDA_VISIBLE_DEVICES"] = os.environ["TRACKER_V5_GPU"]
        log = os.environ.get("TRACKER_V5_LOG")
        self.stderr = open(log, "ab") if log else None
        self.process = subprocess.Popen(
            [os.environ.get("TRACKER_PYTHON", f"{T}/venv/bin/python"), str(CODE / "rma_tracker_v5.py"),
             "--root", os.environ.get("TRACKER_ROOT", f"{T}/assets"), "--legacy", os.environ.get("TRACKER_V5_LEGACY", "lazy")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, env=env, cwd=str(CODE))
        ready = self.process.stdout.readline()
        if ready.strip() != b"READY":
            raise RuntimeError(f"Tracker v5 failed to start: {ready!r}")
        self.k = None

    def _send(self, header, rgb=None):
        self.process.stdin.write(header.encode() + b"\n" + (b"" if rgb is None else np.ascontiguousarray(rgb, np.uint8).tobytes()))
        self.process.stdin.flush()

    def _line(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Tracker v5 process ended ({self.process.poll()})")
        return json.loads(line)

    def _masks(self):
        data = self.process.stdout.read(self.k * PACKED)
        if len(data) != self.k * PACKED:
            raise RuntimeError(f"Tracker v5 process ended ({self.process.poll()})")
        return np.unpackbits(np.frombuffer(data, np.uint8).reshape(self.k, SIZE, SIZE // 8), axis=-1).astype(bool)

    def reset(self, rgb, task_id, state):
        self._send("RESET " + json.dumps({"task": int(task_id), "state": [float(v) for v in state]}), rgb)
        report = self._line()
        self.k = int(report["k"])
        return report, self._masks()

    def step(self, rgb, state, closed, pick):
        self._send("STEP " + json.dumps({"state": [float(v) for v in state], "closed": bool(closed),
                                         "pick": None if pick is None else int(pick)}), rgb)
        info = self._line()
        return info, self._masks()

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


class PerceptionV5(Perception):
    def __init__(self, task_id, tracker=None, renderer=None, label_mode=False):
        super().__init__(task_id, tracker=tracker or TrackerProcessV5(), renderer=renderer)
        self.label_mode = label_mode
        self.timing = {}

    def _signal(self, checklist, cmd, state):
        try:
            return self.signal.update(checklist, cmd, state)
        except Exception as exc:  # ChecklistSignalV5 does not raise; belt and braces
            self.signal_log.append((self.frame, f"signal failure {type(exc).__name__}"))
            return sorted(getattr(self.signal.sig, "retired", [])), None

    def _draw(self, rgb, centres, retired):
        c = np.asarray(centres, np.float32).copy()
        with np.errstate(invalid="ignore"):
            outside = ~((c[:, 0] >= 0) & (c[:, 0] < SIZE) & (c[:, 1] >= 0) & (c[:, 1] < SIZE))
        c[outside] = np.nan
        row = np.zeros(len(c), bool)
        row[list(retired)] = True
        tracks, frame, start = self.overlay.step(c, row)
        png, _ = self.renderer.render(np.ascontiguousarray(rgb, np.uint8), frame, start, self.registry, tracks)
        self.png = png
        drawn = np.isfinite(c).all(axis=1) & ~self.overlay.retired
        self.centres = [[float(v) for v in c[j]] if drawn[j] else None for j in range(len(c))]
        self.visible, self.retired = drawn.tolist(), self.overlay.retired.copy()
        self.frame = self.overlay.frame
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    @staticmethod
    def _check(rgb, state):
        rgb = np.asarray(rgb)
        if rgb.shape != (SIZE, SIZE, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"PerceptionV5 needs a {SIZE}x{SIZE}x3 uint8 front image, got {rgb.shape} {rgb.dtype}")
        if state is None or len(state) < 7:
            raise ValueError("PerceptionV5 needs the 8-d state (ee_pos3, axis-angle3, finger qpos2)")
        return rgb

    def reset(self, rgb, state=None, checklist=None):
        rgb = self._check(rgb, state)
        self.frame = 0
        self.veto = OnlineVetoV5(self.task_id)
        self.signal = LabelSignal(self.task_id) if self.label_mode else ChecklistSignalV5(self.task_id)
        self.signal_log = getattr(self.signal, "log", [])
        self.overlay = overlay_v2r.OverlayState(self.registry, make_tracks)
        t0 = time.perf_counter()
        retired, pick = self._signal(checklist, None, state)
        t1 = time.perf_counter()
        self.report, masks = self.tracker.reset(rgb, self.task_id, state)
        self.tracker_info = self.report
        t2 = time.perf_counter()
        centres, visible = self.veto.reset(masks, rgb, state, retired=retired)
        t3 = time.perf_counter()
        self.masks, self.pick, self.raw_centres = masks, pick, self.veto.raw_centres
        image = self._draw(rgb, centres, retired)
        self.timing = {"signal": t1 - t0, "tracker": t2 - t1, "veto": t3 - t2, "render": time.perf_counter() - t3}
        return image

    def step(self, rgb, state=None, last_gripper_cmd=None, checklist=None):
        rgb = self._check(rgb, state)
        t0 = time.perf_counter()
        retired, pick = self._signal(checklist, last_gripper_cmd, state)
        t1 = time.perf_counter()
        closed = bool(last_gripper_cmd is not None and float(last_gripper_cmd) > 0)
        self.tracker_info, masks = self.tracker.step(rgb, state, closed, pick)
        t2 = time.perf_counter()
        centres, visible = self.veto.step(masks, rgb, state, last_gripper_cmd, retired=retired)
        t3 = time.perf_counter()
        self.masks, self.pick, self.raw_centres = masks, pick, self.veto.raw_centres
        image = self._draw(rgb, centres, retired)
        self.timing = {"signal": t1 - t0, "tracker": t2 - t1, "veto": t3 - t2, "render": time.perf_counter() - t3}
        return image
