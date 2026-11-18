"""Trajectory image for RoboMemArena from RGB only: named-object tracker + the canonical renderer.

Used for both training images (rma_build.py) and evaluation (rma_adapter.py), so the two match.
Runs in the simulator venv (playwright renderer); the tracker runs as a subprocess in its own venv.

Perception(task_id).reset(rgb256, state8) / .step(rgb256, state8) -> overlay RGB uint8[224,224,3]
Also .centres: list of [x, y] or None per object for the latest frame.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import rma_spec  # noqa: E402

DRIVER = Path(os.environ.get("LIBERO_SIMULATOR_DRIVER_ROOT", "/cpfs01/USER/pi05-full-t06-t08"))
sys.path.insert(0, str(DRIVER))
from data_common import make_tracks, measure_entities  # noqa: E402
from render_adapter import CanonicalRenderer  # noqa: E402

SIZE = 256
CADENCE = 20
T = Path("/data/lgd/pi05-abc/tracker")


class TrackerProcess:
    def __init__(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = os.environ.get("TRACKER_PYTHONPATH", f"{T}/gpu_extra:{CODE}")
        env.setdefault("TRACKER_THREADS", "2")
        self.process = subprocess.Popen(
            [os.environ.get("TRACKER_PYTHON", f"{T}/venv/bin/python"), str(CODE / "rma_tracker.py"),
             "--root", os.environ.get("TRACKER_ROOT", f"{T}/assets")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=str(CODE))
        ready = self.process.stdout.readline()
        if ready.strip() != b"READY":
            raise RuntimeError(f"Tracker failed to start: {ready!r}")

    def _send(self, header, rgb):
        self.process.stdin.write(header.encode() + b"\n" + np.ascontiguousarray(rgb, np.uint8).tobytes())
        self.process.stdin.flush()

    def _labels(self):
        data = self.process.stdout.read(SIZE * SIZE)
        if len(data) != SIZE * SIZE:
            raise RuntimeError(f"Tracker process ended ({self.process.poll()})")
        return np.frombuffer(data, np.uint8).reshape(SIZE, SIZE)

    def reset(self, rgb, prompts):
        self._send("RESET " + json.dumps(prompts), rgb)
        report = json.loads(self.process.stdout.readline())
        return report, self._labels()

    def step(self, rgb):
        self._send("STEP", rgb)
        return self._labels()

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()


class Perception:
    def __init__(self, task_id, tracker=None, renderer=None):
        self.task_id = task_id
        self.spec = rma_spec.TASKS[task_id]
        self.registry = rma_spec.registry(task_id)
        self.tracker = tracker or TrackerProcess()
        self.renderer = renderer or CanonicalRenderer(CODE / "assets/entity_paths_model_renderer.js")
        self.report = None

    def _frame(self, rgb, labels):
        entities = measure_entities(labels.astype(np.int64), self.registry, minimum_visible_pixels=rma_spec.MIN_PIXELS)
        self.history.append({"frame": self.frame, "entities": entities})
        self.centres = [entities[sid]["center"] for sid in self.registry]
        tracks = make_tracks(list(self.history), self.registry)
        png, _ = self.renderer.render(np.ascontiguousarray(rgb, np.uint8), self.frame,
                                      max(0, self.frame - CADENCE), self.registry, tracks)
        self.png = png
        self.frame += 1
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def reset(self, rgb, state=None):
        self.frame = 0
        self.history = deque(maxlen=CADENCE + 1)
        self.report, labels = self.tracker.reset(rgb, rma_spec.detector_objects(self.task_id))
        return self._frame(rgb, labels)

    def step(self, rgb, state=None):
        return self._frame(rgb, self.tracker.step(rgb))

    def close(self):
        self.tracker.close()
        self.renderer.close()
