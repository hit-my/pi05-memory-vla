"""RoboMemArena adapter for our A/B/C memory policies.

The benchmark runner calls observe() on every environment step and infer_actions() whenever its action
plan is empty. Per episode this adapter keeps the Ledger-Driver memory text (and, for B/C, the model's own
latents) exactly as closedloop_ledger.py does, feeds every frame to a perception module that draws the
trajectory overlay from RGB + proprioception only, and asks a policy server subprocess for a 20-step chunk.

When the model declares the task complete, or its memory is rejected by the Ledger Driver, the
simulate.py loop would end the episode. The benchmark loop cannot be stopped from an adapter, so the arm
then idles (zero motion, gripper open) until the runner's own limits end the episode.

Factory: build_adapter(checkpoint_dir, code_dir=..., python=..., policy_script=..., gpu=None,
                       log_dir=None, orientation="eval", action_samples=1, action_seed=None)
"""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HOME = Path.home()
IDLE = np.array([0.0] * 6 + [-1.0], np.float32)
HORIZON = 20

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))  # benchmark scripts, when copied there
try:
    from policy_adapter import BasePolicyAdapter
except ImportError:  # outside the benchmark (unit tests)
    class BasePolicyAdapter:  # type: ignore[no-redef]
        pass


def quat2axisangle_libero(quat: np.ndarray) -> np.ndarray:
    """robosuite/LIBERO convention (no w>=0 flip). The dataset's obs/ee_ori uses this; the benchmark's
    observation/state flips the quaternion first, which negates the vector near angle pi."""
    quat = np.asarray(quat, np.float64).copy()
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3, np.float32)
    return (quat[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


def training_state(obs: dict) -> np.ndarray:
    """state8 = ee_pos3 + axis-angle3 + gripper_qpos2, as obs/ee_states + obs/gripper_states in the dataset."""
    raw = obs.get("_raw_obs", obs)
    return np.concatenate([np.asarray(raw["robot0_eef_pos"], np.float32),
                           quat2axisangle_libero(raw["robot0_eef_quat"]),
                           np.asarray(raw["robot0_gripper_qpos"], np.float32)]).astype(np.float32)


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def resize(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    if rgb.shape[:2] == (size, size):
        return np.ascontiguousarray(rgb, np.uint8)
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)).resize((size, size), Image.BILINEAR))


class StubPerception:
    """Placeholder until code/rma_perception.py exists: no overlay, just the resized camera image."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self.last = None

    def reset(self, rgb, state):
        self.last = resize(rgb)
        return self.last

    def step(self, rgb, state):
        self.last = resize(rgb)
        return self.last


class PolicyServer:
    def __init__(self, python: str, script: str, checkpoint: str, code_dir: str, gpu: str | None,
                 log_path: Path, action_samples: int, action_seed: int | None, timeout: float = 600):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MUJOCO_GL", "PYOPENGL_PLATFORM")}
        env["PYTHONPATH"] = code_dir
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["ACTION_SAMPLES"] = str(action_samples)
        if action_seed is not None:
            env["ACTION_SEED"] = str(action_seed)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = open(log_path, "a", encoding="utf-8")
        self.process = subprocess.Popen([python, script, "--serve-json", checkpoint], cwd=code_dir, env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True)
        self.lines: queue.Queue = queue.Queue()
        self.timeout = timeout
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def request(self, payload: dict) -> dict:
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"policy server gave no answer within {self.timeout}s") from exc
        if line is None:
            raise RuntimeError(f"policy server exited with {self.process.poll()}")
        return json.loads(line)

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()
        self.stderr.close()


class MemoryPolicyAdapter(BasePolicyAdapter):
    def __init__(self, checkpoint_dir: str, code_dir: str | None = None, python: str | None = None,
                 policy_script: str | None = None, gpu: str | None = None, log_dir: str | None = None,
                 orientation: str = "eval", action_samples: int = 1, action_seed: int | None = None):
        self.code_dir = str(code_dir or HOME / "pi05-abc/code")
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # rma_perception, rma_spec
        sys.path.insert(0, self.code_dir)  # memory_format of the checkpoint's code
        from memory_format import INITIAL_MEMORY, LATENT_DIM, MAX_ENTITIES  # noqa: E402
        self.initial_memory = INITIAL_MEMORY
        self.blank_latents = [[0.0] * LATENT_DIM for _ in range(MAX_ENTITIES)]
        try:
            from rma_perception import Perception  # noqa: E402
            self.perception_class = Perception
        except ImportError:
            self.perception_class = StubPerception
        from eval_common import TASK_PROMPTS  # noqa: E402  (benchmark module, on sys.path inside the runner)
        self.prompt_to_task = {prompt: int(key[4:]) for key, prompt in TASK_PROMPTS.items()}
        self.orientation = orientation
        self.log_dir = Path(log_dir or os.environ.get("RMA_LOG_DIR", "/tmp/rma_adapter"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.server = PolicyServer(python or str(HOME / "SANE/.venv/bin/python"),
                                   policy_script or str(Path(self.code_dir) / "policy_ens.py"),
                                   checkpoint_dir, self.code_dir, gpu, self.log_dir / "policy_stderr.log",
                                   action_samples, action_seed)
        # One perception object per runner process: its tracker subprocess and headless renderer are
        # started here, before the simulator exists, and re-initialized per episode by reset().
        self.perceptions: dict = {}
        # Object-evidence gate on checklist ticks (rma_evidence.py), on with RMA_EVIDENCE=1
        self.gate_on = os.environ.get("RMA_EVIDENCE") == "1"
        if self.gate_on:
            import ledger_v2  # noqa: E402
            import memory_format  # noqa: E402
            import rma_evidence  # noqa: E402
            import rma_spec  # noqa: E402
            self._ev, self._spec, self._mf, self._lv = rma_evidence, rma_spec, memory_format, ledger_v2
            camera = os.environ.get("RMA_CAMERA", str(HOME / "pi05-abc/rma/camera26.json"))
            self.cameras = json.loads(Path(camera).read_text())
            tasks = os.environ.get("RMA_EVIDENCE_TASKS")
            self.gate_tasks = {int(x) for x in tasks.split(",")} if tasks else set(rma_evidence.GATE_TASKS)
        self.episode = -1
        self.reset()

    # The benchmark gives images after np.flipud of the raw render. Training frames come from the dataset
    # HDF5; `orientation` maps the benchmark image to that convention ("eval" = use as given).
    def _to_training(self, image: np.ndarray) -> np.ndarray:
        if self.orientation == "eval":
            return np.ascontiguousarray(image)
        if self.orientation == "flipud":
            return np.ascontiguousarray(image[::-1])
        if self.orientation == "rot180":
            return np.ascontiguousarray(image[::-1, ::-1])
        if self.orientation == "fliplr":
            return np.ascontiguousarray(image[:, ::-1])
        raise ValueError(self.orientation)

    def reset(self) -> None:
        self.episode += 1
        self.memory = self.initial_memory
        self.latents = self.blank_latents
        self.perception = None
        self.overlay = None
        self.stopped = None
        self.calls = 0
        self.frames = 0
        self.log = []
        self.gate = None

    def observe(self, obs: dict[str, Any], prompt: str, resize_size: int) -> None:
        rgb = self._to_training(obs["observation/image"])
        state = training_state(obs)
        if self.perception is None:
            task = self.prompt_to_task.get(prompt)
            if task not in self.perceptions:
                self.perceptions[task] = self.perception_class(task)
            self.perception = self.perceptions[task]
            self.overlay = self.perception.reset(rgb, state)
            task = self.prompt_to_task.get(prompt)
            if self.gate_on and task in self.gate_tasks:
                self.gate = self._ev.EvidenceGate(task, lambda s, t=task: self._spec.subtask_object(t, s),
                                                  self.cameras[str(task)]["matrix"], self._mf, self._lv)
        else:
            self.overlay = self.perception.step(rgb, state)
        if self.gate is not None:
            centres = getattr(self.perception, "centres", None) or []
            self.gate.observe(state, list(centres))
        self.frames += 1

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        if self.overlay is None:  # runner without observe(): still keep the perception causal
            self.observe(obs, prompt, resize_size)
        if self.stopped:
            return np.tile(IDLE, (HORIZON, 1))
        state = training_state(obs)
        wrist = resize(self._to_training(obs["observation/wrist_image"]))
        request = {"main_png_b64": base64.b64encode(png_bytes(resize(self.overlay))).decode("ascii"),
                   "wrist_png_b64": base64.b64encode(png_bytes(wrist)).decode("ascii"),
                   "state": state.tolist(), "instruction": prompt,
                   "previous_memory_text": self.memory, "previous_latents": self.latents}
        started = time.perf_counter()
        result = self.server.request(request)
        self.calls += 1
        entry = {"episode": self.episode, "call": self.calls, "frame": self.frames,
                 "seconds": round(time.perf_counter() - started, 3)}
        if "error" in result:
            self.stopped = "policy_error"
            entry["error"] = result["error"]
            self._write(entry)
            return np.tile(IDLE, (HORIZON, 1))
        entry.update(generated=result.get("memory_text"), memory_valid=result.get("memory_valid"),
                     sequence_complete=result.get("sequence_complete"))
        actions = np.asarray(result["actions"], np.float32)
        if result.get("latents") is not None:
            self.latents = result["latents"]
        gated = self.gate.apply(result.get("memory_text")) if self.gate is not None else None
        if gated is not None:  # evidence veto replaces the strict driver; malformed text falls through
            self.memory, complete, entry["gate"] = gated
            self.stopped = "model_stop" if complete else None
        elif result.get("memory_valid") is not True or not isinstance(result.get("next_memory_text"), str):
            self.stopped = "invalid_model_memory"  # simulate.py executes this chunk, then ends the episode
        else:
            self.memory = result["next_memory_text"]
            if result.get("sequence_complete") is True:
                self.stopped = "model_stop"
        entry["stopped"] = self.stopped
        self._write(entry)
        return actions

    def _write(self, entry):
        with open(self.log_dir / "memory.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self):
        for perception in self.perceptions.values():
            close = getattr(perception, "close", None)
            if callable(close):
                close()
        self.server.close()


def build_adapter(**kwargs: Any) -> MemoryPolicyAdapter:
    return MemoryPolicyAdapter(**kwargs)
