"""RoboMemArena adapter v5 = rma_adapter_v2.py + rma_adapter_v2.gripper.patch + the v5 perception (code_eval_v5).

Differences from code_causal_v2/rma_adapter_v2.py (with its gripper patch applied):
  * perception = rma_perception_v5.PerceptionV5 (per-task v5 tracking, retirement, overlay_v2r drawing; see its docstring);
  * every observe() passes the policy's current memory text (`self.memory`) as the checklist, and the gripper command
    executed before the frame (the patch's _last_gripper_cmd); frame 0 of the perception = the first frame the runner
    hands to the adapter, i.e. the first frame after the runner's warm-up steps (as frame 0 of the demos);
  * the request carries the three images (overlay, wrist, clean front 224 BILINEAR), as v2;
  * an invalid / unparsable memory never ends the episode (kept previous memory, as v2's lenient branch) and never
    reaches the perception as an exception (rma_checklist_v5 treats it as "no new information");
  * logging: per policy call the perception timing (mean ms per frame since the previous call) and the checklist /
    retirement state; optional frame dumps with RMA_SAVE_FRAMES=<dir> (every RMA_SAVE_EVERY frames, default 10:
    overlay + raw + wrist as sent).
The evidence gate of v2 (RMA_EVIDENCE=1) is kept but is not used in this round.
Factory: build_adapter(checkpoint_dir, code_dir=..., python=..., policy_script=..., gpu=None, log_dir=None,
                       orientation="eval", action_samples=1, action_seed=None)
code_dir = the code of the CHECKPOINT (policy server + memory_format), default ~/pi05-abc/code_causal_v2.
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
        self.code_dir = str(code_dir or HOME / "pi05-abc/code_causal_v2")
        here = str(Path(__file__).resolve().parent)
        sys.path.insert(0, self.code_dir)  # memory_format of the checkpoint's code
        from memory_format import INITIAL_MEMORY, LATENT_DIM, MAX_ENTITIES  # noqa: E402
        sys.path.insert(0, here)           # rma_perception_v5 & co. (code_eval_v5) first
        self.initial_memory = INITIAL_MEMORY
        self.blank_latents = [[0.0] * LATENT_DIM for _ in range(MAX_ENTITIES)]
        from rma_perception_v5 import PerceptionV5  # noqa: E402  (no silent fallback: v5 must load)
        self.perception_class = PerceptionV5
        from eval_common import TASK_PROMPTS  # noqa: E402  (benchmark module, on sys.path inside the runner)
        self.prompt_to_task = {prompt: int(key[4:]) for key, prompt in TASK_PROMPTS.items()}
        self.orientation = orientation
        self.log_dir = Path(log_dir or os.environ.get("RMA_LOG_DIR", "/tmp/rma_adapter"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(os.environ["RMA_SAVE_FRAMES"]) if os.environ.get("RMA_SAVE_FRAMES") else None
        self.save_every = int(os.environ.get("RMA_SAVE_EVERY", "10"))
        self.server = PolicyServer(python or str(HOME / "SANE/.venv/bin/python"),
                                   policy_script or str(Path(self.code_dir) / "policy_ens.py"),
                                   checkpoint_dir, self.code_dir, gpu, self.log_dir / "policy_stderr.log",
                                   action_samples, action_seed)
        self.perceptions: dict = {}
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
        self.raw = None
        self.stopped = None
        self.calls = 0
        self.frames = 0
        self.plan = None  # the action chunk being executed by the runner, and the frame it was issued at
        self.plan_frame = 0
        self.log = []
        self.gate = None
        self.perc_times = []
        self.task = None

    def observe(self, obs: dict[str, Any], prompt: str, resize_size: int) -> None:
        rgb = self._to_training(obs["observation/image"])
        self.raw = rgb  # clean frame of the overlay below
        state = training_state(obs)
        started = time.perf_counter()
        if self.perception is None:
            task = self.prompt_to_task.get(prompt)
            self.task = task
            if task not in self.perceptions:
                self.perceptions[task] = self.perception_class(task)
            self.perception = self.perceptions[task]
            self.overlay = self.perception.reset(rgb, state, checklist=self.memory)
            if self.gate_on and task in self.gate_tasks:
                self.gate = self._ev.EvidenceGate(task, lambda s, t=task: self._spec.subtask_object(t, s),
                                                  self.cameras[str(task)]["matrix"], self._mf, self._lv)
        else:
            self.overlay = self.perception.step(rgb, state, last_gripper_cmd=self._last_gripper_cmd(), checklist=self.memory)
        self.perc_times.append((time.perf_counter() - started, dict(self.perception.timing)))
        if self.gate is not None:
            centres = getattr(self.perception, "centres", None) or []
            self.gate.observe(state, list(centres))
        if self.save_dir is not None and self.frames % self.save_every == 0:
            folder = self.save_dir / f"ep{self.episode}"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(resize(self.overlay)).save(folder / f"{self.frames:05d}_overlay.png")
            Image.fromarray(resize(rgb)).save(folder / f"{self.frames:05d}_raw.png")
            Image.fromarray(resize(self._to_training(obs["observation/wrist_image"]))).save(folder / f"{self.frames:05d}_wrist.png")
        self.frames += 1

    def _last_gripper_cmd(self):
        """Gripper command executed between the previous observed frame and this one (None = none yet = open).
        The runner executes the latest chunk row by row, starting with row 0 right after infer_actions()."""
        if self.plan is None:
            return None
        row = min(max(self.frames - 1 - self.plan_frame, 0), len(self.plan) - 1)
        return float(self.plan[row][6])

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        actions = self._infer_actions(obs, prompt, resize_size)
        self.plan, self.plan_frame = np.asarray(actions, np.float32), self.frames - 1  # issued at the frame just observed
        return actions

    def _perception_summary(self):
        times, self.perc_times = self.perc_times, []
        if not times:
            return {}
        p = self.perception
        out = {"frames": len(times), "perception_ms": round(1000 * float(np.mean([a for a, _ in times])), 1)}
        for key in ("tracker", "veto", "render", "signal"):
            out[f"{key}_ms"] = round(1000 * float(np.mean([b.get(key, 0.0) for _, b in times])), 1)
        try:
            out["checklist_item"] = [p.signal.index, p.signal.text]
            out["retired"] = [int(j) for j in np.flatnonzero(p.retired)]
            out["drawn"] = [c is not None for c in p.centres]
            if p.signal_log:
                out["signal_log"] = p.signal_log[-5:]
        except Exception:
            pass
        return out

    def _infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        if self.overlay is None:  # runner without observe(): still keep the perception causal
            self.observe(obs, prompt, resize_size)
        if self.stopped:
            return np.tile(IDLE, (HORIZON, 1))
        state = training_state(obs)
        wrist = resize(self._to_training(obs["observation/wrist_image"]))
        request = {"main_png_b64": base64.b64encode(png_bytes(resize(self.overlay))).decode("ascii"),
                   "wrist_png_b64": base64.b64encode(png_bytes(wrist)).decode("ascii"),
                   "raw_png_b64": base64.b64encode(png_bytes(resize(self.raw))).decode("ascii"),
                   "state": state.tolist(), "instruction": prompt,
                   "previous_memory_text": self.memory, "previous_latents": self.latents}
        started = time.perf_counter()
        result = self.server.request(request)
        self.calls += 1
        entry = {"episode": self.episode, "call": self.calls, "frame": self.frames,
                 "seconds": round(time.perf_counter() - started, 3), "perception": self._perception_summary()}
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
            entry["lenient"] = "unparsable_kept_previous_memory"  # lenient evaluation: keep going
        else:
            self.memory = result["next_memory_text"]
            if result.get("sequence_complete") is True:
                self.stopped = "model_stop"
        if result.get("lenient"):
            entry["lenient"] = result["lenient"]
        entry["stopped"] = self.stopped
        self._write(entry)
        return actions

    def _write(self, entry):
        with open(self.log_dir / "memory.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def close(self):
        for perception in self.perceptions.values():
            close = getattr(perception, "close", None)
            if callable(close):
                close()
        self.server.close()


def build_adapter(**kwargs: Any) -> MemoryPolicyAdapter:
    return MemoryPolicyAdapter(**kwargs)
