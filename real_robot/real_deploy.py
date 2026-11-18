"""Deployment loop for the real robot (reference implementation) and its offline self-checks.

Every control step:   camera frames -> crop (real_spec.crop_resize) -> bowl tracker -> trajectory overlay
Every 20 steps:       overlay + right-wrist + clean top image, state8, instruction, previous memory, previous latents
                      -> policy server -> 20 actions (right arm, 7-dim) + new memory; the 20 actions are executed
                      open loop, then the next request is made with the memory the model just wrote.
An invalid memory text never stops the run (the previous memory is kept); the run ends when the model writes
Task:Complete, at --max-steps, or on Ctrl-C.

Three ways to run it:
  1. Perception parity (no checkpoint, no robot): replay a recorded episode through the live perception path and
     compare every overlay image with the training image built by real_build.py.
       real_deploy.py --replay LEROBOT_DIR --episode 3 --compare DATASET
  2. Open-loop replay (checkpoint, no robot): as 1, plus the policy; prints the memory the model writes and the error
     of its action chunks against the recorded actions. The scene follows the recording, not the model.
       real_deploy.py --replay LEROBOT_DIR --episode 3 --checkpoint CKPT --code CODE
  3. Robot: a class with  observe() -> {"top": uint8[480,640,3], "right_wrist": uint8[480,640,3], "state": float[14]}
     and  act(action14: float[14])  (absolute joint targets + grippers, the dataset's order), given as module:Class.
       real_deploy.py --robot my_robot:Dobot --checkpoint CKPT --code CODE [--hz 30]

Processes: this script needs the overlay renderer (Playwright Chromium; LIBERO_SIMULATOR_DRIVER_ROOT = the folder with
data_common.py and render_adapter.py). The tracker (TRACKER_PYTHON, TRACKER_PYTHONPATH) and the policy server
(POLICY_PYTHON) run as subprocesses, so the three environments may differ.
"""
import argparse
import base64
import importlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_v2r  # noqa: E402
import real_spec as S  # noqa: E402

INTERVAL = 20


class TrackerClient:
    def __init__(self, log):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if os.environ.get("TRACKER_PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["TRACKER_PYTHONPATH"]
        env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
        self.process = subprocess.Popen([os.environ.get("TRACKER_PYTHON", sys.executable), str(HERE / "real_tracker.py"), "--serve"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "a"), env=env)
        if self.process.stdout.readline().strip() != b"READY":
            raise RuntimeError(f"tracker did not start, see {log}")

    def _call(self, header, rgb):
        self.process.stdin.write(header + b"\n" + np.ascontiguousarray(rgb, np.uint8).tobytes())
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"tracker exited with {self.process.poll()}")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError("tracker: " + reply["error"])
        return np.array([[np.nan, np.nan] if c is None else c for c in reply["centres"]], np.float32), reply

    def reset(self, rgb):
        return self._call(b"RESET", rgb)

    def step(self, rgb):
        return self._call(b"STEP", rgb)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=20)


class Perception:
    """Camera frames -> the three model images. Same crop, tracker, overlay state and renderer as the dataset build."""

    def __init__(self, code, log_dir):
        sys.path.insert(0, os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
        from data_common import make_tracks
        from render_adapter import CanonicalRenderer
        self.registry = S.registry()
        self.renderer = CanonicalRenderer(Path(code) / "assets/entity_paths_model_renderer.js")
        self.state = overlay_v2r.OverlayState(self.registry, make_tracks)
        self.tracker = TrackerClient(Path(log_dir) / "tracker_stderr.log")
        self.started = False

    def reset(self):
        self.state.reset()
        self.started = False

    def step(self, top, right_wrist):
        front = S.crop_resize(top, S.TOP_CROP)
        self.centres, self.report = self.tracker.step(front) if self.started else self.tracker.reset(front)
        self.started = True
        tracks, frame, start = self.state.step(self.centres, None)
        self.overlay_png, _ = self.renderer.render(np.ascontiguousarray(front), frame, start, self.registry, tracks)
        self.raw = S.model_image(front)
        self.wrist = S.model_image(S.crop_resize(right_wrist, S.WRIST_CROP))
        return self.overlay_png

    def close(self):
        self.tracker.close()
        self.renderer.close()


class Policy:
    """policy_ens.py --serve-json CKPT: JSON lines over stdin/stdout (see code/policy.py)."""

    def __init__(self, checkpoint, code, log_dir, samples=1):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", ACTION_SAMPLES=str(samples))
        self.process = subprocess.Popen([os.environ.get("POLICY_PYTHON", sys.executable), str(Path(code) / "policy_ens.py"),
                                         "--serve-json", str(checkpoint)], cwd=str(code), env=env, text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=open(Path(log_dir) / "policy_stderr.log", "a"))

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"policy server exited with {self.process.poll()}")
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)


def png_b64(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Replay:
    """A recorded LeRobot episode played back as if it were the robot (actions are ignored)."""

    def __init__(self, root, index, raw=None):
        from real_track import decode
        root = Path(root)
        if raw is not None:  # real_extract.py output: no pandas needed in this environment
            meta = np.load(Path(raw) / "meta" / f"{S.episode_name(index)}.npz")
            self.state, self.action = meta["state14"], meta["action14"]
        else:
            import pandas as pd
            table = pd.read_parquet(root / f"data/chunk-000/episode_{index:06d}.parquet")
            self.state = np.stack(table["observation.state"].values).astype(np.float32)
            self.action = np.stack(table["action"].values).astype(np.float32)
        self.top = decode(root / f"videos/chunk-000/observation.images.top/episode_{index:06d}.mp4")
        self.wrist = decode(root / f"videos/chunk-000/observation.images.right_wrist/episode_{index:06d}.mp4")
        self.t = 0

    def __len__(self):
        return len(self.state)

    def observe(self):
        t = min(self.t, len(self) - 1)
        return {"top": self.top[t], "right_wrist": self.wrist[t], "state": self.state[t]}

    def act(self, action14):
        self.t += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--raw", type=Path, help="real_extract.py output; replay then reads states from RAW/meta instead of the parquet")
    p.add_argument("--compare", type=Path, help="dataset built by real_build.py: compare overlay images")
    p.add_argument("--robot", help="module:Class with observe() and act(action14)")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--code", type=Path, required=True, help="training code directory (policy_ens.py, memory_format.py, assets/)")
    p.add_argument("--log-dir", type=Path, default=Path("deploy_log"))
    p.add_argument("--hz", type=float, default=S.FPS)
    p.add_argument("--max-steps", type=int, default=900)
    args = p.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code.resolve()))
    from memory_format import INITIAL_MEMORY, LATENT_DIM, MAX_ENTITIES

    if args.replay:
        robot = Replay(args.replay, args.episode, args.raw)
        steps = len(robot)
    else:
        module, name = args.robot.split(":")
        robot = getattr(importlib.import_module(module), name)()
        steps = args.max_steps
    perception = Perception(args.code, args.log_dir)
    policy = Policy(args.checkpoint, args.code, args.log_dir) if args.checkpoint else None
    memory, latents = INITIAL_MEMORY, [[0.0] * LATENT_DIM for _ in range(MAX_ENTITIES)]
    plan, stopped, differing, errors = None, None, 0, []
    left_hold = None
    log = open(args.log_dir / "memory.jsonl", "a")
    try:
        for t in range(steps):
            tick = time.time()
            obs = robot.observe()
            if left_hold is None:
                left_hold = np.asarray(obs["state"], np.float32)[:7].copy()  # the left arm keeps its start pose
            overlay_png = perception.step(obs["top"], obs["right_wrist"])
            if args.compare:
                stored = (args.compare / "images" / S.episode_name(args.episode) / f"{t:06d}.png").read_bytes()
                a = np.asarray(Image.open(io.BytesIO(overlay_png)).convert("RGB"), np.int16)
                b = np.asarray(Image.open(io.BytesIO(stored)).convert("RGB"), np.int16)
                differing += int(np.abs(a - b).max() > 0)
            if policy is not None and t % INTERVAL == 0 and stopped is None:
                state8 = S.state8(np.asarray(obs["state"], np.float32)[None])[0]
                overlay = np.asarray(Image.open(io.BytesIO(overlay_png)).convert("RGB"))
                result = policy.request({"main_png_b64": png_b64(overlay), "wrist_png_b64": png_b64(perception.wrist),
                                         "raw_png_b64": png_b64(perception.raw), "state": [float(v) for v in state8],
                                         "instruction": S.PROMPT, "previous_memory_text": memory, "previous_latents": latents})
                entry = {"frame": t, "previous_memory": memory, "generated": result.get("memory_text"),
                         "memory_valid": result.get("memory_valid"), "error": result.get("error")}
                if "error" in result:
                    stopped = "policy_error"
                else:
                    plan = np.asarray(result["actions"], np.float32)  # [20,7] absolute right-arm targets
                    if result.get("latents") is not None:
                        latents = result["latents"]
                    if result.get("memory_valid") is True and isinstance(result.get("next_memory_text"), str):
                        memory = result["next_memory_text"]
                        if result.get("sequence_complete") is True:
                            stopped = "model_stop"  # this last chunk is still executed
                    else:
                        entry["lenient"] = "unparsable_kept_previous_memory"
                    if args.replay:
                        truth = S.action7(robot.action[t:t + INTERVAL])
                        entry["action_mae"] = float(np.abs(plan[:len(truth)] - truth).mean())
                        errors.append(entry["action_mae"])
                entry["stopped"] = stopped
                log.write(json.dumps(entry) + "\n"), log.flush()
                print(t, (result.get("memory_text") or str(result.get("error"))).replace("\n", " | "), entry.get("action_mae", ""), flush=True)
            if plan is not None and t % INTERVAL < len(plan):
                robot.act(S.action14(plan[t % INTERVAL], left_hold))
            elif args.replay:
                robot.act(None)
            if stopped and t % INTERVAL == INTERVAL - 1:
                break
            if args.robot:
                time.sleep(max(0.0, 1.0 / args.hz - (time.time() - tick)))
    finally:
        perception.close()
        if policy is not None:
            policy.close()
    if args.compare:
        print(f"PARITY overlay images differing from the dataset: {differing} of {steps}")
    if errors:
        print(f"OPEN_LOOP mean |action error| over {len(errors)} chunks: {np.mean(errors):.4f}")
    print("END", stopped)


if __name__ == "__main__":
    main()
