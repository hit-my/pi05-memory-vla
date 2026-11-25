"""Deployment loop of real_deploy.py with the external agent of real_agent.py: checklist verification, memory
maintenance and recovery from a failed grasp.

Every control step:  camera frames -> tracker -> trajectory overlay (as real_deploy.py); the right-arm state is recorded.
Every 20 steps:      the verdicts that have arrived are applied to the memory (a false tick is removed); the policy is
                     asked with that memory; its 20 actions are executed; its new ticks are handed to the checker, which
                     answers in the background.
After a false grasp: the gripper opens, the arm returns along its own recorded joint states to where the attempt began,
                     and the policy continues from the memory it had there with the failed item not ticked.
When the policy writes Task:Complete: the arm holds still until the outstanding verdicts are in (at most
                     --verdict-timeout seconds); a negative one cancels the stop.

    export CHECKER_API_URL=... CHECKER_MODEL=... CHECKER_API_KEY=...        # see real_checker.py
    real_deploy_agent.py --robot my_robot:Dobot --checkpoint CKPT --code CODE [--hz 30] [--save-dir checks]
    real_deploy_agent.py ... --no-recover            # memory maintenance only, the arm is never driven by the agent
    real_deploy_agent.py ... --no-agent              # identical to real_deploy.py (for a paired comparison)

The robot class is the one real_deploy.py uses: observe() -> {"top", "right_wrist", "state"[14]}, act(action14).
Requests are synchronous here (the arm waits while the policy computes), as in real_deploy.py; the agent is not yet
combined with the pause-free loop of real_deploy_async.py.

SAFETY: during a recovery the arm moves without the policy. It retraces states it has already been in, with the gripper
open, no faster than it moved on the way there, but anything that entered the workspace since then is not seen. Keep the
emergency stop at hand, and run with --no-recover first.
"""
import argparse
import base64
import importlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_agent  # noqa: E402
import real_spec as S  # noqa: E402

INTERVAL = 20
OPEN_COMMAND = 0.93   # gripper command of an open gripper in the demonstrations (real_spec.py)


def png_b64(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run(robot, perception, policy, supervisor, initial, max_steps, hz=0.0, verdict_timeout=20.0, log=None, out=print):
    """The control loop. supervisor may be None (then this is real_deploy.py's robot mode). Returns the stop reason."""
    memory, latents = initial
    plan, index, call, stopped, recovery = None, 0, 0, None, None
    left_hold, closed, last, hold_since = None, False, None, None
    for t in range(max_steps):
        tick = time.time()
        obs = robot.observe()
        state14 = np.asarray(obs["state"], np.float32)
        state7 = state14[S.RIGHT]
        if left_hold is None:
            left_hold = state14[:7].copy()   # the left arm keeps its start pose
        overlay_png = perception.step(obs["top"], obs["right_wrist"], closed)
        if supervisor:
            supervisor.record(t, state7)
        action = None

        if recovery is not None:                                   # the agent is driving the arm back
            action = recovery.next(state7)
            if action is None:
                memory, latents = supervisor.finish_recovery(call, t, perception.raw, recovery, state7)
                recovery, plan = None, None
                out(t, "RECOVERED; memory:", memory.replace("\n", " | "))
        if recovery is None and (plan is None or index >= len(plan)):   # a request boundary
            if stopped == "model_stop":                             # the last chunk is done: wait for the verdicts
                if supervisor:
                    supervisor.flush(perception.raw, perception.wrist, t)
                    hold_since = hold_since or time.time()
                    if supervisor.outstanding() and time.time() - hold_since < verdict_timeout:
                        supervisor.wait(0.2)
                    if supervisor.outstanding() and time.time() - hold_since < verdict_timeout:
                        action = last
                    else:
                        checked, latents, recovery = supervisor.apply(call, t, memory, latents, state7, snapshot=False)
                        if checked != memory:
                            memory, stopped, hold_since, plan = checked, None, None, None
                            out(t, "STOP CANCELLED by the agent; memory:", memory.replace("\n", " | "))
                            action = last
                        else:
                            break
                else:
                    break
            elif stopped is None:
                call += 1
                if supervisor:
                    memory, latents, recovery = supervisor.apply(call, t, memory, latents, state7)
                if recovery is not None:
                    plan = None
                    out(t, "RECOVERY started; memory:", memory.replace("\n", " | "))
                    action = recovery.next(state7)
                else:
                    overlay = np.asarray(Image.open(io.BytesIO(overlay_png)).convert("RGB"))
                    result = policy.request({"main_png_b64": png_b64(overlay), "wrist_png_b64": png_b64(perception.wrist),
                                             "raw_png_b64": png_b64(perception.raw), "state": [float(v) for v in S.state8(state14[None])[0]],
                                             "instruction": S.PROMPT, "previous_memory_text": memory, "previous_latents": latents})
                    entry = {"frame": t, "call": call, "previous_memory": memory, "generated": result.get("memory_text"),
                             "memory_valid": result.get("memory_valid"), "error": result.get("error"), "gripper_closed": closed}
                    if "error" in result:
                        stopped = "policy_error"
                    else:
                        given = memory
                        plan, index = np.asarray(result["actions"], np.float32), 0   # [20,7] absolute right-arm targets
                        if result.get("latents") is not None:
                            latents = result["latents"]
                        if result.get("memory_valid") is True and isinstance(result.get("next_memory_text"), str):
                            memory = result["next_memory_text"]
                            if result.get("sequence_complete") is True:
                                stopped = "model_stop"   # this last chunk is still executed
                        else:
                            entry["lenient"] = "unparsable_kept_previous_memory"
                        if supervisor:
                            supervisor.after_reply(call, t, given, result, memory, perception.raw, perception.wrist)
                    entry["stopped"] = stopped
                    if log:
                        log.write(json.dumps(entry) + "\n"), log.flush()
                    out(t, (result.get("memory_text") or str(result.get("error"))).replace("\n", " | "))
                    if stopped == "policy_error":
                        break
        if action is None and plan is not None and index < len(plan):
            action, index = plan[index], index + 1
        if action is not None:
            last = np.asarray(action, np.float32)
            robot.act(S.action14(last, left_hold))
            closed = bool(last[S.GRIPPER - S.RIGHT.start] < S.COMMAND_CLOSED_BELOW)
        if hz:
            time.sleep(max(0.0, 1.0 / hz - (time.time() - tick)))
    out("END", stopped)
    return stopped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, help="module:Class with observe() and act(action14)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True, help="training code directory (policy_ens.py, memory_format.py, assets/)")
    p.add_argument("--log-dir", type=Path, default=Path("deploy_log"))
    p.add_argument("--hz", type=float, default=S.FPS)
    p.add_argument("--max-steps", type=int, default=2400)
    p.add_argument("--no-agent", action="store_true", help="no checker at all (baseline)")
    p.add_argument("--no-recover", action="store_true", help="the agent edits the memory but never moves the arm")
    p.add_argument("--fix-observations", action="store_true", help="also check 'cup: has block / empty' items and rewrite a contradicted one")
    p.add_argument("--threshold", type=float, default=0.5, help="a tick is removed if P(done) is below this")
    p.add_argument("--wait-calls", type=int, default=1, help="policy requests between a tick and the look at its outcome")
    p.add_argument("--back-calls", type=int, default=5, help="recovery: go back to this many requests before the false tick (not before the start of that step)")
    p.add_argument("--rewind-speed", type=float, default=1.0, help="recovery: per-step joint change relative to the fastest recorded step")
    p.add_argument("--verdict-timeout", type=float, default=20.0, help="seconds to wait for verdicts before a stop")
    p.add_argument("--save-dir", type=Path, help="save every checker question (image, prompt, answer)")
    args = p.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code.resolve()))
    from memory_format import INITIAL_MEMORY, LATENT_DIM, MAX_ENTITIES
    from real_deploy import Perception, Policy

    module, name = args.robot.split(":")
    robot = getattr(importlib.import_module(module), name)()
    perception = Perception(args.code, args.log_dir)
    policy = Policy(args.checkpoint, args.code, args.log_dir)
    supervisor = None
    if not args.no_agent:
        from real_checker import VLMChecker
        supervisor = real_agent.Supervisor(VLMChecker(save_dir=args.save_dir), S.PROMPT, fps=args.hz or S.FPS, threshold=args.threshold,
                                           wait_calls=args.wait_calls, recover=not args.no_recover, back_calls=args.back_calls,
                                           open_command=OPEN_COMMAND, fix_observations=args.fix_observations,
                                           log_path=args.log_dir / "agent.jsonl", recovery_options={"speed": args.rewind_speed})
    log = open(args.log_dir / "memory.jsonl", "a")
    try:
        run(robot, perception, policy, supervisor, (INITIAL_MEMORY, [[0.0] * LATENT_DIM for _ in range(MAX_ENTITIES)]),
            args.max_steps, args.hz, args.verdict_timeout, log)
    finally:
        perception.close()
        policy.close()
        if supervisor:
            supervisor.close()


if __name__ == "__main__":
    main()
