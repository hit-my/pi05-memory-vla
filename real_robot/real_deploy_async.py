"""Asynchronous deployment loop: the arm keeps moving while the policy computes, and the model still sees what it was
trained on. Same perception, policy server, robot interface and replay modes as real_deploy.py (imported from it).

What training fixes (real_build.py), and how this loop keeps each of them:
  1. A request made with the observation of control step t returns the 20 actions for steps t .. t+19.
       -> The reply arrives d steps later. Its first d actions are skipped and action[s - t] is executed at step s
          (the actions are absolute joint targets, so this needs no correction). Nothing is executed late.
  2. The "previous memory" and "previous latents" of the request at step t are the ones written at step t - 20,
     and the Action line describes those 20 steps. real_build.py writes such a row for EVERY frame t.
       -> Requests are made every --period steps (default 10), but each request is given the memory and latents
          written by the request made --lag (20) steps earlier, not by the latest one. With period 10 this gives two
          interleaved chains (steps 0,20,40.. and 10,30,50..), each exactly the training situation. If the period does
          not divide 20, the latest request that is at least 20 steps old is used (period 12 -> 24 steps).
          Feeding the latest memory instead (12 steps old) makes the model tick one pick/place twice, run ahead of the
          checklist and write Task:Complete early; see README "异步部署".
  3. The images and the state of a request belong to one control step.
       -> They are captured together at step t, before the request is handed to the worker thread.
Not covered: two consecutive chunks are sampled independently, so the target can jump a little where one takes over
from the other. --blend N cross-fades over the first N steps of a new chunk (default 4, 0 = off). The pause-free
alternative that also removes this is prefix-conditioned sampling ("real-time chunking"), which needs a model change.

Timing rule: a reply must arrive before the previous chunk runs out, i.e. latency (in control steps) <= 20 - period.
The loop prints the measured latency and counts "starved" steps (no fresh action; the last target is held).

  real_deploy_async.py --robot my_robot:Dobot --checkpoint CKPT --code CODE [--period 10] [--hz 30]
  real_deploy_async.py --replay LEROBOT_DIR --episode 47 --checkpoint CKPT --code CODE --sim-latency 8
      (replay: the scene follows the recording; a reply is released --sim-latency steps after its request, so the run
       is deterministic and the memory chain / executed-action error can be checked without a robot)
"""
import argparse
import importlib
import io
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402
from real_deploy import INTERVAL, Perception, Policy, Replay, png_b64  # noqa: E402

HORIZON = INTERVAL  # actions per reply


def previous_for(history, step, lag, initial):
    """Memory and latents for the request at `step`: written by the latest request that is at least `lag` steps old."""
    old = [q for q in history if q <= step - lag]
    return history[max(old)] if old else initial


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--raw", type=Path)
    p.add_argument("--robot", help="module:Class with observe() and act(action14)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, default=Path("deploy_log"))
    p.add_argument("--hz", type=float, default=S.FPS)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--period", type=int, default=10, help="control steps between requests (latency must stay <= 20 - period)")
    p.add_argument("--lag", type=int, default=INTERVAL, help="age in steps of the memory given to a request; 20 = training")
    p.add_argument("--blend", type=int, default=4, help="cross-fade steps when a new chunk takes over (0 = off)")
    p.add_argument("--sim-latency", type=int, default=8, help="replay only: steps between a request and its reply")
    args = p.parse_args()
    if not 1 <= args.period <= HORIZON:
        raise SystemExit("--period must be between 1 and 20")
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
    policy = Policy(args.checkpoint, args.code, args.log_dir)
    initial = (INITIAL_MEMORY, [[0.0] * LATENT_DIM for _ in range(MAX_ENTITIES)])
    history = {}        # request step -> (memory, latents) after that request
    replies = queue.Queue()
    pending = None      # step of the request in flight
    waiting = []        # replay: (release step, reply) not yet released
    plan = old = None   # (request step, actions[20,7]); old = the chunk the current one replaced
    took_over = None    # step at which `plan` became active
    stop_after = None   # last step to execute once the model wrote Task:Complete
    stopped, last, left_hold = None, None, None
    starved, skipped, latencies, errors = 0, 0, [], []
    log = open(args.log_dir / "memory_async.jsonl", "a")

    def ask(step, payload, given):
        started = time.time()
        try:
            result = policy.request(payload)
        except Exception as error:  # the server died: reported on the main thread
            result = {"error": repr(error)[:300]}
        replies.put((step, result, given, time.time() - started))

    try:
        for t in range(steps):
            tick = time.time()
            obs = robot.observe()
            if left_hold is None:
                left_hold = np.asarray(obs["state"], np.float32)[:7].copy()  # the left arm keeps its start pose
            overlay_png = perception.step(obs["top"], obs["right_wrist"])

            # 1. replies that are ready
            ready = []
            if args.replay:
                ready = [r for (release, r) in waiting if release <= t]
                waiting = [(release, r) for (release, r) in waiting if release > t]
            else:
                while not replies.empty():
                    ready.append(replies.get())
            for step, result, given, seconds in ready:
                pending = None
                entry = {"request_step": step, "reply_step": t, "latency_steps": t - step, "latency_s": round(seconds, 3),
                         "previous_memory": given[0], "generated": result.get("memory_text"),
                         "memory_valid": result.get("memory_valid"), "error": result.get("error")}
                latencies.append(t - step)
                if "error" in result:
                    stopped = "policy_error"
                else:
                    memory, latents = given
                    if result.get("latents") is not None:
                        latents = result["latents"]
                    if result.get("memory_valid") is True and isinstance(result.get("next_memory_text"), str):
                        memory = result["next_memory_text"]
                        if result.get("sequence_complete") is True and stop_after is None:
                            stop_after, stopped = step + HORIZON - 1, "model_stop"  # this chunk is still executed
                    else:
                        entry["lenient"] = "unparsable_kept_previous_memory"
                    history[step] = (memory, latents)
                    if t - step < HORIZON:
                        old, plan, took_over = plan, (step, np.asarray(result["actions"], np.float32)), t
                    else:
                        entry["late"] = "reply arrived after its 20 actions were over; not executed"
                entry["stopped"] = stopped
                log.write(json.dumps(entry) + "\n"), log.flush()
                print(f"{step}->{t}", (result.get("memory_text") or str(result.get("error"))).replace("\n", " | "), flush=True)
            if stopped == "policy_error":
                break

            # 2. a new request every --period steps (inputs of this very step; memory from --lag steps ago)
            if t % args.period == 0 and stop_after is None:
                if pending is not None:
                    skipped += 1  # the previous request is still running: latency > period
                else:
                    given = previous_for(history, t, args.lag, initial)
                    state8 = S.state8(np.asarray(obs["state"], np.float32)[None])[0]
                    overlay = np.asarray(Image.open(io.BytesIO(overlay_png)).convert("RGB"))
                    payload = {"main_png_b64": png_b64(overlay), "wrist_png_b64": png_b64(perception.wrist),
                               "raw_png_b64": png_b64(perception.raw), "state": [float(v) for v in state8],
                               "instruction": S.PROMPT, "previous_memory_text": given[0], "previous_latents": given[1]}
                    pending = t
                    if args.replay:
                        ask(t, payload, given)
                        waiting.append((t + args.sim_latency, replies.get()))
                    else:
                        threading.Thread(target=ask, args=(t, payload, given), daemon=True).start()

            # 3. the action of this step
            action = None
            if plan is not None and t - plan[0] < HORIZON:
                action = plan[1][t - plan[0]]
                k = t - took_over
                if args.blend and old is not None and k < args.blend and t - old[0] < HORIZON:
                    w = (k + 1) / (args.blend + 1)
                    action = w * action + (1 - w) * old[1][t - old[0]]
            elif plan is not None:
                starved += 1
                action = last  # hold the last target until the next reply
            if action is not None:
                last = action
                robot.act(S.action14(action, left_hold))
                if args.replay and t < len(robot.action):
                    errors.append(float(np.abs(action - S.action7(robot.action[t:t + 1])[0]).mean()))
            elif args.replay:
                robot.act(None)
            if stop_after is not None and t >= stop_after:
                break
            if args.robot:
                time.sleep(max(0.0, 1.0 / args.hz - (time.time() - tick)))
    finally:
        perception.close()
        policy.close()
    if latencies:
        print(f"LATENCY steps: mean {np.mean(latencies):.1f} max {max(latencies)} (limit {HORIZON - args.period} for period {args.period})")
    print(f"REQUESTS {len(latencies)}  skipped (previous still running) {skipped}  starved steps (held last target) {starved}")
    if errors:
        print(f"REPLAY mean |executed - recorded action| over {len(errors)} steps: {np.mean(errors):.4f}")
    print("END", stopped)


if __name__ == "__main__":
    main()
