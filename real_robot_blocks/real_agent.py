"""External agent for the real robot: it keeps the policy's checklist honest and undoes a failed grasp.

The policy writes its own memory (Action / Subgoal / Task lines). When it writes "Subgoal:done k" it claims that item k
of the checklist is finished. This module, which never touches the policy's weights, does three things:

  1. Verify.   Each claimed physical step (lift / put back / pick up / place) is shown to a vision-language model
               (real_checker.py): the image from the start of the step, the image now, the wrist image now.
  2. Maintain. If the verdict is "not done", the tick of that item (and of every later item) is removed from the
               memory the policy reads next, and Task is set back to Incomplete. Optionally (--fix-observations) an
               observation item ("far cup: has block" / "empty") that the checker contradicts is rewritten.
  3. Recover.  If the removed tick was a grasp (lift / pick up) and the fingers hold nothing, the gripper is opened and
               the arm is driven back along its OWN recorded joint states to where the attempt began, and the policy
               continues from the memory it had there. Nothing but the robot's own past states is used.

The checker runs in worker threads: it never blocks the arm. A verdict takes effect at the next policy request after it
arrives. One exception: when the policy declares the task complete, the stop waits for the verdicts still outstanding.
A failure of the checker (network, bad answer) is logged and otherwise ignored; it never ends a run.

This file has no robot, camera or network code and is covered by real_agent_selftest.py.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

GRASP = {"lift", "pick"}          # steps that end with an object in the gripper: these can be recovered
RELEASE = {"put", "place"}        # steps that end with the gripper open
OBSERVATION = re.compile(r"^(.+ cup): (has block|empty)$")
OBSERVED = ("has block", "empty")


# ---------------------------------------------------------------------------------------------- memory text helpers
def _subgoal(memory):
    return next((line for line in memory.split("\n") if line.startswith("Subgoal:")), "")


def items(memory):
    """{k: (ticked, text)} of the checklist in a memory text."""
    return {int(k): (mark == "x", text.strip()) for mark, k, text in re.findall(r"\[([ x])\](\d+) ([^;\n]*)", _subgoal(memory))}


def checklist_lines(memory):
    return [part.strip() for part in _subgoal(memory)[len("Subgoal:"):].split(";") if part.strip()]


def untick(memory, k):
    """Remove the tick of item k and of every later item; the task is then not complete."""
    lines = memory.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Subgoal:"):
            lines[i] = re.sub(r"\[x\](\d+) ", lambda m: ("[ ]" if int(m.group(1)) >= k else "[x]") + m.group(1) + " ", line)
        elif line.startswith("Task:"):
            lines[i] = "Task:Incomplete"
    return "\n".join(lines)


def rewrite(memory, k, text):
    """Replace the text of item k (its tick is kept)."""
    lines = memory.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Subgoal:"):
            lines[i] = re.sub(r"(\[[ x]\]%d )[^;\n]*" % k, lambda m: m.group(1) + text, line)
    return "\n".join(lines)


def parse_edit(result):
    """Items the policy has just ticked, and item texts it has just revealed, from one reply of the policy server."""
    if result.get("memory_valid") is not True:
        return [], {}
    text = result.get("memory_text") or ""
    sub = next((line for line in text.split("\n") if line.startswith("Subgoal:")), "")[len("Subgoal:"):].strip()
    if not sub.startswith("done"):
        return [], {}
    head, _, tail = sub.partition(" reveal ")
    return [int(x) for x in re.findall(r"\d+", head)], {int(k): t.strip() for k, t in re.findall(r"(\d+)=([^;]*)", tail)}


# ------------------------------------------------------------------------------------------------------ recovery
class Recovery:
    """Open the gripper, then go back along the recorded joint states to an earlier control step.

    trajectory: {control step: right-arm state [6 joints, gripper reading]} recorded by the deployment loop.
    The commanded joints never change by more than `speed` times the largest per-step change the arm itself made on the
    way there, so the way back is no faster than the way out. Commands are absolute joint targets, as the policy's."""

    def __init__(self, trajectory, step_now, step_target, state_now, open_command, open_steps=15, stride=4, speed=1.0):
        steps = [s for s in range(step_target, step_now + 1) if s in trajectory]
        if len(steps) < 2:
            raise ValueError("no recorded states between the target step and now")
        joints = np.stack([np.asarray(trajectory[s], np.float64)[:6] for s in steps])
        self.limit = np.maximum(speed * np.abs(np.diff(joints, axis=0)).max(axis=0), 1e-6)
        self.way = [joints[i] for i in range(len(steps) - 1 - stride, 0, -stride)] + [joints[0]]
        self.command = np.asarray(state_now, np.float64)[:6].copy()
        self.open_command, self.open_left = float(open_command), int(open_steps)
        self.steps, self.max_steps = 0, int(open_steps) + 3 * len(steps) + 60
        self.target = joints[0]

    def next(self, state_now):
        """The next right-arm command [6 joints, gripper], or None when the arm is back."""
        self.steps += 1
        if self.open_left > 0:                       # 1. let go first, without moving
            self.open_left -= 1
        elif self.way and self.steps <= self.max_steps:  # 2. follow the recorded path backwards
            self.command = self.command + np.clip(self.way[0] - self.command, -self.limit, self.limit)
            if np.abs(self.way[0] - self.command).max() < 1e-9:
                self.way.pop(0)
        else:
            return None
        return np.concatenate([self.command, [self.open_command]]).astype(np.float32)

    def distance(self, state_now):
        return float(np.abs(np.asarray(state_now, np.float64)[:6] - self.target).max())


# ---------------------------------------------------------------------------------------------------- supervisor
class Supervisor:
    """One instance per run. The deployment loop calls, in this order:
         record(step, state7)                           every control step
         apply(call, step, memory, latents, state7)     before each policy request -> (memory, latents, Recovery | None)
         after_reply(call, step, given, result, memory, raw, wrist)   after each reply
         finish_recovery(call, raw)                     when a Recovery has ended -> (memory, latents)
         flush(raw, wrist) / outstanding() / wait(s)    when the policy has declared the task complete
    """

    def __init__(self, checker, instruction, fps=30.0, threshold=0.5, wait_calls=1, recover=True, back_calls=5,
                 holding=(0.45, 0.80), open_command=0.93, fix_observations=False, observation_margin=0.3,
                 synchronous=False, workers=2, log_path=None, recovery_options=None):
        self.checker, self.instruction, self.fps = checker, instruction, float(fps)
        self.threshold, self.wait_calls = float(threshold), int(wait_calls)
        self.recover, self.back_calls, self.holding, self.open_command = bool(recover), int(back_calls), holding, float(open_command)
        self.fix_observations, self.observation_margin = bool(fix_observations), float(observation_margin)
        self.synchronous = bool(synchronous)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.log_file = open(log_path, "a") if log_path else None
        self.recovery_options = dict(recovery_options or {})
        self.trajectory, self.snapshot, self.call_step = {}, {}, {}
        self.waiting, self.running, self.ref, self.ref_step, self.ref_call = [], [], None, 0, 0
        self.ref_of, self.recovering, self.events = {}, None, []

    # -- bookkeeping
    def log(self, **event):
        self.events.append(event)
        if self.log_file:
            self.log_file.write(json.dumps(event) + "\n"), self.log_file.flush()

    def record(self, step, state7):
        self.trajectory[step] = np.asarray(state7, np.float32).copy()

    def outstanding(self):
        """Is a verdict still missing (a look not yet taken, or a question not yet answered)?"""
        return bool(self.waiting) or any(not future.done() for future, _ in self.running)

    def wait(self, seconds):
        deadline = time.time() + seconds
        while self.running and time.time() < deadline and not all(f.done() for f, _ in self.running):
            time.sleep(0.02)

    def close(self):
        self.pool.shutdown(wait=False)
        if self.log_file:
            self.log_file.close()

    # -- checks
    def _submit(self, check, raw, wrist, step):
        case = {"kind": check["kind"], "instruction": self.instruction, "k": check["k"], "step": check["step"],
                "checklist": check["checklist"], "seconds": max(0.0, (step - check["ref_step"]) / self.fps),
                "ref": check["ref"], "now": np.asarray(raw).copy(), "wrist": np.asarray(wrist).copy()}
        future = self.pool.submit(self.checker.judge, case)
        self.running.append((future, check))
        if self.synchronous:
            self.wait(600.0)

    def flush(self, raw, wrist, step):
        """Ask now for every check that was still waiting for its look (used when the policy wants to stop)."""
        for check in self.waiting:
            self._submit(check, raw, wrist, step)
        self.waiting = []

    def after_reply(self, call, step, given, result, memory, raw, wrist):
        try:
            if self.ref is None:
                self.ref = np.asarray(raw).copy()
            ticked, revealed = parse_edit(result)
            before = items(given)
            for k in ticked:
                text = revealed.get(k) or before.get(k, (False, ""))[1]
                if not text or text == "?":
                    continue
                kind = "observation" if OBSERVATION.match(text) else "tick"
                if kind == "observation" and not self.fix_observations:
                    continue
                if kind == "tick" and text.split()[0] not in GRASP | RELEASE:
                    continue
                self.waiting.append({"kind": kind, "k": k, "step": text, "ref": self.ref, "ref_step": self.ref_step, "ref_call": self.ref_call,
                                     "tick_call": call, "due": call + (0 if kind == "observation" else self.wait_calls),
                                     "checklist": checklist_lines(given)})
            if ticked:
                self.ref, self.ref_step, self.ref_call = np.asarray(raw).copy(), step, call
            now = items(memory)
            due = [c for c in self.waiting if call >= c["due"]]
            self.waiting = [c for c in self.waiting if call < c["due"]]
            for check in due:   # the policy ticks a grasp when the gripper closes; the outcome is visible a little later
                if now.get(check["k"], (False, ""))[0]:
                    self._submit(check, raw, wrist, step)
        except Exception as error:   # the agent must never end a run
            self.log(event="error", where="after_reply", call=call, error=repr(error)[:300])

    # -- verdicts
    def apply(self, call, step, memory, latents, state7, snapshot=True):
        """Apply the verdicts that have arrived. Returns the memory and latents to give to the policy, and a Recovery if
        the arm has to go back first (the caller runs it and then calls finish_recovery)."""
        recovery = None
        try:
            done = [(f, c) for f, c in self.running if f.done()]
            self.running = [(f, c) for f, c in self.running if not f.done()]
            for future, check in sorted(done, key=lambda fc: fc[1]["k"]):
                k, text = check["k"], check["step"]
                try:
                    p = float(future.result())
                except Exception as error:
                    self.log(event="error", where="checker", call=call, item=k, error=repr(error)[:300])
                    continue
                current = items(memory).get(k, (False, ""))
                if check["kind"] == "observation":   # p = P(a block is visible where the cup stood)
                    place, said = OBSERVATION.match(text).groups()
                    seen = "has block" if p > 0.5 else "empty"
                    change = current[0] and seen != said and abs(p - 0.5) >= self.observation_margin
                    self.log(event="observation_check", call=call, item=k, step=text, p_block=round(p, 3), rewritten=bool(change))
                    if change:
                        memory = rewrite(memory, k, f"{place}: {seen}")
                    continue
                reject = p < self.threshold and current[0]
                self.log(event="tick_check", call=call, tick_call=check["tick_call"], item=k, step=text, p_yes=round(p, 3), reject=bool(reject))
                if not reject:
                    self.ref_of[k] = (check["ref"], check["ref_step"], check["ref_call"])
                    continue
                memory = untick(memory, k)
                self.ref, self.ref_step, self.ref_call = check["ref"], check["ref_step"], check["ref_call"]
                self.waiting = [c for c in self.waiting if c["k"] < k]
                self.running = [(f, c) for f, c in self.running if c["k"] < k]
                self.log(event="untick", call=call, item=k, step=text)
                if recovery is None and self.recover:
                    recovery = self._plan_recovery(call, step, check, state7)
            if snapshot:   # what the policy is given at this request: a recovery can return to it
                self.snapshot[call], self.call_step[call] = (memory, latents), step
        except Exception as error:
            self.log(event="error", where="apply", call=call, error=repr(error)[:300])
        return memory, latents, recovery

    def _plan_recovery(self, call, step, check, state7):
        k, text = check["k"], check["step"]
        verb, reading = text.split()[0], float(np.asarray(state7)[6])
        if verb not in GRASP:
            self.log(event="recovery_skip", call=call, item=k, why="not a grasp step")
            return None
        if self.holding[0] <= reading <= self.holding[1]:
            self.log(event="recovery_skip", call=call, item=k, why="the fingers may hold something", gripper=round(reading, 3))
            return None
        target = max(check["ref_call"] + 1, check["tick_call"] - self.back_calls)
        if target not in self.snapshot or target >= call:
            self.log(event="recovery_skip", call=call, item=k, why="no earlier state to return to")
            return None
        try:
            recovery = Recovery(self.trajectory, step, self.call_step[target], state7, self.open_command, **self.recovery_options)
        except ValueError as error:
            self.log(event="recovery_skip", call=call, item=k, why=str(error))
            return None
        self.recovering = {"k": k, "target_call": target, "started": step}
        self.log(event="recovery_start", call=call, item=k, step=text, gripper=round(reading, 3), from_step=step,
                 to_step=self.call_step[target], to_call=target, waypoints=len(recovery.way))
        return recovery

    def finish_recovery(self, call, step, raw, recovery, state7):
        """The arm is back: the policy continues with the memory it had there, the failed item not ticked."""
        info, self.recovering = self.recovering, None
        memory, latents = self.snapshot[info["target_call"]]
        self.ref, self.ref_step, self.ref_call = np.asarray(raw).copy(), step, call
        self.waiting, self.running = [], []
        self.log(event="recovery_end", call=call, item=info["k"], steps=step - info["started"], joint_error=round(recovery.distance(state7), 4))
        return untick(memory, info["k"]), latents
