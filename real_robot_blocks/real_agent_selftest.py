"""Self-test of the external agent (real_agent.py + the loop of real_deploy_agent.py) with stand-ins for the robot, the
cameras, the policy and the checker. CPU, a few seconds, no data, no network:   python real_agent_selftest.py

The stand-in world has one arm joint that matters and a gripper. The stand-in policy behaves like the real one in the
way that matters here: it ticks a grasp when the gripper has closed, whether or not anything is in it. The first grasp
of every run closes on nothing. The stand-in checker knows the truth.

Checked:
  A. agent with recovery: the false tick is removed from the memory the policy reads; the gripper is opened and the arm
     returns to a state it recorded before the attempt, never moving faster than it did on the way there; the policy
     then repeats the step and the task is completed; correct ticks are left alone.
  B. the last tick is wrong: the stop is held until its verdict arrives and is then cancelled; the run ends later.
  C. the checker fails on every question: the run still ends normally and nothing is changed.
  D. no agent: same loop, the false tick stays (the reference behaviour).
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_agent  # noqa: E402
import real_deploy_agent as D  # noqa: E402
import real_spec as S  # noqa: E402

ITEMS = ["lift the far cup", "far cup: empty", "put the far cup back", "pick up block 0", "place the block on the plate"]
PLACE = {1: 10.0, 2: 10.0, 3: 14.0, 4: 20.0, 5: 26.0}     # where joint 0 has to be for each item
INITIAL = "Action:-\nSubgoal:" + "; ".join(f"[ ]{k + 1} {text}" for k, text in enumerate(ITEMS)) + "\nTask:Incomplete"


class World:
    """observe()/act() of the robot. A closing gripper holds something unless it is the first grasp of the run."""

    def __init__(self, fail_release=False):
        self.joints, self.command, self.grasps, self.holding, self.releases = np.zeros(6, np.float32), 0.93, 0, False, 0
        self.fail_release, self.history = fail_release, []

    def observe(self):
        reading = 0.94 if self.command > 0.4 else (0.67 if self.holding else 0.02)
        state = np.zeros(14, np.float32); state[S.RIGHT] = np.concatenate([self.joints, [reading]])
        return {"top": None, "right_wrist": None, "state": state}

    def act(self, action14):
        joints, command = np.asarray(action14, np.float32)[7:13], float(action14[13])
        if command < 0.4 <= self.command:
            self.grasps += 1; self.holding = self.grasps != 1
        if command >= 0.4 > self.command:
            self.releases += 1; self.holding = False
        self.history.append((joints.copy(), command, self.joints.copy()))
        self.joints, self.command = joints.copy(), command


class Cameras:
    def __init__(self): self.raw = self.wrist = np.zeros((8, 8, 3), np.uint8)
    def step(self, top, wrist, closed=False):
        b = io.BytesIO(); Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(b, format="PNG"); return b.getvalue()


class Policy:
    """Reads the checklist it is given, works on the first open item, and ticks it from the gripper reading alone."""

    def __init__(self): self.given = []

    def request(self, payload):
        memory = payload["previous_memory_text"]; self.given.append(memory)
        joint, reading = payload["state"][0], payload["state"][6]
        todo = [(k, text) for k, (done, text) in sorted(real_agent.items(memory).items()) if not done]
        actions = np.zeros((20, 7), np.float32); actions[:, 0] = joint; actions[:, 6] = 0.93 if reading > 0.8 else 0.0
        edit = "none"
        if todo:
            k, text = todo[0]; verb = text.split()[0]; at = abs(joint - PLACE[k]) < 0.05
            if not at:
                actions[:, 0] = joint + np.clip(PLACE[k] - joint, -2.0, 2.0) * np.arange(1, 21) / 20.0
            elif verb in real_agent.GRASP and reading > 0.8:
                actions[:, 6] = 0.0                      # close; the tick follows at the next request
            elif verb in real_agent.RELEASE and reading <= 0.8:
                actions[:, 6] = 0.93                     # open; the tick follows at the next request
            else:
                edit = f"done {k}"
        after = memory
        if edit != "none":
            after = memory.replace(f"[ ]{k} ", f"[x]{k} ")
        complete = all(done for done, _ in real_agent.items(after).values())
        if complete:
            after = after.replace("Task:Incomplete", "Task:Complete")
        return {"actions": actions.tolist(), "latents": None, "memory_text": f"Action:-\nSubgoal:{edit}\nTask:{'Complete' if complete else 'Incomplete'}",
                "memory_valid": True, "next_memory_text": after, "sequence_complete": complete}


class Checker:
    def __init__(self, world, broken=False): self.world, self.broken, self.asked = world, broken, []

    def judge(self, case):
        if self.broken:
            raise RuntimeError("no network")
        verb = case["step"].split()[0]
        if verb in real_agent.GRASP:
            done = self.world.holding
        else:
            done = not (self.world.fail_release and verb == "place" and sum(a[0] == case["k"] for a in self.asked) == 0)
        self.asked.append((case["k"], case["step"], done))
        return 0.95 if done else 0.05


def scenario(name, agent=True, broken=False, fail_release=False, recover=True):
    world, policy, said = World(fail_release), Policy(), []
    checker = Checker(world, broken)
    supervisor = real_agent.Supervisor(checker, S.PROMPT, synchronous=True, recover=recover) if agent else None
    stopped = D.run(world, Cameras(), policy, supervisor, (INITIAL, None), 4000, out=lambda *a: said.append(" ".join(map(str, a))))
    events = supervisor.events if supervisor else []
    return world, policy, checker, events, stopped, said


def main():
    failures = []

    def check(name, condition, detail=""):
        print(f"  {'ok  ' if condition else 'FAIL'} {name} {detail}")
        if not condition: failures.append(name)

    print("A. agent with recovery")
    world, policy, checker, events, stopped, _ = scenario("A")
    kinds = [e["event"] for e in events]
    check("the false tick of item 1 is removed", any(e["event"] == "untick" and e["item"] == 1 for e in events))
    start = next((e for e in events if e["event"] == "recovery_start"), None); end = next((e for e in events if e["event"] == "recovery_end"), None)
    check("a recovery ran", start is not None and end is not None, str(start))
    if start and end:
        segment = world.history[start["from_step"]:start["from_step"] + end["steps"]]
        check("gripper commanded open during the whole recovery", all(command > 0.4 for _, command, _ in segment))
        moved = max(float(np.abs(joints - before).max()) for joints, _, before in segment)
        fastest = max(float(np.abs(world.history[i][0] - world.history[i][2]).max()) for i in range(start["from_step"]))
        check("never faster than on the way there", moved <= fastest + 1e-6, f"(recovery {moved:.3f}, before {fastest:.3f} per step)")
        check("the arm is back at the recorded state", end["joint_error"] < 1e-3, f"(joint error {end['joint_error']})")
        after = policy.given[[i for i, m in enumerate(policy.given) if "[x]1" in m][0] + 1:]
        check("the policy is then given item 1 as not done", any("[ ]1 lift" in m for m in after))
    check("the task is completed and the run stops by itself", stopped == "model_stop" and all(d for d, _ in real_agent.items(policy.given[-1].replace("[ ]5", "[x]5")).values()))
    check("three grasps: failed lift, lift, pick", world.grasps == 3, f"({world.grasps})")
    check("correct ticks are left alone", sum(e["event"] == "untick" for e in events) == 1 and "error" not in kinds)

    print("B. the last tick is wrong")
    world, policy, checker, events, stopped, said = scenario("B", fail_release=True)
    check("the stop is cancelled once", sum("STOP CANCELLED" in line for line in said) == 1)
    check("item 5 is unticked and done again", any(e["event"] == "untick" and e["item"] == 5 for e in events) and sum(a[0] == 5 for a in checker.asked) == 2)
    check("the run still ends by itself", stopped == "model_stop")

    print("C. the checker fails on every question")
    world, policy, checker, events, stopped, _ = scenario("C", broken=True)
    check("the run ends normally", stopped == "model_stop")
    check("nothing is changed, the failures are logged", not any(e["event"] in ("untick", "recovery_start") for e in events) and any(e["event"] == "error" for e in events))

    print("D. no agent")
    world, policy, checker, events, stopped, _ = scenario("D", agent=False)
    check("the run ends with the false tick in place (two grasps only)", stopped == "model_stop" and world.grasps == 2)

    print("SELFTEST", "PASSED" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
