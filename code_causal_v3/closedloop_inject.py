"""Closed-loop failure injection: does the ledger tick when the action happened but the result did not?

The model runs normally and is told nothing. One goal k of each episode is chosen by episode order
(same in every group). The simulator state is changed once, when the model carries the goal bowl:
  empty_hand : as the bowl starts to lift (>=1.5 cm), it is put back at rest and pinned there until
               the fingers reopen, so the arm completes the motion with nothing in hand.
  drop       : once carried (lift >=5 cm and >=6 cm sideways), the bowl is set down straight below at
               its resting height, out of the fingers, while the arm carries on.
Trigger records go to $INJECT_LOG. Env vars: INJECT_TYPE, INJECT_LOG.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from closedloop_ledger import install

GOALS = {6: ["akita_black_bowl_1"] * 7,
         8: ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3", "akita_black_bowl_1"]}
EMPTY_LIFT, DROP_LIFT, DROP_SIDE = 0.015, 0.05, 0.06
FREEZE_MAX = 300


def chosen_goal(task_id, episode_number):
    """T6: cycles 2..6, T8: steps 1..4, so that injections are spread across the sequence."""
    return episode_number % 5 + 2 if task_id == 6 else episode_number % 4 + 1


def install_injection(simulate, kind, log_path):
    base_class = simulate.OfficialEnvironment

    class InjectingEnvironment(base_class):
        created = 0

        def __init__(self, source, spec, initial, instruction):
            super().__init__(source, spec, initial, instruction)
            self.number = InjectingEnvironment.created
            InjectingEnvironment.created += 1
            self.goal = chosen_goal(spec["task_id"], self.number)
            self.bowl = GOALS[spec["task_id"]][self.goal - 1]
            self.t, self.fired, self.pinned_steps = 0, False, 0
            self.pinned = False
            self.counters_frozen = False
            base = self.env
            while not hasattr(base, "obj_body_id") and hasattr(base, "env"):
                base = base.env
            self.base = base
            sim = self.env.sim
            self.joint = base.get_object(self.bowl).joints[0]
            names = [sim.model.geom_id2name(i) or "" for i in range(sim.model.ngeom)]
            self.grippers = {i for i, n in enumerate(names) if n.startswith("gripper0_") or "finger" in n}
            root = sim.model.body_rootid[base.obj_body_id[self.bowl]]
            self.bowl_geoms = {g for g in range(sim.model.ngeom) if sim.model.body_rootid[sim.model.geom_bodyid[g]] == root}
            self.rest = np.array(sim.data.get_joint_qpos(self.joint), dtype=np.float64)
            self.previous = self.rest.copy()
            self._log({"event": "armed"})

        def _log(self, row):
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"episode_number": self.number, "task_id": self.task_id, "goal": self.goal,
                                         "bowl": self.bowl, "kind": kind, "frame": self.t, **row}) + "\n")

        def _held(self):
            data = self.env.sim.data
            for contact in data.contact[:data.ncon]:
                a, b = int(contact.geom1), int(contact.geom2)
                if (a in self.grippers and b in self.bowl_geoms) or (b in self.grippers and a in self.bowl_geoms):
                    return True
            return False

        def _place(self, qpos):
            sim = self.env.sim
            sim.data.set_joint_qpos(self.joint, qpos)
            sim.data.set_joint_qvel(self.joint, np.zeros(6))
            sim.forward()
            self.obs = self.base._get_observations(force_update=True)

        def _width(self):
            qpos = self.obs["robot0_gripper_qpos"]
            return float(qpos[0] - qpos[1])

        def score_observation(self, frame):
            # Pinning is not a real removal of the bowl, but contact flicker while pinned reads as
            # "not on the plate" and would make the official machine count a goal that never
            # happened. So it does not advance its counters until the bowl is really lifted again.
            if not self.counters_frozen:
                return super().score_observation(frame)
            predicate = bool(self.env._check_success(inc=False))
            count = len(self.env._satisfied_subgoals)
            overshot = bool(self.env._overshot)
            return {"frame": frame, "confirmed_goals": count, "required_goals": self.required_goals,
                    "tiered_success": count / self.required_goals, "overshot": overshot,
                    "sequence_completed_once": count == self.required_goals,
                    "success_now": count == self.required_goals and predicate and not overshot,
                    "physical_satisfied_subgoals": self.env._satisfied_subgoals.copy(),
                    "official_recorder_labels": self.env.get_satisfied_subgoals(self.instruction)}

        def step(self, action):
            result = super().step(action)
            self.t += 1
            if self.pinned:
                self._place(self.rest)
                self.pinned_steps += 1
                if self._width() > 0.05 or self.pinned_steps >= FREEZE_MAX:
                    self.pinned = False
                    self._log({"event": "unpinned", "width": self._width(), "pinned_steps": self.pinned_steps})
            elif self.counters_frozen:
                qpos = np.array(self.env.sim.data.get_joint_qpos(self.joint), dtype=np.float64)
                if self._held() and qpos[2] - self.rest[2] >= EMPTY_LIFT:
                    self.counters_frozen = False
                    self._log({"event": "counters_resumed"})
            elif not self.fired:
                self._maybe_fire()
            return result

        def _maybe_fire(self):
            qpos = np.array(self.env.sim.data.get_joint_qpos(self.joint), dtype=np.float64)
            held = self._held()
            speed = float(np.linalg.norm(qpos[:3] - self.previous[:3]))
            self.previous = qpos.copy()
            if not held and speed < 0.003:
                self.rest = qpos.copy()
            count = len(self.env._satisfied_subgoals)
            if count != self.goal - 1 or not held:
                return
            lift = float(qpos[2] - self.rest[2])
            side = float(np.linalg.norm(qpos[:2] - self.rest[:2]))
            if kind == "empty_hand" and lift >= EMPTY_LIFT:
                self.fired, self.pinned, self.counters_frozen = True, True, True
                self._place(self.rest)
                self._log({"event": "fired", "env_count": count, "lift": lift, "side": side})
            elif kind == "drop" and lift >= DROP_LIFT and side >= DROP_SIDE:
                self.fired = True
                target = self.rest.copy()
                target[:2] = qpos[:2]
                target[2] = self.rest[2] + 0.005
                self._place(target)
                self._log({"event": "fired", "env_count": count, "lift": lift, "side": side})

    simulate.OfficialEnvironment = InjectingEnvironment


if __name__ == "__main__":
    kind = os.environ["INJECT_TYPE"]
    if kind not in ("empty_hand", "drop"):
        raise ValueError(f"Unknown injection {kind}")
    root = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
    sys.path.insert(0, str(root))
    import simulate as simulate_module

    torch_site = os.environ["SIMULATOR_TORCH_SITE"]
    sys.path.insert(0, torch_site)
    try:
        import torch  # noqa: F401
    finally:
        sys.path.remove(torch_site)
    install(simulate_module)
    install_injection(simulate_module, kind, os.environ["INJECT_LOG"])
    simulate_module.main()
