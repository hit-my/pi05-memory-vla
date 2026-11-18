"""Dummy adapter for checking the RoboMemArena install: small random actions, saves the first observation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from policy_adapter import BasePolicyAdapter


class DummyAdapter(BasePolicyAdapter):
    def __init__(self, out_dir: str = "/tmp/rma_dummy", seed: int = 0):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self.episode = -1
        self.reset()

    def reset(self):
        self.episode += 1
        self.saved = False

    def observe(self, obs, prompt, resize_size):
        if self.saved:
            return
        self.saved = True
        raw = obs["_raw_obs"]
        info = {"prompt": prompt, "episode": self.episode,
                "keys": {k: [list(np.asarray(v).shape), str(np.asarray(v).dtype)] for k, v in obs.items()
                         if k not in ("_raw_obs", "prompt") and hasattr(v, "__len__")},
                "raw_keys": {k: list(np.asarray(v).shape) for k, v in raw.items()},
                "state": np.asarray(obs["observation/state"]).round(4).tolist(),
                "raw_eef_pos": np.asarray(raw.get("robot0_eef_pos")).round(4).tolist(),
                "raw_gripper": np.asarray(raw.get("robot0_gripper_qpos")).round(4).tolist()}
        tag = f"{os.getpid()}_{self.episode}"
        np.savez_compressed(self.out / f"obs_{tag}.npz", image=obs["observation/image"],
                            wrist=obs["observation/wrist_image"], raw_agent=raw.get("agentview_image"),
                            raw_wrist=raw.get("robot0_eye_in_hand_image"), state=obs["observation/state"])
        (self.out / f"obs_{tag}.json").write_text(json.dumps(info, indent=1))

    def infer_actions(self, obs, prompt, resize_size):
        actions = self.rng.normal(0, 0.05, (20, 7)).astype(np.float32)
        actions[:, 6] = -1.0
        return actions


def build_adapter(**kwargs):
    return DummyAdapter(**kwargs)
