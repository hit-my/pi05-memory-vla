"""Closed loop with the rule-based tick verifier in the Ledger Driver, optionally with failure injection.

The model and its text are untouched. When a generated edit ticks a goal, code/tick_verifier.py checks
it against the bowl and plate tracks (the perception that draws the overlay). A rejected tick becomes
"Subgoal:none / Task:Incomplete", so the ledger keeps the item open and the model may tick it later.
Env: INJECT_TYPE (none, empty_hand, drop), INJECT_LOG. The verdict of every tick goes in the response.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from closedloop_inject import install_injection
from closedloop_ledger import install
from memory_format import ledger_from_text, step_memory
from tick_verifier import verify

COLOUR = {"蓝": "blue", "橙": "orange", "紫": "purple", "绿": "green", "玫红": "magenta", "黄褐": "ochre", "靛蓝": "indigo", "棕": "brown"}
MIN_PIXELS = 80


def install_verifier(simulate, variant, minimum_visible):
    base_environment = simulate.OfficialEnvironment
    base_policy = simulate.PolicyProcess

    class VerifyingEnvironment(base_environment):
        current = None

        def __init__(self, source, spec, initial, instruction):
            super().__init__(source, spec, initial, instruction)
            self.view = -1
            self.tracks, self.plate_centres = {}, {}
            self.bowl_of = {COLOUR[v["color_name"]]: sid for sid, v in self.registry.items() if v["category"] == "bowl"}
            self.plate_sids = {sid for sid, v in self.registry.items() if v["category"] == "plate"}
            self.anchor = {colour: 0 for colour in self.bowl_of}
            VerifyingEnvironment.current = self

        def observe(self):
            result = super().observe()
            self.view += 1
            entities = simulate.measure_entities(result[2], self.registry, minimum_visible_pixels=minimum_visible)
            for sid, entity in entities.items():
                if entity.get("reliable") and entity.get("center") is not None:
                    centre = np.asarray(entity["center"], float)
                    self.tracks.setdefault(sid, {})[self.view] = centre
                    if sid in self.plate_sids and sid not in self.plate_centres:
                        self.plate_centres[sid] = centre
            return result

    class VerifyingPolicy(base_policy):
        def infer(self, main_png, wrist_png, state, instruction, previous_memory):
            result, latency = super().infer(main_png, wrist_png, state, instruction, previous_memory)
            result["generated_latents"] = self.latents  # kept for offline probes of the model's own z
            delta = result.get("subgoal_delta") or ""
            if result.get("memory_valid") is not True or not delta.startswith("done "):
                return result, latency
            env = VerifyingEnvironment.current
            plates = np.asarray(list(env.plate_centres.values()))
            items = {item.index: item.text for item in ledger_from_text(result["memory_text"]).items}
            verdicts, passed = [], []
            for k in (int(x) for x in delta[len("done "):].split(",")):
                colour = items[k].split(" ")[0]
                relation = "same" if items[k].endswith("same plate") else "empty" if items[k].endswith("empty plate") else None
                ok, moved, distance = verify(env.tracks.get(env.bowl_of[colour], {}), plates, env.anchor[colour], env.view, relation)
                verdicts.append({"goal": k, "colour": colour, "accepted": ok, "moved": moved, "distance": distance, "frame": env.view})
                passed.append((colour, ok))
            result["verifier"] = verdicts
            if all(ok for _, ok in passed):
                for colour, _ in passed:
                    env.anchor[colour] = env.view
                return result, latency
            # Reject: the ledger keeps the item open; the raw generation stays in generated_text.
            text = f"Action:{result['action_text']}\nSubgoal:none\nTask:Incomplete"
            redone = step_memory(previous_memory, text, variant)
            result.update({"memory_text": redone["next_memory_text"], "memory_valid": redone["memory_valid"],
                           "sequence_complete": False, "certified_count": redone["certified_count"],
                           "subgoal_delta": "none", "task_text": "Incomplete", "verifier_rejected": True})
            return result, latency

    simulate.OfficialEnvironment = VerifyingEnvironment
    simulate.PolicyProcess = VerifyingPolicy


if __name__ == "__main__":
    root = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
    sys.path.insert(0, str(root))
    import simulate as simulate_module

    torch_site = os.environ["SIMULATOR_TORCH_SITE"]
    sys.path.insert(0, torch_site)
    try:
        import torch  # noqa: F401
    finally:
        sys.path.remove(torch_site)
    contract = json.loads(Path(sys.argv[sys.argv.index("--contract") + 1]).read_text(encoding="utf-8"))
    install(simulate_module)
    kind = os.environ.get("INJECT_TYPE", "none")
    if kind != "none":
        install_injection(simulate_module, kind, os.environ["INJECT_LOG"])
    install_verifier(simulate_module, contract["model"]["memory_variant"], contract["memory"]["minimum_visible_pixels_for_motion_label"])
    simulate_module.main()
