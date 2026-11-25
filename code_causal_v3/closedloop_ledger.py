"""Run the pinned official simulator with the A/B/C ledger memory.

The official rollout loop in simulate.py stays unchanged. It feeds back whatever
result["memory_text"] holds, so this adapter puts the driver-rendered next memory there
(the raw generation stays in generated_text) and carries generated latents between
rounds. Latents and memory come only from the model; an invalid memory still executes
its finite chunk and then ends with invalid_model_memory, without any repair.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import sys
import time
from pathlib import Path

from memory_format import INITIAL_MEMORY, LATENT_DIM, MAX_ENTITIES


def adapt_response(result):
    """Map a policy response onto what simulate.py reads. Returns (result, next_latents)."""
    result = dict(result)
    if "error" in result:
        return result, None
    result["generated_text"] = result.get("memory_text")
    result["memory_text"] = result.get("next_memory_text")
    return result, result.pop("latents", None)


def validate_ledger_memory(result):
    text = result.get("memory_text")
    if result.get("memory_valid") is not True or not isinstance(text, str):
        return False, "ledger driver rejected the generated memory"
    if type(result.get("sequence_complete")) is not bool:
        return False, "sequence_complete is not boolean"
    return True, None


def blank_latents():
    return [[0.0] * LATENT_DIM for _ in range(MAX_ENTITIES)]


def install(simulate):
    simulate.INITIAL_MEMORY = INITIAL_MEMORY
    simulate.validate_memory = validate_ledger_memory

    class LedgerPolicyProcess(simulate.PolicyProcess):
        latents = None

        def infer(self, main_png, wrist_png, state, instruction, previous_memory):
            # The closed request schema only gains the model's own previous latents.
            if previous_memory == INITIAL_MEMORY or self.latents is None:
                self.latents = blank_latents()
            request = {"main_png_b64": base64.b64encode(main_png).decode("ascii"),
                       "wrist_png_b64": base64.b64encode(wrist_png).decode("ascii"),
                       "state": state.tolist(), "instruction": instruction,
                       "previous_memory_text": previous_memory, "previous_latents": self.latents}
            started = time.perf_counter()
            self.process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
            self.process.stdin.flush()
            try:
                line = self.lines.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError(f"Policy JSON response exceeded {self.timeout}s") from exc
            if line is None:
                raise RuntimeError(f"Policy process exited ({self.process.poll()}); see policy_stderr.log")
            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError("Policy response must be a JSON object")
            result, latents = adapt_response(result)
            if latents is not None:
                self.latents = latents
            return result, time.perf_counter() - started

    simulate.PolicyProcess = LedgerPolicyProcess


if __name__ == "__main__":
    root = Path(os.environ.get("LIBERO_SIMULATOR_DRIVER_ROOT", "xxxx/pi05-full-t06-t08"))
    sys.path.insert(0, str(root))
    import simulate as simulate_module

    torch_site = os.environ.get("SIMULATOR_TORCH_SITE")
    if torch_site is not None:
        if not Path(torch_site).is_dir():
            raise FileNotFoundError(f"Private simulator Torch installation missing: {torch_site}")
        sys.path.insert(0, torch_site)
        try:
            import torch
            if not Path(torch.__file__).is_relative_to(Path(torch_site)):
                raise RuntimeError("Simulator loaded Torch from an unexpected environment")
        finally:
            sys.path.remove(torch_site)
    install(simulate_module)
    simulate_module.main()
