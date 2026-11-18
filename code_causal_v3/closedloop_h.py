"""Closed loop with receding-horizon execution: same model, replans every EXEC_H steps.

Uses simulate_h.py (see tools/patch_simulate_h.py). Each memory lane (frame % 20) keeps its own ledger,
so every query reads memory written exactly 20 frames earlier, as in training. Any lane's Task:Complete
stops the episode; an invalid memory in any lane ends it after the current chunk.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import sys
import time
from pathlib import Path

from closedloop_ledger import adapt_response, blank_latents, validate_ledger_memory
from memory_format import INITIAL_MEMORY


def install(simulate):
    simulate.INITIAL_MEMORY = INITIAL_MEMORY
    simulate.validate_memory = validate_ledger_memory

    class LanePolicyProcess(simulate.PolicyProcess):
        lane = 0

        def infer(self, main_png, wrist_png, state, instruction, previous_memory):
            if not hasattr(self, "lane_latents"):
                self.lane_latents = {}
            if previous_memory == INITIAL_MEMORY:
                self.lane_latents[self.lane] = blank_latents()
            request = {"main_png_b64": base64.b64encode(main_png).decode("ascii"),
                       "wrist_png_b64": base64.b64encode(wrist_png).decode("ascii"),
                       "state": state.tolist(), "instruction": instruction, "previous_memory_text": previous_memory,
                       "previous_latents": self.lane_latents[self.lane]}
            started = time.perf_counter()
            self.process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
            self.process.stdin.flush()
            try:
                line = self.lines.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError(f"Policy JSON response exceeded {self.timeout}s") from exc
            if line is None:
                raise RuntimeError(f"Policy process exited ({self.process.poll()}); see policy_stderr.log")
            result, latents = adapt_response(json.loads(line))
            if latents is not None:
                self.lane_latents[self.lane] = latents
            return result, time.perf_counter() - started

    simulate.PolicyProcess = LanePolicyProcess


if __name__ == "__main__":
    root = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).parent))
    import simulate_h as simulate_module

    torch_site = os.environ["SIMULATOR_TORCH_SITE"]
    sys.path.insert(0, torch_site)
    try:
        import torch  # noqa: F401
    finally:
        sys.path.remove(torch_site)
    install(simulate_module)
    simulate_module.main()
