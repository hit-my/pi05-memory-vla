"""closedloop_t.py with proprioception passed to the tracker (tracker_server_m.py), and state saved in dump mode.

TRACK_MODE=dump writes <dump>.state.npy (float32[N,8]) next to the npz. Simulator segmentation still goes only
to the evaluation log. closedloop_t.py is imported unchanged.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import closedloop_t
from closedloop_ledger import install

LATEST = {"state": None}


class StateClient(closedloop_t.TrackerClient):
    def __init__(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        code = os.environ["TRACKER_CODE"]
        env["PYTHONPATH"] = os.environ.get("TRACKER_PYTHONPATH", code)
        if "TRACKER_CUDA" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["TRACKER_CUDA"]
        self.process = subprocess.Popen(
            [os.environ["TRACKER_PYTHON"], str(Path(code) / "tracker_server_m.py"), "--root", os.environ["TRACKER_ROOT"],
             "--options", os.environ.get("TRACKER_OPTIONS", "{}")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=code)
        ready = self.process.stdout.readline()
        if ready.strip() != b"READY":
            raise RuntimeError(f"Tracker failed to start: {ready!r}")

    def _send(self, command, rgb):
        s = LATEST["state"]
        super()._send(f"{command} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f} {s[6]:.6f} {s[7]:.6f}", rgb)


def install_state(simulate):
    base = simulate.OfficialEnvironment
    mode = os.environ.get("TRACK_MODE", "track")

    class StateEnvironment(base):
        def __init__(self, *args, **kwargs):
            self.state_log = []
            super().__init__(*args, **kwargs)

        def observe(self):
            result = super().observe()
            LATEST["state"] = np.asarray(result[3], np.float64)
            self.state_log.append(np.asarray(result[3], np.float32).copy())
            return result

        def close(self):
            if mode == "dump" and self.state_log:
                np.save(Path(os.environ["TRACK_DIR"]) / f"t{self.track_task:02d}_{self.track_number:03d}.state.npy", np.asarray(self.state_log))
            super().close()

    simulate.OfficialEnvironment = StateEnvironment


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
    install(simulate_module)
    if os.environ.get("TRAIN_STATES"):
        closedloop_t.install_training_states(simulate_module)
    if os.environ.get("INJECT_TYPE"):
        from closedloop_inject import install_injection
        install_injection(simulate_module, os.environ["INJECT_TYPE"], os.environ["INJECT_LOG"])
    install_state(simulate_module)
    closedloop_t.TrackerClient = StateClient
    closedloop_t.install_tracking(simulate_module)
    simulate_module.main()
