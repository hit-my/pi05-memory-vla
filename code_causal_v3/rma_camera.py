"""Dump the agentview camera matrix (world -> pixel, 256 px) for every RoboMemArena task.

Run with the simulator python and the rma_eval.sh PYTHONPATH. Pixel rows are flipped to match the upright
images the policy sees (policy_adapter flips the raw render). Usage: rma_camera.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix

sys.path.insert(0, "xxxx/rma/eval/RoboMemArena-main/evaluation_benchmark/scripts")
import eval_common as ec  # noqa: E402

result = {}
for task in range(1, 27):
    bddl = ec._resolve_bddl_path(task)
    env = None
    for attempt in range(8):  # placement sampling fails now and then
        try:
            env = ec._get_env_class()(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                      ignore_done=True, control_freq=20, initialization_noise=None)
            env.reset()
            break
        except Exception as error:
            print(task, "attempt", attempt, type(error).__name__, flush=True)
            env = None
    if env is None:
        continue
    sim = env.env.sim if hasattr(env, "env") else env.sim
    matrix = get_camera_transform_matrix(sim, "agentview", 256, 256)
    obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    eef = np.asarray(obs["robot0_eef_pos"])
    p = matrix @ np.append(eef, 1.0)
    u, v = p[0] / p[2], p[1] / p[2]
    result[task] = {"bddl": bddl.name, "matrix": matrix.tolist(), "eef_reset": eef.tolist(), "eef_pixel_raw": [float(u), float(v)]}
    print(task, bddl.name, "eef pixel (col,row raw)", round(float(u), 1), round(float(v), 1), flush=True)
    env.close()
Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
