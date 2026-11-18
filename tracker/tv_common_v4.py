"""tv_common with the corrected camera projection (v4).

Cause of the v3 error: camera26.json holds the agentview matrix of a SQUARE 256x256 render (fx = fy = 309 px).
The benchmark does not render square: evaluation_benchmark/scripts/eval_tasks2_26.py calls
ec.patch_env_resolution(480, 640), i.e. the camera is rendered 640 wide x 480 high (same vertical field of view) and
policy_adapter._process_image_match_eval26 then resizes that frame to 256x256 (anisotropic). The demonstrations were
produced the same way. Vertical scale is unchanged (480 -> 256 rows see the same field of view as a 256-row render),
horizontal scale is 256/640 of a 480-row focal length = 0.75 x the square one, about the image centre:
        u_true = 128 + 0.75 * (u_square - 128),   v_true = v_square
So the v3 cross was right at the start pose (u = 128) and wrong in proportion to the distance from the image centre
column (too far left on the left, too far right on the right).

Everything here depends only on the robot state vector and fixed constants (no simulator state at run time).
"""
import numpy as np

from tv_common import *  # noqa: F401,F403  (load, task_of, centres_from_masks, LR, ROOT, ...)
from tv_common import CAM, task_of

RENDER_W, RENDER_H, OUT = 640, 480, 256
SX = (OUT / RENDER_W) / (OUT / RENDER_H)  # 0.75
CX = OUT / 2.0


def _fix(matrix):
    P = np.asarray(matrix, np.float64).copy()
    P[0] = SX * P[0] + (1.0 - SX) * CX * P[2]
    return P


CAM4 = {key: _fix(value["matrix"]) for key, value in CAM.items()}


def project(ep, points):
    """world points [M,3] -> pixels [M,2] (column, row) in the 256x256 frames the policy sees"""
    P = CAM4[str(task_of(ep))]
    h = np.c_[points, np.ones(len(points))] @ P.T
    return h[:, :2] / h[:, 2:3]


def eef_pixels(ep, states):
    """gripper (grip site between the finger pads) pixel for every state row"""
    return project(ep, np.asarray(states)[:, :3])
