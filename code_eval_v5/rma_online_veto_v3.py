"""Step-wise ("online") gripper veto v3 = xxxx/rma/trackv/tv_fuse_v3.py::observe_only_v3, one frame at a time.

The rules live in ONE place, the state machine tv_fuse_v3.VetoV3; the offline observe_only_v3 and this class both feed
it frame by frame (same per-frame measurements tv_fuse_v3.frame_features, same capsule, same per-task configuration
tv_fuse_v3.TASK_CONFIG keyed by task id). What differs between the two callers is only how the inputs arrive: here a
single float32 state and the previous gripper command per call instead of whole arrays. Equality with the offline
function on stored masks is checked by xxxx/rma/trackv/v3/test_veto_v3.py.

Same interface as rma_online_veto.OnlineVeto (v2):
    veto = OnlineVetoV3(task_id)
    centres, visible = veto.reset(masks, rgb, state)                       # frame 0, gripper counted as open
    centres, visible = veto.step(masks, rgb, state, last_gripper_cmd)      # every later frame
plus an optional keyword on both:  retired=<iterable of object indices that are retired at this frame>
(sticky: once an index was passed it stays retired until the next reset). A retired object is never drawn; the rest of
the state machine is unaffected. At evaluation the indices come from the policy's own checklist (the placement of that
object into a drawer / the microwave is ticked); offline from tv_fuse_v3.retired_from_labels.

Inputs per frame (no simulator ground truth, no labels): K boolean masks [K,256,256], the RGB frame [256,256,3] uint8,
the 8-d proprioceptive state (ee_pos3, axis-angle3, finger qpos2) and the gripper command issued BEFORE this frame.
centres: float64 [K,2], nan where the object is not drawn; visible: bool [K]. Note that v3 may draw the object's
ANCHOR (last accepted resting position) instead of the mask centroid while the mask is partly occluded.
Also set after each call: raw_centres, raw_visible, area, eef, events (as in v2).
"""
from __future__ import annotations

import sys

import numpy as np

TRACKV = "xxxx/rma/trackv"


def _import_offline():
    before = list(sys.path)
    sys.path.insert(0, TRACKV)
    try:
        import tv_common
        import tv_fuse_v3
    finally:
        sys.path[:] = before
    return tv_common, tv_fuse_v3


tv_common, tv_fuse_v3 = _import_offline()


class OnlineVetoV3:
    def __init__(self, task_id, cfg=None):
        self.task_id = int(task_id)
        self.ep = f"t{self.task_id:02d}_online"  # tv_common.task_of(ep) == int(ep[1:3]) selects the camera matrix
        self.cfg = cfg
        self.core = None

    def reset(self, masks, rgb, state, retired=None):
        self.k = len(masks)
        self.core = tv_fuse_v3.VetoV3(self.task_id, self.k, self.cfg)
        return self._frame(masks, rgb, state, False, retired)

    def step(self, masks, rgb, state, last_gripper_cmd=None, retired=None):
        if self.core is None:
            raise RuntimeError("OnlineVetoV3.step before reset")
        closed = bool(last_gripper_cmd is not None and float(last_gripper_cmd) > 0)
        return self._frame(masks, rgb, state, closed, retired)

    @property
    def events(self):
        return self.core.events if self.core is not None else []

    @property
    def t(self):
        return None if self.core is None else self.core.t

    def _frame(self, masks, rgb, state, closed, retired):
        masks = np.asarray(masks, bool)
        if masks.shape != (self.k, 256, 256):
            raise ValueError(f"expected masks [{self.k},256,256], got {masks.shape}")
        states = np.asarray(state, np.float64).reshape(1, -1)  # tv_common.load() also casts float32 -> float64
        e = tv_common.eef_pixels(self.ep, states)[0]
        f = tv_fuse_v3.frame_features(masks, np.asarray(rgb), e, tv_fuse_v3.capsule(self.ep, states[0]))
        flags = None
        if retired is not None:
            flags = np.zeros(self.k, bool)
            for j in retired:
                flags[int(j)] = True
        centres, vis = self.core.step(f, e, closed, float(states[0, 6]), flags)
        self.raw_centres, self.raw_visible, self.area, self.eef = f["c"], f["v"], f["area"], e
        return centres, vis
