"""Step-wise ("online") version of tv_fuse.observe_only: "seen only + gripper veto", one frame at a time.

The offline function xxxx/rma/trackv/tv_fuse.py::observe_only loops over whole arrays but only ever reads
index t and its own running state, so it is causal. OnlineVeto keeps that running state (anchors q, held object,
pending grasp, recent centroids, first-frame area and colour, previous gripper command) and executes the SAME
statements per frame. Everything that can be shared is imported from the offline code, not copied:
  constants                      tv_fuse   (GRASP_PX, NEAR_PX, FAR_PX, SETTLE, MIN_WIDTH, HELD_EDGE_PX, STABLE_N,
                                            STABLE_PX, AREA_ABS, AREA_RATIO, AREA_MIN, CHOOSE_PX, CHOOSE_WAIT, COLOUR_TOL)
  mask centroid / visibility     tv_common.centres_from_masks  (>= 20 px)
  end-effector pixel             tv_common.eef_pixels          (camera26.json)
  projected gripper/wrist capsule tv_render.robot_masks
Only numpy + cv2 are needed, so this module runs in the simulator venv and in the tracker venv.

Inputs per frame (no simulator ground truth): K boolean masks [K,256,256] from the mask tracker, the RGB frame,
the 8-d proprioceptive state (ee_pos3, axis-angle3, finger qpos2) and the gripper command that was issued BEFORE
this frame (offline: closed = np.r_[False, actions[:-1, 6] > 0]).

    veto = OnlineVeto(task_id)
    centres, visible = veto.reset(masks, rgb, state)                       # frame 0, gripper counted as open
    centres, visible = veto.step(masks, rgb, state, last_gripper_cmd)      # every later frame
centres: float64 [K,2], nan where the object is not drawn; visible: bool [K]. veto.events as in observe_only.
"""
from __future__ import annotations

import sys

import numpy as np

TRACKV = "xxxx/rma/trackv"  # tv_common / rma_sam3_stream hard-code this root themselves


def _import_offline():
    """Import the offline modules without leaving their directories on sys.path of the host process."""
    before = list(sys.path)
    sys.path.insert(0, TRACKV)
    try:
        import tv_common
        import tv_fuse
        import tv_render
    finally:
        sys.path[:] = before  # also drops the tracker 'deps' folder that tv_render appends
    return tv_common, tv_fuse, tv_render


tv_common, tv_fuse, tv_render = _import_offline()
from tv_common import centres_from_masks, eef_pixels  # noqa: E402
from tv_fuse import (AREA_ABS, AREA_MIN, AREA_RATIO, CHOOSE_PX, CHOOSE_WAIT, COLOUR_TOL, FAR_PX, GRASP_PX,  # noqa: E402
                     HELD_EDGE_PX, MIN_WIDTH, NEAR_PX, SETTLE, STABLE_N, STABLE_PX)
from tv_render import robot_masks  # noqa: E402


class OnlineVeto:
    def __init__(self, task_id, grasp_px=GRASP_PX):
        self.task_id = int(task_id)
        self.ep = f"t{self.task_id:02d}_online"  # tv_common.task_of(ep) == int(ep[1:3]) selects the camera matrix
        self.grasp_px = grasp_px
        self.t = None

    # ------------------------------------------------------------------ public
    def reset(self, masks, rgb, state):
        self.t = 0
        self.k = len(masks)
        self.q = None
        self.area0 = None
        self.colour0 = None
        self.held, self.pending, self.events = None, None, []
        self.recent = [[] for _ in range(self.k)]
        self.closed_prev = False
        return self._frame(masks, rgb, state, False)

    def step(self, masks, rgb, state, last_gripper_cmd=None):
        if self.t is None:
            raise RuntimeError("OnlineVeto.step before reset")
        closed = bool(last_gripper_cmd is not None and float(last_gripper_cmd) > 0)
        return self._frame(masks, rgb, state, closed)

    # ------------------------------------------------------------------ one iteration of observe_only's loop
    def _frame(self, masks, rgb, state, closed_t):
        t, k = self.t, self.k
        masks = np.asarray(masks, bool)
        if masks.shape != (k, 256, 256):
            raise ValueError(f"expected masks [{k},256,256], got {masks.shape}")
        states = np.asarray(state, np.float64).reshape(1, -1)  # tv_common.load() also casts to float64
        c_all, v_all = centres_from_masks(masks[None])
        c_in, v_in = c_all[0], v_all[0]
        e = eef_pixels(self.ep, states)[0]
        width = states[0, 6]
        area = masks.sum(axis=(1, 2))
        robot = None  # projected gripper/wrist capsule, only needed for a re-acquisition test
        if t == 0:
            self.area0 = area
            self.colour0 = [rgb[masks[j]].mean(axis=0) if AREA_MIN <= area[j] <= AREA_ABS else None for j in range(k)]
            self.q = [c_in[j].copy() if v_in[j] else None for j in range(k)]
        area0, colour0, q, recent, events = self.area0, self.colour0, self.q, self.recent, self.events
        held, pending = self.held, self.pending
        centres, vis = np.full((k, 2), np.nan), np.zeros(k, bool)

        if t and closed_t and not self.closed_prev and held is None:
            cand = sorted((np.hypot(*(q[j] - e)), j) for j in range(k) if q[j] is not None)
            near = [j for d, j in cand if d < self.grasp_px]
            pending = (near, t, e.copy(), {j: q[j].copy() for j in near}) if near else None
        if pending is not None and not closed_t:
            pending = None
        if pending is not None and t - pending[1] >= SETTLE:
            near, t0, e0, q0 = pending
            moved = e - e0
            if width < MIN_WIDTH * 0.6:
                pending = None
            elif len(near) == 1 and width >= MIN_WIDTH:
                held, pending = near[0], None
                events.append((t, "grasp", held))
            elif np.hypot(*moved) > CHOOSE_PX:
                score = {}
                for j in near:
                    if v_in[j]:
                        d = c_in[j] - q0[j]
                        if np.hypot(*d) > 0.4 * np.hypot(*moved):
                            score[j] = np.hypot(*(d - moved))
                best = min(score, key=score.get) if score else None
                if best is not None and score[best] < 0.6 * np.hypot(*moved) + 4:
                    held = best
                    events.append((t, "grasp", held))
                    pending = None
                elif np.hypot(*moved) > 3 * CHOOSE_PX or t - t0 > CHOOSE_WAIT:
                    pending = None
            elif t - t0 > CHOOSE_WAIT and len(near) == 1 and width >= MIN_WIDTH:
                held, pending = near[0], None
                events.append((t, "grasp", held))
        if held is not None and (not closed_t or width < MIN_WIDTH * 0.6):
            events.append((t, "release", held))
            held = None
        for j in range(k):
            if not v_in[j]:
                recent[j] = []
                continue
            c = c_in[j]
            if area[j] > AREA_ABS or (AREA_MIN <= area0[j] <= AREA_ABS and area[j] > AREA_RATIO * area0[j]):
                recent[j] = []
                continue
            cx, cy = int(np.clip(round(c[0]), 0, 255)), int(np.clip(round(c[1]), 0, 255))
            recent[j] = (recent[j] + [c])[-STABLE_N:]
            reacq = False
            if j == held:
                ys_, xs_ = np.nonzero(masks[j])
                ok = bool(len(xs_)) and float(np.min(np.hypot(xs_ - e[0], ys_ - e[1]))) < HELD_EDGE_PX
            elif q[j] is not None and np.hypot(*(c - q[j])) < NEAR_PX:
                ok = True
            else:
                still = len(recent[j]) == STABLE_N and np.ptp(np.asarray(recent[j]), axis=0).max() < STABLE_PX
                if robot is None:
                    robot = robot_masks(self.ep, states)[0]
                ok = bool(still and not robot[cy, cx] and np.hypot(*(c - e)) > FAR_PX)
                if ok and colour0[j] is not None:
                    ok = float(np.abs(rgb[masks[j]].mean(axis=0) - colour0[j]).mean()) < COLOUR_TOL
                if ok:
                    reacq = True
                    events.append((t, "reacquired", j))
            if ok:
                if j == held or reacq or q[j] is None:
                    q[j] = c.copy()
                elif np.hypot(*(c - e)) > FAR_PX:
                    q[j] = 0.8 * q[j] + 0.2 * c
                centres[j], vis[j] = c, True

        self.held, self.pending = held, pending
        self.closed_prev = closed_t
        self.raw_centres, self.raw_visible, self.area, self.eef = c_in, v_in, area, e
        self.t = t + 1
        return centres, vis
