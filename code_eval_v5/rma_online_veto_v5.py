"""Step-wise ("online") tracking rules v5 = /data/lgd/rma/trackv/v5/tv_fuse_v5.py, one frame at a time.

The rules live in ONE place, the state machine tv_fuse_v5.VetoV5 (per-task rule set: mode raw / v5 / v4 with the constants of
rma_veto_v5_constants.json next to this file = a copy of /data/lgd/rma/trackv/v5/constants_v5.json). The offline functions
(observe_only_v5 on masks, observe_from_features on cached measurements) and this class feed it frame by frame with the same
per-frame measurements (tv_fuse_v5.frame_features5) and the corrected gripper projection (tv_common_v4).
Equality with the offline functions on stored masks: /data/lgd/rma/trackv/v5/test_veto_v5.py.

Same interface as rma_online_veto_v4.OnlineVetoV4:
    veto = OnlineVetoV5(task_id)
    centres, visible = veto.reset(masks, rgb, state)                       # frame 0, gripper counted as open
    centres, visible = veto.step(masks, rgb, state, last_gripper_cmd)      # every later frame
    optional keyword on both: retired=<iterable of object indices retired at this frame> (sticky)
Inputs per frame: K boolean masks [K,256,256], the RGB frame [256,256,3] uint8, the 8-d proprioceptive state, the gripper
command issued BEFORE this frame. No simulator state, no labels, no future frames.
Also set after each call: raw_centres, raw_visible, area, eef, events.

WHEN to pass `retired` (v5: per task in the constants, key "retire"; triggers of tv_fuse_v5.retirement_plan_v5):
    place_open   the gripper command opens while the checklist's current item is "place <obj> into the <container>"
    place_end    the checklist ticks "place <obj> into the <container>"
    close_end    the checklist ticks the first "close the <container>" after that placement
    close_moveNN the gripper closed inside that "close the <container>" item and then moved >= NN px in the image
    (close_start / close_grip exist as well; no task uses them)
RetireSignalV5 below turns (current checklist item text, last gripper command, state) into those indices step by step;
offline the same comes from tv_fuse_v5.retired_from_labels_v5 (test: equal on the demos).

What the evaluation-time tracker must do besides calling this class (see constants "first_frame" / "first_frame_appearance"):
    tasks 1-17, 20-26: unchanged first frame (method B; B2 for tasks 25/26), SAM3 stream tracker, masks -> this class.
    tasks 18, 19 (first frame by appearance; reference implementation /data/lgd/rma/trackv/v5/track5.py, constants under
        "first_frame_appearance"): no track before frame t0 (2 / 10: the objects are still dropping in the stored frame 0).
        At t0: zoomed SAM3 text detections on the fixed cabinet-1 crop -> appearance classifier (weights in the constants) ->
        joint choice of one detection per object (ff_lib.select_joint). Task 18: the choice is repeated at frame 6 and an
        object whose new detection is not where its mask is, is re-started. Every 10 frames an object without a track, or
        whose mask has been (nearly) empty for 10 frames, is looked for again. At the first grasp of each object (checklist
        item "pick up the <object>"): if its mask did not travel with the closed gripper it is re-started from the detection
        at the gripper (Stream5.add). The rules of this class then run on those masks with min_area from the constants.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

TRACKV = "/data/lgd/rma/trackv"
CONSTANTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rma_veto_v5_constants.json")


def _import_offline():
    before = list(sys.path)
    sys.path.insert(0, TRACKV)
    sys.path.insert(0, TRACKV + "/v5")
    try:
        import tv_common_v4 as tv_common
        import tv_fuse_v5
    finally:
        sys.path[:] = before
    return tv_common, tv_fuse_v5


tv_common, tv_fuse_v5 = _import_offline()
_CONST = None


def constants():
    global _CONST
    if _CONST is None:
        with open(CONSTANTS) as fh:
            _CONST = json.load(fh)
    return _CONST


def task_config(task_id):
    """full rule configuration of a task: CONFIG5 defaults <- constants[tasks][task][cfg]"""
    cfg = dict(tv_fuse_v5.CONFIG5)
    cfg.update(constants()["tasks"][str(int(task_id))]["cfg"])
    return cfg


class OnlineVetoV5:
    def __init__(self, task_id, cfg=None):
        self.task_id = int(task_id)
        self.ep = f"t{self.task_id:02d}_online"  # tv_common.task_of(ep) selects the camera matrix
        full = task_config(self.task_id)
        if cfg:
            full.update(cfg)
        full["_no_constants"] = True
        self.cfg = full
        self.core = None

    def reset(self, masks, rgb, state, retired=None):
        self.k = len(masks)
        self.core = tv_fuse_v5.VetoV5(self.task_id, self.k, self.cfg)
        return self._frame(masks, rgb, state, False, retired)

    def step(self, masks, rgb, state, last_gripper_cmd=None, retired=None):
        if self.core is None:
            raise RuntimeError("OnlineVetoV5.step before reset")
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
        states = np.asarray(state, np.float64).reshape(1, -1)
        e = tv_common.eef_pixels(self.ep, states)[0]
        f = tv_fuse_v5.frame_features5(masks, np.asarray(rgb), e, tv_fuse_v5.capsule(self.ep, states[0]), tv_fuse_v5.feature_floor(self.core.cfg),
                                       blobs=self.core.cfg["blob"] != "")
        flags = None
        if retired is not None:
            flags = np.zeros(self.k, bool)
            for j in retired:
                flags[int(j)] = True
        centres, vis = self.core.step(f, e, closed, float(states[0, 6]), flags)
        self.raw_centres, self.raw_visible, self.area, self.eef = f["c"], f["v"], f["area"], e
        return centres, vis


class RetireSignalV5:
    """Checklist -> retired object indices, one frame per update() (causal twin of tv_fuse_v5.retired_from_labels_v5).
    update(current_item_text, last_gripper_cmd, state) returns the list of indices retired at this frame (sticky)."""

    def __init__(self, task_id, spec=None):
        self.task_id = int(task_id)
        self.spec = constants()["tasks"][str(self.task_id)]["retire"] if spec is None else spec
        self.ep = f"t{self.task_id:02d}_online"
        before = list(sys.path)
        sys.path.insert(0, TRACKV + "/code")
        try:
            import rma_spec
        finally:
            sys.path[:] = before
        self._obj = lambda s: rma_spec.subtask_object(self.task_id, s)
        self.reset()

    def reset(self):
        self.t = 0
        self.text = None
        self.retired = set()
        self.waiting = []      # (container, trigger, object): placements waiting for their "close the <container>" item
        self.active = []       # triggers armed inside the current item: dicts(kind, obj, ...)
        self.prev_closed = False

    def _start(self, s, closed):
        """a new checklist item became current"""
        box = tv_fuse_v5._container(s)
        act = []
        if s.startswith("place") and box is not None:
            j = self._obj(s)
            kind = "drawer" if "drawer" in box else box
            trig = self.spec.get(kind, "none")
            if j is not None and trig != "none":
                if trig in ("place_end", "place_open"):
                    act.append(dict(when=trig.split("_")[1], obj=j))
                else:
                    self.waiting.append((box, trig.split("_")[1], j))
        if s.startswith("close") and box is not None:
            keep = []
            for b, when, j in self.waiting:
                if b == box:
                    act.append(dict(when=when, obj=j, g0=None))
                else:
                    keep.append((b, when, j))
            self.waiting = keep
        self.active = act

    def update(self, text, last_gripper_cmd=None, state=None):
        s = str(text).lower()
        closed = bool(self.t > 0 and last_gripper_cmd is not None and float(last_gripper_cmd) > 0)
        new = s != self.text
        if new:
            for a in self.active:      # the previous item ended (checklist tick): every trigger armed in it fires now at the latest
                self.retired.add(a["obj"])
            self._start(s, closed)
            self.text = s
        e = None
        for a in self.active:
            if a["obj"] in self.retired:
                continue
            w = a["when"]
            if w == "start":
                self.retired.add(a["obj"])
            elif w == "open":
                if self.prev_closed and not closed:
                    self.retired.add(a["obj"])
            elif w == "grip":
                if closed:
                    self.retired.add(a["obj"])
            elif w.startswith("move"):
                if closed or a["g0"] is not None:
                    if e is None:
                        e = tv_common.eef_pixels(self.ep, np.asarray(state, np.float64).reshape(1, -1))[0]
                    if a["g0"] is None:
                        a["g0"] = e.copy()
                    if float(np.hypot(*(e - a["g0"]))) >= float(w[4:]):
                        self.retired.add(a["obj"])
        self.prev_closed = closed
        self.t += 1
        return sorted(self.retired)
