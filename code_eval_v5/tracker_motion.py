"""CUTIE with a motion prior from proprioception: a bowl only moves if the gripper is there to move it.

Inputs are RGB, end-effector position and gripper width, all available on a real robot. The eef-to-pixel
map comes from a calibration file. Here it is fitted on training replays (tools/tracker_calib.py); on a real
robot this is the hand-eye calibration step. No simulator segmentation or object id enters this file.

Motion is never invented from the gripper: a moving bowl's mask always comes from CUTIE or a detection in
the image, so an empty grasp cannot draw a bowl that moves. Only a static bowl is held in place, and only
while the pixels at its place still look as they did.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

import numpy as np

from tracker_core import SIZE, Options, TrackerCore


@dataclass
class MotionOptions(Options):
    motion_gate: bool = False
    calib: str = ""             # JSON with a 3x4 projection "P"
    near_px: float = 32.0       # a bowl this close to the gripper in the image may move
    tol_px: float = 8.0         # a static bowl's mask may wander this far before it counts as wrong
    closed_width: float = 0.04  # finger opening below this is a closed gripper
    grace: int = 15             # frames after release in which the released bowl may still fall or settle
    match_diff: int = 20        # grey levels; a pixel within this of its reference is unchanged
    match_ok: float = 0.8       # share of unchanged pixels that says the bowl is still at its place
    reseed: bool = True         # correct CUTIE's memory, not just the reported mask
    reseed_every: int = 10
    redetect: bool = True
    detect_every: int = 20
    detect_px: float = 40.0     # a detection must lie this close to where the lost bowl should be
    stable_frames: int = 10
    stable_px: float = 1.5
    adopt_frames: int = 40      # a mask stable elsewhere this long wins over a place that no longer looks right


class MotionTracker(TrackerCore):
    def __init__(self, root, options=None):
        super().__init__(root, options or MotionOptions())
        self._calib_path, self.P = None, None

    def _load_calib(self):
        if self.options.calib != self._calib_path:
            with open(self.options.calib, encoding="utf-8") as stream:
                self.P = np.asarray(json.load(stream)["P"], np.float64)
            self._calib_path = self.options.calib

    def _project(self, xyz):
        u = self.P @ np.append(np.asarray(xyz, np.float64), 1.0)
        return u[:2] / u[2]

    @staticmethod
    def _centre(mask):
        ys, xs = np.nonzero(mask)
        return np.array([xs.mean(), ys.mean()]) if len(xs) >= 80 else None

    def _set_anchor(self, k, mask, rgb):
        self.anchor[k] = {"mask": mask.copy(), "c": self._centre(mask), "ref": rgb[mask].astype(np.int16)}
        self.wait[k] = 0

    def _match(self, k, rgb):
        a = self.anchor[k]
        return float((np.abs(rgb[a["mask"]].astype(np.int16) - a["ref"]).max(axis=1) <= self.options.match_diff).mean())

    def initialize(self, rgb, n_bowls, proprio=None):
        report, labels = super().initialize(rgb, n_bowls)
        if getattr(self.options, "motion_gate", False):
            self._load_calib()
            self.anchor, self.wait = {}, {}
            self.valid = {}
            self.recent = {k: deque(maxlen=self.options.stable_frames) for k in range(1, self.n_bowls + 1)}
            self.expected = {}
            for k in range(1, self.n_bowls + 1):
                mask = labels == k
                self.valid[k] = mask.sum() >= 80
                if self.valid[k]:
                    self._set_anchor(k, mask, rgb)
                else:
                    self.anchor[k], self.wait[k] = None, 0
            self.carry, self.offset, self.released, self.release_frame = None, None, None, -10 ** 6
            self.was_closed = False
            self.last_reseed, self.last_detect = -10 ** 6, -10 ** 6
            self.events = {"held": 0, "reseeds": 0, "detects": 0, "redetected": 0, "stolen": 0, "left_behind": 0, "adopted": 0}
        return report, labels

    def _settled(self, k, centre):
        recent = self.recent[k]
        recent.append(centre)
        if len(recent) < recent.maxlen or any(c is None for c in recent):
            return False
        points = np.asarray(recent)
        return float(np.linalg.norm(points - points.mean(axis=0), axis=1).max()) <= self.options.stable_px

    def _on_other(self, k, mask, rgb):
        """The mask sits mostly on another bowl that is still at its place."""
        for j, a in self.anchor.items():
            if j != k and a is not None and self.valid[j] and j != self.carry:
                if (mask & a["mask"]).sum() > 0.5 * max(mask.sum(), 1) and self._match(j, rgb) >= self.options.match_ok:
                    return True
        return False

    def step(self, rgb, proprio=None):
        options = self.options
        if not getattr(options, "motion_gate", False) or proprio is None:
            return super().step(rgb)
        if self.core is None:
            return np.zeros((SIZE, SIZE), np.uint8)
        raw = self._raw(rgb)
        self.frame += 1
        eef = self._project(proprio[:3])
        closed = float(proprio[3]) - float(proprio[4]) < options.closed_width
        centres = {k: self._centre(raw == k) for k in range(1, self.n_bowls + 1)}

        def place(k):
            a = self.anchor[k]
            return a["c"] if self.valid[k] and a is not None else centres[k]

        if closed and not self.was_closed:
            near = [(float(np.linalg.norm(place(k) - eef)), k) for k in centres if place(k) is not None]
            near = [item for item in near if item[0] < options.near_px]
            if near:
                self.carry = min(near)[1]
                self.offset = place(self.carry) - eef
        if self.was_closed and not closed and self.carry is not None:
            self.released, self.release_frame = self.carry, self.frame
            self.expected[self.carry] = eef + self.offset
            self.carry = None
        self.was_closed = closed
        if self.carry is not None:
            self.expected[self.carry] = eef + self.offset

        output = np.zeros((SIZE, SIZE), np.uint8)
        reseed, lost = [], []
        for k in centres:
            mask, centre = raw == k, centres[k]
            a = self.anchor[k] if self.valid[k] else None
            near = a is not None and float(np.linalg.norm(a["c"] - eef)) < options.near_px
            if a is not None and k == self.carry and not near:
                # The gripper left its place holding this bowl, or holding nothing.
                if self._match(k, rgb) >= options.match_ok:
                    self.carry = None  # nothing was lifted: the bowl is still there
                else:
                    self.valid[k] = False
                    a = None
            settling = k == self.released and self.frame - self.release_frame <= options.grace
            if k == self.carry or settling or near:  # may move: trust the image, unless the mask is on another bowl
                if a is not None and k == self.carry and centre is not None and float(np.linalg.norm(centre - a["c"])) > options.tol_px:
                    self.valid[k] = False
                if centre is not None and self._on_other(k, mask, rgb):
                    self.events["stolen"] += 1
                    lost.append(k)
                elif centre is not None and k == self.carry and not self.valid[k] \
                        and float(np.linalg.norm(centre - self.expected[k])) > 2 * options.near_px:
                    self.events["left_behind"] += 1  # the mask stayed behind while the bowl went with the gripper
                    lost.append(k)
                else:
                    output[mask] = k
                self.recent[k].clear()
                continue
            if a is not None:  # must be static
                if centre is not None and float(np.linalg.norm(centre - a["c"])) < options.tol_px:
                    output[mask] = k
                    self.wait[k] = 0
                    if self._settled(k, centre) and 0.7 * a["mask"].sum() <= mask.sum() <= 1.3 * a["mask"].sum() and self._match(k, rgb) >= options.match_ok:
                        self._set_anchor(k, mask, rgb)
                elif self._match(k, rgb) >= options.match_ok:
                    output[a["mask"]] = k
                    self.events["held"] += 1
                    reseed.append(k)
                else:  # hidden, or really gone. Report nothing unless the mask stays stable somewhere else for long.
                    self.wait[k] = self.wait[k] + 1 if centre is not None and self._settled(k, centre) and not self._on_other(k, mask, rgb) else 0
                    if self.wait[k] >= options.adopt_frames:
                        self._set_anchor(k, mask, rgb)
                        output[mask] = k
                        self.events["adopted"] += 1
                continue
            # The bowl has moved and the gripper is done with it: wait for it to settle, or find it again.
            if centre is not None and not self._on_other(k, mask, rgb):
                output[mask] = k
                if self._settled(k, centre) and float(np.linalg.norm(centre - eef)) >= options.near_px:
                    self._set_anchor(k, mask, rgb)
                    self.valid[k] = True
            else:
                lost.append(k)
        output, raw = self._repair(rgb, raw, output, reseed, lost, eef)
        return output

    def _repair(self, rgb, raw, output, reseed, lost, eef):
        options = self.options
        seeds = {}
        if lost and options.redetect and self.frame - self.last_detect >= options.detect_every:
            self.last_detect = self.frame
            self.events["detects"] += 1
            found, _, _ = self._appearance_bowls(rgb, 8)
            taken = np.zeros((SIZE, SIZE), bool)
            for k, a in self.anchor.items():
                if a is not None and self.valid[k] and k not in lost:
                    taken |= a["mask"]
            for k in lost:
                want = self.expected.get(k)
                if want is None:
                    continue
                if float(np.linalg.norm(want - eef)) < options.near_px and k != self.carry:
                    continue  # the gripper is in the way
                free = [d for d in found if (d["mask"] & taken).sum() < 0.25 * d["mask"].sum()]
                if not free:
                    continue
                best = min(free, key=lambda d: (d["cx"] - want[0]) ** 2 + (d["cy"] - want[1]) ** 2)
                if np.hypot(best["cx"] - want[0], best["cy"] - want[1]) <= options.detect_px:
                    seeds[k] = best["mask"]
                    taken |= best["mask"]
                    output[output == k] = 0
                    output[best["mask"] & (output == 0)] = k
                    self.events["redetected"] += 1
                    if k != self.carry:
                        self._set_anchor(k, best["mask"], rgb)
                        self.valid[k] = True
        if options.reseed and reseed and self.frame - self.last_reseed >= options.reseed_every:
            for k in reseed:
                seeds[k] = self.anchor[k]["mask"]
        if seeds:
            self.last_reseed = self.frame
            self.events["reseeds"] += 1
            seed = raw.copy()
            for k in seeds:
                seed[seed == k] = 0
            for k in lost:
                if k not in seeds:
                    seed[seed == k] = 0
            for k, mask in seeds.items():
                seed[mask] = k
            raw = self._raw(rgb, seed, self.objects)
        return output, raw
