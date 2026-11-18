"""Online entity positions: the mask tracker "sees", gripper kinematics decides what can move.

Causal, frame by frame; inputs are the mask tracker's centroid per object, the RGB frame, the proprioceptive state
(ee_pos3 + axis-angle3 + gripper_qpos2) and the gripper command executed before this frame. No simulator state.

Per object: position p and whether it is held.
  resting : p follows the tracker's mask only if the mask is outside the projected gripper/wrist region, within NEAR_PX
            of p (objects do not teleport) and the gripper is not next to it.
  held    : a gripper-close within GRASP_PX of a resting object, confirmed SETTLE frames later by the finger opening
            (an object keeps the fingers apart; a handle or an empty grasp does not), until the gripper opens:
            p = gripper pixel + the offset at the grasp.
Visible = held, or mask evidence at p, or (tracker elsewhere) the colour at p still matches the object.
Validated on the 2600 demonstrations against the subtask annotation: see /data/lgd/rma/trackv/tv_all.py.
"""
import cv2
import numpy as np

GRASP_PX, NEAR_PX, FAR_PX = 45, 12, 30
SETTLE, MIN_WIDTH = 12, 0.014
MIN_PIXELS = 20


class FusedTracks:
    def __init__(self, camera_matrix, size=256):
        self.P = np.asarray(camera_matrix, float)
        self.size = size
        self.ys, self.xs = np.mgrid[0:size, 0:size]

    def _project(self, points):
        h = np.c_[points, np.ones(len(points))] @ self.P.T
        return h[:, :2] / h[:, 2:3]

    def robot_region(self, state):
        pos = np.asarray(state[:3], float)
        R, _ = cv2.Rodrigues(np.asarray(state[3:6], float))
        pts = self._project(np.stack([pos + 0.03 * R[:, 2], pos - 0.24 * R[:, 2], pos + 0.05 * R[:, 0], pos + 0.05 * R[:, 1], pos]))
        radius = int(np.clip(max(np.hypot(*(pts[2] - pts[4])), np.hypot(*(pts[3] - pts[4]))), 6, 30))
        canvas = np.zeros((self.size, self.size), np.uint8)
        if np.all(np.isfinite(pts[:2])) and np.all(np.abs(pts[:2]) < 2000):
            cv2.line(canvas, tuple(int(round(v)) for v in pts[0]), tuple(int(round(v)) for v in pts[1]), 1, thickness=2 * radius)
        return canvas > 0, pts[4]

    def _colour(self, rgb, p):
        x, y = int(np.clip(round(p[0]), 3, self.size - 4)), int(np.clip(round(p[1]), 3, self.size - 4))
        return np.median(np.asarray(rgb[y - 3:y + 4, x - 3:x + 4]).reshape(-1, 3), axis=0)

    def _centroid(self, mask):
        return np.array([self.xs[mask].mean(), self.ys[mask].mean()]) if int(mask.sum()) >= MIN_PIXELS else None

    def reset(self, tracker_centres, rgb, state):
        """tracker_centres: per object [x, y] or None (the mask tracker's reliable centroid). Returns like step()."""
        self.k, self.t = len(tracker_centres), 0
        self.p = [None if c is None else np.asarray(c, float) for c in tracker_centres]
        self.ref = [None if c is None else self._colour(rgb, c) for c in self.p]
        self.held, self.offset, self.pending, self.was_closed = None, None, None, False
        self.events = []
        return [None if c is None else c.tolist() for c in self.p], [c is not None for c in self.p]

    def step(self, tracker_centres, rgb, state, closed):
        """closed: the gripper command executed before this frame is "close". Returns (centres [x,y]|None, visible)."""
        self.t += 1
        t, k = self.t, self.k
        robot, e = self.robot_region(state)
        closed = bool(closed)
        if closed and not self.was_closed and self.held is None:
            cand = [(np.hypot(*(self.p[j] - e)), j) for j in range(k) if self.p[j] is not None]
            self.pending = (min(cand)[1], self.p[min(cand)[1]] - e, t) if cand and min(cand)[0] < GRASP_PX else None
        if self.pending is not None and not closed:
            self.pending = None
        if self.pending is not None and t - self.pending[2] == SETTLE:
            if float(state[6]) >= MIN_WIDTH:
                self.held, self.offset = self.pending[0], self.pending[1]
                self.events.append((t, "grasp", self.held))
            else:
                self.events.append((t, "not_a_grasp", self.pending[0]))
            self.pending = None
        released = None
        if self.held is not None:
            self.p[self.held] = e + self.offset
            if not closed:
                self.events.append((t, "release", self.held))
                released, self.held = self.held, None
        self.was_closed = closed
        centres, visible = [], []
        for j in range(k):
            p = self.p[j]
            if p is None:
                centres.append(None)
                visible.append(False)
                continue
            if j == self.held:
                centres.append(p.tolist())
                visible.append(True)
                continue
            x, y = int(np.clip(round(p[0]), 0, self.size - 1)), int(np.clip(round(p[1]), 0, self.size - 1))
            evidence = False
            c = None if tracker_centres[j] is None else np.asarray(tracker_centres[j], float)
            if c is not None:
                cx, cy = int(np.clip(round(c[0]), 0, self.size - 1)), int(np.clip(round(c[1]), 0, self.size - 1))
                if robot[cy, cx]:
                    c = None
            if c is not None and np.hypot(*(c - p)) < NEAR_PX:
                evidence = True
                if np.hypot(*(c - e)) > FAR_PX:
                    p = self.p[j] = 0.8 * p + 0.2 * c
                    self.ref[j] = self._colour(rgb, p)
            if not evidence and self.ref[j] is not None and not robot[y, x]:
                evidence = bool(np.abs(self._colour(rgb, p) - self.ref[j]).mean() < 30)
            centres.append(p.tolist())
            visible.append(evidence)
        return centres, visible
