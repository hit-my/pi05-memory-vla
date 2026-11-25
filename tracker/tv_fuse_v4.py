"""Gripper veto v4 for the SAM3 object tracks = v3 (tv_fuse_v3.py, unchanged) with
  1. the corrected camera projection (tv_common_v4: the frames are a 640x480 render squeezed to 256x256, so the
     horizontal focal length is 0.75 x the one v3 used). Gripper pixel, wrist capsule and every distance to them change.
  2. thresholds re-chosen on the selection half with the corrected gripper pixel:
       grasp_edge_px 12 -> 3, grasp_anchor_px 20 -> 14   (a true grasp has the mask edge < 1 px from the gripper pixel in
       95 % of cases; a drawer handle next to an object in the drawer is >= 3.4 px away)
  3. release (new): a just-released object may be re-acquired within rel_px of its release place while the gripper is
       still next to it (v3 required the gripper to be far_px away, which only "worked" because its gripper pixel was
       off), the area test then also accepts the area the object had before the grasp (rel_area_rest) and the colour
       test is skipped there (rel_colour=False: a dropped object often shows another face / lies in shadow).
  4. duplicate masks (new, dup_tie): two masks on one thing while both objects rest there -> the object released
       last keeps it (it lies on top / in front), the other is not drawn.
  5. retirement: see retirement_plan (drawer: when the drawer has been closed; microwave: when the placement ends).
  6. tasks 18 and 19: no rules (enabled=False), the SAM3 centroid is drawn as is (chosen on the selection half).
The text below is the v3 description; it still applies.

Gripper veto v3 for the SAM3 object tracks ("seen only + gripper veto", revised; baseline v2 = tv_fuse.observe_only).

Causal: per frame it reads the K object masks, the RGB frame, the proprioceptive state, the gripper command issued before
the frame, the camera calibration and the task id. It never reads demo labels, future frames or simulator state.
(One optional input, `retired`, is supplied by the caller: at evaluation from the policy's own checklist, offline from the
subtask boundaries -- see retired_from_labels. The veto itself does not look at labels.)

One state machine (class VetoV3) is used by the offline function observe_only_v3 and by the step-wise
rma_online_veto_v3.OnlineVetoV3, so both run the same statements.

Rules (all thresholds in CONFIG; per-task overrides in TASK_CONFIG, keyed by task id)
  anchor      Each object has an anchor = the place where it rests. It is set at frame 0 and afterwards ONLY (a) while the
              object is carried / at its release, (b) once after a release when the object is seen still with the gripper
              away, (c) by a re-acquisition. A resting object's anchor never drifts (v2 moved it by an EMA, which let a
              mask that slides along the forearm drag the point).
  rest        Not held: the mask centroid is drawn only if it is within near_px of the anchor.
  occlusion   Not held, mask still at the anchor but touched by the projected gripper/wrist region or clearly smaller
              than its usual area: the visible part's centroid is unreliable, so the ANCHOR is drawn instead of the
              shifted centroid (last seen position; the object cannot have moved). Optionally (full_occl="hold") the
              anchor is also drawn when no usable mask exists while the gripper/wrist region covers the anchor.
  area        A mask larger than area_abs, or larger than max(area_ratio x running median of accepted areas, area_floor),
              is not the object (v2 compared with the frame-0 area, which is tiny when objects occlude each other).
  grasp       At a close command the candidates are the objects whose MASK (when it is at the anchor) lies within
              grasp_edge_px of the gripper pixel, or whose anchor is within grasp_anchor_px when the mask is away/unseen
              (v2: every anchor within 45 px). One candidate with the fingers kept apart is held; several candidates are
              decided by which one is seen travelling with the gripper; fingers fully closed = nothing held.
  held        The mask is accepted if its centroid is within held_px of the gripper pixel or its edge within
              held_edge_px, unless the mask stayed behind at the pre-grasp place while the gripper left. The anchor follows.
  release     The anchor stays at the last accepted carried position and the object may be re-acquired elsewhere.
  relocation  A resting object's mask that is seen somewhere else is believed only if the object may have moved
              (it was released, or the gripper closed on / next to it, and it has not since been seen undisturbed at
              the anchor for relock_n frames with the gripper away), the mask has been still for stable_n
              frames, is outside the gripper/wrist region, far from the gripper, has the first-frame colour and an area
              similar to the object's usual area. An object the gripper never closed on cannot relocate.
  retired     Objects flagged by the caller are not drawn and become inert: their mask is ignored from then on (the mask
              tracker may drop them), they are no grasp candidates and cannot claim another object's mask.
"""
import cv2
import numpy as np

from tv_common_v4 import centres_from_masks, eef_pixels, project, task_of

CONFIG = dict(
    enabled=True,          # False: no veto at all, the SAM3 centroid is drawn whenever the mask has >= 20 px
    near_px=12.0,          # a resting object's centroid must be this close to its anchor
    far_px=30.0,           # "gripper away" for re-acquisition / refinement
    anchor_ema=0.0,        # 0 = hard-locked anchor; v2 behaviour = 0.2
    area_abs=1200, area_ratio=3.0, area_floor=200, area_min=20, area_ref_n=30,
    occl_freeze=True, occl_area=0.65, occl_robot_px=3.0, occl_near_px=20.0,
    full_occl="blank",     # "hold": draw the anchor while the gripper/wrist region covers it and no usable mask exists
    grasp_edge_px=3.0, grasp_anchor_px=14.0,   # v3: 12 / 20 (tuned against the wrong gripper pixel)
    settle=12, min_width=0.014, choose_px=12.0, choose_wait=60,
    held_px=45.0, held_edge_px=25.0, behind_px=25.0,
    multi_hold=False,      # True: every candidate seen travelling with the gripper counts as held (tasks whose first-frame
                           # masks often put two masks on one physical object)
    loose_on_open=True, touch_px=0.0, relock_n=30,  # touch_px > 0: the gripper pixel merely passing this close also unlocks an object (off)
    stable_n=10, stable_px=3.0, colour_tol=35.0, reacq_area=(0.4, 2.5), dup_frac=0.5,
    rel_px=45.0,           # > 0 (v4): a just-released object may be re-acquired within rel_px of its release place while the gripper is still near
    rel_area_rest=True,    # True (v4): the area test of a just-released object also accepts the object's area before it was grasped
                           # (while carried the gripper hides part of it, so the carried area is too small a reference)
    rel_colour=False,      # False (v4): no colour test for a just-released object re-acquired near its release place
    dup_tie=True,          # True (v4): two masks on one thing, both objects resting there -> the one released last keeps it
)
# Tasks 18/19 (cabinet): the objects stand so close that the first-frame masks often put two masks on one physical
# object (identity errors the veto cannot repair). With the corrected gripper pixel the rules were compared with the raw
# SAM3 centroids on the selection half: the rules remove more right points than wrong ones there, so v4 switches them
# off for these two tasks (enabled=False). multi_hold / dup_frac are kept for anyone who switches them on again.
TASK_CONFIG = {18: dict(enabled=False, multi_hold=True, dup_frac=0.0), 19: dict(enabled=False, multi_hold=True, dup_frac=0.0)}


def config_for(task_id, override=None):
    cfg = dict(CONFIG)
    cfg.update(TASK_CONFIG.get(int(task_id), {}))
    if override:
        cfg.update(override)
    return cfg


def capsule(ep, state):
    """The projected gripper/wrist capsule of tv_render.robot_masks as numbers: [ax, ay, bx, by, radius] in pixels
    (nan when the projection is unusable). Proprioception + calibrated camera only."""
    pos = np.asarray(state[:3], np.float64)
    R, _ = cv2.Rodrigues(np.asarray(state[3:6], np.float64))
    z = R[:, 2]
    pts = project(ep, np.stack([pos + 0.03 * z, pos - 0.24 * z, pos + 0.05 * R[:, 0], pos + 0.05 * R[:, 1], pos]))
    radius = int(np.clip(max(np.hypot(*(pts[2] - pts[4])), np.hypot(*(pts[3] - pts[4]))), 6, 30))
    a, b = pts[0], pts[1]
    if not (np.all(np.isfinite([a, b])) and np.all(np.abs([a, b]) < 2000)):
        return np.full(5, np.nan)
    return np.array([round(a[0]), round(a[1]), round(b[0]), round(b[1]), radius], np.float64)


def cap_dist(cap, x, y):
    """Distance (px) from points to the capsule surface; <= 0 inside; inf when there is no capsule."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if not np.isfinite(cap[0]):
        return np.full(x.shape, np.inf)
    ax, ay, bx, by, r = cap
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    u = np.clip(((x - ax) * dx + (y - ay) * dy) / L, 0.0, 1.0) if L > 0 else np.zeros(x.shape)
    return np.hypot(x - (ax + u * dx), y - (ay + u * dy)) - r


def frame_features(masks, rgb, e, cap):
    """Everything the rules read from one frame. masks bool [K,256,256], rgb uint8 [256,256,3], e gripper pixel [2],
    cap = capsule(...)."""
    k = len(masks)
    c, v = centres_from_masks(masks[None])
    c, v = c[0], v[0]
    f = dict(c=c, v=v, area=masks.reshape(k, -1).sum(1).astype(np.int64), edge=np.full(k, np.inf), rob_dist=np.full(k, np.inf),
             rob_c=np.zeros(k, bool), rob_frac=np.zeros(k), col=np.full((k, 3), np.nan), ext=np.zeros((k, 2)),
             inter=np.zeros((k, k), np.int64), cap=np.asarray(cap, np.float64))
    for j in range(k):
        if not v[j]:
            continue
        ys, xs = np.nonzero(masks[j])
        f["edge"][j] = float(np.min(np.hypot(xs - e[0], ys - e[1])))
        d = cap_dist(cap, xs, ys)
        f["rob_dist"][j] = float(d.min())
        f["rob_frac"][j] = float((d <= 0).mean())
        f["rob_c"][j] = bool(cap_dist(cap, c[j][0], c[j][1]) <= 0)
        f["col"][j] = rgb[ys, xs].mean(axis=0)
        f["ext"][j] = (np.ptp(xs), np.ptp(ys))
        for i in range(k):
            f["inter"][j, i] = int((masks[j] & masks[i]).sum())
    return f


def features(ep, masks, rgb, e, states):
    rows = [frame_features(masks[t], rgb[t], e[t], capsule(ep, states[t])) for t in range(len(masks))]
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class VetoV4:
    """The rule state machine, one frame per step()."""

    def __init__(self, task_id, k, cfg=None):
        self.cfg = config_for(task_id, cfg)
        self.k = k
        self.t = 0
        self.q = [None] * k            # anchor
        self.aref = [[] for _ in range(k)]  # recent accepted areas
        self.colour0 = [None] * k
        self.loose = [False] * k       # the object may have been moved: a re-acquisition elsewhere is allowed
        self.refine = [False] * k      # released: snap the anchor once when the object is seen still, gripper away
        self.rel = [False] * k         # released and not yet seen settled: its usual area is unknown
        self.calm = [0] * k
        self.recent = [[] for _ in range(k)]
        self.held, self.pending, self.events = [], None, []  # held: normally one object (several only with multi_hold)
        self.grasp_e, self.grasp_q = {}, {}
        self.closed_prev = False
        self.retired = np.zeros(k, bool)
        self.last_rel = [-1] * k       # frame of the object's last release
        self.area_rest = [None] * k    # usual area before the last grasp

    def _area_ok(self, j, area):
        c = self.cfg
        ref = float(np.median(self.aref[j])) if self.aref[j] else None
        if area > c["area_abs"]:
            return False, ref
        if ref is not None and area > max(c["area_ratio"] * ref, c["area_floor"]):
            return False, ref
        return True, ref

    def step(self, f, e, closed, width, retired=None):
        cfg, k, t = self.cfg, self.k, self.t
        c_in, v_in, area = f["c"], f["v"], f["area"]
        centres, vis = np.full((k, 2), np.nan), np.zeros(k, bool)
        if retired is not None:
            self.retired |= np.asarray(retired, bool)
        if not cfg["enabled"]:
            for j in range(k):
                if v_in[j] and not self.retired[j]:
                    centres[j], vis[j] = c_in[j], True
            self.t += 1
            return centres, vis
        q, events = self.q, self.events
        if t == 0:
            for j in range(k):
                if v_in[j]:
                    q[j] = c_in[j].copy()
                    self.aref[j] = [int(area[j])]
                    if cfg["area_min"] <= area[j] <= cfg["area_abs"]:
                        self.colour0[j] = f["col"][j].copy()
                else:
                    self.loose[j] = True  # never seen: may be acquired wherever it first appears
        held, pending = self.held, self.pending
        if self.retired.any():
            # a retired object is inert: its mask is ignored (so the mask tracker may stop tracking it), it is no grasp
            # candidate and it cannot claim a mask from another object
            v_in = v_in & ~self.retired
            for h in [h for h in held if self.retired[h]]:
                events.append((t, "release", h))
                held.remove(h)
            if pending is not None:
                keep = [j for j in pending[0] if not self.retired[j]]
                pending = (keep,) + tuple(pending[1:]) if keep else None
        ok_area, aref = [False] * k, [None] * k
        at_anchor = [False] * k  # a usable mask at the anchor (also when partly occluded)
        for j in range(k):
            if v_in[j]:
                ok_area[j], aref[j] = self._area_ok(j, area[j])
                at_anchor[j] = ok_area[j] and q[j] is not None and _dist(c_in[j], q[j]) < cfg["occl_near_px"]
        # two masks on one thing: the mask belongs to the object that is being carried, else to one that rests at its
        # own anchor or was just released; for an object of lower standing it is not usable
        dup = [False] * k
        if cfg["dup_frac"] > 0:
            rank = [2 if j in held else 1 if (self.rel[j] or at_anchor[j]) else 0 for j in range(k)]
            for j in range(k):
                if v_in[j]:
                    dup[j] = any(i != j and v_in[i] and (rank[i] > rank[j] or (cfg["dup_tie"] and rank[i] == rank[j] == 1 and self.last_rel[i] > self.last_rel[j]))
                                 and f["inter"][j, i] > cfg["dup_frac"] * min(area[j], area[i]) for i in range(k))
            for j in range(k):
                if dup[j]:
                    at_anchor[j] = False
        # ---- which objects are at the gripper (mask touching it, or anchor next to it when the mask is away)
        near = []
        if t and closed != self.closed_prev:
            for j in range(k):
                if q[j] is None or j in held or self.retired[j]:
                    continue
                d = float(f["edge"][j]) if at_anchor[j] else None
                if (d is not None and d < cfg["grasp_edge_px"]) or (d is None and _dist(q[j], e) < cfg["grasp_anchor_px"]):
                    near.append((d if d is not None else _dist(q[j], e), j))
            near = [j for _, j in sorted(near)]
            if cfg["loose_on_open"] or closed:
                for j in near:  # the gripper closed / opened right at it: it may get moved even if no grasp is recognised
                    self.loose[j], self.calm[j] = True, 0
        # ---- grasp attribution
        if t and closed and not self.closed_prev and not held:
            pending = (near, t, e.copy(), {j: q[j].copy() for j in near}) if near else None
        if pending is not None and not closed:
            pending = None
        if pending is not None and t - pending[1] >= cfg["settle"]:
            near, t0, e0, q0 = pending
            moved = e - e0
            mv = float(np.hypot(*moved))
            grasp = []
            if width < cfg["min_width"] * 0.6:
                pending = None  # fingers fully closed: nothing between them
            elif len(near) == 1 and width >= cfg["min_width"]:
                grasp = [near[0]]
            elif mv > cfg["choose_px"]:
                score = {}
                for j in near:
                    if v_in[j]:
                        d = c_in[j] - q0[j]
                        if np.hypot(*d) > 0.4 * mv:
                            score[j] = float(np.hypot(*(d - moved)))
                best = min(score, key=score.get) if score else None
                if best is not None and score[best] < 0.6 * mv + 4:
                    grasp = sorted(j for j in score if score[j] < 0.6 * mv + 4) if cfg["multi_hold"] else [best]
                elif mv > 3 * cfg["choose_px"] or t - t0 > cfg["choose_wait"]:
                    pending = None  # the gripper left and nothing came along: it closed on a handle
            elif t - t0 > cfg["choose_wait"] and len(near) == 1 and width >= cfg["min_width"]:
                grasp = [near[0]]
            if grasp:
                held, pending = list(grasp), None
                for h in held:
                    self.grasp_e[h], self.grasp_q[h] = e0.copy(), q0[h].copy()
                    if self.aref[h]:
                        self.area_rest[h] = float(np.median(self.aref[h]))
                    events.append((t, "grasp", h))
        if held and (not closed or width < cfg["min_width"] * 0.6):
            for h in held:
                events.append((t, "release", h))
                self.loose[h], self.refine[h], self.rel[h], self.calm[h] = True, True, True, 0
                self.last_rel[h] = t
            held = []
        # ---- per object: accept / freeze / reject
        for j in range(k):
            drawn = None
            if not v_in[j] or not ok_area[j] or dup[j]:
                self.recent[j] = []
            else:
                c = c_in[j]
                self.recent[j] = (self.recent[j] + [c])[-cfg["stable_n"]:]
                still = len(self.recent[j]) == cfg["stable_n"] and np.ptp(np.asarray(self.recent[j]), axis=0).max() < cfg["stable_px"]
                away = _dist(c, e) > cfg["far_px"]
                if j in held:
                    ok = _dist(c, e) < cfg["held_px"] or f["edge"][j] < cfg["held_edge_px"]
                    if ok and _dist(e, self.grasp_e[j]) > cfg["behind_px"] and _dist(c, self.grasp_q[j]) < cfg["near_px"]:
                        ok = False  # the mask stayed where the object stood before the grasp
                    if ok:
                        q[j] = c.copy()
                        self.aref[j] = (self.aref[j] + [int(area[j])])[-cfg["area_ref_n"]:]
                        drawn = c
                else:
                    touched = f["rob_dist"][j] <= cfg["occl_robot_px"]  # the gripper/wrist region touches the mask
                    occluded = cfg["occl_freeze"] and q[j] is not None and not self.rel[j] and (
                        touched or (aref[j] is not None and area[j] < cfg["occl_area"] * aref[j]))
                    if cfg["occl_freeze"] and self.rel[j] and touched and not still:
                        pass  # just released, at the gripper and moving: neither the centroid nor the anchor can be trusted
                    elif q[j] is not None and _dist(c, q[j]) < cfg["near_px"] and not occluded:
                        drawn = c
                        self.aref[j] = (self.aref[j] + [int(area[j])])[-cfg["area_ref_n"]:]
                        if self.refine[j] and still and away:
                            q[j] = c.copy()
                            self.refine[j], self.rel[j] = False, False
                            self.aref[j] = [int(area[j])]
                        elif cfg["anchor_ema"] and away:
                            q[j] = (1 - cfg["anchor_ema"]) * q[j] + cfg["anchor_ema"] * c
                    elif at_anchor[j] and occluded:
                        drawn = q[j].copy()  # partly hidden: last seen position instead of the shifted centroid
                    elif self.loose[j] or q[j] is None:
                        near_release = bool(cfg["rel_px"] > 0 and self.rel[j] and q[j] is not None and _dist(c, q[j]) < cfg["rel_px"])
                        ok = bool(still and (away or near_release) and not f["rob_c"][j])
                        if ok and self.colour0[j] is not None and (cfg["rel_colour"] or not near_release):
                            ok = float(np.abs(f["col"][j] - self.colour0[j]).mean()) < cfg["colour_tol"]
                        if ok and aref[j] is not None:
                            hi = aref[j]
                            if cfg["rel_area_rest"] and self.rel[j] and self.area_rest[j] is not None:
                                hi = max(hi, self.area_rest[j])
                            ok = (self.rel[j] or cfg["reacq_area"][0] * aref[j] <= area[j]) and area[j] <= cfg["reacq_area"][1] * hi
                        if ok:
                            events.append((t, "reacquired", j))
                            q[j] = c.copy()
                            self.refine[j], self.rel[j], self.calm[j] = False, False, 0  # stays loose until seen calm
                            self.aref[j] = [int(area[j])]
                            drawn = c
            if j not in held and q[j] is not None:
                # may the object have been moved?  gripper near the anchor -> yes; seen undisturbed for a while -> no
                ge = _dist(e, q[j])
                if cfg["touch_px"] > 0 and ge < cfg["touch_px"]:
                    self.loose[j], self.calm[j] = True, 0
                elif self.loose[j] and drawn is not None and ge > cfg["far_px"] + 10 and not self.refine[j]:
                    self.calm[j] += 1
                    if self.calm[j] >= cfg["relock_n"]:
                        self.loose[j], self.calm[j] = False, 0
                else:
                    self.calm[j] = 0
                if drawn is None and cfg["full_occl"] == "hold" and not (pending is not None and j in pending[0]) \
                        and float(cap_dist(f["cap"], q[j][0], q[j][1])) <= cfg["occl_robot_px"]:
                    drawn = q[j].copy()
            if drawn is not None and not self.retired[j]:
                centres[j], vis[j] = drawn, True
        self.held, self.pending = held, pending
        self.closed_prev = bool(closed)
        self.t = t + 1
        return centres, vis


def observe_from_features(ep, F, retired=None, cfg=None):
    """observe_only_v4 on cached per-frame measurements (features(...) plus e / closed / width)."""
    n, k = F["v"].shape
    core = VetoV4(task_of(ep), k, cfg)
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    keys = ("c", "v", "area", "edge", "rob_dist", "rob_c", "col", "cap", "inter")
    for t in range(n):
        centres[t], vis[t] = core.step({key: F[key][t] for key in keys}, F["e"][t], bool(F["closed"][t]), float(F["width"][t]),
                                       None if retired is None else retired[t])
    return centres, vis, core.events


def observe_only_v4(ep, masks, robot, rgb, states, actions, retired=None, cfg=None):
    """v3 of tv_fuse.observe_only; same arguments (`robot` is accepted for compatibility: v3 computes the same
    gripper/wrist capsule analytically from `states`).
    retired: optional bool [N,K], True from the frame an object is retired (nothing is drawn for it afterwards and its
    mask is ignored; sticky).
    Returns centres [N,K,2] (nan when nothing is drawn), visible [N,K], events."""
    n, k = masks.shape[:2]
    e = eef_pixels(ep, states)
    closed = np.r_[False, actions[:-1, 6] > 0]
    core = VetoV4(task_of(ep), k, cfg)
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    for t in range(n):
        f = frame_features(masks[t], rgb[t], e[t], capsule(ep, states[t]))
        centres[t], vis[t] = core.step(f, e[t], bool(closed[t]), float(states[t, 6]), None if retired is None else retired[t])
    return centres, vis, core.events


# ---------------------------------------------------------------- retirement signal (offline: from the subtask labels)
def _container(text):
    """'top drawer' / 'middle drawer' / 'bottom drawer' / 'microwave' named in a subtask text, else None"""
    s = str(text).lower()
    for name in ("top drawer", "middle drawer", "bottom drawer", "microwave"):
        if name in s:
            return name
    return None


def retirement_plan(task_id, texts):
    """texts: the episode's subtask texts in the order they occur. Returns a list of (index of the subtask whose END
    retires, object index, text of that subtask).
      drawer    : object j is retired when the subtask that CLOSES the drawer it was placed into ends (the first
                  'close the <same> drawer' after 'place <j> into the <same> drawer'). While the drawer is open the
                  object lies visibly in it and the normal rules apply. No such close subtask -> never retired.
      microwave : object j is retired when 'place <j> into the microwave' ends. Seen from the front camera the inside
                  of the microwave is hidden also while its door is open (checked on frames), so the object is out of
                  view from the placement on; waiting for the door would only leave the mask tracker's point on the
                  door edge / the gripper.
    """
    import sys
    sys.path.insert(0, "xxxx/rma/trackv/code")
    import rma_spec
    plan = []
    for i, text in enumerate(texts):
        s = str(text).lower()
        box = _container(s)
        j = rma_spec.subtask_object(task_id, s)
        if j is None or box is None or not s.startswith("place"):
            continue
        if box == "microwave":
            plan.append((i, j, str(text)))
            continue
        for i2 in range(i + 1, len(texts)):
            s2 = str(texts[i2]).lower()
            if s2.startswith("close") and _container(s2) == box:
                plan.append((i2, j, str(texts[i2])))
                break
    return plan


def retired_from_labels(ep, n=None, mode="closed"):
    """bool [N,K], True from the first frame after the end of the subtask that retires the object (retirement_plan).
    Offline stand-in for the policy's checklist tick of that subtask; reads the demo's subtask labels.
    mode="closed" (the v4 output): as above.
    mode="handle" (alternative, stored as *_handle in all/final_v4): a drawer object is retired already when the gripper
        closes on the handle inside the closing subtask (first close command there), i.e. when the drawer starts to be
        pushed shut. At evaluation: the checklist's current item is "close the <drawer>" and the gripper command closes.
        Reason: while the drawer is pushed shut the object moves without being grasped and then disappears; the rules
        cannot know that and may keep a dot at the old place until the closing subtask ends."""
    import sys
    sys.path.insert(0, "xxxx/rma/trackv/code")
    import rma_spec
    task = task_of(ep)
    m = np.load(f"xxxx/rma/lr_eps/meta/{ep}.npz")
    sub = m["subtask"] if n is None else m["subtask"][:n]
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    starts = [0] + [t for t in range(1, len(sub)) if sub[t] != sub[t - 1]]
    seq = [text[int(sub[t])] for t in starts]
    ends = starts[1:] + [len(sub)]
    out = np.zeros((len(sub), len(rma_spec.TASKS[task]["objects"])), bool)
    cmd = np.r_[False, m["actions"][:len(sub) - 1, 6] > 0]
    for i, j, text_i in retirement_plan(task, seq):
        t0 = ends[i]
        if mode == "handle" and str(text_i).lower().startswith("close"):
            hit = np.flatnonzero(cmd[starts[i]:ends[i]])
            if len(hit):
                t0 = starts[i] + int(hit[0])
        out[t0:, j] = True
    return out
