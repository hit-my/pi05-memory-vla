"""Tracking rules v5 for the SAM3 object tracks (per-task rule sets chosen against simulator ground truth on the selection half).

Causal. Per frame the rules read: the K object masks (as per-frame measurements tv_fuse_v4.frame_features: centroid, area,
distance of the mask edge to the gripper pixel, mean colour, overlaps between the masks), the corrected gripper pixel
(tv_common_v4, from the robot state), the gripper command issued before the frame, the finger opening (state[6]), fixed
per-task constants (constants_v5.json) and - for retirement only - the checklist signal `retired` supplied by the caller.
No future frames, no simulator state, no ground truth.

Two rule sets ("mode", chosen per task in constants_v5.json):
  raw   the SAM3 centroid is drawn whenever the mask has >= 20 px. Optional colour gate (raw_colour_tol > 0): not drawn while
        the mask's mean colour differs from its first-frame colour by more than the tolerance. Plus retirement.
  v5    the state machine below ("an object only moves when the gripper moves it"):
     anchor     every object has an anchor = where it rests. Frame 0: the mask centroid (it follows the mask during the first
                start_settle frames while the scene settles).
     rest       an object the gripper has not grasped cannot move: its mask centroid is drawn only while it is within near_px
                of the anchor (a mask that jumps to the arm, to another object, to the background is not drawn; when the mask
                comes back it is drawn again at once). The centroid of the visible part is drawn as is (no freezing).
     grasp      when the gripper command closes, objects whose mask edge is within grab_edge_px of the gripper pixel (or whose
                anchor is within grab_anchor_px while the mask is away) during the next grab_win frames are grasp candidates.
                A candidate is "carried" once its mask has left its anchor together with the gripper for conf_n frames (or, if
                it is the only candidate and its mask is hidden, when the fingers stay apart after settle frames).
     carried    the mask is drawn while it is at the gripper (centroid within held_px of the gripper pixel or mask edge within
                held_edge_px). Hidden in the hand -> nothing is drawn; the anchor travels with the gripper. A "carried" mask
                that stays at the pre-grasp place while the gripper leaves was never carried (back to rest).
     release    gripper command opens (or the fingers close completely): the object rests where it was released. Its mask is
                drawn at once while it is within rel_px of the release place - unless it leaves together with the gripper
                (it is on the gripper). When the mask has been still for stable_n frames with the gripper >= lock_far_px away,
                the anchor is locked there (-> rest); a released object may lock again within rel_px of its release place.
     push       (switch) a resting object's anchor follows its mask while the mask moves together with the closed gripper
                (the object lies in a drawer that is pushed / pulled).
     dup        (dup_frac > 0) two masks on one thing: the carried object, else the object released later, keeps the mask
                (it lies on top); the other object is not drawn there.
     hold_rest  (switch "pre") an object the gripper has not carried yet is drawn at its anchor while its mask is away / lost
                (the arm passed in front of it and the mask stayed on the arm).
     hold_dup   (switch; task 19) a resting object whose mask was taken by the object released later (dup: the newcomer stands in
                front of it and hides most of it) is drawn at its anchor: it still stands there.
     hold_rel   (switch; task 22) after a release, while no usable mask exists, the gripper pixel of the release is drawn
                (the object lies where it was put; only a sliver of it is visible) until it is retired.
     area       a mask larger than area_abs or than area_ratio x the running median of accepted areas is not the object.
     retired    objects flagged by the caller are never drawn again and are inert.
Tried on the selection half and removed (see v5/PROGRESS.md): freezing at the anchor while partly hidden (v4), re-anchoring when
the mask grows, rejecting masks inside the projected gripper capsule, a tight "mask touches the gripper" test while carried,
a minimum carried area, soft / confirmed retirement.
"""
import json
import sys

import numpy as np

if "xxxx/rma/trackv" not in sys.path:
    sys.path.append("xxxx/rma/trackv")
from tv_fuse_v4 import _dist, cap_dist, capsule, frame_features  # noqa: F401  (capsule is used by the callers)

CONFIG5 = dict(
    mode="v5",             # "raw" | "v5"
    min_area=20,           # a mask with fewer pixels does not exist for the rules (tasks 18/19 use a lower value)
    blob="",               # "largest": the centroid / area of the largest connected part of the mask are used instead of those of
                           #            the whole mask (a mask that sits on the object AND on a second thing; needs f["cc_c"], f["cc_a"])
    raw_colour_tol=0.0,    # raw mode: > 0 = colour gate (mean abs RGB difference to the first-frame colour)
    near_px=12.0, start_settle=15,
    grab_edge_px=4.0, grab_anchor_px=14.0, grab_win=12,
    settle=12, min_width=0.014, lift_px=8.0, conf_n=3,
    held_px=45.0, held_edge_px=25.0, behind_px=25.0, behind_n=5,
    rel_px=30.0, stable_n=10, stable_px=3.0, lock_far_px=20.0,
    push=True, push_tol=0.5, push_abs=5.0,
    dup_frac=0.5,          # 0 = off
    hold_rest="",          # "" | "pre"
    hold_rel=False,
    hold_dup=False,        # True: a resting object that loses its mask to a later-released object (dup) is drawn at its anchor
    area_abs=1200, area_ratio=3.0, area_floor=200, area_ref_n=30,
)
REST, MOV, REL = 0, 1, 2
def blobs_of(masks, n=3):
    """the n largest 8-connected parts of every mask: centroids cc_c [K,n,2] (nan), areas cc_a [K,n] (0)"""
    import cv2
    k = len(masks)
    cc_c, cc_a = np.full((k, n, 2), np.nan), np.zeros((k, n), np.int64)
    for j in range(k):
        if not masks[j].any():
            continue
        num, lab, stats, cen = cv2.connectedComponentsWithStats(masks[j].astype(np.uint8), connectivity=8)
        order = sorted(range(1, num), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))[:n]
        for r, i in enumerate(order):
            cc_c[j, r], cc_a[j, r] = cen[i], int(stats[i, cv2.CC_STAT_AREA])
    return cc_c, cc_a


def frame_features5(masks, rgb, e, cap, min_pixels=20, blobs=False):
    """Per-frame measurements = tv_fuse_v4.frame_features (used as is when min_pixels == 20, the v4 mask threshold).
    With a lower threshold the same quantities are measured for masks of >= min_pixels pixels.
    blobs=True adds cc_c / cc_a (blobs_of)."""
    if blobs:
        f = frame_features5(masks, rgb, e, cap, min_pixels)
        f["cc_c"], f["cc_a"] = blobs_of(masks)
        return f
    if min_pixels == 20:
        return frame_features(masks, rgb, e, cap)
    k = len(masks)
    f = dict(c=np.full((k, 2), np.nan), v=np.zeros(k, bool), area=masks.reshape(k, -1).sum(1).astype(np.int64), edge=np.full(k, np.inf),
             rob_dist=np.full(k, np.inf), rob_c=np.zeros(k, bool), rob_frac=np.zeros(k), col=np.full((k, 3), np.nan), ext=np.zeros((k, 2)),
             inter=np.zeros((k, k), np.int64), cap=np.asarray(cap, np.float64))
    for j in range(k):
        if f["area"][j] < min_pixels:
            continue
        ys, xs = np.nonzero(masks[j])
        f["c"][j], f["v"][j] = (xs.mean(), ys.mean()), True
        f["edge"][j] = float(np.min(np.hypot(xs - e[0], ys - e[1])))
        d = cap_dist(cap, xs, ys)
        f["rob_dist"][j] = float(d.min())
        f["rob_frac"][j] = float((d <= 0).mean())
        f["rob_c"][j] = bool(cap_dist(cap, f["c"][j][0], f["c"][j][1]) <= 0)
        f["col"][j] = rgb[ys, xs].mean(axis=0)
        f["ext"][j] = (np.ptp(xs), np.ptp(ys))
        for i in range(k):
            f["inter"][j, i] = int((masks[j] & masks[i]).sum())
    return f


def feature_floor(cfg):
    """mask-size threshold of the measurements for a rule configuration"""
    return int(min(20, cfg["min_area"]))


def load_constants(path=None):
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants_v5.json")
    with open(path) as fh:
        return json.load(fh)


_CONST = None


def config_for(task_id, override=None):
    """per-task configuration: CONFIG5 <- constants_v5.json["tasks"][task]["cfg"] <- override
    (override["_no_constants"] = True skips the constants file)"""
    global _CONST
    cfg = dict(CONFIG5)
    if override is None or not override.get("_no_constants"):
        if _CONST is None:
            try:
                _CONST = load_constants()
            except FileNotFoundError:
                _CONST = {"tasks": {}}
        cfg.update(_CONST.get("tasks", {}).get(str(int(task_id)), {}).get("cfg", {}))
    if override:
        cfg.update({k: v for k, v in override.items() if k != "_no_constants"})
    return cfg


class VetoV5:
    """The rule state machine, one frame per step()."""

    def __init__(self, task_id, k, cfg=None):
        self.cfg = config_for(task_id, cfg)
        self.k, self.t = k, 0
        self.events = []
        self.retired = np.zeros(k, bool)
        self.q = [None] * k                 # anchor
        self.st = [REST] * k
        self.conf = [False] * k             # MOV: confirmed carried
        self.q0, self.e0, self.off = [None] * k, [None] * k, [None] * k   # anchor / gripper pixel at the grasp, mask - gripper offset
        self.grab_t = [-1] * k
        self.recent = [[] for _ in range(k)]
        self.aref = [[] for _ in range(k)]  # recent accepted areas
        self.colour0 = [None] * k
        self.rel_t = [-1] * k
        self.cn, self.bn = [0] * k, [0] * k
        self.q_rel, self.e_rel = [None] * k, [None] * k   # release place (last carried position) / gripper pixel at the release
        self.ever = [False] * k             # has (possibly) been carried
        self.far = [False] * k
        self.e_close, self.q_close = None, [None] * k
        self.closed_prev = False
        self.close_t = -10 ** 6
        self.gave_up = False

    def _area_ok(self, j, area):
        c = self.cfg
        if area > c["area_abs"]:
            return False
        if self.aref[j]:
            ref = float(np.median(self.aref[j]))
            if area > max(c["area_ratio"] * ref, c["area_floor"]):
                return False
        return True

    def _accept(self, j, area):
        self.aref[j] = (self.aref[j] + [int(area)])[-self.cfg["area_ref_n"]:]

    def step(self, f, e, closed, width, retired=None):
        """f: per-frame measurements (c [K,2], v [K], area [K], edge [K], col [K,3], inter [K,K]); e gripper pixel [2];
        closed: gripper command issued before this frame is 'close'; width: finger opening state[6];
        retired: optional bool [K]. Returns centres [K,2] (nan = not drawn), visible [K]."""
        cfg, k, t = self.cfg, self.k, self.t
        if retired is not None:
            self.retired |= np.asarray(retired, bool)
        c_in, area = f["c"], f["area"]
        if cfg["blob"] == "largest":  # the largest connected part of each mask stands for the mask
            c_in, area = f["cc_c"][:, 0], f["cc_a"][:, 0]
        v_in = np.asarray(f["v"], bool) & (np.asarray(area) >= cfg["min_area"])
        centres, vis = np.full((k, 2), np.nan), np.zeros(k, bool)
        if cfg["mode"] == "raw":
            for j in range(k):
                if t == 0 and v_in[j]:
                    self.colour0[j] = f["col"][j].copy()
                if v_in[j] and not self.retired[j]:
                    if cfg["raw_colour_tol"] > 0 and self.colour0[j] is not None and float(np.abs(f["col"][j] - self.colour0[j]).mean()) > cfg["raw_colour_tol"]:
                        continue  # the mask no longer has the colour it had in the first frame: it is on something else
                    centres[j], vis[j] = c_in[j], True
            self.t += 1
            return centres, vis
        q, st, ev = self.q, self.st, self.events
        usable = [bool(v_in[j]) and not self.retired[j] and self._area_ok(j, area[j]) for j in range(k)]
        lose = [False] * k
        if cfg["dup_frac"] > 0:  # two masks on one thing: the carried object, else the one released later, keeps it
            rank = [(2 if st[j] == MOV and self.conf[j] else 1, self.rel_t[j]) for j in range(k)]
            lose = [usable[j] and any(i != j and v_in[i] and not self.retired[i] and rank[i] > rank[j]
                                      and f["inter"][j, i] > cfg["dup_frac"] * min(area[j], area[i]) for i in range(k)) for j in range(k)]
            usable = [usable[j] and not lose[j] for j in range(k)]
        if t == 0:
            for j in range(k):
                if usable[j]:
                    q[j] = c_in[j].copy()
                    self._accept(j, area[j])
                    self.colour0[j] = f["col"][j].copy()
        # ---- gripper events
        if t and closed and not self.closed_prev:
            self.close_t, self.gave_up = t, False
            self.e_close = np.array(e, float)
            self.q_close = [None if x is None else x.copy() for x in q]
        tiny = width < cfg["min_width"] * 0.6
        release = t and ((not closed and self.closed_prev) or (closed and tiny and not self.gave_up and t - self.close_t >= cfg["settle"]))
        if release:
            if closed:
                self.gave_up = True
            for j in range(k):
                if st[j] == MOV:
                    if self.conf[j]:
                        st[j], self.rel_t[j], self.recent[j] = REL, t, []
                        # rounded: the release place may be drawn (hold) and must not depend on last-bit differences of the projection
                        self.q_rel[j], self.e_rel[j] = np.round(q[j], 3), np.round(np.array(e, float), 3)
                        ev.append((t, "release", j))
                    else:
                        st[j] = REST
                        if self.far[j]:
                            self.ever[j] = True  # a grasp candidate, the gripper went away closed and its mask was not seen staying behind: it may have been carried
                    self.conf[j] = False
        # ---- grasp candidates (for grab_win frames after the close command)
        if closed and not self.gave_up and 0 <= t - self.close_t <= cfg["grab_win"]:
            for j in range(k):
                if self.retired[j] or st[j] == MOV or q[j] is None:
                    continue
                at = usable[j] and _dist(c_in[j], q[j]) < (cfg["rel_px"] if st[j] == REL else cfg["near_px"])
                if (at and f["edge"][j] < cfg["grab_edge_px"]) or (not at and _dist(q[j], e) < cfg["grab_anchor_px"]):
                    st[j], self.conf[j], self.cn[j], self.bn[j], self.far[j] = MOV, False, 0, 0, False
                    self.q0[j], self.e0[j], self.off[j], self.grab_t[j] = q[j].copy(), np.array(e, float), None, t
        n_cand = sum(1 for j in range(k) if st[j] == MOV)
        # ---- per object
        for j in range(k):
            if self.retired[j]:
                if st[j] == MOV:
                    st[j], self.conf[j] = REST, False
                continue
            drawn = None
            c = c_in[j]
            if st[j] == MOV:
                moved = e - self.e0[j]
                mv = float(np.hypot(*moved))
                if mv > cfg["behind_px"]:
                    self.far[j] = True
                if not self.conf[j]:
                    if usable[j]:
                        d0 = c - self.q0[j]
                        if np.hypot(*d0) > cfg["lift_px"] and _dist(c, e) < cfg["held_px"] and float(np.hypot(*(d0 - moved))) < 0.6 * mv + 4:
                            self.cn[j] += 1
                            if self.cn[j] >= cfg["conf_n"]:
                                self.conf[j] = self.ever[j] = True
                                ev.append((t, "grasp", j))
                        else:
                            self.cn[j] = 0
                        if not self.conf[j] and _dist(c, q[j]) < cfg["near_px"]:
                            drawn = c  # still at / near its place
                            if mv > cfg["behind_px"] and _dist(c, self.q0[j]) < cfg["near_px"]:
                                st[j], self.far[j] = REST, False  # the gripper left without it
                    if not self.conf[j] and st[j] == MOV and not usable[j] and n_cand == 1 and t - self.grab_t[j] >= cfg["settle"] and width >= cfg["min_width"]:
                        self.conf[j] = self.ever[j] = True  # hidden in the hand, only candidate, fingers kept apart
                        ev.append((t, "grasp", j))
                if self.conf[j]:
                    ok = usable[j] and (_dist(c, e) < cfg["held_px"] or f["edge"][j] < cfg["held_edge_px"])
                    if ok and mv > cfg["behind_px"] and _dist(c, self.q0[j]) < cfg["near_px"]:
                        ok = False  # the mask stayed where the object stood before the grasp
                        self.bn[j] += 1
                        if self.bn[j] >= cfg["behind_n"]:  # ... for several frames: it was never carried
                            st[j], self.conf[j], q[j] = REST, False, self.q0[j].copy()
                            ev.append((t, "ungrasp", j))
                            centres[j], vis[j] = c, True
                            continue
                    if ok:
                        q[j], self.off[j] = c.copy(), c - e
                        self._accept(j, area[j])
                        drawn = c
                    else:
                        q[j] = e + self.off[j] if self.off[j] is not None else np.array(e, float)
                        drawn = None
            elif st[j] == REL:
                ok = usable[j] and _dist(c, self.q_rel[j]) < cfg["rel_px"]
                if ok:
                    de = e - self.e_rel[j]
                    md = float(np.hypot(*de))
                    if md > 6 and float(np.hypot(*(c - self.q_rel[j] - de))) < 0.5 * md + 3:
                        ok = False  # the mask leaves with the gripper: it is on the gripper, not on the released object
                if ok:
                    drawn = c
                    self.recent[j] = (self.recent[j] + [c])[-cfg["stable_n"]:]
                    if len(self.recent[j]) == cfg["stable_n"] and np.ptp(np.asarray(self.recent[j]), axis=0).max() < cfg["stable_px"] \
                            and _dist(c, e) > cfg["lock_far_px"]:
                        st[j], q[j] = REST, c.copy()
                        self.aref[j] = [int(area[j])]
                        ev.append((t, "settled", j))
                else:
                    self.recent[j] = []
                    if cfg["hold_rel"]:
                        drawn = self.e_rel[j].copy()
            else:  # REST
                if q[j] is None:
                    if usable[j]:
                        q[j] = c.copy()
                        drawn = c
                elif usable[j] and _dist(c, q[j]) < cfg["near_px"]:
                    drawn = c
                    self._accept(j, area[j])
                    if t < cfg["start_settle"]:
                        q[j] = c.copy()
                    elif cfg["push"] and closed and self.e_close is not None and self.q_close[j] is not None:
                        de = e - self.e_close
                        md = float(np.hypot(*de))
                        if md > 3 and float(np.hypot(*(c - self.q_close[j] - de))) < cfg["push_tol"] * md + cfg["push_abs"]:
                            q[j] = c.copy()  # it moves together with the closed gripper (in a drawer that is being pushed/pulled)
                elif usable[j] and self.q_rel[j] is not None and _dist(c, self.q_rel[j]) < cfg["rel_px"]:
                    # a released object lies within rel_px of its release place: lock again where its mask is seen still
                    self.recent[j] = (self.recent[j] + [c])[-cfg["stable_n"]:]
                    if len(self.recent[j]) == cfg["stable_n"] and np.ptp(np.asarray(self.recent[j]), axis=0).max() < cfg["stable_px"] \
                            and _dist(c, e) > cfg["lock_far_px"] and not f["rob_c"][j]:
                        q[j] = c.copy()
                        self.aref[j] = [int(area[j])]
                        drawn = c
                        ev.append((t, "relocked", j))
                if drawn is None and q[j] is not None:
                    if cfg["hold_rel"] and self.q_rel[j] is not None:
                        drawn = self.e_rel[j].copy()  # released, no usable mask: the gripper pixel of its release
                    elif cfg["hold_rest"] == "pre" and not self.ever[j]:
                        drawn = q[j].copy()  # not carried yet, mask away / lost: it still stands at its anchor
                    elif cfg["hold_dup"] and lose[j]:
                        drawn = q[j].copy()  # its mask now sits on the object put in front of it: it still stands at its anchor
            if drawn is not None:
                centres[j], vis[j] = drawn, True
        self.closed_prev = bool(closed)
        self.t = t + 1
        return centres, vis


FKEYS = ("c", "v", "area", "edge", "rob_c", "col", "inter", "cc_c", "cc_a")


def observe_from_features(task_id, F, retired=None, cfg=None, masks=None):
    """the rules on cached per-frame measurements (v5/feat5/<ep>.npz: frame_features5 stacked + e, closed, width).
    (masks: unused, no rule in use reads the masks themselves)"""
    n, k = F["v"].shape
    core = VetoV5(task_id, k, cfg)
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    for t in range(n):
        f = {key: F[key][t] for key in FKEYS if key in F}
        centres[t], vis[t] = core.step(f, F["e"][t], bool(F["closed"][t]), float(F["width"][t]), None if retired is None else retired[t])
    return centres, vis, core.events


def needs_masks(cfg):
    return False


def load_masks(ep):
    """the stored masks of an episode [N,K,256,256] (v5/track > all/final_fix > all/final)"""
    from pathlib import Path
    for d in ("v5/track_split", "v5/track", "all/final_fix", "all/final"):
        p = Path("xxxx/rma/trackv") / d / f"{ep}.npz"
        if p.exists():
            return np.unpackbits(np.load(p)["masks"], axis=-1)[..., :256].astype(bool)
    raise FileNotFoundError(ep)


def observe_only_v5(ep, masks, rgb, states, actions, retired=None, cfg=None):
    """masks bool [N,K,256,256], rgb uint8 [N,256,256,3], states [N,8], actions [N,7]. Returns centres, visible, events."""
    from tv_common_v4 import eef_pixels, task_of
    n, k = masks.shape[:2]
    e = eef_pixels(ep, states)
    closed = np.r_[False, actions[:-1, 6] > 0]
    core = VetoV5(task_of(ep), k, cfg)
    floor = feature_floor(core.cfg)
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    for t in range(n):
        f = frame_features5(masks[t], rgb[t], e[t], capsule(ep, states[t]), floor, blobs=core.cfg["blob"] != "")
        centres[t], vis[t] = core.step(f, e[t], bool(closed[t]), float(states[t, 6]), None if retired is None else retired[t])
    return centres, vis, core.events


# ---------------------------------------------------------------- retirement signal (offline stand-in for the checklist)
def _container(text):
    s = str(text).lower()
    for name in ("top drawer", "middle drawer", "bottom drawer", "microwave", "basket"):
        if name in s:
            return name
    return None


def retirement_plan_v5(task_id, texts, spec):
    """spec: {"drawer": trig, "microwave": trig, "basket": trig}; trig in
         none | place_open | place_end | close_start | close_grip | close_moveNN | close_end
       place_open   : the gripper command opens while the checklist's current item is 'place <obj> into the <container>'
       place_end    : the subtask 'place <obj> into the <container>' ends (checklist tick)
       close_*      : about the first 'close the <container>' subtask after the placement:
                      start = it becomes the current item; grip = the gripper command closes inside it;
                      moveNN = the gripper closed inside it and then moved >= NN px in the image (the drawer is being pushed);
                      end = it ends (checklist tick).
    Returns list of (subtask index, when, object index)."""
    sys.path.insert(0, "xxxx/rma/trackv/code")
    import rma_spec
    plan = []
    for i, text in enumerate(texts):
        s = str(text).lower()
        box = _container(s)
        j = rma_spec.subtask_object(task_id, s)
        if j is None or box is None or not s.startswith("place"):
            continue
        kind = "drawer" if "drawer" in box else box
        trig = spec.get(kind, "none")
        if trig == "none":
            continue
        if trig in ("place_end", "place_open"):
            plan.append((i, trig.split("_")[1], j))
            continue
        for i2 in range(i + 1, len(texts)):
            s2 = str(texts[i2]).lower()
            if s2.startswith("close") and _container(s2) == box:
                plan.append((i2, trig.split("_")[1], j))
                break
    return plan


def retired_from_labels_v5(ep, n, spec):
    """bool [N,K], True from the frame the object is retired. Offline stand-in for the policy's checklist (reads the demo's
    subtask labels, the gripper commands and - for close_moveNN - the gripper pixel). Causal twin: rma_online_veto_v5.RetireSignalV5."""
    sys.path.insert(0, "xxxx/rma/trackv/code")
    import rma_spec
    task = int(ep[1:3])
    m = np.load(f"xxxx/rma/lr_eps/meta/{ep}.npz")
    sub = m["subtask"][:n]
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    starts = [0] + [t for t in range(1, len(sub)) if sub[t] != sub[t - 1]]
    seq = [text[int(sub[t])] for t in starts]
    ends = starts[1:] + [len(sub)]
    out = np.zeros((len(sub), len(rma_spec.TASKS[task]["objects"])), bool)
    cmd = np.r_[False, m["actions"][:len(sub) - 1, 6] > 0]
    for i, when, j in retirement_plan_v5(task, seq, spec or {}):
        t0 = ends[i]
        if when == "start":
            t0 = starts[i]
        elif when == "grip":
            hit = np.flatnonzero(cmd[starts[i]:ends[i]])
            t0 = starts[i] + int(hit[0]) if len(hit) else ends[i]
        elif when.startswith("move"):
            from tv_common_v4 import eef_pixels
            e = eef_pixels(ep, m["states"][:len(sub)].astype(np.float64))
            hit = np.flatnonzero(cmd[starts[i]:ends[i]])
            if len(hit):
                g0 = starts[i] + int(hit[0])
                far = np.flatnonzero(np.hypot(*(e[g0:ends[i]] - e[g0]).T) >= float(when[4:]))
                if len(far):
                    t0 = g0 + int(far[0])
        elif when == "open":
            seg = cmd[max(starts[i] - 1, 0):ends[i]]
            hit = np.flatnonzero(seg[:-1] & ~seg[1:])
            t0 = max(starts[i] - 1, 0) + int(hit[0]) + 1 if len(hit) else ends[i]
        out[t0:, j] = True
    return out
