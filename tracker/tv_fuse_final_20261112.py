"""Fused tracker: a mask tracker "sees", gripper kinematics decides what can move.  Causal; RGB + proprioception only.

State per object: position p, mode in {free, held, hidden}.
  free   : the object rests somewhere. p only follows the mask tracker when the mask is plausible:
           not in the projected robot region, close to p (objects do not teleport), gripper not next to it.
  held   : a gripper-close near a free object, confirmed SETTLE frames later by the finger opening (an object keeps the
           fingers apart; a handle or an empty grasp does not), until the gripper opens: p = gripper pixel + offset.
  hidden : not seen for HIDE frames (inside a drawer / microwave). A hidden object cannot be grasped again until it is
           seen at p for SEEN frames in a row.
Visibility = mask evidence at p, or colour evidence at p when the mask tracker is elsewhere.
"""
import numpy as np

from tv_common import centres_from_masks, eef_pixels

GRASP_PX, NEAR_PX, FAR_PX, HIDE, SEEN = 45, 12, 30, 15, 5
SETTLE, MIN_WIDTH = 12, 0.014  # finger qpos 12 frames after the close command: objects p1 0.018, handles p50 0.010


def fuse(ep, masks, robot, rgb, states, actions, **kw):
    c, v = centres_from_masks(masks & ~robot[:, None])
    c0, _ = centres_from_masks(masks[:1])
    ref = [np.median(rgb[0][masks[0, j]], axis=0) if masks[0, j].sum() else None for j in range(masks.shape[1])]
    return fuse_centres(ep, c, v, c0[0], ref, robot, rgb, states, actions, **kw)


def load_dets(path):
    """out/sam3det/<ep>.npz -> {frame: centres [M,2]}"""
    z = np.load(path, allow_pickle=True)
    return {int(t): np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2] for t, b in zip(z["frames"], z["boxes"])}


def patch_colour(frame, p):
    x, y = int(np.clip(round(p[0]), 3, 252)), int(np.clip(round(p[1]), 3, 252))
    return np.median(np.asarray(frame[y - 3:y + 4, x - 3:x + 4]).reshape(-1, 3), axis=0)


def fuse_centres(ep, c_in, v_in, start, ref, robot, rgb, states, actions, grasp_px=GRASP_PX, width_check=True, hide=False, dets=None, closed=None, settle=SETTLE):
    """c_in/v_in: the mask tracker's centroids [N,K,2] and visibility [N,K]; start: positions at frame 0;
    ref: reference colour per object (None -> taken from frame 0 at start)."""
    n, k = v_in.shape
    e = eef_pixels(ep, states)
    if closed is None:
        closed = np.r_[False, actions[:-1, 6] > 0]
    p = np.array(start, float).copy()
    ref = [r if r is not None else (patch_colour(rgb[0], p[j]) if np.all(np.isfinite(p[j])) else None) for j, r in enumerate(ref)]
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    mode = ["free"] * k
    unseen, seen = [0] * k, [SEEN] * k
    held, off, pending = None, None, None
    events = []
    for t in range(n):
        if t and closed[t] and not closed[t - 1] and held is None:
            cand = [(np.hypot(*(p[j] - e[t])), j) for j in range(k) if np.all(np.isfinite(p[j])) and (mode[j] == "free" or not hide)]
            pending = (min(cand)[1], p[min(cand)[1]] - e[t], t) if cand and min(cand)[0] < grasp_px else None
        if pending is not None and not closed[t]:
            pending = None
        if pending is not None and t - pending[2] == settle:
            if not width_check or states[t, 6] >= MIN_WIDTH:
                held, off = pending[0], pending[1]
                mode[held] = "held"
                events.append((t, "grasp", held))
            else:
                events.append((t, "not_a_grasp", pending[0]))
            pending = None
        if held is not None:
            p[held] = e[t] + off
            if not closed[t]:
                events.append((t, "release", held))
                mode[held], unseen[held], seen[held] = "free", 0, SEEN
                held = None
        for j in range(k):
            if not np.all(np.isfinite(p[j])):
                continue
            if j == held:
                centres[t, j], vis[t, j] = p[j], True
                continue
            x, y = int(np.clip(round(p[j][0]), 0, 255)), int(np.clip(round(p[j][1]), 0, 255))
            covered = bool(robot[t, y, x])
            evidence = False
            seen_now, c = bool(v_in[t, j]), c_in[t, j]
            if dets is not None:  # re-detection: the detection nearest to where the object is believed to be
                cand = dets.get(t)
                seen_now = cand is not None and len(cand) > 0
                if seen_now:
                    c = cand[np.argmin(np.hypot(*(cand - p[j]).T))]
            if seen_now:
                cx, cy = int(np.clip(round(c[0]), 0, 255)), int(np.clip(round(c[1]), 0, 255))
                if not robot[t, cy, cx] and np.hypot(*(c - p[j])) < NEAR_PX:
                    evidence = True
                    if np.hypot(*(c - e[t])) > FAR_PX:
                        p[j] = 0.8 * p[j] + 0.2 * c
                        ref[j] = patch_colour(rgb[t], p[j])
            if not evidence and ref[j] is not None and not covered:
                evidence = bool(np.abs(patch_colour(rgb[t], p[j]) - ref[j]).mean() < 30)
            if evidence:
                unseen[j], seen[j] = 0, seen[j] + 1
                if mode[j] == "hidden" and seen[j] >= SEEN:
                    mode[j] = "free"
            elif not covered:  # the arm in front of it is not evidence that it is gone
                unseen[j], seen[j] = unseen[j] + 1, 0
                if mode[j] == "free" and unseen[j] >= HIDE:
                    mode[j] = "hidden"
                    events.append((t, "hidden", j))
            centres[t, j], vis[t, j] = p[j], evidence
    return centres, vis, events


HELD_PX, STABLE_N, STABLE_PX = 45, 10, 3.0


HELD_EDGE_PX = 25  # gripper pixel to nearest mask pixel while the object is carried


AREA_ABS, AREA_RATIO, AREA_MIN = 1200, 3.0, 20  # mask-area sanity (objects are 65-375 px at 256 px)


CHOOSE_PX, CHOOSE_WAIT = 12, 60  # gripper travel (px) / frames before deciding which of several nearby objects is held


COLOUR_TOL = 35  # mean abs RGB difference allowed between a re-acquired mask and the object's first-frame mask


def observe_only(ep, masks, robot, rgb, states, actions, grasp_px=GRASP_PX):
    """Every drawn position is what the mask tracker SEES. Gripper kinematics never supplies a position; it only vetoes:
      not held : the object cannot move. A centroid within NEAR_PX of the last accepted position is accepted. A centroid
                 elsewhere is rejected (shown as not visible) unless it has been still for STABLE_N frames, outside the
                 projected gripper/wrist region and away from the gripper: then the object really is somewhere new
                 (dropped, pushed) and is re-acquired there.
      held     : (gripper closed near the object, fingers kept apart) the centroid must stay within HELD_PX of the
                 gripper pixel, otherwise it is rejected for that frame.
    Returns centres [N,K,2] (nan when rejected / unseen), visible [N,K], events."""
    n, k = masks.shape[:2]
    c_in, v_in = centres_from_masks(masks)
    e = eef_pixels(ep, states)
    closed = np.r_[False, actions[:-1, 6] > 0]
    area = masks.sum(axis=(2, 3))
    area0 = area[0]
    colour0 = [rgb[0][masks[0, j]].mean(axis=0) if AREA_MIN <= area0[j] <= AREA_ABS else None for j in range(k)]
    q = [c_in[0, j].copy() if v_in[0, j] else None for j in range(k)]
    centres, vis = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    held, pending, events = None, None, []
    recent = [[] for _ in range(k)]
    for t in range(n):
        if t and closed[t] and not closed[t - 1] and held is None:
            cand = sorted((np.hypot(*(q[j] - e[t])), j) for j in range(k) if q[j] is not None)
            near = [j for d, j in cand if d < grasp_px]
            # several objects stand within reach of the gripper (two boxes on one plate): do not guess by distance,
            # wait and see which one is SEEN moving with the gripper
            pending = (near, t, e[t].copy(), {j: q[j].copy() for j in near}) if near else None
        if pending is not None and not closed[t]:
            pending = None
        if pending is not None and t - pending[1] >= SETTLE:
            near, t0, e0, q0 = pending
            moved = e[t] - e0
            if states[t, 6] < MIN_WIDTH * 0.6:
                pending = None  # fingers fully closed: nothing between them
            elif len(near) == 1 and states[t, 6] >= MIN_WIDTH:
                held, pending = near[0], None  # one object in reach and the fingers are kept apart: unambiguous
                events.append((t, "grasp", held))
            elif np.hypot(*moved) > CHOOSE_PX:
                # held = the nearby object that is SEEN travelling with the gripper (thin objects leave the fingers
                # almost closed, so finger width alone cannot tell an object from a handle)
                score = {}
                for j in near:
                    if v_in[t, j]:
                        d = c_in[t, j] - q0[j]
                        if np.hypot(*d) > 0.4 * np.hypot(*moved):
                            score[j] = np.hypot(*(d - moved))
                best = min(score, key=score.get) if score else None
                if best is not None and score[best] < 0.6 * np.hypot(*moved) + 4:
                    held = best
                    events.append((t, "grasp", held))
                    pending = None
                elif np.hypot(*moved) > 3 * CHOOSE_PX or t - t0 > CHOOSE_WAIT:
                    pending = None  # the gripper left and nothing came along: it closed on a handle
            elif t - t0 > CHOOSE_WAIT and len(near) == 1 and states[t, 6] >= MIN_WIDTH:
                held, pending = near[0], None  # holding still for a long time with something between the fingers
                events.append((t, "grasp", held))
        if held is not None and (not closed[t] or states[t, 6] < MIN_WIDTH * 0.6):
            events.append((t, "release", held))  # opened, or the fingers closed on nothing: the object is no longer carried
            held = None
        for j in range(k):
            if not v_in[t, j]:
                recent[j] = []
                continue
            c = c_in[t, j]
            # area veto: a mask far larger than the object (it swallowed the table, a drawer front, the arm) is not the object
            if area[t, j] > AREA_ABS or (AREA_MIN <= area0[j] <= AREA_ABS and area[t, j] > AREA_RATIO * area0[j]):
                recent[j] = []
                continue
            cx, cy = int(np.clip(round(c[0]), 0, 255)), int(np.clip(round(c[1]), 0, 255))
            recent[j] = (recent[j] + [c])[-STABLE_N:]
            reacq = False
            if j == held:
                # held: the gripper touches the object's EDGE, not its centre (a bottle is grasped at the top),
                # so measure from the gripper pixel to the nearest mask pixel
                ys_, xs_ = np.nonzero(masks[t, j])
                ok = bool(len(xs_)) and float(np.min(np.hypot(xs_ - e[t][0], ys_ - e[t][1]))) < HELD_EDGE_PX
            elif q[j] is not None and np.hypot(*(c - q[j])) < NEAR_PX:
                ok = True
            else:
                still = len(recent[j]) == STABLE_N and np.ptp(np.asarray(recent[j]), axis=0).max() < STABLE_PX
                ok = bool(still and not robot[t, cy, cx] and np.hypot(*(c - e[t])) > FAR_PX)
                if ok and colour0[j] is not None:
                    # re-acquiring somewhere new is only believed if the thing there looks like the object (the arm is
                    # white, handles are black): mean colour of the mask against the first-frame mask
                    ok = float(np.abs(rgb[t][masks[t, j]].mean(axis=0) - colour0[j]).mean()) < COLOUR_TOL
                if ok:
                    reacq = True
                    events.append((t, "reacquired", j))
            if ok:
                # the anchor q may not be dragged along: while not held it only follows the object when the gripper is away
                if j == held or reacq or q[j] is None:
                    q[j] = c.copy()
                elif np.hypot(*(c - e[t])) > FAR_PX:
                    q[j] = 0.8 * q[j] + 0.2 * c
                centres[t, j], vis[t, j] = c, True
    return centres, vis, events