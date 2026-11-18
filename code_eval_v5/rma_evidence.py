"""Object-centric evidence for checklist items (pick / place / pour), from tracker centres + proprioception.

The checklist may tick item k only after its evidence has been seen. Signals, all available at test time:
  eef pixel   : end-effector position projected with the agentview camera matrix (rma_camera.py)
  closed      : gripper finger width below a threshold (state dims 6:8)
  held(k)     : closed, and object k's tracked centre within HOLD_PX of the eef pixel
  tilt        : rotation angle of the end effector relative to its orientation at the grasp
Item kinds come from the subtask text (rma_spec.subtask_object names the object):
  pick X  : held(X) for HOLD_FRAMES consecutive frames and X's centre moved MOVE_PX from where it started
  place X : after the pick, gripper open for RELEASE_FRAMES, X not held, and X's centre (or last seen centre,
            if it vanished into a container) is MOVE_PX from where it started
  pour X n: while X is held, tilt rises above POUR_DEG and falls back below RETURN_DEG; the n-th such event
  other   : open/close drawer or door -> no object evidence yet (kind "articulated"; always allowed for now)

EvidenceTracker.update(t, state, centres, visible) is called every frame; satisfied(k) says whether item k's
evidence has been observed (given items are worked in order). Offline check: rma_evidence.py DATASET CAMERA.json
"""
import copy
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HOLD_PX, HOLD_FRAMES, MOVE_PX, RELEASE_FRAMES = 18.0, 8, 8.0, 5
POUR_DEG, RETURN_DEG = 35.0, 15.0  # demo pours tilt 45-47 (sauce, milk->pan), 61 (milk->mug), 93 (wine) deg
CLOSED_WIDTH = 0.07  # finger width (m): ~0.062 when holding the benchmark objects, ~0.078-0.080 open


# Articulated parts (drawers, microwave door): arm-motion evidence for now. While the gripper holds the handle
# (closed segment, nothing picked) the end effector must travel along the part's direction (robot base frame,
# metres). Demo values: drawer open dy -17..-24 cm, close dy +15..+17 cm; door open (-37,-19) cm, close
# (+25..+30, +12..+20) cm. Drawer level from the eef height at the grasp: bottom ~0.53, middle ~0.61, top ~0.68 m.
ARTIC = {("open", "drawer"): (np.array([0.0, -1.0, 0.0]), 0.10), ("close", "drawer"): (np.array([0.0, 1.0, 0.0]), 0.09),
         ("open", "door"): (np.array([-0.89, -0.45, 0.0]), 0.20), ("close", "door"): (np.array([0.83, 0.55, 0.0]), 0.15)}
LEVELS = (("bottom", 0.0, 0.57), ("middle", 0.57, 0.645), ("top", 0.645, 9.0))
# Where the motion starts (eef, base frame) and the finger width while doing it, from the demos (spread ~2.5 cm
# for drawers, 4 cm door open, 10 cm door close). Closing in the air (width ~0.001) far from a handle and
# sweeping sideways no longer counts (seen at test time with the pilot policy).
HANDLES = {("open", "drawer"): {"top": (0.139, 0.455, 0.677), "middle": (0.140, 0.455, 0.606), "bottom": (0.140, 0.451, 0.529)},
           ("close", "drawer"): {"top": (0.137, 0.290, 0.678), "middle": (0.140, 0.292, 0.607), "bottom": (0.140, 0.295, 0.533)},
           ("open", "door"): {None: (0.064, 0.409, 0.582)}, ("close", "door"): {None: (-0.265, 0.266, 0.639)}}
RADIUS = {("open", "drawer"): 0.05, ("close", "drawer"): 0.05, ("open", "door"): 0.06, ("close", "door"): 0.15}
WIDTH = {("open", "drawer"): (0.006, 0.035), ("close", "drawer"): (0.0, 0.035),
         ("open", "door"): (0.015, 0.045), ("close", "door"): (0.0, 0.035)}


def artic_spec(text):
    words = text.lower().split()
    action = "open" if "open" in words else "close" if "close" in words else None
    part = "drawer" if "drawer" in words else "door" if ("door" in words or "microwave" in words) else None
    level = next((w for w in ("top", "middle", "bottom") if w in words), None)
    return (action, part, level) if action and part else None


def item_kind(text):
    words = text.lower()
    if re.match(r"^(top|middle|bottom) drawer: ", words):  # an observation item (causal labels): nothing to check
        return "observe", None
    if artic_spec(text) and not words.startswith(("place", "put", "pick")):
        return "articulated", artic_spec(text)
    if "pour" in words:
        m = re.search(r"(first|second|1st|2nd)", words)
        return "pour", (2 if m and m.group(1) in ("second", "2nd") else 1)
    if words.startswith("pick") or " pick " in words:
        return "pick", None
    if words.startswith("place") or words.startswith("put"):
        return "place", None
    return "articulated", None


class EvidenceTracker:
    """Sequential: only the first item still lacking evidence is checked, so one object's tracking glitch while
    another is carried cannot confirm a later item. satisfied(i) == evidence seen for items 0..i."""

    def __init__(self, items, object_of, matrix):
        self.object_of = object_of
        self.matrix = np.asarray(matrix)
        self.set_items(items)
        self.pointer = 0         # items[:pointer] have evidence
        self.start = {}          # object -> first visible centre
        self.last_seen = {}
        self.held_run = 0        # consecutive frames the pending item's object is held
        self.open_run = 0
        self.grasp_rot = None
        self.tilted = False
        self.pours = 0           # completed tilt events while the current object is held
        self.carrying = None     # object picked (with evidence) and not yet placed
        self.prev_closed = False
        self.seg_start = None    # eef position at the last closing of the gripper
        self.ref_proj = None     # min projection on the pending articulated direction
        self.ref_pos = None      # eef position at that point (must be at the handle)
        self.events = []         # (t, item index)

    def set_items(self, items):
        """Item texts, None = not yet known ("?", causal labels); called again whenever an item is revealed."""
        self.items = list(items)
        self.kinds = [("unknown", None) if t is None else item_kind(t) for t in self.items]
        self.objects = [None if t is None else self.object_of(t) for t in self.items]
        # which drawer is used after the checking round ("again", after a place) is the model's own call
        self.check_level = []
        placed = False
        for text, (kind, _) in zip(self.items, self.kinds):
            self.check_level.append(text is not None and not placed and "again" not in text.lower())
            placed = placed or kind == "place"

    def eef_pixel(self, state):
        p = self.matrix @ np.append(state[:3], 1.0)
        return p[:2] / p[2]

    def update(self, t, state, centres, visible, closed=None):
        """closed: the executed gripper command (True = close); wide objects (milk) keep the fingers apart, so the
        finger width is only a fallback."""
        state = np.asarray(state, float)
        if closed is None:
            closed = abs(state[6] - state[7]) < CLOSED_WIDTH
        eef = self.eef_pixel(state)
        for k, (c, v) in enumerate(zip(centres, visible)):
            if v:
                self.start.setdefault(k, np.asarray(c, float))
                self.last_seen[k] = np.asarray(c, float)
        self.open_run = 0 if closed else self.open_run + 1
        if closed and not self.prev_closed:
            self.seg_start = state[:3].copy()
        self.prev_closed = closed
        # tilt events while carrying (orientation reference = at the grasp)
        if closed and self.grasp_rot is None:
            self.grasp_rot = Rotation.from_rotvec(state[3:6])
        if not closed:
            self.grasp_rot, self.tilted = None, False
        elif self.carrying is not None:
            angle = np.degrees((Rotation.from_rotvec(state[3:6]) * self.grasp_rot.inv()).magnitude())
            if not self.tilted and angle > POUR_DEG:
                self.tilted = True
            elif self.tilted and angle < RETURN_DEG:
                self.tilted = False
                self.pours += 1
        if self.pointer >= len(self.items):
            return
        kind, extra = self.kinds[self.pointer]
        k = self.objects[self.pointer]
        ok = False
        if kind == "unknown":  # nothing to check until the model names the item
            return
        if kind == "observe":  # what the camera showed; the model's own reading, not gated
            self.pointer += 1
            return

        def grasped():  # held for HOLD_FRAMES and moved away from where it started
            near = k in self.start and visible[k] and np.linalg.norm(np.asarray(centres[k], float) - eef) < HOLD_PX
            self.held_run = self.held_run + 1 if (closed and near) else 0
            moved = k in self.start and np.linalg.norm(self.last_seen[k] - self.start[k]) >= MOVE_PX
            if self.held_run >= HOLD_FRAMES and moved:
                self.carrying, self.pours = k, 0
                return True
            return False

        if kind == "articulated":
            action, part, level = extra
            direction, distance = ARTIC[(action, part)]
            width = abs(state[6] - state[7])
            lo_w, hi_w = WIDTH[(action, part)]
            if closed and self.carrying is None and lo_w <= width <= hi_w:
                # travel measured from the farthest point the other way while this item is pending; that point
                # must be at the part's handle (the level named in the item, or any level after the checking round)
                proj = float(np.dot(state[:3], direction))
                if self.ref_proj is None or proj < self.ref_proj:
                    self.ref_proj, self.ref_pos = proj, state[:3].copy()
                anchors = HANDLES[(action, part)]
                if part == "drawer" and level in anchors and self.check_level[self.pointer]:
                    anchors = {level: anchors[level]}
                near = min(np.linalg.norm(self.ref_pos - np.asarray(a)) for a in anchors.values()) <= RADIUS[(action, part)]
                ok = near and proj - self.ref_proj >= distance
            else:
                self.ref_proj = None
        elif k is None:
            ok = True  # item names no tracked object and no articulated part
        elif kind == "pick":
            ok = grasped()
        elif kind == "place":
            if self.carrying != k:
                grasped()  # the list may have no separate pick item: the grasp must still be seen first
            else:
                ok = self.open_run >= RELEASE_FRAMES
                if ok:
                    self.carrying = None
        elif kind == "pour":
            if self.carrying != k:
                grasped()
            else:
                ok = self.pours >= extra
        if ok:
            self.events.append((t, self.pointer))
            self.pointer += 1
            self.held_run = 0
            self.ref_proj = None

    def satisfied(self, index):
        return index < self.pointer


# Tasks gated: offline misses <~10% and no adjacent look-alike objects (25/26 swap at test time).
GATE_TASKS = (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24)
# Pours are not gated: the policy's pours at test time are less clean than the demos and the tilt count both
# missed and double-counted real pours (pilot, task 10); counting stays with the model.
GATED_KINDS = ("pick", "place", "articulated")
VALVE_CALLS = 15  # a tick vetoed this many consecutive calls (~300 frames) is let through: tracking may be lost


class EvidenceGate:
    """Applies the model's generated memory with object evidence as a veto.

    The model stays in charge of what it claims; a tick of item k is accepted only once EvidenceTracker has seen
    item k's evidence (or after VALVE_CALLS consecutive vetoes), and Task:Complete only when every item is ticked.
    Unlike the strict Ledger Driver, a tick ahead of the accepted progress or a repeated tick is not an error:
    the gate keeps its own ledger and re-renders the previous memory the model will read next round.
    """

    def __init__(self, task_id, object_of, matrix, memory_format, ledger_v2, variants=("A", "C")):
        self.task_id, self.object_of, self.matrix = task_id, object_of, matrix
        self.mf, self.lv, self.variants = memory_format, ledger_v2, variants
        self.ledger = ledger_v2.Ledger()
        self.tracker = None
        self.veto_run = 0
        self.frames = 0

    def observe(self, state, centres):
        self.frames += 1
        if self.tracker is not None:
            visible = [c is not None for c in centres]
            pts = [(0.0, 0.0) if c is None else c for c in centres]
            self.tracker.update(self.frames, state, pts, visible)

    def apply(self, generated):
        """-> (next_memory_text, sequence_complete, info) or None when the text is malformed (caller decides)."""
        if not isinstance(generated, str):
            return None
        lines = generated.split("\n")
        if len(lines) != 3 or not lines[0].startswith("Action:") or not lines[1].startswith("Subgoal:") \
                or not lines[2].startswith("Task:"):
            return None
        action = lines[0][len("Action:"):]
        if all(self.mf.action_entities(action, v) is None for v in self.variants):
            return None
        try:
            delta = self.lv.parse_delta("\n".join(lines[1:]))
        except self.lv.InvalidDelta:
            delta = self.lv.Delta("none", "Incomplete")
        info = {"claim": delta.kind}
        if delta.kind == "init" and not self.ledger.initialized:
            self.ledger = self.lv.Ledger(initialized=True, open_ended=delta.open_ended,
                                         items=[self.lv.SubgoalItem(i + 1, t) for i, t in enumerate(delta.items)])
            self.tracker = EvidenceTracker(delta.items, self.object_of, self.matrix)
        elif not self.ledger.initialized:
            return None
        if delta.kind != "init" and delta.reveals:
            # causal labels: the model names a "?" item once it has decided it (a bad reveal is ignored)
            try:
                trial = copy.deepcopy(self.ledger)
                self.lv.apply_reveals(trial, delta.reveals)
                self.ledger = trial
                self.tracker.set_items([it.text for it in self.ledger.items])
                info["revealed"] = sorted(delta.reveals)
            except self.lv.InvalidDelta:
                info["bad_reveal"] = True
        if delta.kind in ("done", "at"):
            claimed = max(delta.indices)
            info["claimed"] = claimed
            if delta.kind == "at":  # v2 absolute progress: lowering it is a correction and needs no evidence
                for item in self.ledger.items:
                    if item.index > claimed and item.done:
                        item.done = False
                        info["untick"] = info.get("untick", 0) + 1
            vetoed = False
            for item in self.ledger.items:
                if item.done or item.index > claimed:
                    continue
                if item.text is None:  # an unknown item cannot be ticked; neither can anything after it
                    info["vetoed_unknown"] = item.index
                    break
                kind = self.tracker.kinds[item.index - 1][0]
                if kind not in GATED_KINDS:  # accept the claim and keep the evidence pointer in step
                    item.done = True
                    self.tracker.pointer = max(self.tracker.pointer, item.index)
                    continue
                if self.tracker.satisfied(item.index - 1):
                    item.done = True
                    continue
                self.veto_run += 1
                if self.veto_run >= VALVE_CALLS:  # let it through and move the evidence pointer past it
                    item.done = True
                    self.tracker.pointer = max(self.tracker.pointer, item.index)
                    self.veto_run = 0
                    info["valve"] = item.index
                    continue
                vetoed = True
                info["vetoed"] = item.index
                break
            if not vetoed:
                self.veto_run = 0
        done = sum(it.done for it in self.ledger.items)
        complete = delta.task == "Complete" and done == len(self.ledger.items)
        if complete:
            self.ledger.open_ended = False
        self.ledger.task_status = "Complete" if complete else "Incomplete"
        info.update(accepted=done, evidence=self.tracker.pointer if self.tracker else 0)
        return self.mf.render_previous(action, self.ledger), complete, info


def offline(dataset, camera):
    """For every built episode: when does each item's evidence first hold, versus the true end of the item?"""
    import glob
    import json
    from collections import defaultdict
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rma_spec
    cams = json.loads(Path(camera).read_text())
    stats = defaultdict(list)
    per_task = int(__import__("os").environ.get("PER_TASK", 8))
    chosen, count = [], defaultdict(int)
    for path in sorted(glob.glob(f"{dataset}/episodes/*.npz")):
        task = int(Path(path).stem[1:3])
        if count[task] < per_task:
            chosen.append(path); count[task] += 1
    for path in chosen:
        ep = Path(path).stem
        task = int(ep[1:3])
        z = np.load(path)
        log = json.loads(Path(f"{dataset}/log/{ep}.json").read_text())
        items, ends = log["items"], list(z["ends"])
        ev = EvidenceTracker(items, lambda s: rma_spec.subtask_object(task, s), cams[str(task)]["matrix"])
        first = [None] * len(items)
        for t in range(len(z["states"])):
            cmd = None if __import__("os").environ.get("USE_WIDTH") else bool(z["actions"][t, 6] > 0)
            ev.update(t, z["states"][t], z["centres"][t], z["visibility"][t], closed=cmd)
            for i in range(len(items)):
                if first[i] is None and ev.satisfied(i):
                    first[i] = t
        for i, text in enumerate(items):
            kind, extra = ev.kinds[i]
            if kind == "articulated":
                kind = f"{extra[0]}-{extra[1]}"
            stats[(task, kind)].append(None if first[i] is None else first[i] - ends[i])
    print("task kind      n   miss  early(<-40)  median_offset  p90|offset|")
    for (task, kind), offs in sorted(stats.items()):
        got = [o for o in offs if o is not None]
        miss = len(offs) - len(got)
        early = sum(o < -40 for o in got)
        med = int(np.median(got)) if got else None
        p90 = int(np.percentile(np.abs(got), 90)) if got else None
        print(f"{task:4d} {kind:6s} {len(offs):5d} {miss:5d} {early:8d} {str(med):>12s} {str(p90):>10s}")


if __name__ == "__main__":
    offline(sys.argv[1], sys.argv[2])
