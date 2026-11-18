"""Rule-based check of a ledger tick, from the same perception that draws the trajectory overlay.

A tick for a bowl is accepted only if, since the previous accepted tick of that bowl:
  - the bowl really moved (rules out an empty hand), and
  - it now sits near a plate that fits the subgoal's own words (rules out a drop): "-> empty plate"
    must be a different plate from where it started, "-> same plate" must be the same one.
No goal oracle and no plate numbers. Off-centre placements on the right plate pass, as agreed.
"""
import numpy as np

MOVE_PX = 15.0
PLATE_PX = 30.0


def nearest_plate(plates, point):
    distances = np.linalg.norm(np.asarray(plates) - point, axis=1)
    return int(np.argmin(distances)), float(distances.min())


def verify(track, plates, anchor_frame, frame, relation=None, move_px=MOVE_PX, plate_px=PLATE_PX):
    """track: {frame: centre}, visible frames only. plates: array[P,2]. relation: "empty", "same" or None.

    Returns (accepted, moved, distance). An unseen bowl gives no evidence, so the tick is rejected.
    """
    seen = sorted(f for f in track if anchor_frame <= f <= frame)
    if not seen:
        return False, 0.0, None
    earlier = [f for f in track if f <= anchor_frame]
    start = track[max(earlier)] if earlier else track[seen[0]]
    moved = max(float(np.linalg.norm(track[f] - start)) for f in seen)
    start_plate, _ = nearest_plate(plates, start)
    end_plate, distance = nearest_plate(plates, track[seen[-1]])
    fits = relation is None or (end_plate != start_plate if relation == "empty" else end_plate == start_plate)
    return moved >= move_px and distance <= plate_px and fits, moved, distance
