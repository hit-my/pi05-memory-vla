"""Per-task rule table from constants_v5.json -> v5/RULES_v5.md"""
import json
from pathlib import Path

TV = Path("xxxx/rma/trackv")
C = json.load(open(TV / "v5/constants_v5.json"))
D = C["common"]
TRIG = {"place_end": "checklist ticks 'place <obj> into the <container>'", "place_open": "gripper command opens while the current checklist item is 'place <obj> into the <container>'",
        "close_end": "checklist ticks the first 'close the <container>' after the placement",
        "close_move20": "inside the first 'close the <container>' item after the placement: gripper closed and then moved >= 20 px in the image"}
rows = ["| task | objects | first frame | rule set | active rules | task-specific parameters | retirement trigger | err/100 on the selection half (raw) |", "|" + "---|" * 8]
for t in range(1, 27):
    e = C["tasks"][str(t)]
    cfg = dict(D); cfg.update(e["cfg"])
    if t in (18, 19):
        ff = f"appearance: frame {C['first_frame_appearance'][str(t)]['t0']}, zoomed SAM3 text detections + appearance classifier + joint choice; late acquisition every 10 frames; identity check at the first grasp" \
            if str(t) in C.get("first_frame_appearance", {}) else "appearance (v5)"
    else:
        ff = "B2: box-like SAM3 detection nearest to the fixed click" if t in (25, 26) else "B: SAM3 generic detection nearest to the fixed click"
    if cfg["mode"] == "raw":
        rs = "raw"
        act = ["SAM3 centroid drawn as is" + (f" (masks down to {cfg['min_area']} px)" if cfg["min_area"] != 20 else "")] + ([f"colour gate (not drawn while the mask colour differs from its first-frame colour by > {cfg['raw_colour_tol']:.0f})"] if cfg["raw_colour_tol"] > 0 else [])
        par = {k: v for k, v in e["cfg"].items() if k != "mode"}
    else:
        rs = "v5"
        act = ["rest lock", "grasp / carried / release"]
        act += ["push (object in a moving drawer)"] if cfg["push"] else []
        act += ["dup (later-released object keeps a shared mask)"] if cfg["dup_frac"] > 0 else []
        act += ["hold_rest=pre (not yet carried: anchor drawn while the mask is away)"] if cfg["hold_rest"] == "pre" else []
        act += ["hold_rel (after release without a usable mask: the gripper pixel of the release is drawn)"] if cfg["hold_rel"] else []
        act += ["hold_dup (a resting object whose mask was taken by the object put in front of it is drawn at its anchor)"] if cfg["hold_dup"] else []
        act += [f"masks down to {cfg['min_area']} px"] if cfg["min_area"] != 20 else []
        act += [f"area rule only above {cfg['area_floor']} px"] if cfg["area_floor"] != D["area_floor"] else []
        par = e["cfg"]
    ret = "; ".join(f"{k}: {TRIG.get(v, v)}" for k, v in e["retire"].items() if (k == "basket" and t in (1, 2, 3)) or (k == "drawer" and t in (4, 5, 11, 12, 13, 14, 17))
                    or (k == "microwave" and t in (20, 21, 22, 23, 24))) or "none"
    if t == 22 and "close_end" in ret:
        ret += " (= the end of the episode in the demos: never fires there)"
    sel = f"{e.get('selection_half_err100', float('nan')):.2f} ({e.get('selection_half_raw_err100', float('nan')):.2f})" if "selection_half_err100" in e else ""
    rows.append(f"| {t} | {', '.join(e['objects'])} | {ff} | {rs} | {'; '.join(act)} | {json.dumps(par) if par else '-'} | {ret} | {sel} |")
common = f"""# v5 rule sets

## Common rules (tv_fuse_v5.py; the same thresholds for every task unless the table says otherwise)
- **raw**: the centroid of the SAM3 mask is drawn whenever the mask has at least 20 px. Nothing else (plus retirement where the table names a trigger).
- **v5** ("an object only moves when the gripper moves it"):
  - *anchor / rest lock*: every object has an anchor (its resting place; frame 0: the mask centroid, following the mask for the first {D['start_settle']} frames).
    A resting object's centroid is drawn only while it is within {D['near_px']:.0f} px of the anchor; a mask that jumped elsewhere is not drawn, and is drawn again as soon as it is back.
    The centroid of the visible part is drawn as it is (no freezing while partly hidden).
  - *grasp*: when the gripper command closes, objects whose mask edge is within {D['grab_edge_px']:.0f} px of the gripper pixel (or whose anchor is within {D['grab_anchor_px']:.0f} px while the mask is away)
    during the next {D['grab_win']} frames are candidates; a candidate is carried once its mask has left the anchor together with the gripper for {D['conf_n']} frames
    (or, single candidate with hidden mask, when the fingers stay apart after {D['settle']} frames).
  - *carried*: drawn while the mask is at the gripper (centroid within {D['held_px']:.0f} px of the gripper pixel or edge within {D['held_edge_px']:.0f} px); hidden in the hand = nothing drawn.
  - *release*: gripper command opens (or fingers fully closed). The mask is drawn at once while within {D['rel_px']:.0f} px of the release place, unless it leaves together with the gripper.
    The anchor is locked when the mask has been still ({D['stable_n']} frames within {D['stable_px']:.0f} px) with the gripper at least {D['lock_far_px']:.0f} px away.
  - *area*: a mask larger than {D['area_abs']} px or {D['area_ratio']:.0f}x the running median area is not the object.
  - switches: *push* (anchor follows a mask that moves together with the closed gripper), *dup* (two masks on one thing: the carried object, else the one released later, keeps it),
    *hold_rest=pre* (an object not carried yet is drawn at its anchor while its mask is away), *hold_rel* (after a release without usable mask the gripper pixel of the release is drawn),
    *hold_dup* (a resting object that loses its mask by dup is drawn at its anchor), *min_area* (smallest mask the rules accept, default 20 px).
- **retirement**: an object flagged by the checklist signal is never drawn again.
- Gripper pixel: projection of the grip site with the corrected camera (tv_common_v4); "closed" = the gripper command issued before the frame.

## Per task
"""
open(TV / "v5/RULES_v5.md", "w").write(common + "\n".join(rows) + "\n")
print("\n".join(rows))
