"""Write v5/constants_v5.json (the one file with every per-task constant of v5) from the selection results.
mkconst5.py select_json [select_json ...]   (later files override earlier ones per task)"""
import json
import sys
from pathlib import Path

import tv_fuse_v5

TV = Path("xxxx/rma/trackv")
sys.path.insert(0, str(TV / "code"))
import rma_spec  # noqa: E402

tasks = {}
for f in sys.argv[1:]:
    for t, r in json.load(open(f)).items():
        tasks[str(int(t))] = dict(chosen=r["chosen"], cfg=r["cfg"], retire=r["retire"], selection_half_err100=round(r["A_all"][r["chosen"]]["err100"], 3),
                                  selection_half_raw_err100=round(r["A_all"]["raw|none"]["err100"], 3))
for t in range(1, 27):
    e = tasks.setdefault(str(t), dict(chosen="raw|none", cfg={"mode": "raw"}, retire={}))
    e["objects"] = [o[0] for o in rma_spec.TASKS[t]["objects"]]
    e["first_frame"] = "appearance (frame-0 selection v5: zoomed SAM3 detections + appearance classifier, see first_frame_appearance)" if t in (18, 19) else \
        ("B2 (box-like detection nearest to the fixed click, v4/initB2.py)" if t in (25, 26) else "B (SAM3 generic detection nearest to the fixed click, initB)")
    e["clicks"] = [list(c) for c in rma_spec.CLICKS[t]]
ff = {}
for t in (18, 19):
    p = TV / f"v5/ff_clf_{t}_d.json"
    if p.exists():
        import det5_const
        j = json.load(open(p))
        ff[str(t)] = dict(
            reference_implementation="xxxx/rma/trackv/v5/track5.py (detector det5.py, features/selection ff_lib.py)",
            t0=det5_const.T0[t], second_look_frame=det5_const.RECHECK.get(t), crop_xyxy=list(det5_const.CROP[t]), crop_zoom_to=1008,
            prompts=det5_const.PROMPTS, detector_threshold=det5_const.THR, area_range=[20, 700], nms_iou=0.7,
            selection=dict(p_min=0.5, pair_distance_max_px=j["dmax"], pair_penalty_per_px=0.25, usual_area_px=j["atyp"], area_exponent=j["beta"], no_overlap=0.3,
                           note="ff_lib.select_joint: maximise sum log P + area_exponent*log(min(1, area/usual_area)) - pair_penalty*max(0, distance - pair_distance_max); image positions are not used"),
            late=dict(every=10, p_min=0.7, min_area_ratio=0.4, not_within_px_of_gripper=15, no_overlap=0.3, lost_after_frames=10,
                      lost_within_px_of_last_seen=15, first_within_px_of_another_objects_selection=j["dmax"] + 8,
                      note="objects without a track are looked for next to where another object was selected; objects whose mask had < 20 px for "
                           "lost_after_frames frames are looked for where the mask was last seen"),
            grasp_check=dict(wait_frames=15, gripper_moved_px=18, mask_farther_than_px_from_gripper=12, detection_within_px_of_gripper=6, p_min=0.02, p_min_if_mask_empty=0.5,
                             crop="96 px square centred on the gripper pixel",
                             note="uses the checklist item 'pick up the <object>': when the closed gripper has moved away and the object's mask is not at the gripper "
                                  "(or the object has no track), it is re-started from the detection at the gripper; a track of another object that sat on it is dropped"),
            classes=j["names"] + ["other"],
            features="ff_lib.feat (24-d): mean RGB(3), std RGB(3), hue histogram weighted by saturation(8), saturation hist(3), value hist(3), area/200, w/20, h/20, fill",
            classifier=j["clf"], trained_on="selection half; zoomed detections of frames 4, 6, 10 (task 18) / 10 (task 19) labelled with the GT boxes of the same frame")
out = dict(version="v5", note="per-task rule sets chosen on the selection half (even positions of the sorted episode list) against simulator GT; "
                              "cfg overrides tv_fuse_v5.CONFIG5; retire = checklist triggers per container (tv_fuse_v5.retirement_plan_v5)",
           common=tv_fuse_v5.CONFIG5, tasks=tasks, first_frame_appearance=ff)
json.dump(out, open(TV / "v5/constants_v5.json", "w"), indent=1)
print("wrote constants_v5.json;", {t: v["chosen"] for t, v in sorted(tasks.items(), key=lambda x: int(x[0]))})
