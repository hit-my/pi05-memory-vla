"""Equivalence of the evaluation perception v5 with the training data (all/final_v5 + rma_build_v2r overlays).

  test_perception_v5.py OUT SHARD NSHARDS PER_TASK MODE [ep ...]       (simulator venv; env as in rma_eval_v5.sh)
MODE live   : PerceptionV5 with the real tracker subprocess (rma_tracker_v5.py, SAM3 on the GPU)
     stored : the tracker replaced by a stub returning the STORED masks (feat5.src: v5/track_split > v5/track >
              all/final_fix > all/final) -> tests signal + rules + drawing alone (must be exact)
Episodes: PER_TASK per task chosen exactly as test_veto_v5.py does (rng 5000 + task over v5/feat5), or the given ones.
Each frame is fed as the evaluation loop does: front frame, state as stored, previous gripper command actions[t-1, 6]
(None at t = 0), and as checklist the demo's current subtask text (LabelSignal: the label-derived checklist sequence).
Compared per frame:
  masks     vs the stored masks the final tracks were computed from (live mode)
  centres   vs final_v5 centres (nan-equal; max |diff| px), visibility, retired
  overlay   renderer PNG bytes vs the training overlay = overlay_v2r.OverlayState on final_v5 rendered by the same
            CanonicalRenderer (the code path of rma_build_v2r.py); where the running build already finished the episode
            (xxxx/rma/dataset_v2r/dataset/rows_base/<ep>.jsonl) also vs its stored PNG
Writes OUT/<ep>.json and prints one line per episode.
"""
import sys

sys.dont_write_bytecode = True

import json
import os
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import overlay_v2r  # noqa: E402
import rma_perception_v5 as pv5  # noqa: E402
import rma_spec  # noqa: E402
from rma_perception import make_tracks  # noqa: E402

TV = Path("xxxx/rma/trackv")
LR = Path("xxxx/rma/lr_eps")
BUILD = Path(os.environ.get("BUILD_DATASET", "xxxx/rma/dataset_v2r/dataset"))
out_dir, shard, shards, per_task, mode = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
given = sys.argv[6:]
out_dir.mkdir(parents=True, exist_ok=True)


def mask_source(ep):
    for d in ("v5/track_split", "v5/track", "all/final_fix", "all/final"):
        p = TV / d / f"{ep}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(ep)


def stored_masks(ep):
    return np.unpackbits(np.load(mask_source(ep))["masks"], axis=-1)[..., :256].astype(bool)


class StubTracker:
    def __init__(self):
        self.m = None
        self.t = 0

    def reset(self, rgb, task_id, state):
        self.t = 0
        return {"k": self.m.shape[1], "stub": True}, self.m[0]

    def step(self, rgb, state, closed, pick):
        self.t += 1
        return {}, self.m[self.t]

    def close(self):
        pass


if given:
    eps = given
else:
    by = {}
    for p in sorted((TV / "v5/feat5").glob("t*_seed*.npz")):
        if "tmp" not in p.name:
            by.setdefault(p.stem[:3], []).append(p.stem)
    eps = []
    for task, names in sorted(by.items()):
        rng = np.random.default_rng(5000 + int(task[1:]))
        eps += sorted(rng.choice(names, min(per_task, len(names)), replace=False).tolist())
eps = [e for i, e in enumerate(eps) if i % shards == shard]
print("episodes", len(eps), eps, flush=True)

tracker = StubTracker() if mode == "stored" else pv5.TrackerProcessV5()
perception = None
same = lambda a, b: bool(np.array_equal(a, b, equal_nan=True))
for ep in eps:
    task = int(ep[1:3])
    rgb = np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")
    m = np.load(LR / "meta" / f"{ep}.npz")
    states, actions = m["states"], m["actions"]
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    sub = m["subtask"]
    z = np.load(TV / "all/final_v5" / f"{ep}.npz")
    ref_c, ref_v, ref_r = z["centres"], z["visibility"], z["retired"]
    n, k = ref_v.shape
    masks_ref = stored_masks(ep)
    assert masks_ref.shape[:2] == (n, k) and len(rgb) >= n, (ep, masks_ref.shape, len(rgb), n)
    if mode == "stored":
        tracker.m = masks_ref
    if perception is None:
        perception = pv5.PerceptionV5(task, tracker=tracker, label_mode=True)
    perception.task_id, perception.spec, perception.registry = task, rma_spec.TASKS[task], rma_spec.registry(task)
    # training path: overlay_v2r on the stored final_v5 track, same renderer
    tc, drawn, sticky, _ = overlay_v2r.load_tracks(TV / "all/final_v5" / f"{ep}.npz", n, k)
    train_state = overlay_v2r.OverlayState(rma_spec.registry(task), make_tracks)
    built = (BUILD / "rows_base" / f"{ep}.jsonl").exists()
    c_on, v_on, r_on = np.full((n, k, 2), np.nan, np.float32), np.zeros((n, k), bool), np.zeros((n, k), bool)
    mask_diff = np.zeros((n, k), np.int64)
    png_eq = png_eq_build = 0
    png_diff_frames, info_events = [], []
    timing = {"tracker": 0.0, "veto": 0.0, "render": 0.0, "signal": 0.0}
    start = time.time()
    for t in range(n):
        frame = np.ascontiguousarray(rgb[t])
        cmd = None if t == 0 else float(actions[t - 1, 6])
        item = text[int(sub[t])]
        if t == 0:
            perception.reset(frame, states[t], checklist=item)
        else:
            perception.step(frame, states[t], last_gripper_cmd=cmd, checklist=item)
        for key in timing:
            timing[key] += perception.timing.get(key, 0.0)
        if perception.tracker_info and perception.tracker_info.get("detections"):
            info_events.append([t, perception.tracker_info["detections"]])
        c_on[t] = [c if c is not None else [np.nan, np.nan] for c in perception.centres]
        v_on[t] = perception.visible
        r_on[t] = perception.retired
        mask_diff[t] = (perception.masks != masks_ref[t]).reshape(k, -1).sum(-1)
        tracks, fr, st = train_state.step(tc[t], sticky[t])
        png_ref, _ = perception.renderer.render(frame, fr, st, rma_spec.registry(task), tracks)
        if perception.png == png_ref:
            png_eq += 1
        else:
            png_diff_frames.append(t)
        if built:
            png_eq_build += (BUILD / "images" / ep / f"{t:06d}.png").read_bytes() == perception.png
    secs = time.time() - start
    ref_cf = np.where(ref_v[..., None], ref_c, np.nan).astype(np.float32)
    both = np.isfinite(c_on).all(-1) & np.isfinite(ref_cf).all(-1)
    dc = np.abs(c_on - ref_cf).max(-1)
    frames_c = (~(np.isclose(c_on, ref_cf, rtol=0, atol=0, equal_nan=True).all(-1))).any(-1)
    res = dict(ep=ep, task=task, mode=mode, frames=n, k=k, cfg_mode=perception.veto.cfg["mode"],
               masks_frames_differ=int((mask_diff.sum(1) > 0).sum()), masks_first_diff=int(np.argmax(mask_diff.sum(1) > 0)) if mask_diff.any() else None,
               masks_pixels_differ=int(mask_diff.sum()),
               centres_equal=same(c_on, ref_cf), centres_frames_differ=int(frames_c.sum()),
               centres_max_px=round(float(dc[both].max()) if both.any() else 0.0, 4),
               visibility_equal=same(v_on, ref_v), visibility_frames_differ=int((v_on != ref_v).any(-1).sum()),
               retired_equal=same(r_on, ref_r),
               png_equal=png_eq, png_frames_differ=len(png_diff_frames), png_first_diff=png_diff_frames[:5],
               build_checked=built, png_equal_to_build=png_eq_build if built else None,
               seconds=round(secs, 1), ms_per_frame={a: round(1000 * b / n, 2) for a, b in timing.items()},
               detections=info_events[:40], tracker_report=perception.report, signal_log=perception.signal_log[:10])
    (out_dir / f"{ep}.json").write_text(json.dumps(res))
    np.savez_compressed(out_dir / f"{ep}.npz", centres=c_on, visibility=v_on, retired=r_on, mask_diff=mask_diff)
    ok = res["centres_equal"] and res["visibility_equal"] and res["retired_equal"] and png_eq == n
    print("OK  " if ok else "DIFF", json.dumps({a: b for a, b in res.items() if a not in ("detections", "tracker_report", "signal_log")}), flush=True)
perception.close()
print("TEST_V5_DONE", shard, flush=True)
