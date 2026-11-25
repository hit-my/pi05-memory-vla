"""Is a mask difference of the live tracker a logic difference or GPU nondeterminism?   (tracker venv, one GPU)
  test_tracker_v5.py OUT ep [ep ...]
Per episode, in ONE process:
  online   rma_tracker_v5.TrackerV5 fed frame by frame (state, closed = actions[t-1,6] > 0, pick from the subtask labels)
  offline  the offline reference code re-run here: tasks 18/19 track5b.track_episode (CHECKS=split) on the same SAM3
           models; other tasks the tv_full_b.py / tv_full_fix.py loop (Sam3Stream from the STORED first-frame labels
           initB/<ep>.npy, or v4/initB2/<ep>.npy for 25/26)
  stored   the masks the final tracks were made from (v5/track_split > v5/track > all/final_fix > all/final)
and the first-frame labels of the online method B / B2 vs the stored ones (tasks != 18/19).
online == offline (same process) but != stored -> cross-process nondeterminism; online != offline -> logic."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import rma_tracker_v5 as tv5  # noqa: E402
import track5  # noqa: E402
import track5b  # noqa: E402
from tv_common import load  # noqa: E402

TV = Path("xxxx/rma/trackv")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)


def stored(ep):
    for d in ("v5/track_split", "v5/track", "all/final_fix", "all/final"):
        p = TV / d / f"{ep}.npz"
        if p.exists():
            return np.unpackbits(np.load(p)["masks"], axis=-1)[..., :256].astype(bool), d
    raise FileNotFoundError(ep)


def cmp(a, b):
    d = (a != b).reshape(len(a), a.shape[1], -1).sum(-1)
    f = d.sum(1) > 0
    return dict(frames_differ=int(f.sum()), first=int(np.argmax(f)) if f.any() else None, pixels=int(d.sum()))


tr = tv5.TrackerV5("xxxx/pi05-abc/tracker/assets")
for ep in sys.argv[2:]:
    task = int(ep[1:3])
    rgb, states, actions = load(ep)
    n = len(rgb)
    m = np.load(f"xxxx/rma/lr_eps/meta/{ep}.npz")
    st_raw = m["states"][:n]
    sub_text = {int(s): str(x).lower() for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    picks = [None] * n
    for t in range(n):
        s = sub_text[int(m["subtask"][t])]
        picks[t] = tv5.rma_spec.subtask_object(task, s) if s.startswith("pick") else None
    ref, src = stored(ep)
    K = ref.shape[1]
    start = time.time()
    on = np.zeros_like(ref)
    report, on[0] = tr.reset(np.ascontiguousarray(rgb[0]), task, st_raw[0])
    for t in range(1, n):
        info, on[t] = tr.step(np.ascontiguousarray(rgb[t]), st_raw[t], bool(actions[t - 1, 6] > 0), picks[t])
    s_on = time.time() - start
    res = dict(ep=ep, task=task, frames=n, k=K, src=src, report={a: b for a, b in report.items() if a != "pick"},
               online_vs_stored=cmp(on, ref), online_s=round(s_on, 1))
    if task in (18, 19):
        res["info_online"] = tr.app.info
        z = np.load(TV / src / f"{ep}.npz")
        res["info_stored"] = json.loads(str(z["info"]))
    import os
    import torch
    res["torch_threads"] = torch.get_num_threads()
    if os.environ.get("ONLINE_ONLY") == "1":
        (out / f"{ep}.json").write_text(json.dumps(res))
        print(json.dumps(res), flush=True)
        continue
    start = time.time()
    if task in (18, 19):
        C = {task: tv5.appearance_constants(task)}
        off, first, info = track5b.track_episode(ep, tr.base.stream, tr.base.proc, C)
        res["info_offline"] = info
    else:
        lab = np.load(TV / ("v4/initB2" if task in (25, 26) else "initB") / f"{ep}.npy")
        first = np.stack([lab == j + 1 for j in range(K)])
        res["first_frame_labels_equal_stored"] = None  # filled below from the online run
        off = np.zeros_like(ref)
        ms = tr.base.stream.reset(rgb[0], [first[j] for j in range(K)])
        for j in range(K):
            off[0, j] = ms[j]
        for t in range(1, n):
            ms = tr.base.stream.step(rgb[t])
            for j in range(K):
                off[t, j] = ms[j]
        # first-frame labels of the online selection (method B / B2) vs the stored ones
        labels, rep = tr.base.first_frame(np.ascontiguousarray(rgb[0]), task)
        if task in (25, 26):
            dm, ds, _ = tr.base.detect(np.ascontiguousarray(rgb[0]))
            cand, sc = tv5.b2_candidates(dm, ds)
            labels, _, _ = tv5.b2_select(tv5.rma_spec.CLICKS[task], cand, sc, labels)
        res["first_frame_labels_equal_stored"] = bool(np.array_equal(labels, lab))
        res["first_frame_label_pixels_differ"] = int((labels != lab).sum())
    res["offline_s"] = round(time.time() - start, 1)
    res["online_vs_offline"] = cmp(on, off)
    res["offline_vs_stored"] = cmp(off, ref)
    (out / f"{ep}.json").write_text(json.dumps(res))
    print(json.dumps(res), flush=True)
print("TEST_TRACKER_DONE", flush=True)
