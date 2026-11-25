"""debug the grasp check on stored data: dbg_grasp.py ep k"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("xxxx/rma/trackv")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE / "v5"))
import rma_spec  # noqa
from tv_common import load, task_of  # noqa
from tv_common_v4 import eef_pixels  # noqa
import det5, ff_lib, track5  # noqa
from sam3.model.sam3_image_processor import Sam3Processor  # noqa
from sam3.model_builder import build_sam3_image_model  # noqa

ep, k = sys.argv[1], int(sys.argv[2])
task = task_of(ep)
rgb, states, actions = load(ep)
e = eef_pixels(ep, states)
closed = np.r_[False, actions[:-1, 6] > 0]
meta = np.load(f"xxxx/rma/lr_eps/meta/{ep}.npz")
text = {int(s): str(x).lower() for s, x in zip(meta["subtask_order"], meta["subtask_texts"])}
K = len(rma_spec.TASKS[task]["objects"])
J = json.load(open(HERE / f"v5/ff_clf_{task}_d.json"))
clf = ff_lib.Softmax.from_json(J["clf"])
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False), confidence_threshold=det5.THR)
ys, xs = np.mgrid[0:256, 0:256]
for tc in np.flatnonzero(closed[1:] & ~closed[:-1]) + 1:
    s = text[int(meta["subtask"][tc])]
    kk = rma_spec.subtask_object(task, s) if s.startswith("pick") else None
    print("close at", tc, "subtask", s, "object", kk)
    if kk != k:
        continue
    for dt in (15, 25, 40, 60, 90):
        t = tc + dt
        mv = float(np.hypot(*(e[t] - e[tc])))
        cx, cy = int(np.clip(e[t][0] - 48, 0, 160)), int(np.clip(e[t][1] - 48, 0, 160))
        cand, sc = track5.candidates(proc, rgb[t], task, crop=(cx, cy, cx + 96, cy + 96))
        P = track5.probs(clf, rgb[t], cand, K)
        rows = []
        for i in np.argsort(-P[:, k])[:6]:
            m = cand[i]
            rows.append((round(float(P[i, k]), 3), int(m.sum()), round(float(np.min(np.hypot(xs[m] - e[t][0], ys[m] - e[t][1]))), 1), [round(float(x)) for x in ff_lib.centroid(m)]))
        near = sorted([(round(float(np.min(np.hypot(xs[m] - e[t][0], ys[m] - e[t][1]))), 1), round(float(P[i, k]), 3), int(m.sum()), np.round(P[i], 2).tolist()) for i, m in enumerate(cand)])[:5]
        print(f"  t={t} closed={closed[t]} gripper moved {mv:.1f} px, e={e[t].round(0).tolist()} n_cand {len(cand)}; top by P[k]: (P, area, edge, centroid) {rows}")
        print(f"      nearest to the gripper pixel: (edge, P[k], area, P all) {near}")
