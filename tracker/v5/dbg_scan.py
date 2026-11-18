import json, sys
from pathlib import Path
import numpy as np
HERE = Path("/data/lgd/rma/trackv")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE / "v5"))
import track5b, track5, ff_lib, det5
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
ep = sys.argv[1]
print("scan json:", open(HERE / "v5/scan" / f"{ep}.json").read() if (HERE / "v5/scan" / f"{ep}.json").exists() else None)
task, rgb, e, closed, picks = track5b.episode_inputs(ep)
z = np.load(HERE / "v5/track" / f"{ep}.npz")
masks = np.unpackbits(z["masks"], axis=-1)[..., :256].astype(bool)
info = json.loads(str(z["info"]))
print("info", {k: v for k, v in info.items()})
tog = np.flatnonzero(closed[1:] != closed[:-1]) + 1
print("toggles", tog.tolist(), "picks at toggles", [picks[t] for t in tog])
J = json.load(open(HERE / "v5/ff_clf_19_d.json" if task == 19 else HERE / "v5/ff_clf_18_d.json"))
clf = ff_lib.Softmax.from_json(J["clf"]); atyp = J["atyp"]
K = masks.shape[1]
for tc in tog[::2]:
    k = picks[tc]
    if k is None:
        continue
    print("grasp of", k, "at", tc)
    for t in range(tc, min(len(rgb), tc + 400), 15):
        if not closed[t]:
            break
        mk = masks[t, k]
        ys, xs = np.nonzero(mk)
        d = float(np.min(np.hypot(xs - e[t][0], ys - e[t][1]))) if len(xs) else None
        print(f"   t={t} moved {np.hypot(*(e[t] - e[tc])):.0f} mask area {int(mk.sum())} edge-to-gripper {d} far {track5b.mask_far(mk, e[t])}")
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False), confidence_threshold=det5.THR)
for to in tog[1::2]:
    k = picks[tog[list(tog).index(to) - 1]]
    t = to + track5b.R_WAIT
    if k is None or t >= len(rgb):
        continue
    cand, sc = track5.candidates(proc, rgb[t], task, crop=track5b.crop_at(e[to]))
    P = track5.probs(clf, rgb[t], cand, K)
    rows = [(round(float(P[i, k]), 3), int(cand[i].sum()), [round(float(x)) for x in ff_lib.centroid(cand[i])], round(float(np.hypot(*(ff_lib.centroid(cand[i]) - e[to]))), 1),
             bool(track5b.overlap(cand[i], masks[t, k], track5b.R_OVER))) for i in np.argsort(-P[:, k])[:5]]
    print(f"release of {k} at {to}: check at {t}: e_rel {e[to].round(0).tolist()} e_now {e[t].round(0).tolist()} mask area {int(masks[t, k].sum())} centroid {ff_lib.centroid(masks[t, k]).round(0).tolist() if masks[t, k].any() else None}")
    print("    top (P, area, centroid, dist to e_rel, overlaps mask):", rows, "atyp", atyp[k], "->", track5b.release_candidate(P, cand, k, e[to], e[t], atyp))
