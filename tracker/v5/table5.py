"""Score table against GT: raw (v4 masks) / v4 retire / v5 per task, held-out half (B) and all episodes -> v5/TABLE_v5.md, v5/table_v5.json
columns: err/100 valid object-frames (wrong dot + drawn-hidden >10 px from projected centre + missing while >=30 px visible),
% of scored (GT visible >= 8 px and drawn) within 10 px, wrong-dot % of scored, missing % of the >=30 px visible, drawn-while-hidden % of
hidden frames (any distance), first frame: episodes with every object's dot on the right object / usable episodes."""
import json
import sys
from multiprocessing import Pool

import numpy as np

import ev5

V4 = ev5.TV / "all/final_v4"
V5 = ev5.TV / "all/final_v5"
M = {"raw": (V4, "raw_centres", "raw_visibility"), "v4": (V4, "centres", "visibility"), "v5": (V5, "centres", "visibility"), "v5raw": (V5, "raw_centres", "raw_visibility")}
FF_FRAME = {18: 10, 19: 10}


def inside(p, b, pad=2.0):
    return bool(np.isfinite(p).all() and np.isfinite(b).all() and b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad)


def one(ep):
    g = ev5.G(ep)
    task = int(ep[1:3])
    out = {"ok": g.ok, "half": ev5.half_of(ep)}
    z = {d: np.load(d / f"{ep}.npz") for d in (V4, V5)}
    if g.ok:
        for name, (d, kc, kv) in M.items():
            out[name] = ev5.score(g, z[d][kc], z[d][kv])
    # first frame (firstframe.py definition; for tasks 18/19 v5 starts at frame 4 / 10: the dot of frame 10 is checked)
    q = ev5.qa().get(ep, {})
    if q.get("settled_ok", False) and q.get("status") != "error":
        gg = np.load(ev5.GT / f"{ep}.npz")
        bb = gg["bbox"]
        for name, d in (("v4", V4), ("v5", V5)):
            f = FF_FRAME.get(task, 0) if name == "v5" else 0
            c, v = z[d]["raw_centres"][f].astype(np.float64), z[d]["raw_visibility"][f]
            # right = the dot is on the object; an object that is hidden at that frame (GT area < 8 px) and has no dot also counts as right
            out["ff_" + name] = [bool((v[k] and (inside(c[k], bb[10, k]) or inside(c[k], bb[f, k]))) or (not v[k] and gg["mask_area"][max(f, 10), k] < 8))
                                 for k in range(bb.shape[1])]
    return ep, out


if __name__ == "__main__":
    tasks = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 27))
    eps = [e for t in tasks for e in ev5.task_eps(t)]
    with Pool(26) as p:
        R = dict(p.map(one, eps, chunksize=4))
    const = json.load(open(ev5.TV / "v5/constants_v5.json"))
    res, lines = {}, []
    hdr = "| task | half | method | err/100 | within 10 px % | wrong-dot % | missing % | drawn-hidden % | hid-far /100 | first frame all objects right | targets |"
    lines += ["# v5 against simulator ground truth", "",
              "err/100 = (wrong dot on a visible object + dot drawn while hidden and >10 px from the projected centre + missing while >=30 px visible) per 100 valid object-frames.",
              "raw = SAM3 centroids of the v4 masks; v4 = all/final_v4 centres (with retirement); v5 = all/final_v5 centres. Half B = held-out (odd positions), not used for any choice.",
              "targets: err <= 1.0 (stretch 0.5), not worse than raw by more than 0.1, >= 97 % within 10 px, first frame right in >= 98 % of the usable episodes.", "", hdr, "|" + "---|" * 11]
    worse = []
    for t in tasks:
        res[t] = {}
        for half in ("B", "all"):
            sel = [ep for ep in ev5.task_eps(t) if half == "all" or R[ep]["half"] == half]
            row = {}
            for name in M:
                acc = {}
                for ep in sel:
                    if R[ep]["ok"]:
                        ev5.add(acc, R[ep][name])
                s = ev5.summary(acc)
                s["hid_all_per100"] = 100.0 * acc["dh"] / acc["n"]
                row[name] = s
            for name in ("v4", "v5"):
                ff = [R[ep]["ff_" + name] for ep in sel if "ff_" + name in R[ep]]
                row[name]["ff_all"] = int(sum(all(x) for x in ff)); row[name]["ff_n"] = len(ff)
                row[name]["ff_obj"] = np.sum(np.array(ff), axis=0).tolist() if ff else []
            res[t][half] = row
            for name in ("raw", "v4", "v5"):
                s = row[name]
                tg = ""
                if name == "v5":
                    ok_err = s["err100"] <= 1.0 and s["err100"] <= row["raw"]["err100"] + 0.1
                    ok_w = s["w10"] >= 97.0
                    ok_ff = s["ff_n"] > 0 and min(s["ff_obj"]) >= 0.98 * s["ff_n"]
                    tg = ("err OK" if ok_err else "err MISS") + (" (<=0.5)" if s["err100"] <= 0.5 else "") + ", " + ("10px OK" if ok_w else "10px MISS") + ", " + ("first OK" if ok_ff else "first MISS")
                    s["meets"] = dict(err=bool(ok_err), w10=bool(ok_w), first=bool(ok_ff))
                ffs = f"{s['ff_all']}/{s['ff_n']} (per object {s['ff_obj']})" if "ff_all" in s else ""
                lines.append(f"| {t} | {half} | {name} | {s['err100']:.2f} | {s['w10']:.2f} | {s['wrong_sc']:.2f} | {s['miss_v']:.2f} | {s['dh_hid']:.2f} | {s['dhf']:.2f} | {ffs} | {tg} |")
            if half == "B":
                for other in ("raw", "v4"):
                    cols = [c for c in ("err100", "wrong_sc", "miss_v", "dh_hid") if row["v5"][c] > row[other][c] + 0.05] + (["w10"] if row["v5"]["w10"] < row[other]["w10"] - 0.05 else [])
                    if cols:
                        worse.append(f"task {t}: v5 worse than {other} on " + ", ".join(f"{c} ({row['v5'][c]:.2f} vs {row[other][c]:.2f})" for c in cols))
    lines += ["", "## held-out half: columns where v5 is worse than raw or v4 (by more than 0.05)"] + (worse or ["none"])
    lines += ["", "## chosen rule set per task (constants_v5.json)"] + [f"- task {t}: {const['tasks'][str(t)]['chosen']}  cfg {json.dumps(const['tasks'][str(t)]['cfg'])} retire {json.dumps(const['tasks'][str(t)]['retire'])}" for t in tasks]
    open(ev5.TV / "v5/TABLE_v5.md", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(ev5.TV / "v5/table_v5.json", "w"), indent=1)
    for t in tasks:
        b, a = res[t]["B"], res[t]["all"]
        print(f"t{t:02d} B: raw {b['raw']['err100']:6.2f} v4 {b['v4']['err100']:6.2f} v5 {b['v5']['err100']:6.2f} w10 {b['v5']['w10']:6.2f} ff {b['v5']['ff_obj']}/{b['v5']['ff_n']} | all: raw {a['raw']['err100']:6.2f} v4 {a['v4']['err100']:6.2f} "
              f"v5 {a['v5']['err100']:6.2f} | {b['v5']['meets']}")
    print("\n".join(worse))
