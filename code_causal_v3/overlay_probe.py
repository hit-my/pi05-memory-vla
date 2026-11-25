"""Attribution: same saved raw image, same memory, same state; only the trajectory lines differ. Does the Subgoal decision change?

Frames come from closed-loop runs with the tracker. Three sets: false ticks, true ticks, and ordinary frames with no tick.
Variants: track (as run), truth (simulator centres, for attribution only, offline), plus optional transforms of the
tracker's own centres that use no truth (smoothing, gap filling, offset). Nothing is executed.

Usage (under the simulator venv): overlay_probe.py GROUP CKPT_GROUP RUNS TASK OUT.json VARIANTS [MAX_NORMAL]
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from closedloop_ledger import install  # noqa: E402

OLD = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
sys.path.insert(0, str(OLD))
import simulate  # noqa: E402

group, ckpt_group, runs, task, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4], Path(sys.argv[5])
variants = sys.argv[6].split(",")
max_normal = int(sys.argv[7]) if len(sys.argv) > 7 else 80
install(simulate)
CL = Path(os.environ.get("CL_ROOT", "xxxx/pi05-abc/closedloop"))
HOME = Path.home()
contract = json.loads((CODE / f"contract_{ckpt_group}.json").read_text())
random.seed(0)
cadence = simulate.CADENCE


def centres(series, key):
    """{frame: {sid: [x, y] or None}} from track.jsonl. key: pred or true."""
    out = {}
    for row in series:
        out[row["frame"]] = {sid: (b[key] if b[f"{key}_area"] >= 80 else None) for sid, b in row["bowls"].items()}
    return out


def transform(name, track):
    """Causal: each output uses only frames up to its own."""
    frames = sorted(track)
    sids = list(track[frames[0]])
    if name == "track" or name == "truth":
        return track
    if name.startswith("hold"):  # fill short gaps with the last seen centre
        limit = int(name[4:] or 1000)
        result, last = {}, {s: None for s in sids}
        age = {s: 0 for s in sids}
        for f in frames:
            row = {}
            for s in sids:
                if track[f][s] is not None:
                    last[s], age[s] = track[f][s], 0
                    row[s] = track[f][s]
                else:
                    age[s] += 1
                    row[s] = last[s] if age[s] <= limit else None
            result[f] = row
        return result
    if name.startswith("ema"):  # exponential smoothing
        alpha = float(name[3:])
        result, value = {}, {s: None for s in sids}
        for f in frames:
            row = {}
            for s in sids:
                c = track[f][s]
                if c is None:
                    row[s] = None
                    continue
                value[s] = np.array(c, float) if value[s] is None else alpha * np.array(c, float) + (1 - alpha) * value[s]
                row[s] = value[s].tolist()
            result[f] = row
        return result
    if name == "still":  # hold a bowl exactly still unless it clearly moved
        result, anchor = {}, {s: None for s in sids}
        for f in frames:
            row = {}
            for s in sids:
                c = track[f][s]
                if c is None:
                    row[s] = None
                    continue
                c = np.array(c, float)
                if anchor[s] is None or np.linalg.norm(c - anchor[s]) > 1.5:
                    anchor[s] = c
                row[s] = anchor[s].tolist()
            result[f] = row
        return result
    raise ValueError(name)


def history(track, frame, registry):
    rows = []
    for f in range(max(0, frame - cadence), frame + 1):
        entities = {}
        for sid in registry:
            c = track.get(f, {}).get(sid)
            entities[sid] = {"center": c, "raw_center": c, "area": 500 if c is not None else 0, "reliable": c is not None}
        rows.append({"frame": f, "entities": entities})
    return rows


jobs = []
for run in runs:
    base = CL / run / group / task
    run_data = json.loads((base / "run.json").read_text())
    log = {}
    for line in Path(str(base) + ".track/track.jsonl").open():
        row = json.loads(line)
        log.setdefault(row["episode_number"], []).append(row)
    for number, episode in enumerate(run_data["episodes"]):
        folder = base / f"{task}_{episode['episode_id']}"
        scores = {}
        for line in (folder / "oracle_scores.jsonl").read_text().splitlines():
            s = json.loads(line)
            scores[s["frame"]] = s["confirmed_goals"]
        first = {}
        for f in sorted(scores):
            for k in range(1, scores[f] + 1):
                first.setdefault(k, f)
        replies = [json.loads(l) for l in (folder / "queries.jsonl").read_text().splitlines()]
        replies = [r for r in replies if r.get("event") == "response"]
        shift = int(os.environ.get("PROBE_SHIFT", "0"))
        by_frame = {r["frame"]: r for r in replies}
        for r in replies:
            delta = str(r["response"].get("subgoal_delta") or "")
            kind = "normal"
            if delta.startswith("done"):
                k = int(delta.split()[1].split(",")[0])
                kind = "false" if first.get(k) is None or first[k] > r["frame"] + 20 else "true"
            q = r
            if shift and kind != "normal":
                q = by_frame.get(r["frame"] - shift)
                if q is None:
                    continue
            jobs.append({"kind": kind, "run": run, "number": number, "demo": episode["episode_id"], "frame": q["frame"], "folder": str(folder),
                         "delta": delta, "seen_action": q["response"].get("action_text"), "memory": q["previous_memory_text"], "state": q["state"], "instruction": q["instruction"],
                         "main_raw": str(folder / "query_images" / f"{q['frame']:06d}_raw.png"), "wrist": str(folder / q["wrist_image"])})
    tracks_cache = None
false = [j for j in jobs if j["kind"] == "false"]
true = [j for j in jobs if j["kind"] == "true"]
normal = [j for j in jobs if j["kind"] == "normal" and j["frame"] >= 40]
random.shuffle(true)
random.shuffle(normal)
chosen = false + true[:len(false) + 10] + normal[:max_normal]
print("false", len(false), "true", len(true), "normal", len(normal), "chosen", len(chosen), flush=True)

logs = {}


def series_for(job):
    key = (job["run"], job["number"])
    if key not in logs:
        all_rows = {}
        for line in Path(str(CL / job["run"] / group / task) + ".track/track.jsonl").open():
            row = json.loads(line)
            all_rows.setdefault(row["episode_number"], []).append(row)
        for number, rows in all_rows.items():
            logs[(job["run"], number)] = rows
    return logs[key]


renderer_js = CODE / contract["visual"]["renderer_asset"]
checkpoint = HOME / f"pi05-abc-ckpt/{ckpt_group}/checkpoint_{os.environ.get('CKPT_STEP', '010000')}"
policy_log = Path(tempfile.mkdtemp()) / "policy.log"
os.environ.setdefault("ACTION_SAMPLES", "1")
os.environ.setdefault("ACTION_SEED", "5")
policy = simulate.PolicyProcess(HOME / "SANE/.venv/bin/python", CODE / "policy_ens.py", checkpoint, policy_log, 600.0)
results = []
with simulate.CanonicalRenderer(renderer_js) as renderer:
    cache = {}
    for index, job in enumerate(chosen):
        registry = json.loads((Path(job["folder"]) / "environment.json").read_text())["registry"]
        series = series_for(job)
        ckey = (job["run"], job["number"])
        if ckey not in cache:
            cache[ckey] = {"pred": centres(series, "pred"), "true": centres(series, "true")}
        raw = np.asarray(Image.open(job["main_raw"]).convert("RGB"))
        wrist = Path(job["wrist"]).read_bytes()
        row = {k: job[k] for k in ("kind", "run", "demo", "frame", "delta", "seen_action")}
        for name in variants:
            base_track = cache[ckey]["true" if name == "truth" else "pred"]
            track = transform(name, base_track)
            tracks = simulate.make_tracks(history(track, job["frame"], registry), registry)
            png, _ = renderer.render(raw, job["frame"], max(0, job["frame"] - cadence), registry, tracks)
            policy.latents = None
            result, _ = policy.infer(png, wrist, np.asarray(job["state"]), job["instruction"], job["memory"])
            row[name] = {"delta": result.get("subgoal_delta"), "action": result.get("action_text"), "valid": result.get("memory_valid"), "png_sha": __import__("hashlib").sha256(png).hexdigest()[:12]}
        results.append(row)
        if index % 20 == 0:
            print(index, row["kind"], row["delta"], {n: row[n]["delta"] for n in variants}, flush=True)
            out.write_text(json.dumps(results))
out.write_text(json.dumps(results))
policy.close()
for kind in ("false", "true", "normal"):
    part = [r for r in results if r["kind"] == kind]
    if not part:
        continue
    text = ", ".join(f"{n}: {sum(str(r[n]['delta'] or '').startswith('done') for r in part)}/{len(part)}" for n in variants)
    print(f"{kind:7s} ticks under each overlay -> {text}", flush=True)
    placed = ", ".join(f"{n}: {sum('placed' in str(r[n]['action'] or '') for r in part)}" for n in variants)
    same = ", ".join(f"{n}: {sum(r[n]['action'] == r[variants[0]]['action'] for r in part)}" for n in variants)
    picture = ", ".join(f"{n}: {sum(r[n]['png_sha'] != r[variants[0]]['png_sha'] for r in part)}" for n in variants)
    print(f"        placed in Action -> {placed}\n        Action same as first variant -> {same}\n        picture differs from first variant -> {picture}", flush=True)
print("PROBE_DONE", flush=True)
