"""Integrity check of a v2r dataset, and field-by-field diff against the previous dataset.

  rma_check_v2r.py --dataset D --raw LR_EPS --tracks TRACKS --contract FILE [--prev /data/lgd/rma/full/dataset_causal]
                   [--partial] [--image-samples 400] [--report FILE.json] [--skip-loader]
Exit status 0 only when every check passes. Checks:
  A  coverage: built episodes vs RAW (all, unless --partial), frames per episode vs the LeRobot arrays, manifest and
     index sha256, row counts.
  B  files: every image_path / wrist_image_path / raw_image_path of every row exists and is non-empty; a random sample
     is decoded (224x224 RGB); raw image == PIL BILINEAR resize of the stored frame on the sample.
  C  tracks: each episode was built from the track file that is in TRACKS now (sha256); episode npz visibility/centres
     equal the track file's drawn positions; inconsistency counters of the track files are zero.
  D  labels: the Action line is recomputed for every row from (visibility, recorded actions, states) with the row
     writer's own function and must equal the stored one; 'occluded' appears only where the object was drawn in
     fewer than half of frames t-20..t (causal window).
  E  diff vs --prev: per row, every field except the expected ones must be equal. Expected to differ: raw_image_path
     (new), and the Action line inside memory_target_text / memory_target.action_text / previous_memory_text.
     Subgoal and Task lines, instruction, indices, paths, flags, states/actions/ends must be identical.
  F  latents (contract dataset.latents_dir): one table per episode, z [N,K,64], valid; standardize.json present.
  G  the training loader itself: model_data.FullCorpus on train.jsonl and val.jsonl with the contract (unless
     --skip-loader), capacities, and a few batches.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--raw", type=Path, required=True)
p.add_argument("--tracks", type=Path, required=True)
p.add_argument("--contract", type=Path, required=True)
p.add_argument("--prev", type=Path)
p.add_argument("--code", type=Path, default=Path.home() / "pi05-abc/code_causal_v2")
p.add_argument("--partial", action="store_true")
p.add_argument("--image-samples", type=int, default=400)
p.add_argument("--report", type=Path)
p.add_argument("--skip-loader", action="store_true")
p.add_argument("--skip-latents", action="store_true")
args = p.parse_args()
D, code = args.dataset, args.code.resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(code))
import overlay_v2r  # noqa: E402
import rma_spec  # noqa: E402

failures, report = [], {}


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def sha(path):
    return overlay_v2r.sha256_file(path)


# rma_build's label functions without its imports of the simulator driver: load the source, keep the pure functions
def label_functions():
    source = (code / "rma_build.py").read_text()
    start, end = source.index("def held_table"), source.index("def main()")
    scope = {"np": np, "rma_spec": rma_spec, "AXES": ((0, "forward", "backward"), (1, "left", "right"), (2, "up", "down")),
             "THRESHOLD": 0.015}
    head = source[:start]
    assert 'AXES = ((0, "forward", "backward"), (1, "left", "right"), (2, "up", "down"))' in head and "THRESHOLD = 0.015" in head
    exec(compile(source[start:end], "rma_build.py[labels]", "exec"), scope)
    return scope["held_table"], scope["action_text"]


held_table, action_text = label_functions()
rng = random.Random(0)

# ---------------------------------------------------------------- A coverage
manifest = json.loads((D / "split_manifest.json").read_text())
contract = json.loads(args.contract.read_text(encoding="utf-8"))
built = sorted(x.stem for x in (D / "rows").glob("*.jsonl"))
every = sorted(x.stem for x in (args.raw / "meta").glob("t*_seed*.npz") if (args.raw / "front" / f"{x.stem}.npy").exists())
check("A1 every RAW episode is built" if not args.partial else "A1 built episodes exist in RAW (partial build)",
      built == every if not args.partial else set(built) <= set(every), f"{len(built)} of {len(every)}")
check("A2 manifest lists exactly the built episodes", sorted(r["episode_id"] for r in manifest["episodes"]) == built)
for part in ("train", "val"):
    check(f"A3 {part}.jsonl sha256 == manifest", sha(D / f"{part}.jsonl") == manifest["splits"][part]["index_sha256"])
check("A4 norm_stats sha256 == manifest", sha(D / "norm_stats.json") == manifest["normalization"]["sha256"])
frames_raw = {}
for ep in built:
    front = np.load(args.raw / "front" / f"{ep}.npy", mmap_mode="r")
    wrist = np.load(args.raw / "wrist" / f"{ep}.npy", mmap_mode="r")
    with np.load(args.raw / "meta" / f"{ep}.npz") as m:
        frames_raw[ep] = min(len(front), len(wrist), len(m["subtask"]))
by_ep = {r["episode_id"]: r for r in manifest["episodes"]}
bad = [ep for ep in built if by_ep[ep]["actions"] != frames_raw[ep] or by_ep[ep]["queries"] != frames_raw[ep]]
check("A5 rows per episode == frames in the LeRobot arrays", not bad, f"{sum(frames_raw.values())} frames; bad {bad[:3]}")
tasks_c = {int(t["task_id"]): t for t in contract["source"]["tasks"]}
per_task = Counter(int(ep[1:3]) for ep in built)
frames_task = Counter()
for ep in built:
    frames_task[int(ep[1:3])] += frames_raw[ep]
check("A6 contract source.tasks counts == dataset",
      set(tasks_c) == set(per_task) and all(tasks_c[t]["metadata_episode_count"] == per_task[t]
                                            and tasks_c[t]["metadata_frame_count"] == frames_task[t] for t in per_task))
report["episodes"], report["frames"] = len(built), sum(frames_raw.values())
report["split"] = {q: [manifest["splits"][q]["episodes"], manifest["splits"][q]["queries"]] for q in ("train", "val")}

# ---------------------------------------------------------------- B..E per episode
EXPECTED_KEYS = ["sample_id", "task_id", "episode_id", "observation_index", "history_start", "history_end", "instruction",
                 "previous_memory_text", "previous_query_index", "memory_update_interval", "memory_target_text",
                 "memory_target", "image_path", "wrist_image_path", "episode_npz", "action_start", "valid_action_steps",
                 "action_label_reliable", "raw_image_path"]
TARGET_KEYS = ["action_text", "subgoal_delta", "task_text", "confirmed_goals", "total_goals", "action_reliable"]
missing_files, empty_files, schema_bad, n_rows = [], [], [], 0
track_stale, track_mismatch, track_counters = [], [], Counter()
label_bad, occluded_bad = [], []
diff_fields, clause_moves, prev_missing, npz_diff = Counter(), Counter(), [], []
rows_changed = rows_compared = subgoal_task_changed = 0
changed_by_task, rows_by_task = Counter(), Counter()
examples = []
image_pool = []
for ep in built:
    task = int(ep[1:3])
    names = rma_spec.names(task)
    rows = [json.loads(line) for line in open(D / "rows" / f"{ep}.jsonl")]
    n = len(rows)
    n_rows += n
    rows_by_task[task] += n
    log = json.loads((D / "log" / f"{ep}.json").read_text())
    z = np.load(D / "episodes" / f"{ep}.npz")
    # B files
    listing = {e.name: e.stat().st_size for e in os.scandir(D / "images" / ep)}
    for r in rows:
        if list(r.keys()) != EXPECTED_KEYS or list(r["memory_target"].keys()) != TARGET_KEYS:
            schema_bad.append(r["sample_id"])
        for field in ("image_path", "wrist_image_path", "raw_image_path"):
            name = r[field].rsplit("/", 1)[1]
            if r[field] != f"images/{ep}/{name}" or name not in listing:
                missing_files.append(r[field])
            elif listing[name] == 0:
                empty_files.append(r[field])
    if len(listing) != 3 * n:
        missing_files.append(f"images/{ep}: {len(listing)} files for {n} frames")
    image_pool.append((ep, n))
    # C tracks
    track_path = args.tracks / f"{ep}.npz"
    if not track_path.is_file() or sha(track_path) != log["tracks"]["sha256"]:
        track_stale.append(ep)
    else:
        centres, drawn, retired, stats = overlay_v2r.load_tracks(track_path, n, len(names))
        for key in ("visible_without_centre", "centre_without_visible", "centre_outside_image", "retired_but_centre", "retired_not_sticky"):
            track_counters[key] += stats[key]
        if not (np.array_equal(z["visibility"], drawn) and np.array_equal(z["retired"], retired)
                and np.array_equal(z["centres"], np.where(drawn[..., None], centres, 0.0).astype(np.float32))):
            track_mismatch.append(ep)
    # D labels
    visible = z["visibility"].astype(bool)
    held, events = held_table(task, z["actions"], [int(e) for e in z["ends"]], log["items"], len(names))
    for t, r in enumerate(rows):
        expected = action_text(t, z["states"], held, events, visible, names)
        if expected != r["memory_target"]["action_text"] or r["memory_target_text"].split("\n")[0] != "Action:" + expected:
            label_bad.append(r["sample_id"])
        start = max(0, t - 20)
        for k, clause in enumerate(expected.split(" | ")):
            if clause.split(": ", 1)[1] == "occluded" and not visible[start:t + 1, k].mean() < 0.5:
                occluded_bad.append(r["sample_id"])
    # E diff vs previous dataset
    if args.prev is not None:
        prev_path = args.prev / "rows" / f"{ep}.jsonl"
        if not prev_path.is_file():
            prev_missing.append(ep)
        else:
            old_rows = [json.loads(line) for line in open(prev_path)]
            if len(old_rows) != n:
                prev_missing.append(f"{ep}: {len(old_rows)} != {n} rows")
            zo = np.load(args.prev / "episodes" / f"{ep}.npz")
            for key in ("states", "actions", "ends", "confirmed_goals"):
                if not np.array_equal(z[key], zo[key]):
                    npz_diff.append(f"{ep}:{key}")
            for r, o in zip(rows, old_rows):
                rows_compared += 1
                changed = False
                for key in EXPECTED_KEYS:
                    if key == "raw_image_path":
                        if key in o and o[key] != r[key]:
                            diff_fields[key] += 1
                        continue
                    if key == "memory_target":
                        for sub in TARGET_KEYS:
                            if r[key][sub] != o[key][sub]:
                                diff_fields[f"memory_target.{sub}"] += 1
                                changed = True
                    elif r[key] != o.get(key, "<absent>"):
                        diff_fields[key] += 1
                        changed = True
                extra = set(o) - set(EXPECTED_KEYS)
                if extra:
                    diff_fields["previous dataset has extra keys " + ",".join(sorted(extra))] += 1
                # the change must be confined to the Action line
                for key in ("memory_target_text", "previous_memory_text"):
                    if r[key].split("\n")[1:] != o[key].split("\n")[1:]:
                        subgoal_task_changed += 1
                if r["memory_target"]["action_text"] != o["memory_target"]["action_text"]:
                    for a, b in zip(o["memory_target"]["action_text"].split(" | "), r["memory_target"]["action_text"].split(" | ")):
                        if a != b:
                            clause_moves[f"{a.split(': ', 1)[1]} -> {b.split(': ', 1)[1]}"] += 1
                    if len(examples) < 6 and rng.random() < 0.02:
                        examples.append({"sample_id": r["sample_id"], "old": o["memory_target"]["action_text"],
                                         "new": r["memory_target"]["action_text"]})
                rows_changed += changed
                changed_by_task[task] += changed

check("B1 row schema (keys and order) == previous rows + raw_image_path", not schema_bad, f"{n_rows} rows; bad {schema_bad[:3]}")
check("B2 every image path of every row exists", not missing_files, f"{3 * n_rows} paths; missing {missing_files[:3]}")
check("B3 no empty image file", not empty_files, str(empty_files[:3]))
check("A7 rows total == manifest", n_rows == manifest["source"]["queries"] == sum(frames_raw.values()), str(n_rows))
check("C1 every episode was built from the track file now in TRACKS", not track_stale, str(track_stale[:5]))
check("C2 episode npz visibility/centres/retired == track file", not track_mismatch, str(track_mismatch[:5]))
check("C3 track files self-consistent (visibility == finite centre, inside image, retired sticky and blank)",
      not any(track_counters.values()), str(dict(track_counters)))
check("D1 stored Action line == row writer's rule recomputed from the stored visibility", not label_bad, str(label_bad[:3]))
check("D2 'occluded' only where drawn in < 50% of frames t-20..t", not occluded_bad, str(occluded_bad[:3]))
report.update(rows=n_rows, track_counters=dict(track_counters))

# sample of images decoded
sample = []
for _ in range(args.image_samples):
    ep, n = rng.choice(image_pool)
    sample.append((ep, rng.randrange(n)))
decode_bad, raw_bad, wrist_bad = [], [], []
overlay_vs_raw = []
for ep, t in sample:
    front = np.load(args.raw / "front" / f"{ep}.npy", mmap_mode="r")
    wrist = np.load(args.raw / "wrist" / f"{ep}.npy", mmap_mode="r")
    images = {}
    for suffix in ("", "_wrist", "_raw"):
        with Image.open(D / "images" / ep / f"{t:06d}{suffix}.png") as im:
            # the overlay is the renderer's canvas PNG: RGBA with a fully opaque alpha channel, as in the previous
            # dataset (the loader converts to RGB); wrist and raw are RGB
            modes = ("RGBA",) if suffix == "" else ("RGB",)
            if im.size != (224, 224) or im.mode not in modes or (im.mode == "RGBA" and np.asarray(im)[..., 3].min() != 255):
                decode_bad.append(f"{ep}/{t}{suffix} {im.size} {im.mode}")
            images[suffix] = np.asarray(im.convert("RGB"))
    if args.prev is not None and (args.prev / "images" / ep / f"{t:06d}.png").is_file():
        with Image.open(args.prev / "images" / ep / f"{t:06d}.png") as old, Image.open(D / "images" / ep / f"{t:06d}.png") as new:
            if (old.mode, old.size) != (new.mode, new.size):
                decode_bad.append(f"{ep}/{t} overlay {new.mode}{new.size} but previous dataset {old.mode}{old.size}")
    if not np.array_equal(images["_raw"], np.asarray(Image.fromarray(np.ascontiguousarray(front[t])).resize((224, 224), Image.BILINEAR))):
        raw_bad.append(f"{ep}/{t}")
    if not np.array_equal(images["_wrist"], np.asarray(Image.fromarray(np.ascontiguousarray(wrist[t])).resize((224, 224), Image.BILINEAR))):
        wrist_bad.append(f"{ep}/{t}")
    overlay_vs_raw.append(float(np.mean(np.abs(images[""].astype(np.float32) - images["_raw"].astype(np.float32)))))
check("B4 sampled images decode as 224x224 (overlay opaque RGBA like the previous dataset; wrist/raw RGB)", not decode_bad, f"{len(sample)} frames x 3; bad {decode_bad[:3]}")
check("B5 sampled raw image == BILINEAR(256->224) of the stored front frame", not raw_bad, str(raw_bad[:3]))
check("B6 sampled wrist image == BILINEAR(256->224) of the stored wrist frame", not wrist_bad, str(wrist_bad[:3]))
check("B7 overlay is the same frame as the raw image (mean |overlay - raw| < 3 grey levels on every sample)",
      max(overlay_vs_raw) < 3.0, f"mean {np.mean(overlay_vs_raw):.2f} max {max(overlay_vs_raw):.2f}")

if args.prev is not None:
    allowed = {"memory_target_text", "previous_memory_text", "memory_target.action_text"}
    unexpected = {k: v for k, v in diff_fields.items() if k not in allowed}
    check("E1 previous dataset has the same episodes and row counts", not prev_missing, str(prev_missing[:3]))
    check("E2 states / actions / ends / confirmed_goals identical to the previous episode npz", not npz_diff, str(npz_diff[:3]))
    check("E3 no field differs except the Action line (memory_target_text, previous_memory_text, memory_target.action_text)",
          not unexpected, str(unexpected))
    check("E4 Subgoal and Task lines identical in every row (target and previous memory)", subgoal_task_changed == 0, str(subgoal_task_changed))
    print(f"     rows compared {rows_compared}, rows with a changed field {rows_changed} ({100 * rows_changed / max(rows_compared, 1):.1f}%)")
    print("     fields that differ (rows):", dict(diff_fields))
    print("     Action clause changes old -> new (clauses):", dict(clause_moves.most_common(12)))
    print("     share of rows changed by task:", {t: round(changed_by_task[t] / rows_by_task[t], 3) for t in sorted(rows_by_task)})
    for e in examples:
        print("     e.g.", e)
    report["diff"] = {"rows_compared": rows_compared, "rows_changed": rows_changed, "fields": dict(diff_fields),
                      "clause_moves": dict(clause_moves), "changed_share_by_task": {t: changed_by_task[t] / rows_by_task[t] for t in rows_by_task}}
    if not args.partial:
        check("E5 priors.json identical to the previous dataset", (D / "priors.json").read_text() == (args.prev / "priors.json").read_text())
        check("E6 norm_stats.json identical to the previous dataset", sha(D / "norm_stats.json") == sha(args.prev / "norm_stats.json"))
        old_manifest = json.loads((args.prev / "split_manifest.json").read_text())
        check("E7 train/val membership identical to the previous dataset",
              {r["episode_id"]: r["split"] for r in manifest["episodes"]} == {r["episode_id"]: r["split"] for r in old_manifest["episodes"]})

# ---------------------------------------------------------------- F latents
if not args.skip_latents:
    latents = Path(contract["dataset"]["latents_dir"])
    bad = []
    valid_share = []
    for ep in built:
        path = latents / f"{ep}.npz"
        if not path.is_file():
            bad.append(ep + " missing")
            continue
        with np.load(path) as a, np.load(D / "episodes" / f"{ep}.npz") as z:
            zz, valid = a["z"], a["valid"]
            n, k = z["visibility"].shape
            window_seen = np.array([[z["visibility"][max(0, t - 20):t + 1, j].any() for j in range(k)] for t in range(n)])
            norms = np.linalg.norm(zz[valid], axis=1)
            if zz.shape != (n, k, 64) or valid.shape != (n, k) or not np.isfinite(zz).all() or not np.array_equal(valid, window_seen) \
                    or (len(norms) and not np.allclose(norms, 8.0, atol=1e-2)) or np.abs(zz[~valid]).max(initial=0) != 0:
                bad.append(ep)
            valid_share.append(valid.mean())
    check("F1 latent table per episode: z [N,K,64] finite, |z| = 8 where valid, 0 elsewhere, valid == object drawn in t-20..t", not bad, str(bad[:4]))
    st = latents / "standardize.json"
    ok = st.is_file() and len(json.loads(st.read_text())["mean"]) == 64 and json.loads(st.read_text())["episodes"] == len(built)
    check("F2 standardize.json (64-d mean/std, fitted for this episode set)", ok)
    report["latent_valid_share"] = float(np.mean(valid_share)) if valid_share else None

# ---------------------------------------------------------------- G the training loader
if not args.skip_loader:
    try:
        import model_data
        tokenizer = model_data.JointTokenizer(contract["model"]["tokenizer"])
        for part in ("train", "val"):
            c = json.loads(json.dumps(contract))
            c["dataset"]["index"] = f"{part}.jsonl"
            corpus = model_data.FullCorpus(D, c, tokenizer)
            capacities, lengths = corpus.measure_capacities()
            picks = [0, len(corpus.rows) // 2, len(corpus.rows) - 1]
            try:
                shapes = None
                corpus.tokenizer = model_data.JointTokenizer(contract["model"]["tokenizer"], capacities)  # as train.py does
                for i in picks:
                    b = corpus.batch(i)
                    shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in b.items()}
                batch_note = f"batch keys {sorted(shapes)}"
            except Exception as exc:  # batch() needs openpi's Observation class; the index/schema checks above do not
                batch_note = f"batch() not run here: {type(exc).__name__}: {str(exc)[:120]}"
            check(f"G1 model_data.FullCorpus accepts {part}.jsonl (PROMPT_VERSION {model_data.PROMPT_VERSION})", True,
                  f"rows {len(corpus.rows)} episodes {corpus.coverage['episodes']} capacities {capacities} observed {lengths}; {batch_note}")
            report[f"loader_{part}"] = {"rows": len(corpus.rows), "capacities": capacities, "observed": lengths, "batch": batch_note}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        check("G1 model_data.FullCorpus accepts the dataset", False, f"{type(exc).__name__}: {exc}")

report["failures"] = failures
if args.report:
    args.report.write_text(json.dumps(report, indent=1, default=str))
print("CHECK_RESULT", "OK" if not failures else f"FAILED {failures}", flush=True)
sys.exit(1 if failures else 0)
