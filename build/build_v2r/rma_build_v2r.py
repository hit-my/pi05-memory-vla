"""Row writer v2r: rma_build.py with the tracker replaced by STORED object tracks, plus the clean front image.

What is the same as rma_build.py (imported from it, not copied: lerobot_episodes, lerobot_load, held_table, directions,
action_text): episode loading from the LeRobot arrays, the Action-line rule, wrist image, row fields, episode npz.
What changes:
  * centres / visibility come from --tracks <dir>/<ep>.npz (overlay_v2r.load_tracks), not from a live tracker;
    visibility[t,k] = "object k is drawn at frame t" (finite centre, not retired). The Action line's "occluded"
    (drawn in fewer than half of the frames t-20..t) is recomputed from it; it only looks at frames <= t.
  * images/<ep>/NNNNNN.png is drawn by the unchanged CanonicalRenderer from those tracks (overlay_v2r.OverlayState);
  * images/<ep>/NNNNNN_raw.png = the clean front frame, 256 -> 224 PIL BILINEAR; every row gets raw_image_path;
  * rows go to OUT/rows_base/<ep>.jsonl (rma_build format: whole subtask list in the init). The causal Subgoal labels
    are applied afterwards by rma_assemble_v2r.py (same two-stage order as dataset -> dataset_causal before).
Nothing else is rewritten (states, actions, first-frame finger reading: as stored in the LeRobot arrays).

Usage (simulator venv, needs LIBERO_SIMULATOR_DRIVER_ROOT and PLAYWRIGHT_BROWSERS_PATH; see env_v2r.sh):
  rma_build_v2r.py --raw LR_EPS --tracks TRACKS --out OUT --shard I --shards N [--episodes-file F] [--tasks 1,2]
                   [--code ~/pi05-abc/code_causal_v2] [--rebuild-stale]
Resumable: an episode is done when OUT/rows_base/<ep>.jsonl exists (written last, atomically). OUT/log/<ep>.json records
the sha256 of the track file used; a done episode whose track file has changed stops the build unless --rebuild-stale.
Writes OUT/images/<ep>/NNNNNN{,_wrist,_raw}.png, OUT/episodes/<ep>.npz, OUT/rows_base/<ep>.jsonl, OUT/log/<ep>.json
"""
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ in a code directory a training run hashes

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_v2r  # noqa: E402

BUILDER = "rma_build_v2r/1"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--episodes-file", type=Path, help="one episode id per line; default: every episode in RAW")
    p.add_argument("--tasks", default="", help="comma-separated task ids (default all)")
    p.add_argument("--code", type=Path, default=Path.home() / "pi05-abc/code_causal_v2")
    p.add_argument("--rebuild-stale", action="store_true")
    return p.parse_args()


def import_rma_build(code, raw, out):
    """rma_build.py reads sys.argv at import; give it RAW/OUT and take its functions unchanged."""
    os.environ["SOURCE"] = "lerobot"
    sys.path.insert(0, str(code))
    saved = sys.argv
    sys.argv = [str(code / "rma_build.py"), str(raw), str(out), "0", "1"]
    try:
        spec = importlib.util.spec_from_file_location("rma_build_orig", code / "rma_build.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def main():
    args = parse()
    os.environ["TASKS"] = args.tasks
    rb = import_rma_build(args.code.resolve(), args.raw, args.out)
    import rma_spec
    from ledger_v2 import delta_text, ledger_from_count
    from memory_format import INITIAL_MEMORY, render_previous
    from data_common import make_tracks          # simulator driver dir (on sys.path via rma_perception)
    from render_adapter import CanonicalRenderer

    jobs = rb.lerobot_episodes()
    if args.episodes_file:
        wanted = [x.strip() for x in args.episodes_file.read_text().split() if x.strip()]
        known = {j[2] for j in jobs}
        absent = [x for x in wanted if x not in known]
        if absent:
            raise SystemExit(f"episodes not in RAW: {absent[:5]}")
        jobs = [j for j in jobs if j[2] in set(wanted)]
    jobs = [j for i, j in enumerate(jobs) if i % args.shards == args.shard]
    print("episodes", len(jobs), "tracks", args.tracks, flush=True)
    out = args.out
    for sub in ("images", "episodes", "rows_base", "log"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    renderer_js = args.code.resolve() / "assets/entity_paths_model_renderer.js"
    renderer_sha = overlay_v2r.sha256_file(renderer_js)
    renderer = None
    for task_id, seed, stem in jobs:
        ep = f"t{task_id:02d}_seed{seed}"
        track_path = args.tracks / f"{ep}.npz"
        if not track_path.is_file():
            raise SystemExit(f"no track file for {ep}: {track_path}")
        track_sha = overlay_v2r.sha256_file(track_path)
        done, log_path = out / "rows_base" / f"{ep}.jsonl", out / "log" / f"{ep}.json"
        if done.exists():
            used = json.loads(log_path.read_text())["tracks"]["sha256"]
            if used == track_sha:
                continue
            if not args.rebuild_stale:
                raise SystemExit(f"{ep} was built from a different track file ({used[:12]} != {track_sha[:12]}); "
                                 f"use a fresh OUT or --rebuild-stale")
            done.unlink()
            print("STALE", ep, "rebuilding", flush=True)
        started = time.time()
        rgb, wrist, states, actions, ends, texts = rb.lerobot_load(stem)
        n = len(rgb)
        names = rma_spec.names(task_id)
        registry = rma_spec.registry(task_id)
        track_centres, visible, retired, track_stats = overlay_v2r.load_tracks(track_path, n, len(names))
        if renderer is None:
            renderer = CanonicalRenderer(renderer_js)
        state = overlay_v2r.OverlayState(registry, make_tracks)
        folder = out / "images" / ep
        folder.mkdir(parents=True, exist_ok=True)
        render_s = 0.0
        for t in range(n):
            frame_rgb = np.ascontiguousarray(rgb[t], np.uint8)
            tracks, frame, start = state.step(track_centres[t], retired[t])
            t0 = time.perf_counter()
            png, _ = renderer.render(frame_rgb, frame, start, registry, tracks)
            render_s += time.perf_counter() - t0
            (folder / f"{t:06d}.png").write_bytes(png)
            Image.fromarray(wrist[t]).resize((224, 224), Image.BILINEAR).save(folder / f"{t:06d}_wrist.png")
            Image.fromarray(frame_rgb).resize((224, 224), Image.BILINEAR).save(folder / f"{t:06d}_raw.png")
        centres = np.where(visible[..., None], track_centres, 0.0).astype(np.float32)  # rma_build convention: 0 where not seen
        held, events = rb.held_table(task_id, actions, ends, texts, len(names))
        confirmed = np.array([sum(1 for e in ends if e <= t) for t in range(n)])
        items = texts
        acts = [rb.action_text(t, states, held, events, visible, names) for t in range(n)]
        rows = []
        for t in range(n):
            previous = INITIAL_MEMORY if t < 20 else render_previous(acts[t - 20], ledger_from_count(items, int(confirmed[t - 20])))
            delta = delta_text(None if t < 20 else int(confirmed[t - 20]), int(confirmed[t]), items)
            task_text = delta.split("\nTask:")[1]
            target = {"action_text": acts[t], "subgoal_delta": delta.split("\n")[0][len("Subgoal:"):], "task_text": task_text,
                      "confirmed_goals": int(confirmed[t]), "total_goals": len(items), "action_reliable": True}
            rows.append({"sample_id": f"{ep}_{t:06d}", "task_id": task_id, "episode_id": ep, "observation_index": t,
                         "history_start": max(0, t - 20), "history_end": t, "instruction": rma_spec.TASKS[task_id]["prompt"],
                         "previous_memory_text": previous, "previous_query_index": t - 20 if t >= 20 else None,
                         "memory_update_interval": 20,
                         "memory_target_text": f"Action:{acts[t]}\n{delta}", "memory_target": target,
                         "image_path": f"images/{ep}/{t:06d}.png", "wrist_image_path": f"images/{ep}/{t:06d}_wrist.png",
                         "episode_npz": f"episodes/{ep}.npz", "action_start": t, "valid_action_steps": min(20, n - t),
                         "action_label_reliable": True,
                         "raw_image_path": f"images/{ep}/{t:06d}_raw.png"})
        np.savez_compressed(out / "episodes" / f"{ep}.npz", states=states, actions=actions, centres=centres,
                            visibility=visible, action_text=np.array(acts), confirmed_goals=confirmed, ends=np.array(ends),
                            retired=retired)
        seconds = round(time.time() - started, 1)
        log_path.write_text(json.dumps({
            "frames": n, "items": items, "builder": BUILDER,
            "tracks": {"dir": str(args.tracks), "file": track_path.name, "sha256": track_sha, **track_stats},
            "renderer_js_sha256": renderer_sha, "visible": visible.mean(axis=0).round(3).tolist(),
            "seconds": seconds, "render_seconds": round(render_s, 1)}))
        temporary = done.with_suffix(".jsonl.tmp")
        with open(temporary, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        temporary.replace(done)  # the done marker
        print("DONE", ep, n, "visible", visible.mean(axis=0).round(2).tolist(), "s", seconds, "render", round(render_s, 1), flush=True)
    if renderer is not None:
        renderer.close()
    print("BUILD_DONE", args.shard, flush=True)


if __name__ == "__main__":
    main()
