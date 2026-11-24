"""Step 2: tracks + frames -> the trainer's row format (overlay / wrist / clean images, episode arrays, memory labels).

  real_build.py --raw RAW --out OUT --code CODE [--shard I --shards N]
RAW  = output of real_extract.py + real_track.py (front/, wrist/, meta/, tracks/)
CODE = the training code directory (memory_format.py, ledger_v2.py, causal_labels.py, assets/, overlay_v2r.py)
Needs the overlay renderer: LIBERO_SIMULATOR_DRIVER_ROOT (data_common.py, render_adapter.py) and a Playwright
Chromium (PLAYWRIGHT_BROWSERS_PATH). Resumable: an episode is done when OUT/rows/<ep>.jsonl exists.

Labels (the same rules as the simulator dataset, see README):
  * checklist = real_extract.py's items plus one observation item per cup ("far cup: has block" / "far cup: empty"),
    inserted by causal_labels.with_observations: it is revealed and ticked at the frame the spot under the lifted cup is
    in view, never earlier. "pick up block N" is not the same in every demonstration, so it is "?" until it starts
    (causal_labels.task_prior over the TRAINING episodes); what decides N is then already in the checklist;
  * item k is ticked from its last frame on, Task:Complete once all are ticked (causal_labels.episode_labels, unchanged);
  * window Action line (last 20 frames), one clause per tracked object (three cups, three numbered blocks): picked /
    placed at the gripper events of the four grasps, direction words from the motion of the held object's centre in the
    cropped top image, otherwise still / occluded;
  * the Action line that is trained is the accumulated chain within the current item (memory_format.accumulate_action,
    the "scheme B" rule of the simulator run); the window line is kept as window_action_text.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_spec as S  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--raw", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--code", type=Path, required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--shards", type=int, default=1)
p.add_argument("--no-images", action="store_true", help="labels only (no renderer needed)")
args = p.parse_args()
sys.path.insert(0, str(args.code.resolve()))
import causal_labels as cl  # noqa: E402
import overlay_v2r  # noqa: E402
from memory_format import INITIAL_MEMORY, accumulate_action, action_entities, cap_action  # noqa: E402

INTERVAL = 20


def directions(xy):
    """xy [M,2] centres of the held bowl over the window (nan = not seen) -> direction words, image frame."""
    words = []
    for half in (xy[: len(xy) // 2 + 1], xy[len(xy) // 2:]):
        seen = half[np.isfinite(half).all(-1)]
        if len(seen) < 2:
            continue
        d = seen[-1] - seen[0]
        axis = int(np.argmax(np.abs(d)))
        if abs(d[axis]) >= S.MOTION_PIXELS:
            word = ("right" if d[0] > 0 else "left") if axis == 0 else ("backward" if d[1] > 0 else "forward")
            if not words or words[-1] != word:
                words.append(word)
    return words


def window_action(t, centres, visible, held, events):
    start = max(0, t - INTERVAL)
    clauses = []
    for k, name in enumerate(S.NAMES):
        evs = [e for f in range(start + 1, t + 1) for (kk, e) in events.get(f, []) if kk == k]
        moved = directions(centres[start:t + 1, k]) if held[start:t + 1, k].any() else []
        if moved:
            body = " -> ".join(moved) + "".join(f" ; {e}" for e in evs)
        elif evs:
            body = " ; ".join(evs)
        elif visible[start:t + 1, k].mean() < 0.5:
            body = "occluded"
        else:
            body = "still"
        clauses.append(f"{name}: {body}")
    return " | ".join(clauses)


def main():
    raw, out = args.raw, args.out
    names = sorted((path.stem for path in (raw / "tracks").glob("t*_seed*.npz")), key=lambda s: int(s.split("seed")[1]))
    names = [n for i, n in enumerate(names) if i % args.shards == args.shard]
    for sub in ("images", "episodes", "rows", "log"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    registry = S.registry()
    every = sorted((path.stem for path in (raw / "meta").glob("t*_seed*.npz")), key=lambda s: int(s.split("seed")[1]))
    checklists = {}
    for ep in every:
        meta = np.load(raw / "meta" / f"{ep}.npz")
        observed = [tuple(o) for o in json.loads(str(meta["observed"]))]
        checklists[ep] = cl.with_observations([str(x) for x in meta["items"]], [int(e) for e in meta["ends"]], observed) \
            + (cl.observation_positions(observed),)
    training = [ep for ep in every if int(ep.split("seed")[1]) not in S.VAL_EPISODES]
    prior = cl.task_prior([checklists[ep][0] for ep in training], set().union(*(checklists[ep][2] for ep in training)))
    (out / "prior.json").write_text(json.dumps({"fixed": {str(k): v for k, v in prior.fixed.items()}, "length": prior.length,
                                                "demos": prior.demos}, indent=1))
    renderer = make_tracks = None
    if not args.no_images:
        sys.path.insert(0, os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
        from data_common import make_tracks
        from render_adapter import CanonicalRenderer
        renderer = CanonicalRenderer(args.code.resolve() / "assets/entity_paths_model_renderer.js")
    for ep in names:
        done = out / "rows" / f"{ep}.jsonl"
        if done.exists():
            continue
        started = time.time()
        meta = np.load(raw / "meta" / f"{ep}.npz")
        states, actions = meta["states"], meta["actions"]
        closed, ev_frames = meta["closed"], [int(e) for e in meta["events"]]
        items, ends, _ = checklists[ep]
        n = len(states)
        centres, visible, retired, stats = overlay_v2r.load_tracks(raw / "tracks" / f"{ep}.npz", n, len(S.NAMES))
        if not args.no_images:
            front, wrist = np.load(raw / "front" / f"{ep}.npy"), np.load(raw / "wrist" / f"{ep}.npy")
            state = overlay_v2r.OverlayState(registry, make_tracks)
            folder = out / "images" / ep
            folder.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                rgb = np.ascontiguousarray(front[t], np.uint8)
                tracks, frame, start = state.step(centres[t], retired[t])
                png, _ = renderer.render(rgb, frame, start, registry, tracks)
                (folder / f"{t:06d}.png").write_bytes(png)
                Image.fromarray(wrist[t]).resize((224, 224), Image.BILINEAR).save(folder / f"{t:06d}_wrist.png")
                Image.fromarray(rgb).resize((224, 224), Image.BILINEAR).save(folder / f"{t:06d}_raw.png")
        # held object and events from the gripper: grasp i (i = 0, 1, 2) carries cup i, the fourth the numbered block = the count
        held = np.zeros((n, len(S.NAMES)), bool)
        events = {}
        for i in range(4):
            k, close, release = (i if i < 3 else S.BLOCK_OBJECT[int(meta["count"])]), ev_frames[2 * i], ev_frames[2 * i + 1]
            held[close:release, k] = True
            events.setdefault(close, []).append((k, "picked"))
            events.setdefault(release, []).append((k, "placed"))
        window = [window_action(t, centres, visible, held, events) for t in range(n)]
        labels = cl.episode_labels(prior, items, ends, n, window)
        confirmed = [sum(it.done for it in now.items) for _, _, now in labels]
        acc = [None] * n
        rows = []
        for t in range(n):
            reset = t >= INTERVAL and confirmed[t] > confirmed[t - INTERVAL]
            acc[t] = accumulate_action(acc[t - INTERVAL] if t >= INTERVAL else None, window[t], reset)
            assert action_entities(acc[t], "A") is not None, (ep, t, acc[t])
            previous, delta, _ = labels[t]
            if t >= INTERVAL:
                previous = "\n".join(["Action:" + cap_action(acc[t - INTERVAL])] + previous.split("\n")[1:])
            else:
                assert previous == INITIAL_MEMORY
            sub, task_line = delta.split("\n")
            target = {"action_text": acc[t], "window_action_text": window[t], "subgoal_delta": sub[len("Subgoal:"):],
                      "task_text": task_line[len("Task:"):], "confirmed_goals": confirmed[t], "total_goals": len(items),
                      "action_reliable": True}
            rows.append({"sample_id": f"{ep}_{t:06d}", "task_id": S.TASK_ID, "episode_id": ep, "observation_index": t,
                         "history_start": max(0, t - INTERVAL), "history_end": t, "instruction": S.PROMPT,
                         "previous_memory_text": previous, "previous_query_index": t - INTERVAL if t >= INTERVAL else None,
                         "memory_update_interval": INTERVAL, "memory_target_text": f"Action:{acc[t]}\n{delta}",
                         "memory_target": target, "image_path": f"images/{ep}/{t:06d}.png",
                         "wrist_image_path": f"images/{ep}/{t:06d}_wrist.png", "episode_npz": f"episodes/{ep}.npz",
                         "action_start": t, "valid_action_steps": min(INTERVAL, n - t), "action_label_reliable": True,
                         "raw_image_path": f"images/{ep}/{t:06d}_raw.png"})
        np.savez_compressed(out / "episodes" / f"{ep}.npz", states=states, actions=actions,
                            centres=np.where(visible[..., None], centres, 0.0).astype(np.float32), visibility=visible,
                            action_text=np.array(window), confirmed_goals=np.array(confirmed), ends=np.array(ends),
                            retired=retired, state14=meta["state14"], action14=meta["action14"], closed=closed,
                            gripper_events=np.array(ev_frames), has_block=meta["has_block"], view=meta["view"])
        (out / "log" / f"{ep}.json").write_text(json.dumps({
            "frames": n, "items": items, "ends": ends, "has_block": [bool(h) for h in meta["has_block"]], "gripper_events": ev_frames, "tracks": stats,
            "visible": visible.mean(axis=0).round(3).tolist(), "seconds": round(time.time() - started, 1)}))
        temporary = done.with_suffix(".jsonl.tmp")
        with open(temporary, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        temporary.replace(done)
        print("DONE", ep, n, "visible", visible.mean(axis=0).round(2).tolist(), "s", round(time.time() - started, 1), flush=True)
    if renderer is not None:
        renderer.close()
    print("BUILD_DONE", args.shard, flush=True)


if __name__ == "__main__":
    main()
