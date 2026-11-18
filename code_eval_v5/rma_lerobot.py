"""Convert the LeRobot RoboMemArena dataset (yinchenghust/robomemarena_lerobot) into per-episode arrays.

Usage: rma_lerobot.py LEROBOT OUT SHARD NSHARDS
Each video file is decoded once, sequentially; its episodes are sliced by from/to timestamps (20 fps).
Writes OUT/<camera>/tNN_seedS.npy (uint8 T x 256 x 256 x 3) for camera in front, wrist, and once per episode
OUT/meta/tNN_seedS.npz with states (T, 8), actions (T, 7), subtask (T,), keyframe (T,) and the subtask texts.
Work is sharded by video file, so a shard writes whole files; the meta pass runs in shard 0.
"""
import glob
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

src, out, shard, shards = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
episodes = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(src / "meta/episodes/*/*.parquet")))])
sources = pd.read_parquet(src / "meta/episode_sources.parquet").set_index("episode_index")
info = json.loads((src / "meta/info.json").read_text())
fps = info["fps"]


def name(ep):
    s = sources.loc[ep]
    return f"t{int(s.benchmark_task_id):02d}_seed{int(s.collection_seed)}"


def video_jobs():
    jobs = []
    for camera in ("front", "wrist"):
        key = f"videos/observation.images.{camera}"
        for (chunk, index), group in episodes.groupby([f"{key}/chunk_index", f"{key}/file_index"]):
            path = src / info["video_path"].format(video_key=f"observation.images.{camera}", chunk_index=chunk, file_index=index)
            spans = [(int(r.episode_index), float(r[f"{key}/from_timestamp"]), int(r.length)) for _, r in group.iterrows()]
            jobs.append((camera, path, spans))
    return jobs


def decode(camera, path, spans):
    """Decode one file; frame i of the file has timestamp i / fps."""
    wanted = {round(start * fps): (ep, length) for ep, start, length in spans}
    todo = [ep for ep, _, _ in spans if not (out / camera / f"{name(ep)}.npy").exists()]
    if not todo:
        return
    (out / camera).mkdir(parents=True, exist_ok=True)
    current, frames, remaining = None, [], 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if current is None and i in wanted:
                current, remaining = wanted[i]
                frames = []
            if current is not None:
                frames.append(frame.to_ndarray(format="rgb24"))
                remaining -= 1
                if remaining == 0:
                    target = out / camera / f"{name(current)}.npy"
                    np.save(str(target) + ".tmp.npy", np.stack(frames))
                    Path(str(target) + ".tmp.npy").rename(target)
                    current = None
    print("decoded", camera, path.name, len(spans), flush=True)


def meta():
    subtasks = pd.read_parquet(src / "meta/subtasks.parquet")
    texts = {int(v): k for k, v in subtasks["subtask_index"].items()}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(str(src / "data/*/*.parquet"))):
        data = pd.read_parquet(path)
        for ep, rows in data.groupby("episode_index"):
            rows = rows.sort_values("frame_index")
            sub = rows["subtask_index"].to_numpy().astype(np.int64)
            order = [int(s) for k, s in enumerate(sub) if k == 0 or s != sub[k - 1]]
            np.savez(out / "meta" / f"{name(ep)}.npz",
                     states=np.stack(rows["observation.state"].to_numpy()).astype(np.float32),
                     actions=np.stack(rows["action"].to_numpy()).astype(np.float32),
                     subtask=sub, keyframe=rows["is_keyframe"].to_numpy().astype(bool).reshape(-1),
                     subtask_order=np.array(order), subtask_texts=np.array([texts[s] for s in order]),
                     task_id=int(sources.loc[ep].benchmark_task_id), seed=int(sources.loc[ep].collection_seed))
        print("meta", Path(path).name, flush=True)


if __name__ == "__main__":
    if shard == 0:
        meta()
    for k, job in enumerate(video_jobs()):
        if k % shards == shard:
            decode(*job)
    print("CONVERT_DONE", shard, flush=True)
