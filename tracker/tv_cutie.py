"""Baseline: the current pipeline (clicks -> SAM2.1 mask on frame 0 -> causal CUTIE), re-run to keep the masks.
Usage: tv_cutie.py SHARD NSHARDS   (tracker venv; PYTHONPATH gpu_extra)"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
from tv_common import ROOT, episodes, load, save_masks, task_of  # noqa: E402
import rma_spec  # noqa: E402
from rma_tracker import NamedTracker  # noqa: E402
from tracker_core import Options  # noqa: E402

shard, shards = int(sys.argv[1]), int(sys.argv[2])
tracker = NamedTracker("xxxx/pi05-abc/tracker/assets", Options(selection="detector", max_mem_frames=10))
for i, ep in enumerate(episodes()):
    if i % shards != shard or (ROOT / "out/cutie" / f"{ep}.npz").exists():
        continue
    rgb, _, _ = load(ep)
    k = len(rma_spec.TASKS[task_of(ep)]["objects"])
    labels = np.zeros(rgb.shape[:3], np.uint8)
    _, labels[0] = tracker.initialize_named(rgb[0].copy(), rma_spec.detector_objects(task_of(ep)))
    for t in range(1, len(rgb)):
        labels[t] = tracker.step(rgb[t].copy())
    masks = np.stack([labels == j + 1 for j in range(k)], axis=1)
    save_masks("cutie", ep, masks)
    print("cutie", ep, len(rgb), masks.sum(axis=(0, 2, 3)), flush=True)
