"""Draw the first-frame detections for a few episodes per task, to check the named-object detector by eye.
Usage (tracker venv): rma_detect_check.py RAW OUT.png --root ASSETS"""
import glob
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

import rma_spec
from rma_tracker import NamedTracker
from tracker_core import Options

raw, out = Path(sys.argv[1]), sys.argv[2]
tracker = NamedTracker(sys.argv[sys.argv.index("--root") + 1], Options(selection="detector", max_mem_frames=10))
rows = []
for task_id, spec in sorted(rma_spec.TASKS.items()):
    files = sorted(glob.glob(str(raw / spec["folder"] / "*_0_seed*_task*.hdf5")))[:4]
    tiles = []
    for path in files:
        with h5py.File(path, "r") as f:
            rgb = f["data/demo_0/obs/agentview_rgb"][0]
        report, labels = tracker.initialize_named(rgb, rma_spec.detector_objects(task_id))
        image = rgb.copy()
        colours = [(0, 93, 170), (199, 86, 0), (116, 66, 184)]
        for k, obj in enumerate(report["objects"]):
            image[labels == k + 1] = (0.5 * image[labels == k + 1] + 0.5 * np.array(colours[k])).astype(np.uint8)
            if obj["box"]:
                x0, y0, x1, y1 = map(int, obj["box"])
                cv2.rectangle(image, (x0, y0), (x1, y1), colours[k], 1)
        print(task_id, Path(path).name, [(o["prompt"], o["score"], o["pixels"]) for o in report["objects"]], flush=True)
        tiles.append(image)
    if tiles:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(tiles, 1))
cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
