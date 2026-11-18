"""Tracker process for the motion-prior tracker. Same pipes as tracker_server.py, plus proprioception:
"RESET <n_bowls> x y z g0 g1" and "STEP x y z g0 g1", followed by the RGB bytes. RGB and proprioception only."""
import argparse
import json
import sys

import numpy as np

from tracker_core import SIZE
from tracker_motion import MotionOptions, MotionTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args()
    tracker = MotionTracker(args.root, MotionOptions(**json.loads(args.options)))
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    sink.write(b"READY\n")
    sink.flush()
    while True:
        line = source.readline()
        if not line:
            return
        command = line.decode().split()
        rgb = np.frombuffer(source.read(SIZE * SIZE * 3), np.uint8).reshape(SIZE, SIZE, 3)
        if command[0] == "RESET":
            report, labels = tracker.initialize(rgb, int(command[1]), np.array(command[2:], float))
            sink.write(json.dumps(report).encode() + b"\n")
        else:
            labels = tracker.step(rgb, np.array(command[1:], float))
        sink.write(labels.tobytes())
        sink.flush()


if __name__ == "__main__":
    main()
