"""Tracker process for closed loop. Runs in its own Python 3.12 venv, talks over binary pipes.

Request:  one text line, then 256*256*3 RGB bytes.  "RESET <n_bowls>" starts an episode, "STEP" continues it.
Response: for RESET a JSON line (report); then, for both, 256*256 uint8 labels (0 background, 1..K bowls left to right).
Only RGB ever reaches this process.
"""
import argparse
import json
import sys

import numpy as np

from tracker_core import SIZE, Options, TrackerCore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--options", default="{}", help="JSON overrides for tracker_core.Options")
    args = parser.parse_args()
    tracker = TrackerCore(args.root, Options(**json.loads(args.options)))
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    sink.write(b"READY\n")
    sink.flush()
    while True:
        line = source.readline()
        if not line:
            return
        command = line.decode().split()
        data = source.read(SIZE * SIZE * 3)
        rgb = np.frombuffer(data, np.uint8).reshape(SIZE, SIZE, 3)
        if command[0] == "RESET":
            report, labels = tracker.initialize(rgb, int(command[1]))
            sink.write(json.dumps(report).encode() + b"\n")
        else:
            labels = tracker.step(rgb)
        sink.write(labels.tobytes())
        sink.flush()


if __name__ == "__main__":
    main()
