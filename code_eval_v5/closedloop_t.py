"""Closed loop where the trajectory image comes from an RGB-only tracker instead of simulator segmentation.

The simulator still does the physics and the official scoring. Only the segmentation returned by
OfficialEnvironment.observe changes, and everything downstream (measure_entities, make_tracks, renderer,
policy) is untouched. Simulator segmentation goes only to the evaluation log, never to the model.

Env vars:
  TRACK_MODE    track (default) | dump. dump keeps simulator segmentation in the loop and saves RGB and
                segmentation per episode to TRACK_DIR, for offline tuning of the tracker.
  TRACK_DIR     per-process folder for track.jsonl, init.jsonl or dumps
  TRACKER_PYTHON, TRACKER_ROOT, TRACKER_CODE, TRACKER_OPTIONS (JSON)
  INJECT_TYPE, INJECT_LOG as in closedloop_inject.py (optional)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from closedloop_ledger import install

SIZE = 256
SOURCE = "RGB-only tracker (Grounding DINO, SAM2.1, CUTIE); simulator segmentation used for evaluation logs only"


class TrackerClient:
    def __init__(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        code = os.environ["TRACKER_CODE"]
        env["PYTHONPATH"] = os.environ.get("TRACKER_PYTHONPATH", code)
        if "TRACKER_CUDA" in os.environ:  # empty runs the tracker on CPU, leaving the simulator's GPU setting alone
            env["CUDA_VISIBLE_DEVICES"] = os.environ["TRACKER_CUDA"]
        self.process = subprocess.Popen(
            [os.environ["TRACKER_PYTHON"], str(Path(code) / "tracker_server.py"), "--root", os.environ["TRACKER_ROOT"],
             "--options", os.environ.get("TRACKER_OPTIONS", "{}")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=code)
        ready = self.process.stdout.readline()
        if ready.strip() != b"READY":
            raise RuntimeError(f"Tracker failed to start: {ready!r}")

    def _send(self, command, rgb):
        self.process.stdin.write(command.encode() + b"\n" + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.process.stdin.flush()

    def _labels(self):
        data = self.process.stdout.read(SIZE * SIZE)
        if len(data) != SIZE * SIZE:
            raise RuntimeError(f"Tracker process ended ({self.process.poll()})")
        return np.frombuffer(data, np.uint8).reshape(SIZE, SIZE)

    def reset(self, rgb, n_bowls):
        self._send(f"RESET {n_bowls}", rgb)
        report = json.loads(self.process.stdout.readline())
        return report, self._labels()

    def step(self, rgb):
        self._send("STEP", rgb)
        return self._labels()


def centre(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.mean()), float(ys.mean())] if len(xs) else None


def install_tracking(simulate):
    mode = os.environ.get("TRACK_MODE", "track")
    folder = Path(os.environ["TRACK_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    base_class = simulate.OfficialEnvironment
    client = TrackerClient() if mode == "track" else None
    original_save_json = simulate.save_json

    def save_json(path, value):
        if mode == "track" and isinstance(value, dict) and "perception_source" in value:
            value = {**value, "perception_source": SOURCE}
        return original_save_json(path, value)
    simulate.save_json = save_json

    class TrackedEnvironment(base_class):
        created = 0

        def __init__(self, source, spec, initial, instruction):
            super().__init__(source, spec, initial, instruction)
            self.track_number = TrackedEnvironment.created
            TrackedEnvironment.created += 1
            self.track_frame = 0
            self.track_task = spec["task_id"]
            # Left to right matches ascending ids in these scenes, so colours stay what the model was trained on.
            self.bowl_sids = sorted((sid for sid, row in self.registry.items() if row["category"] == "bowl"), key=int)
            self.dump_rgb, self.dump_seg = [], []
            self.last_masks = {}

        def observe(self):
            main, wrist, truth, state = super().observe()
            frame = self.track_frame
            self.track_frame += 1
            if mode == "passthrough":  # null control: same code path, simulator segmentation
                return main, wrist, truth, state
            if mode == "dump":
                self.dump_rgb.append(main.copy())
                self.dump_seg.append(truth.copy())
                return main, wrist, truth, state
            if frame == 0:
                report, labels = client.reset(main, len(self.bowl_sids))
                with open(folder / "init.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"episode_number": self.track_number, "task_id": self.track_task, **report}) + "\n")
            else:
                labels = client.step(main)
            segmentation = np.zeros(truth.shape, dtype=np.uint8)
            row = {"episode_number": self.track_number, "task_id": self.track_task, "frame": frame, "bowls": {}}
            hold = int(os.environ.get("TRACK_HOLD", "0"))
            for index, sid in enumerate(self.bowl_sids, 1):
                predicted = labels == index
                if hold:  # a vanished mask keeps its last place for a while, so the bowl does not drop out of the picture
                    if predicted.sum() >= 80:
                        self.last_masks[index] = (predicted.copy(), frame)
                    elif index in self.last_masks and frame - self.last_masks[index][1] <= hold:
                        predicted = self.last_masks[index][0] & (labels == 0)
                segmentation[predicted] = int(sid)
                actual = truth == int(sid)
                union = int((predicted | actual).sum())
                row["bowls"][sid] = {"pred": centre(predicted), "true": centre(actual), "pred_area": int(predicted.sum()),
                                     "true_area": int(actual.sum()), "iou": (int((predicted & actual).sum()) / union) if union else None}
            with open(folder / "track.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            return main, wrist, segmentation, state

        def close(self):
            if mode == "dump" and self.dump_rgb:
                np.savez_compressed(folder / f"t{self.track_task:02d}_{self.track_number:03d}.npz",
                                    rgb=np.asarray(self.dump_rgb, np.uint8), seg=np.asarray(self.dump_seg, np.uint8),
                                    bowl_sids=np.asarray([int(s) for s in self.bowl_sids]), task_id=self.track_task)
            super().close()

    simulate.OfficialEnvironment = TrackedEnvironment


def install_training_states(simulate):
    """Closed loop on TRAINING initial states, so that iterating never touches the held-out states.
    TRAIN_OFFSET picks which ones; initial states and instructions come from the replay inputs."""
    home = Path.home()
    inputs = home / "pi05-ledger-A/inputs"
    manifest = json.loads((home / "pi05-abc/dataset/split_manifest.orig.json").read_text())
    train = {f"t{int(e['task_id']):02d}_{e['episode_id']}" for e in manifest["episodes"] if e["split"] == "train"}
    index = {e["name"]: e for e in json.loads((inputs / "index.json").read_text())["episodes"]}
    lost = set(json.loads((home / "pi05-ledger-A/labels_A2/report.json").read_text())["summary"]["episodes_with_lost_replay"])
    offset = int(os.environ.get("TRAIN_OFFSET", "3"))

    def prepare_episodes(args, contract, spec, indexed):
        prefix = f"t{spec['task_id']:02d}"
        names = sorted((n for n in train - lost if n.startswith(prefix)), key=lambda n: int(n.split("_")[-1]))
        step = max(len(names) // args.episodes_per_task, 1)
        chosen = [names[(i * step + offset) % len(names)] for i in range(args.episodes_per_task)]
        items = []
        for name in chosen:
            with np.load(inputs / f"{name}.npz", allow_pickle=False) as z:
                initial = np.asarray(z["initial_state"], dtype=np.float64)
            items.append({"episode_id": name.split("_", 1)[1], "initial_state": initial, "instruction": index[name]["instruction"],
                          "provenance": {"split": "training-closed-loop", "held_out": False, "source_kind": "training-inputs"}})
        return items, {"split": "training-closed-loop", "held_out": False, "selected_episodes": chosen}
    simulate.prepare_episodes = prepare_episodes


if __name__ == "__main__":
    root = Path(os.environ["LIBERO_SIMULATOR_DRIVER_ROOT"])
    sys.path.insert(0, str(root))
    import simulate as simulate_module

    torch_site = os.environ["SIMULATOR_TORCH_SITE"]
    sys.path.insert(0, torch_site)
    try:
        import torch  # noqa: F401
    finally:
        sys.path.remove(torch_site)
    install(simulate_module)
    if os.environ.get("TRAIN_STATES"):
        install_training_states(simulate_module)
    if os.environ.get("INJECT_TYPE"):
        from closedloop_inject import install_injection
        install_injection(simulate_module, os.environ["INJECT_TYPE"], os.environ["INJECT_LOG"])
    install_tracking(simulate_module)
    simulate_module.main()
