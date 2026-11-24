"""Trajectory overlay from STORED object tracks (v2r) -- the one place that turns a track file into renderer input.

Track file (one npz per episode, e.g. /path/to/<ep>.npz):
    centres    float32 [N,K,2]  256-px source-image coordinates (x, y); nan = not drawn at this frame
    visibility bool    [N,K]
    retired    bool    [N,K]    sticky; a retired object is never drawn again

Drawing rule (user decision for this round):
  * only seen positions are drawn: a point exists at frame f for object k iff centres[f,k] is finite
    (and inside the image) and the object is not retired at f;
  * a retired object is not drawn and its trail is cleared: from its first retired frame on, the object has NO points in
    the 21-frame history window, also not the ones seen before the retirement.
Everything else is the unchanged drawing path of rma_perception.Perception._frame / rma_perception_v2.PerceptionV2._frame_v2:
    history = deque(maxlen=CADENCE + 1) of {"frame": t, "entities": {sid: {"center", "reliable"}}}
    tracks  = data_common.make_tracks(list(history), registry)
    png, _  = CanonicalRenderer.render(rgb256, t, max(0, t - CADENCE), registry, tracks)       # 224x224 PNG bytes
Centres are passed as float(np.float32(c)), exactly as PerceptionV2._frame_v2 does.

OverlayState is step-wise (one frame per call, causal), so an evaluation adapter can use the same class:
    state = OverlayState(registry); state.reset()
    tracks, frame, start = state.step(centres_row [K,2], retired_row [K] or None)
    png, _ = renderer.render(rgb, frame, start, registry, tracks)
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

SIZE = 256
CADENCE = 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tracks(path, n, k):
    """-> centres float32 [n,k,2] (nan where not drawn), drawn bool [n,k], retired bool [n,k], stats dict.
    Raises when the file does not fit the episode (length / object count); disagreements between the three arrays
    are counted in stats, never silently repaired in a way that draws more than `centres` says."""
    with np.load(path, allow_pickle=False) as z:
        missing = {"centres", "visibility", "retired"} - set(z.files)
        if missing:
            raise ValueError(f"{path}: track file lacks {sorted(missing)}")
        centres = np.asarray(z["centres"], np.float32)
        visibility = np.asarray(z["visibility"], bool)
        retired = np.asarray(z["retired"], bool)
    if centres.shape != (n, k, 2) or visibility.shape != (n, k) or retired.shape != (n, k):
        raise ValueError(f"{path}: track shapes {centres.shape}/{visibility.shape}/{retired.shape} for episode [{n},{k}]")
    finite = np.isfinite(centres).all(axis=-1)
    with np.errstate(invalid="ignore"):
        inside = finite & (centres[..., 0] >= 0) & (centres[..., 0] < SIZE) & (centres[..., 1] >= 0) & (centres[..., 1] < SIZE)
    sticky = np.logical_or.accumulate(retired, axis=0)
    drawn = inside & ~sticky
    stats = {"visible_without_centre": int((visibility & ~finite).sum()),
             "centre_without_visible": int((finite & ~visibility).sum()),
             "centre_outside_image": int((finite & ~inside).sum()),
             "retired_but_centre": int((finite & sticky).sum()),
             "retired_not_sticky": int((sticky & ~retired).sum()),
             "retired_frames": sticky.sum(axis=0).astype(int).tolist(),
             "drawn_fraction": drawn.mean(axis=0).round(4).tolist()}
    centres = np.where(drawn[..., None], centres, np.nan).astype(np.float32)
    return centres, drawn, sticky, stats


class OverlayState:
    """History of the last CADENCE + 1 frames; step() returns the renderer's `tracks` for the new frame."""

    def __init__(self, registry, make_tracks):
        self.registry = registry
        self.sids = list(registry)  # "1".."K" in object order (rma_spec.registry)
        self.make_tracks = make_tracks
        self.reset()

    def reset(self):
        self.frame = 0
        self.history = deque(maxlen=CADENCE + 1)
        self.retired = np.zeros(len(self.sids), bool)

    def step(self, centres_row, retired_row=None):
        centres_row = np.asarray(centres_row, np.float32)
        if retired_row is not None:
            newly = np.asarray(retired_row, bool) & ~self.retired
            self.retired |= newly
            for j in np.flatnonzero(newly):  # trail cleared: forget what was seen before the retirement
                for record in self.history:
                    record["entities"][self.sids[j]] = {"center": None, "reliable": False}
        entities = {}
        for j, sid in enumerate(self.sids):
            seen = bool(np.isfinite(centres_row[j]).all()) and not self.retired[j]
            entities[sid] = {"center": [float(v) for v in centres_row[j]] if seen else None, "reliable": seen}
        self.history.append({"frame": self.frame, "entities": entities})
        tracks = self.make_tracks(list(self.history), self.registry)
        frame = self.frame
        self.frame += 1
        return tracks, frame, max(0, frame - CADENCE)
