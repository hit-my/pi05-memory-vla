"""Shared LIBERO-Mem identity and observed-mask geometry (no goal inference).

Public simulator API:
  build_registry(episode_metadata: dict) -> dict[str, dict]
    episode_metadata has equally long exo_boxes / ego_boxes lists of frames;
    each frame maps the official entity name to [segmentation_id, bbox_xywh, trace].
    A complete one-frame synthetic registry is sufficient for a live episode.
  measure_entities(segmentation, registry, minimum_visible_pixels=80) -> dict
  make_tracks(history, registry) -> list[dict]

The registry is built once from the full episode, never from moving objects or
current positions. Geometry matches build_adjacent_changes.clean: 8-connected
components of area >2, then the mean of zero-based pixel indices. raw_center
retains that measurement even when motion reliability rejects the observation.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

PALETTE = ("#005daa", "#c75600", "#7442b8", "#00866a", "#b13076", "#6a5c00", "#4056a1", "#8d4b24")
COLOR_NAMES = ("蓝", "橙", "紫", "绿", "玫红", "黄褐", "靛蓝", "棕")
CAMERAS = ("exo_boxes", "ego_boxes")


def build_registry(episode_metadata: dict) -> dict[str, dict]:
    """Validate full sid/name identity across both cameras and all observations.

    Returns sid-string keys, each with name, entity_id, physical_name,
    segmentation_id, category, id, color, color_name and short_label. Robot
    category is robot_end_effector; bowls are bowl. Non-robot palette allocation
    follows the canonical renderer's numeric sid ordering, including stationary
    and currently invisible entities. Trace values are not read by this helper.
    """
    if not isinstance(episode_metadata, dict):
        raise ValueError("Episode metadata must be an object")
    for camera in CAMERAS:
        if not isinstance(episode_metadata.get(camera), list):
            raise ValueError(f"Missing metadata camera list: {camera}")
    if len(episode_metadata[CAMERAS[0]]) != len(episode_metadata[CAMERAS[1]]):
        raise ValueError("Cross-camera metadata frame counts differ")
    by_sid: dict[str, dict] = {}
    by_name: dict[str, str] = {}
    for camera in CAMERAS:
        for frame, entries in enumerate(episode_metadata[camera]):
            if not isinstance(entries, dict):
                raise ValueError(f"Invalid entity dictionary: {camera}/{frame}")
            for name, entry in entries.items():
                location = f"{camera}/{frame}/{name}"
                if not isinstance(name, str) or not name or name != " ".join(name.split()):
                    raise ValueError(f"Noncanonical entity name: {location}")
                if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                    raise ValueError(f"Expected [sid, bbox, trace]: {location}")
                if isinstance(entry[0], bool) or not isinstance(entry[0], (int, np.integer)) or entry[0] <= 0:
                    raise ValueError(f"Invalid segmentation ID: {location}")
                bbox = np.asarray(entry[1], dtype=np.float64)
                if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox < 0):
                    raise ValueError(f"Invalid normalized bbox: {location}")
                if np.any(bbox[:2] + bbox[2:] > 1 + 1e-6):
                    raise ValueError(f"Out-of-image normalized bbox: {location}")
                sid = str(int(entry[0]))
                if name in by_name and by_name[name] != sid:
                    raise ValueError(f"Entity changed segmentation ID: {location}")
                if sid in by_sid and by_sid[sid]["name"] != name:
                    raise ValueError(f"Segmentation ID aliases two entities: {location}")
                by_name[name] = sid
                if sid not in by_sid:
                    entity_id = name.replace(" ", "_")
                    category = "robot_end_effector" if name == "robot" else name.rsplit(" ", 1)[0]
                    bowl = re.fullmatch(r"bowl ([1-9][0-9]*)", name)
                    by_sid[sid] = {
                        "name": name, "entity_id": entity_id,
                        "physical_name": f"akita_black_bowl_{bowl[1]}" if bowl else entity_id,
                        "segmentation_id": int(sid), "category": category,
                    }
    if not by_sid:
        raise ValueError("Empty episode identity registry")
    registry = {}
    entity_index = robot_index = 0
    for sid in sorted(by_sid, key=int):
        label = by_sid[sid]
        if label["category"] == "robot_end_effector":
            robot_index += 1
            label.update(id=f"R{robot_index}", color="#55585f", color_name="灰", short_label="机械手")
        else:
            color_index = entity_index % len(PALETTE)
            entity_index += 1
            color_name = COLOR_NAMES[color_index]
            label.update(id=f"E{entity_index}", color=PALETTE[color_index], color_name=color_name,
                         short_label=color_name + "碗" if label["category"] == "bowl" else label["name"])
        registry[sid] = label
    return registry


def measure_entities(segmentation, registry, minimum_visible_pixels=80) -> dict:
    """Return cleaned mask centres; reliability is separate from raw visibility.

    Each sid maps to {center:[x,y]|None, raw_center:[x,y]|None, area:int,
    reliable:bool}. No occluded position is extrapolated or carried forward.
    """
    segmentation = np.asarray(segmentation)
    if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
        segmentation = segmentation[..., 0]
    if segmentation.ndim != 2 or segmentation.dtype.kind not in "iu":
        raise ValueError("Expected a two-dimensional integer segmentation image")
    if minimum_visible_pixels < 1:
        raise ValueError("minimum_visible_pixels must be positive")
    measured = {}
    for sid in registry:
        mask = np.asarray(segmentation == int(sid), dtype=np.uint8)
        count, components, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(count, dtype=np.bool_)
        keep[1:] = statistics[1:, cv2.CC_STAT_AREA] > 2
        cleaned = keep[components]
        y, x = np.nonzero(cleaned)
        area = int(x.size)
        raw_center = [float(x.mean()), float(y.mean())] if area else None
        reliable = area >= minimum_visible_pixels
        measured[str(sid)] = {"center": raw_center if reliable else None,
                              "raw_center": raw_center, "area": area, "reliable": reliable}
    return measured


def make_tracks(history, registry) -> list[dict]:
    """Consume chronological {frame, entities} records; preserve source-frame gaps.

    Only bowls are rendered. Every retained point has i == frame == its real
    source observation index. The renderer, not this helper, owns drawing-time
    simplification/indexing. Unreliable points are omitted, never interpolated.
    """
    tracks = {str(sid): {"sid": str(sid), "points": []}
              for sid, label in registry.items() if label["category"] == "bowl"}
    previous = None
    for row in history:
        frame = row["frame"]
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or frame < 0:
            raise ValueError("History frame indices must be nonnegative integers")
        frame = int(frame)
        if previous is not None and frame <= previous:
            raise ValueError("History must be strictly chronological")
        previous = frame
        for sid, track in tracks.items():
            if sid not in row["entities"]:
                raise ValueError(f"Missing measurement record for registered entity {sid}")
            entity = row["entities"][sid]
            if not entity["reliable"]:
                continue
            center = np.asarray(entity["center"], dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError(f"Invalid reliable centre at frame {frame}, sid {sid}")
            track["points"].append({"i": frame, "frame": frame, "center": center.tolist()})
    return list(tracks.values())
