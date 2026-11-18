"""Render causal dense histories with the unchanged approved Canvas renderer.

All observations participate in time-aware point selection. Selected vertices
are actual measured centres; interpolation is used only to bound omission error.
Visibility transitions remain explicit, so simplifying never bridges a gap.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def select_history_frames(tracks, history_start, frame, tolerance=1.5):
    """Return common source-frame indices, preserving all visibility boundaries."""
    start, end = int(history_start), int(frame)
    if start < 0 or end < start or tolerance < 0:
        raise ValueError('Invalid causal history window or simplification tolerance')
    count = end - start + 1
    positions = np.full((count, len(tracks), 2), np.nan, dtype=np.float64)
    visible = np.zeros((count, len(tracks)), dtype=bool)
    for entity_index, track in enumerate(tracks):
        seen = set()
        for point in track['points']:
            source_frame = int(point['frame'])
            if source_frame > end:
                raise ValueError(f'Future trajectory point {source_frame} at observation {end}')
            if source_frame < start:
                continue
            if source_frame in seen:
                raise ValueError(f'Duplicate trajectory frame for {track["sid"]}: {source_frame}')
            seen.add(source_frame)
            center = np.asarray(point['center'], dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError('Track points must contain finite measured centres')
            positions[source_frame - start, entity_index] = center
            visible[source_frame - start, entity_index] = True
    mandatory = {0, count - 1}
    if count > 1:
        changes = np.flatnonzero(np.any(visible[1:] != visible[:-1], axis=1)) + 1
        for boundary in changes:
            mandatory.update((int(boundary) - 1, int(boundary)))
    selected = set(mandatory)
    ordered = sorted(mandatory)
    pending = list(zip(ordered[:-1], ordered[1:]))
    while pending:
        left, right = pending.pop()
        if right - left <= 1:
            continue
        active = np.all(visible[left:right + 1], axis=0)
        if not np.any(active):
            continue
        actual = positions[left + 1:right, active]
        fraction = np.arange(1, right - left, dtype=np.float64)[:, None, None] / (right - left)
        expected = positions[left, active] + fraction * (positions[right, active] - positions[left, active])
        error = np.max(np.linalg.norm(actual - expected, axis=-1), axis=1)
        relative = int(np.argmax(error))
        if error[relative] > tolerance:
            split = left + 1 + relative
            selected.add(split)
            pending.extend(((left, split), (split, right)))
    return [start + index for index in sorted(selected)]


def resize_rgb(rgb, size=224):
    """Bilinear RGB resize with black letterboxing, without crop/rotation/jitter."""
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB')
    height, width = array.shape[:2]
    ratio = min(size / width, size / height)
    resized = Image.fromarray(array).resize(
        (max(1, round(width * ratio)), max(1, round(height * ratio))), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas)


_DOCUMENT = '''<!doctype html><html><head><meta charset="utf-8"></head>
<body data-overlay-style="rgb-soft" data-entity-labels="off">
<canvas id="observation" width="224" height="224"></canvas>
<canvas id="scene" width="224" height="224"></canvas></body></html>'''

_RENDER = '''window.renderFrame = async function(payload) {
  const image = new Image();
  image.src = 'data:image/png;base64,' + payload.rgb;
  await image.decode();
  const canvas = document.getElementById('scene');
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
  ctx.filter = 'none'; ctx.setLineDash([]);
  ctx.clearRect(0, 0, 224, 224); ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 224, 224);
  const result = window.EntityPathVariant.render({
    ctx, frame: payload.frame, index: payload.index, episodeStartFrame: 0,
    rawImage: image, sourceSize: {width: image.naturalWidth, height: image.naturalHeight},
    identities: window.EntityPathVariant.identities(payload.registry),
    tracks: payload.tracks,
    options: {mode: 'gray', initial: true, arrows: true, samples: false, arm: false, zoom: false},
    variantOptions: {lookbackFrames: payload.frame - payload.history_start, previewScale: 1}
  });
  return {png: canvas.toDataURL('image/png').split(',')[1], diagnostics: result.diagnostics};
};'''


class CanonicalRenderer:
    """One persistent Chromium page; no alternate drawing or caption rendering."""

    def __init__(self, renderer_js, *, tolerance=1.5):
        from playwright.sync_api import sync_playwright

        self.tolerance = float(tolerance)
        self._closed = False
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=True, args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'])
            self._page = self._browser.new_page(viewport={'width': 224, 'height': 448}, device_scale_factor=1)
            self._page.set_content(_DOCUMENT)
            self._page.add_script_tag(path=str(Path(renderer_js).resolve()))
            self._page.add_script_tag(content=_RENDER)
        except BaseException:
            self._playwright.stop()
            raise

    def render(self, rgb, frame, history_start, registry, tracks):
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError('Renderer expects stored-orientation HWC uint8 RGB')
        selected = select_history_frames(tracks, history_start, frame, self.tolerance)
        indices = {source_frame: index for index, source_frame in enumerate(selected)}
        prepared = []
        dense_count = 0
        height, width = array.shape[:2]
        for track in tracks:
            sid = str(track['sid'])
            if sid not in registry:
                raise ValueError(f'Trajectory identity absent from registry: {sid}')
            points = []
            for point in track['points']:
                source_frame = int(point['frame'])
                if history_start <= source_frame <= frame:
                    dense_count += 1
                if source_frame not in indices:
                    continue
                x, y = map(float, point['center'])
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f'Centre outside source image: {sid} frame={source_frame}')
                points.append({'i': indices[source_frame], 'frame': source_frame, 'center': [x, y]})
            points.sort(key=lambda point: point['frame'])
            prepared.append({'sid': sid, 'points': points})
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format='PNG')
        result = self._page.evaluate('payload => window.renderFrame(payload)', {
            'rgb': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'frame': int(frame), 'history_start': int(history_start), 'index': len(selected) - 1,
            'registry': registry, 'tracks': prepared,
        })
        diagnostics = result['diagnostics']
        if diagnostics['colorConflicts']:
            raise ValueError(f'Color-only identity collision: {diagnostics["colorConflicts"]}')
        if diagnostics['labels'] or diagnostics['canvasSize'] != [224, 224]:
            raise RuntimeError('Canonical model-input rendering contract changed')
        diagnostics.update({
            'identityScope': 'episode-annotation-registry',
            'selectedSourceFrames': selected,
            'denseWindowPoints': dense_count,
            'renderIndexScope': 'index into selectedSourceFrames; source frames are unchanged',
            'simplificationToleranceSourcePixels': self.tolerance,
            'frameSelection': 'Joint time-linear error over every reliable dense observation; actual vertices only; preserve visibility boundaries',
        })
        return base64.b64decode(result['png'], validate=True), diagnostics

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._browser.close()
            finally:
                self._playwright.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
