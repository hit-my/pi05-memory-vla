(() => {
  'use strict';
  const SIZE = 224;
  const COLOR_ONLY = document.body.dataset.entityLabels === 'off';
  const SOFT_BACKGROUND = document.body.dataset.overlayStyle === 'rgb-soft';
  const RGB_OVERLAY = SOFT_BACKGROUND || document.body.dataset.overlayStyle === 'rgb';
  const BLUR_PX = SOFT_BACKGROUND ? 0.55 : 0;
  const STYLE = RGB_OVERLAY
    ? {pathWidth: 1.4, pathCasing: 2.2, pathAlpha: 0.8, diamondRadius: 3.4, dotRadius: 2.2, markerCasing: 2, markerWidth: 1.1, labelCasing: 1.5}
    : {pathWidth: 2.2, pathCasing: 4.2, pathAlpha: 1, diamondRadius: 4.5, dotRadius: 3.1, markerCasing: 3.2, markerWidth: 1.5, labelCasing: 3.5};
  if (SOFT_BACKGROUND) Object.assign(STYLE, {
    pathWidth: 2.2, pathCasing: 3.6, pathAlpha: 1, diamondRadius: 3.8,
    dotRadius: 2.7, markerCasing: 2.6, markerWidth: 1.4, labelCasing: 2.4,
  });
  const PALETTE = ['#005daa', '#c75600', '#7442b8', '#00866a', '#b13076', '#6a5c00', '#4056a1', '#8d4b24'];
  const registries = new WeakMap();

  function identities(labels) {
    if (registries.has(labels)) return registries.get(labels);
    const registry = {}; let entity = 0, robot = 0;
    for (const sid of Object.keys(labels).sort((a, b) => a.localeCompare(b, 'en', {numeric: true}))) {
      const isRobot = labels[sid].category === 'robot_end_effector';
      const index = isRobot ? robot++ : entity++;
      registry[sid] = Object.freeze({id: `${isRobot ? 'R' : 'E'}${index + 1}`,
        color: isRobot ? '#55585f' : PALETTE[index % PALETTE.length]});
    }
    Object.freeze(registry); registries.set(labels, registry);
    return registry;
  }

  function overlap(a, b) {
    return Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
      Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  }

  window.EntityPathVariant = {
    id: 'model',
    title: COLOR_ONLY ? '224×224 · 颜色标识实体轨迹' : (SOFT_BACKGROUND ? '224×224 · 轻虚化RGB与清晰轨迹' : (RGB_OVERLAY ? '224×224 · RGB轻量实体轨迹' : '224×224 · 模型专用实体轨迹')),
    intro: COLOR_ONLY
      ? '图内去掉E1等文字和文字引线，只保留固定颜色的轨迹、箭头和观测标记；沿用轻微虚化背景。右侧表格仅供人看，不能假设模型看到了图外对照。文字记忆应明确使用轨迹颜色指代实体。'
      : (SOFT_BACKGROUND
        ? '左侧为原始RGB；右侧背景仅做0.55像素轻微模糊、保留颜色，轨迹和箭头单独清晰绘制。坐标、顺序和位移保持原样；虚化只用于绘图，不用于重新计算轨迹或文字标签。'
        : (RGB_OVERLAY
          ? '左侧为当前RGB，右侧在同一RGB上轻量叠加近期轨迹、观测标记和短ID。背景不转灰、不淡化；两图共用固定取景和缩放。叠加区域仍会改变局部像素，不能视为无损图像。'
          : '左侧保留当前RGB观测；右侧直接在224×224画布上绘制近期轨迹与短ID，不再缩小展示页。两图使用完全相同的取景和缩放，不随轨迹单独放大。')),
    initialLabel: '窗口内首次观测（菱形）',
    canvasLayout: {width: SIZE, height: SIZE, decorate: false},
    identities,
    controlsHTML: `
      <label>回看 H（原始帧）<input id="model-lookback" type="number" min="0" step="1" value="50"></label>
      <label>浏览显示<select id="model-preview-scale"><option value="1">原生224×224</option><option value="2">放大2倍查看像素</option></select></label>
      <p class="note">只绘窗口内实际观测，旧轨迹完全移除。浏览放大不会改变导出尺寸。H=50是回看参数，不是实际query边界。</p>`,
    readOptions() {
      const value = document.getElementById('model-lookback').valueAsNumber;
      return {lookbackFrames: Number.isFinite(value) ? Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.floor(value))) : 50,
        previewScale: document.getElementById('model-preview-scale').value === '2' ? 2 : 1};
    },
    render(view) {
      const {ctx, frame, index, options, variantOptions} = view;
      const startFrame = Math.max(view.episodeStartFrame, frame - variantOptions.lookbackFrames);
      const ratio = Math.min(SIZE / view.sourceSize.width, SIZE / view.sourceSize.height);
      const resizedWidth = Math.round(view.sourceSize.width * ratio);
      const resizedHeight = Math.round(view.sourceSize.height * ratio);
      const padX = Math.floor((SIZE - resizedWidth) / 2), padY = Math.floor((SIZE - resizedHeight) / 2);
      const scaleX = resizedWidth / view.sourceSize.width, scaleY = resizedHeight / view.sourceSize.height;
      // Stored centres are mean zero-based pixel indices; Canvas uses pixel-edge coordinates.
      const project = p => [padX + (p[0] + 0.5) * scaleX, padY + (p[1] + 0.5) * scaleY];
      const transform = {sourceWidth: view.sourceSize.width, sourceHeight: view.sourceSize.height,
        resizedWidth, resizedHeight, padX, padY, scaleX, scaleY, sourcePixelCenterOffset: 0.5,
        crop: null, flip: false, viewportDependsOnHistory: false};
      document.documentElement.style.setProperty('--preview-scale', String(variantOptions.previewScale));
      const observation = document.getElementById('observation'), observationContext = observation.getContext('2d');
      observationContext.setTransform(1, 0, 0, 1, 0, 0);
      observationContext.clearRect(0, 0, SIZE, SIZE); observationContext.fillStyle = 'white';
      observationContext.fillRect(0, 0, SIZE, SIZE);
      observationContext.imageSmoothingEnabled = true; observationContext.imageSmoothingQuality = 'high';
      observationContext.drawImage(view.rawImage, padX, padY, resizedWidth, resizedHeight);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      const backgroundDrawn = options.mode === 'gray';
      if (backgroundDrawn) {
        ctx.save(); ctx.filter = RGB_OVERLAY ? 'none' : 'grayscale(1)'; ctx.globalAlpha = RGB_OVERLAY ? 1 : 0.36;
        ctx.drawImage(view.rawImage, padX, padY, resizedWidth, resizedHeight);
        if (SOFT_BACKGROUND) {
          // Keep an opaque RGB base so the filtered image does not create a pale border.
          ctx.filter = `blur(${BLUR_PX}px)`;
          ctx.drawImage(view.rawImage, padX, padY, resizedWidth, resizedHeight);
        }
        ctx.restore();
      }

      const records = [], rows = [], segments = [], markers = [], sampleMarkers = [], arrows = [], labels = [];
      const drawnPoints = [], omittedEntities = [];
      for (const track of view.tracks) {
        const identity = view.identities[track.sid];
        const points = track.points.filter(p => p.i <= index && p.frame >= startFrame && p.frame <= frame)
          .map(p => ({...p, screen: project(p.center)}));
        const lastKnown = track.points.at(-1);
        if (!points.length) {
          omittedEntities.push({sid: track.sid, id: identity.id, lastObservedFrame: lastKnown?.frame ?? null});
          rows.push({sid: track.sid, displayId: identity.id, color: identity.color,
            historyText: '窗口内无观测', stateText: '不画窗口外陈旧位置'});
          continue;
        }
        const first = points[0], last = points.at(-1), visible = last.i === index && last.frame === frame;
        const gaps = [];
        const record = {sid: track.sid, identity, points, first, last, visible, gaps};
        for (let k = 1; k < points.length; k++) {
          const a = points[k - 1], b = points[k];
          if (b.i !== a.i + 1) {gaps.push([a, b]); continue;}
          segments.push({sid: track.sid, id: identity.id, color: identity.color,
            fromFrame: a.frame, toFrame: b.frame, fromIndex: a.i, toIndex: b.i,
            from: a.screen, to: b.screen, width: STYLE.pathWidth});
        }
        for (const p of points) drawnPoints.push({sid: track.sid, id: identity.id, color: identity.color,
          frame: p.frame, index: p.i, source: p.center, screen: p.screen});
        records.push(record);
        rows.push({sid: track.sid, displayId: identity.id, color: identity.color,
          historyText: `${points.length}次 · f${first.frame}–${last.frame}`,
          stateText: visible ? '当前可见' : `当前缺测；末见f${last.frame}`});
      }

      // Only the background is softened; every annotation retains its measured geometry.
      function strokeSegment(segment, color, width) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        ctx.beginPath(); ctx.moveTo(...segment.from); ctx.lineTo(...segment.to); ctx.stroke();
      }
      ctx.save();
      ctx.globalAlpha = STYLE.pathAlpha;
      for (const s of segments) strokeSegment(s, 'white', STYLE.pathCasing);
      for (const s of segments) strokeSegment(s, s.color, s.width);
      ctx.globalAlpha = 1;

      const endpointPositions = records.flatMap(r => options.initial ? [r.first.screen, r.last.screen] : [r.last.screen]);
      if (options.arrows) {
        for (const s of segments) {
          const dx = s.to[0] - s.from[0], dy = s.to[1] - s.from[1], length = Math.hypot(dx, dy);
          if (length < 14) continue;
          let best = null;
          for (const fraction of [0.5, 0.3, 0.7]) {
            const tip = [s.from[0] + fraction * dx, s.from[1] + fraction * dy];
            const markerClearance = Math.min(...endpointPositions.map(p => Math.hypot(p[0] - tip[0], p[1] - tip[1]))) - 5;
            const arrowClearance = arrows.length ? Math.min(...arrows.map(a => Math.hypot(a.tip[0] - tip[0], a.tip[1] - tip[1]))) - 3.5 : Infinity;
            const score = Math.min(markerClearance, arrowClearance);
            if (!best || score > best.score) best = {tip, score, fraction};
          }
          if (best.score < 0) continue;
          const ux = dx / length, uy = dy / length, tip = best.tip;
          const arrowLength = RGB_OVERLAY && !SOFT_BACKGROUND ? 3.5 : 4.5;
          const arrowHalfWidth = RGB_OVERLAY && !SOFT_BACKGROUND ? 1.9 : 2.5;
          const left = [tip[0] - arrowLength * ux - arrowHalfWidth * uy, tip[1] - arrowLength * uy + arrowHalfWidth * ux];
          const right = [tip[0] - arrowLength * ux + arrowHalfWidth * uy, tip[1] - arrowLength * uy - arrowHalfWidth * ux];
          ctx.beginPath(); ctx.moveTo(...tip); ctx.lineTo(...left); ctx.lineTo(...right); ctx.closePath();
          ctx.strokeStyle = 'white'; ctx.lineWidth = RGB_OVERLAY && !SOFT_BACKGROUND ? 0.8 : 1.2; ctx.stroke(); ctx.fillStyle = s.color; ctx.fill();
          arrows.push({sid: s.sid, fromFrame: s.fromFrame, toFrame: s.toFrame, tip, left, right, fraction: best.fraction});
        }
      }

      function sample(record, point, reason) {
        ctx.fillStyle = record.identity.color; ctx.beginPath(); ctx.arc(...point.screen, 1.2, 0, 2 * Math.PI); ctx.fill();
        sampleMarkers.push({sid: record.sid, sourceFrame: point.frame, sourceIndex: point.i, screen: point.screen, reason});
      }
      function marker(record, point, kind) {
        const p = point.screen, radius = kind === 'diamond' ? STYLE.diamondRadius : STYLE.dotRadius;
        ctx.beginPath();
        if (kind === 'diamond') {
          ctx.moveTo(p[0], p[1] - radius); ctx.lineTo(p[0] + radius, p[1]);
          ctx.lineTo(p[0], p[1] + radius); ctx.lineTo(p[0] - radius, p[1]); ctx.closePath();
        } else ctx.arc(...p, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = 'white'; ctx.lineWidth = STYLE.markerCasing; ctx.stroke();
        if (kind === 'current' || !RGB_OVERLAY) {
          ctx.fillStyle = kind === 'current' ? record.identity.color : 'white'; ctx.fill();
        }
        ctx.strokeStyle = record.identity.color; ctx.lineWidth = STYLE.markerWidth; ctx.stroke();
        markers.push({sid: record.sid, id: record.identity.id, color: record.identity.color, kind,
          sourceFrame: point.frame, sourceIndex: point.i, source: point.center, screen: p, radius});
      }
      for (const record of records) {
        if (options.samples) for (const point of record.points) sample(record, point, 'sample');
        else for (const gap of record.gaps) for (const point of gap) sample(record, point, 'gap-boundary');
        if (options.initial) marker(record, record.first, 'diamond');
        marker(record, record.last, record.visible ? 'current' : 'last');
      }

      // Local short labels: never reserve a rail or shrink the image to fit annotations.
      ctx.font = '700 13px Arial,sans-serif'; ctx.textBaseline = 'top'; ctx.lineJoin = 'round';
      if (!COLOR_ONLY) for (const record of records) {
        const anchor = record.last.screen, width = Math.ceil(ctx.measureText(record.identity.id).width), height = 16;
        const offsets = [[8,-19],[8,7],[-width-8,-19],[-width-8,7],[-width/2,-24],[-width/2,12]];
        let best = null;
        for (const [dx, dy] of offsets) {
          const box = {x: Math.max(2, Math.min(SIZE - width - 2, anchor[0] + dx)),
            y: Math.max(2, Math.min(SIZE - height - 2, anchor[1] + dy)), w: width, h: height};
          let cost = labels.reduce((sum, l) => sum + 100 * overlap(box, l.box), 0);
          for (const p of endpointPositions) cost += 10 * overlap(box, {x:p[0]-5,y:p[1]-5,w:10,h:10});
          for (const s of segments) cost += overlap(box, {x:Math.min(s.from[0],s.to[0])-2,y:Math.min(s.from[1],s.to[1])-2,w:Math.abs(s.to[0]-s.from[0])+4,h:Math.abs(s.to[1]-s.from[1])+4});
          if (!best || cost < best.cost) best = {box, cost};
          if (cost === 0) break;
        }
        const {box} = best;
        const leader = [Math.max(box.x, Math.min(box.x + box.w, anchor[0])), Math.max(box.y, Math.min(box.y + box.h, anchor[1]))];
        ctx.strokeStyle = '#667085'; ctx.lineWidth = 0.7; ctx.setLineDash([1,2]);
        ctx.beginPath(); ctx.moveTo(...anchor); ctx.lineTo(...leader); ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = 'white'; ctx.lineWidth = STYLE.labelCasing; ctx.strokeText(record.identity.id, box.x, box.y);
        ctx.fillStyle = record.identity.color; ctx.fillText(record.identity.id, box.x, box.y);
        labels.push({sid: record.sid, id: record.identity.id, color: record.identity.color,
          anchorSourceFrame: record.last.frame, anchor, box, leader, fontSize: 13});
      }
      ctx.restore();
      const sampleFrames = [...new Set(drawnPoints.map(p => p.frame))].sort((a,b) => a-b);
      const gapCount = records.reduce((sum,r) => sum + r.gaps.length, 0);
      const colorConflicts = [];
      if (COLOR_ONLY) {
        const byColor = new Map();
        for (const record of records) {
          const {color, id} = record.identity;
          if (!byColor.has(color)) byColor.set(color, []);
          byColor.get(color).push(id);
        }
        for (const [color, ids] of byColor) if (ids.length > 1) colorConflicts.push({color, ids});
      }
      return {subtitle: '', footer: '', rows,
        statusText: `原始窗口 [${startFrame}, ${frame}]；${records.length}个实体有窗口内观测，${sampleFrames.length}个采样时刻，${segments.length}段连续连线，${gapCount}处缺测断开。原生224×224，无裁剪、无独立轨迹缩放；${COLOR_ONLY ? '只绘彩色轨迹和观测标记，不叠加ID文字或文字引线。' : '只绘短ID和观测几何。'}图中无标题、帧号、说明或侧栏。${colorConflicts.length ? ' 警告：当前不同实体存在重复轨迹颜色，不能仅凭颜色唯一指代；请使用带ID版本。' : ''}`,
        diagnostics: {
          variant:'model', canvasSize:[SIZE,SIZE], decorations:false, titleDrawn:false, explanationDrawn:false,
          colorOnly:COLOR_ONLY, idLabelsEnabled:!COLOR_ONLY, colorConflicts,
          overlayStyle:SOFT_BACKGROUND?'rgb-soft-strong':(RGB_OVERLAY?'rgb-light':'gray-contrast'), annotationStyle:STYLE,
          requestedWindow:{startFrame,endFrame:frame,lookbackFrames:variantOptions.lookbackFrames},
          contextMode:'hidden', recentPointFrames:drawnPoints.map(p=>({sid:p.sid,frame:p.frame,index:p.index})),
          drawnPoints, segmentFrames:segments, recentSegments:segments, olderSegments:[], arrows, markers,
          initialMarkers:markers.filter(m=>m.kind==='diamond'), endpointMarkers:markers.filter(m=>m.kind!=='diamond'),
          sampleMarkers, labels, omittedEntities, identities:view.identities, identityScope:'task-annotation-registry',
          sampleFrames, gapCount, historyCount:drawnPoints.length, segmentCount:segments.length,
          sourceToCanvas:transform, observationTransform:{...transform}, observationFrame:frame,
          backgroundDrawn, backgroundFrame:backgroundDrawn?frame:null,
          backgroundMode:backgroundDrawn?(SOFT_BACKGROUND?'rgb-soft':(RGB_OVERLAY?'rgb':'grayscale')):'none',
          backgroundOpacity:backgroundDrawn?(RGB_OVERLAY?1:0.36):0,
          backgroundBlurPx:backgroundDrawn?BLUR_PX:0,
          viewport:{width:SIZE,height:SIZE,fit:'fixed-source-image',independentZoom:false},
          entityBoxes:0, entitySprites:0
        }
      };
    }
  };
})();
