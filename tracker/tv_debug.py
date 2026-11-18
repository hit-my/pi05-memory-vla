import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv_common import ROOT, LR, centres_from_masks, eef_pixels, load, load_masks, project

ep = sys.argv[1]
frames = [int(x) for x in sys.argv[2:]]
rgb, states, actions = load(ep)
meta = np.load(LR / "meta" / f"{ep}.npz")
texts = {int(s): str(t) for s, t in zip(meta["subtask_order"], meta["subtask_texts"])}
e = eef_pixels(ep, states)
cmd = actions[:, 6] > 0
base, _ = load_masks("cutie", ep)
c, v = centres_from_masks(base)
print("init centres", c[0].round(1), "pixels", base[0].sum(axis=(1, 2)))
for t in range(1, len(cmd)):
    if cmd[t] != cmd[t - 1]:
        print(t, "CLOSE" if cmd[t] else "open ", "eef", e[t].round(1), "state", states[t, :6].round(3), "finger", states[t, 6].round(4),
              "|", texts[int(meta["subtask"][t])], "| cutie", c[t].round(0).tolist())
panels = []
for t in frames:
    img = cv2.resize(rgb[t][..., ::-1], (768, 768), interpolation=cv2.INTER_CUBIC)
    R, _ = cv2.Rodrigues(states[t, 3:6])
    o = e[t] * 3
    for axis, col in zip(range(3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        p = project(ep, (states[t, :3] + 0.12 * R[:, axis])[None])[0] * 3
        cv2.line(img, tuple(int(a) for a in o), tuple(int(a) for a in p), col, 2)
    cv2.putText(img, f"{ep} f{t} {'CLOSED' if cmd[t] else 'open'} x=red y=green z=blue", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panels.append(img)
cv2.imwrite(str(ROOT / "view" / f"debug_{ep}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
