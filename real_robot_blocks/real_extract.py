"""Step 0: yx_blocks hdf5 episodes -> OUT/meta/<ep>.npz (any Python with h5py + Pillow).

  real_extract.py HDF5_DIR OUT
meta/<ep>.npz: states [N,8], actions [N,7] (what the model sees), state14, action14 (unchanged originals),
               closed [N] bool (gripper COMMAND), events [8] (close/open frames of the four grasps),
               has_block [3] bool (far, middle, near cup), count, items [8], ends [8] (last frame of each item),
               observed (JSON [[k, text]]: the observation written right after item k), view [3] (its frame)
An episode is kept only if it is complete and consistent: four grasps (three cups, one numbered block), and the
numbered block that left its place is the one equal to the number of pink blocks seen under the cups.
Everything here is for labelling; none of it runs at deployment.
"""
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_spec as S  # noqa: E402

# numbered blocks in the top camera (labels only): 0 green, 1 blue, 2 red
BLOCKS = (lambda r, g, b: (g > 85) & (r < 60) & (b < 90) & (g - r > 40),
          lambda r, g, b: (b > 80) & (r < 40) & (g < 70),
          lambda r, g, b: (r > 120) & (g < 70) & (b < 80))


def image(f, camera, t):
    return np.asarray(Image.open(io.BytesIO(f[f"observations/images/{camera}"][t].tobytes())).convert("RGB"))


def block_centre(rgb, k):
    im = rgb.astype(np.int16)
    m = BLOCKS[k](im[..., 0], im[..., 1], im[..., 2])
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) > 60 else None


def analyse(f):
    """-> dict of labels, or a string saying why the episode is not used."""
    action = f["action"][:].astype(np.float32)
    n = len(action)
    closed = action[:, S.GRIPPER] < S.COMMAND_CLOSED_BELOW
    events = [t for t in range(1, n) if closed[t] != closed[t - 1]]
    if len(events) != 8 or [bool(closed[e]) for e in events] != [True, False] * 4:
        return f"expected close/open x4, got {len(events) // 2} grasps"
    has, view = [], []
    for close, release in zip(events[0:6:2], events[1:6:2]):
        seen = next((t for t in range(close, release) if int(S.pink_mask(image(f, "top", t)).sum()) > S.SEEN_PIXELS), None)
        has.append(seen is not None)
        view.append(seen + S.VIEW_AFTER_SEEN if seen is not None else close + min(S.VIEW_EMPTY, (release - close) // 2))
    first, last = image(f, "top", 0), image(f, "top", n - 1)
    moved = []
    for k in range(3):
        a, b = block_centre(first, k), block_centre(last, k)
        if a is None:
            return f"numbered block {k} not found in the first frame"
        moved.append(b is None or float(np.hypot(*(a - b))) > 60)
    count = int(sum(has))
    if moved != [k == count for k in range(3)]:
        return f"blocks seen {has} (count {count}) but numbered blocks moved {moved}"
    items, observed = S.items_for(has)
    ends = []
    for i in range(3):
        ends += [view[i] - 1, events[2 * i + 1] + S.PLACE_MARGIN]
    ends += [events[6] + S.PICK_MARGIN, min(events[7] + S.PLACE_MARGIN, n - 1)]
    if any(b <= a + 1 for a, b in zip(ends, ends[1:])):
        return f"item ends not increasing with room for the observations: {ends}"
    state = f["observations/qpos"][:].astype(np.float32)
    return {"states": S.state8(state), "actions": S.action7(action), "state14": state, "action14": action, "closed": closed,
            "events": np.array(events), "has_block": np.array(has), "count": count, "items": np.array(items),
            "ends": np.array(ends), "observed": json.dumps(observed), "view": np.array(view)}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "meta").mkdir(parents=True, exist_ok=True)
    kept, dropped, table = [], {}, {}
    for path in sorted(src.glob("episode_init_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1])):
        index = int(path.stem.split("_")[-1])
        with h5py.File(path, "r") as f:
            result = analyse(f)
        if isinstance(result, str):
            dropped[index] = result
            print("DROPPED", index, result, flush=True)
            continue
        np.savez_compressed(out / "meta" / f"{S.episode_name(index)}.npz", **result)
        kept.append(index)
        table.setdefault(tuple(bool(h) for h in result["has_block"]), []).append(index)
        print("OK", index, len(result["states"]), [bool(h) for h in result["has_block"]], "ends", result["ends"].tolist(), flush=True)
    (out / "extract.json").write_text(json.dumps({"kept": kept, "dropped": dropped,
                                                  "has_block": {str(k): v for k, v in table.items()}}, indent=1))
    print("EXTRACT_DONE kept", len(kept), "dropped", dropped, {str(k): len(v) for k, v in table.items()})


if __name__ == "__main__":
    main()
