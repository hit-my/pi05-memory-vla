"""Step 0: LeRobot v2.1 parquet -> OUT/meta/<ep>.npz (any Python with pandas + pyarrow).

  real_extract.py LEROBOT_DIR OUT
meta/<ep>.npz: states [N,8], actions [N,7] (what the model sees, real_spec.state8 / action7),
               state14, action14 (the unchanged originals), closed [N] bool, events [6] (close/open frames),
               ends [6] (last frame of each checklist item), subtask [N] (index of the item in progress)
The six checklist items are cut from the right gripper alone: every demonstration is close-open-close-open-close-open.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_spec as S  # noqa: E402


def segment(gripper):
    """-> closed [N] bool, events (frames where the reading changes side), ends (one per checklist item)."""
    closed = gripper < S.CLOSED_BELOW
    events = [t for t in range(1, len(closed)) if closed[t] != closed[t - 1]]
    if len(events) != 6 or [bool(closed[e]) for e in events] != [True, False] * 3:
        raise ValueError(f"expected close/open x3, got {[(e, bool(closed[e])) for e in events]}")
    n = len(closed)
    ends = [min(e + (S.PICK_MARGIN if i % 2 == 0 else S.PLACE_MARGIN), n - 1) for i, e in enumerate(events)]
    if any(b <= a for a, b in zip(ends, ends[1:])):
        raise ValueError(f"item ends not increasing: {ends}")
    return closed, events, ends


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "meta").mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "meta/info.json").read_text())
    for index in range(info["total_episodes"]):
        table = pd.read_parquet(src / f"data/chunk-000/episode_{index:06d}.parquet")
        state = np.stack(table["observation.state"].values).astype(np.float32)
        action = np.stack(table["action"].values).astype(np.float32)
        closed, events, ends = segment(state[:, S.GRIPPER])
        subtask = np.searchsorted(np.asarray(ends), np.arange(len(state))).clip(max=len(S.ITEMS) - 1)
        np.savez_compressed(out / "meta" / f"{S.episode_name(index)}.npz", states=S.state8(state), actions=S.action7(action),
                            state14=state, action14=action, closed=closed, events=np.array(events), ends=np.array(ends),
                            subtask=subtask, subtask_texts=np.array(S.ITEMS))
    print("EXTRACT_DONE", info["total_episodes"])


if __name__ == "__main__":
    main()
