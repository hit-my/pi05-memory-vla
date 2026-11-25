"""mem_tracker_v5.py ep : GPU memory and speed of the v5 tracker over one long demo (tracker venv); prints every 250 frames."""
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rma_tracker_v5 as tv5  # noqa: E402
import torch  # noqa: E402
from tv_common import load  # noqa: E402

torch.set_num_threads(2)
ep = sys.argv[1]
task = int(ep[1:3])
rgb, states, actions = load(ep)
tr = tv5.TrackerV5("xxxx/pi05-abc/tracker/assets")
start = time.time()
tr.reset(np.ascontiguousarray(rgb[0]), task, states[0])
last = time.time()
for t in range(1, len(rgb)):
    tr.step(np.ascontiguousarray(rgb[t]), states[t], bool(actions[t - 1, 6] > 0), None)
    if t % 250 == 0 or t == len(rgb) - 1:
        now = time.time()
        print(f"{ep} frame {t} allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB "
              f"max {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB  {1000 * (now - last) / 250:.0f} ms/frame", flush=True)
        last = now
print("MEM_DONE", round(time.time() - start, 1), flush=True)
