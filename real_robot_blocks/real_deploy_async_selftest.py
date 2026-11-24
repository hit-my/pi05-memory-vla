"""Scheduling self-test of real_deploy_async.py with stand-ins for the camera, tracker and model (CPU, seconds, no data).
The fake model answers the request of step t with actions t, t+1, .. t+19 and the memory "m<t>"; the fake recording has
action t at step t. Checked for several (period, latency) pairs:
  * the action executed at step s is exactly s (replies are aligned by request step, nothing is executed late),
  * every request is given the memory written `lag` steps earlier (the newest one that is at least 20 steps old),
  * the run ends 19 steps after the request whose reply says the task is complete.
  python real_deploy_async_selftest.py CODE     (CODE = training code directory, for memory_format.py)"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CODE = sys.argv[1]
sys.path.insert(0, str(Path(CODE).resolve()))
import real_deploy  # noqa: E402
import real_spec as S  # noqa: E402
from memory_format import INITIAL_MEMORY  # noqa: E402

N, COMPLETE_AT = 200, 150
seen, executed = [], []


class FakePerception:
    def __init__(self, *a): self.wrist = self.raw = np.zeros((8, 8, 3), np.uint8)
    def step(self, top, wrist, closed=False):
        import io
        from PIL import Image
        b = io.BytesIO(); Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(b, format="PNG"); return b.getvalue()
    def close(self): pass


class FakePolicy:
    def __init__(self, *a, **k): pass
    def request(self, payload):
        t = int(round(payload["state"][0]))
        seen.append((t, payload["previous_memory_text"]))
        return {"actions": [[float(t + i)] * 7 for i in range(20)], "latents": [[float(t)]], "memory_text": f"m{t}",
                "memory_valid": True, "next_memory_text": f"m{t}", "sequence_complete": t >= COMPLETE_AT}
    def close(self): pass


class FakeReplay:
    def __init__(self, *a):
        self.t = 0
        self.action = np.tile(np.arange(N, dtype=np.float32)[:, None], (1, 14))
    def __len__(self): return N
    def observe(self):
        return {"top": None, "right_wrist": None, "state": np.full(14, float(self.t), np.float32)}
    def act(self, action14):
        if action14 is not None:
            executed.append((self.t, float(np.asarray(action14)[S.RIGHT][0])))
        self.t += 1


real_deploy.Perception, real_deploy.Policy, real_deploy.Replay = FakePerception, FakePolicy, FakeReplay
import real_deploy_async as A  # noqa: E402

failures = 0
for period, latency in ((10, 8), (10, 1), (5, 4), (12, 8), (10, 10), (4, 3)):
    seen.clear(), executed.clear()
    sys.argv = ["x", "--replay", "none", "--checkpoint", "none", "--code", CODE,
                "--period", str(period), "--sim-latency", str(latency), "--log-dir", str(HERE / "_selftest_log")]
    import contextlib, io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        A.main()
    wrong_action = [(s, a) for s, a in executed if abs(a - s) > 1e-4]
    wrong_memory = []
    steps = [t for t, _ in seen]
    for t, given in seen:
        old = [q for q in steps if q <= t - 20]
        want = f"m{max(old)}" if old else INITIAL_MEMORY
        if given != want: wrong_memory.append((t, given, want))
    first_complete = min(t for t in steps if t >= COMPLETE_AT)
    ok = not wrong_action and not wrong_memory and executed[-1][0] == first_complete + 19 and "END model_stop" in out.getvalue()
    failures += not ok
    lags = sorted({t - int(g[1:]) for t, g in seen if g.startswith("m")})
    print(f"period {period:2d} latency {latency}: {'OK' if ok else 'FAIL'}  requests {len(seen)}  memory age {lags}  first action at step "
          f"{executed[0][0]}  last step {executed[-1][0]}  wrong actions {len(wrong_action)}  wrong memories {len(wrong_memory)}")
print("SELFTEST", "PASSED" if not failures else f"FAILED ({failures})")
sys.exit(1 if failures else 0)
