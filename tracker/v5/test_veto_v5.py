"""Online == offline for v5. For PER_TASK seeded-random episodes of every task:
  OnlineVetoV5 + RetireSignalV5 fed frame by frame (float32 state, previous gripper command, current subtask text)
  == tv_fuse_v5.observe_only_v5 on the stored masks with retired_from_labels_v5
  == tv_fuse_v5.observe_from_features on the cached measurements (the path that produced all/final_v5)
  == the stored all/final_v5/<ep>.npz
exactly (centres incl. nan, visibility, events, retirement flags). test_veto_v5.py SHARD NSHARDS PER_TASK"""
import filecmp
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/lgd/rma/trackv")
sys.path.insert(0, "/data/lgd/rma/trackv/v5")
sys.path.insert(0, "/pfs/pfs-7jnepv/lgd/pi05-abc/code_causal_v2")  # the online twin is imported from here
from rma_online_veto_v5 import CONSTANTS, OnlineVetoV5, RetireSignalV5  # noqa: E402
from tv_common_v4 import load, task_of  # noqa: E402
import tv_fuse_v5  # noqa: E402
from feat5 import src  # noqa: E402

TV = Path("/data/lgd/rma/trackv")
shard, shards, per_task = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
assert filecmp.cmp(CONSTANTS, TV / "v5/constants_v5.json", shallow=False), "constants next to the online twin differ from v5/constants_v5.json"
CONST = tv_fuse_v5.load_constants()
by = {}
for p in sorted((TV / "v5/feat5").glob("t*_seed*.npz")):
    if "tmp" not in p.name:
        by.setdefault(p.stem[:3], []).append(p.stem)
eps = []
for task, names in sorted(by.items()):
    rng = np.random.default_rng(5000 + int(task[1:]))
    eps += sorted(rng.choice(names, min(per_task, len(names)), replace=False).tolist())
same = lambda a, b: bool(np.array_equal(a, b, equal_nan=True))
bad = 0
for i, ep in enumerate(eps):
    if i % shards != shard:
        continue
    task = task_of(ep)
    z = np.load(src(ep))
    rgb, states, actions = load(ep)
    masks = np.unpackbits(z["masks"], axis=-1)[..., :256].astype(bool)
    n, k = masks.shape[:2]
    m = np.load(f"/data/lgd/rma/lr_eps/meta/{ep}.npz")
    text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
    raw_states = m["states"]
    spec = CONST["tasks"][str(task)]["retire"]
    ret = tv_fuse_v5.retired_from_labels_v5(ep, n, spec)
    F = dict(np.load(TV / "v5/feat5" / f"{ep}.npz"))
    veto, sig = OnlineVetoV5(task), RetireSignalV5(task)
    c_on, v_on, r_on = np.full((n, k, 2), np.nan), np.zeros((n, k), bool), np.zeros((n, k), bool)
    start = time.time()
    for t in range(n):
        st = raw_states[t]  # as stored (float32 or float64; np.asarray(..., float64) inside the twin is then lossless)
        cmd = None if t == 0 else float(actions[t - 1, 6])
        rr = sig.update(text[int(m["subtask"][t])], cmd, st)
        r_on[t, rr] = True
        if t == 0:
            c_on[t], v_on[t] = veto.reset(masks[0], rgb[0], st, retired=rr)
        else:
            c_on[t], v_on[t] = veto.step(masks[t], rgb[t], st, cmd, retired=rr)
    ms = 1000 * (time.time() - start) / n
    # second pass with the state cast to float32 (what a float32 observation pipeline would deliver): same visibility, centres within 1e-3 px
    veto32, sig32 = OnlineVetoV5(task), RetireSignalV5(task)
    c32, v32 = np.full((n, k, 2), np.nan), np.zeros((n, k), bool)
    for t in range(n):
        st = raw_states[t].astype(np.float32)
        cmd = None if t == 0 else float(actions[t - 1, 6])
        rr = sig32.update(text[int(m["subtask"][t])], cmd, st)
        c32[t], v32[t] = veto32.reset(masks[0], rgb[0], st, retired=rr) if t == 0 else veto32.step(masks[t], rgb[t], st, cmd, retired=rr)
    f32_ok = same(v32, v_on) and float(np.nanmax(np.abs(c32 - c_on), initial=0.0)) < 1e-3
    c_off, v_off, ev_off = tv_fuse_v5.observe_only_v5(ep, masks, rgb, states, actions, retired=ret if ret.any() else None)
    c_f, v_f, ev_f = tv_fuse_v5.observe_from_features(task, F, retired=ret if ret.any() else None, masks=masks if tv_fuse_v5.needs_masks(veto.cfg) else None)
    s = np.load(TV / "all/final_v5" / f"{ep}.npz")
    ev_on = [(int(t), a, int(j)) for t, a, j in veto.events]
    checks = dict(retire_online_eq_labels=same(r_on, ret), online_eq_masks=same(c_on, c_off) and same(v_on, v_off),
                  events_eq=ev_on == [(int(t), a, int(j)) for t, a, j in ev_off] == [(int(t), a, int(j)) for t, a, j in ev_f],
                  masks_eq_features=same(c_f, c_off) and same(v_f, v_off),
                  stored_eq=same(s["centres"], c_f.astype(np.float32)) and same(s["visibility"], v_f) and same(s["retired"], ret),
                  retired_never_drawn=not v_on[r_on].any(), float32_state_ok=bool(f32_ok))
    ok = all(checks.values())
    bad += not ok
    print("OK  " if ok else "FAIL", json.dumps(dict(ep=ep, frames=n, k=k, mode=veto.cfg["mode"], state_dtype=str(raw_states.dtype), retired_frames=int(ret.sum()), drawn=round(float(v_on.mean()), 3),
                                                   events=len(ev_on), ms_per_frame=round(ms, 2), **{a: b for a, b in checks.items() if not b})), flush=True)
print("V5_TEST_SHARD", shard, "failed", bad, flush=True)
