import shutil
from pathlib import Path
f = Path("/data/lgd/rma/trackv/tv_fuse.py")
if not f.with_suffix(".py.bak_grasp").exists():
    shutil.copy(f, f.with_suffix(".py.bak_grasp"))
s = f.read_text()
i = s.index("def observe_only")
head, body = s[:i], s[i:]
old = """            cand = [(np.hypot(*(q[j] - e[t])), j) for j in range(k) if q[j] is not None]
            pending = (min(cand)[1], t) if cand and min(cand)[0] < grasp_px else None
        if pending is not None and not closed[t]:
            pending = None
        if pending is not None and t - pending[1] == SETTLE:
            if states[t, 6] >= MIN_WIDTH:
                held = pending[0]
                events.append((t, "grasp", held))
            pending = None
"""
new = """            cand = sorted((np.hypot(*(q[j] - e[t])), j) for j in range(k) if q[j] is not None)
            near = [j for d, j in cand if d < grasp_px]
            # several objects stand within reach of the gripper (two boxes on one plate): do not guess by distance,
            # wait and see which one is SEEN moving with the gripper
            pending = (near, t, e[t].copy(), {j: q[j].copy() for j in near}) if near else None
        if pending is not None and not closed[t]:
            pending = None
        if pending is not None and t - pending[1] >= SETTLE:
            near, t0, e0, q0 = pending
            if states[t, 6] < MIN_WIDTH:
                pending = None  # fingers closed on a handle or on nothing
            elif len(near) == 1:
                held, pending = near[0], None
                events.append((t, "grasp", held))
            else:
                moved = e[t] - e0
                if np.hypot(*moved) > CHOOSE_PX or t - t0 > CHOOSE_WAIT:
                    score = {j: np.hypot(*((c_in[t, j] - q0[j]) - moved)) for j in near if v_in[t, j]}
                    held = min(score, key=score.get) if score else near[0]
                    events.append((t, "grasp", held))
                    pending = None
"""
assert body.count(old) == 1, body.count(old)
body = body.replace(old, new)
head += "CHOOSE_PX, CHOOSE_WAIT = 12, 60  # gripper travel (px) / frames before deciding which of several nearby objects is held\n\n\n"
f.write_text(head + body)
print("patched")
