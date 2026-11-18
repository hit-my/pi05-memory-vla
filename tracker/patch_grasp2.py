from pathlib import Path
f = Path("/data/lgd/rma/trackv/tv_fuse.py")
s = f.read_text()
old = """        if pending is not None and t - pending[1] >= SETTLE:
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
new = """        if pending is not None and t - pending[1] >= SETTLE:
            near, t0, e0, q0 = pending
            moved = e[t] - e0
            if states[t, 6] < MIN_WIDTH * 0.6:
                pending = None  # fingers fully closed: nothing between them
            elif np.hypot(*moved) > CHOOSE_PX:
                # held = the nearby object that is SEEN travelling with the gripper (thin objects leave the fingers
                # almost closed, so finger width alone cannot tell an object from a handle)
                score = {}
                for j in near:
                    if v_in[t, j]:
                        d = c_in[t, j] - q0[j]
                        if np.hypot(*d) > 0.4 * np.hypot(*moved):
                            score[j] = np.hypot(*(d - moved))
                best = min(score, key=score.get) if score else None
                if best is not None and score[best] < 0.6 * np.hypot(*moved) + 4:
                    held = best
                    events.append((t, "grasp", held))
                    pending = None
                elif np.hypot(*moved) > 3 * CHOOSE_PX or t - t0 > CHOOSE_WAIT:
                    pending = None  # the gripper left and nothing came along: it closed on a handle
            elif t - t0 > CHOOSE_WAIT and len(near) == 1 and states[t, 6] >= MIN_WIDTH:
                held, pending = near[0], None  # holding still for a long time with something between the fingers
                events.append((t, "grasp", held))
"""
assert s.count(old) == 1, s.count(old)
f.write_text(s.replace(old, new))
print("patched")
