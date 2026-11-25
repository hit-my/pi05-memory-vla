from pathlib import Path
f = Path("xxxx/rma/trackv/tv_fuse.py")
s = f.read_text()
old = """            if states[t, 6] < MIN_WIDTH * 0.6:
                pending = None  # fingers fully closed: nothing between them
            elif np.hypot(*moved) > CHOOSE_PX:
"""
new = """            if states[t, 6] < MIN_WIDTH * 0.6:
                pending = None  # fingers fully closed: nothing between them
            elif len(near) == 1 and states[t, 6] >= MIN_WIDTH:
                held, pending = near[0], None  # one object in reach and the fingers are kept apart: unambiguous
                events.append((t, "grasp", held))
            elif np.hypot(*moved) > CHOOSE_PX:
"""
assert s.count(old) == 1, s.count(old)
f.write_text(s.replace(old, new))
print("patched")
