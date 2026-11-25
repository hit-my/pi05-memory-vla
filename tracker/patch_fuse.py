import shutil
from pathlib import Path
f = Path("xxxx/rma/trackv/tv_fuse.py")
if not f.with_suffix(".py.bak_held").exists():
    shutil.copy(f, f.with_suffix(".py.bak_held"))
s = f.read_text()
old = "                ok = np.hypot(*(c - e[t])) < HELD_PX\n"
new = ("                # held: the gripper touches the object's EDGE, not its centre (a bottle is grasped at the top),\n"
       "                # so measure from the gripper pixel to the nearest mask pixel\n"
       "                ys_, xs_ = np.nonzero(masks[t, j])\n"
       "                ok = bool(len(xs_)) and float(np.min(np.hypot(xs_ - e[t][0], ys_ - e[t][1]))) < HELD_EDGE_PX\n")
if old in s:
    s = s.replace(old, new)
    i = s.index("def observe_only")
    s = s[:i] + "HELD_EDGE_PX = 25  # gripper pixel to nearest mask pixel while the object is carried\n\n\n" + s[i:]
    f.write_text(s)
    print("patched")
else:
    print("already patched or pattern missing")
import re
print([l for l in s.splitlines() if re.match(r"^[A-Z_]+ = ", l)])
