import shutil
import sys
from pathlib import Path
f = Path("xxxx/rma/trackv/tv_fuse.py")
if not f.with_suffix(".py.bak_colour").exists():
    shutil.copy(f, f.with_suffix(".py.bak_colour"))
s = f.read_text()
tol = sys.argv[1] if len(sys.argv) > 1 else "35"
if "COLOUR_TOL" in s:
    import re
    s = re.sub(r"^COLOUR_TOL = \d+", f"COLOUR_TOL = {tol}", s, flags=re.M)
    f.write_text(s)
    print("tolerance set", tol)
    raise SystemExit
a = "                ok = bool(still and not robot[t, cy, cx] and np.hypot(*(c - e[t])) > FAR_PX)\n"
b = ("                ok = bool(still and not robot[t, cy, cx] and np.hypot(*(c - e[t])) > FAR_PX)\n"
     "                if ok and colour0[j] is not None:\n"
     "                    # re-acquiring somewhere new is only believed if the thing there looks like the object (the arm is\n"
     "                    # white, handles are black): mean colour of the mask against the first-frame mask\n"
     "                    ok = float(np.abs(rgb[t][masks[t, j]].mean(axis=0) - colour0[j]).mean()) < COLOUR_TOL\n")
c1 = "    area = masks.sum(axis=(2, 3))\n    area0 = area[0]\n"
d1 = c1 + "    colour0 = [rgb[0][masks[0, j]].mean(axis=0) if AREA_MIN <= area0[j] <= AREA_ABS else None for j in range(k)]\n"
assert s.count(a) == 1 and s.count(c1) == 1, (s.count(a), s.count(c1))
i = s.index("def observe_only")
head, body = s[:i], s[i:]
body = body.replace(a, b).replace(c1, d1)
head += f"COLOUR_TOL = {tol}  # mean abs RGB difference allowed between a re-acquired mask and the object's first-frame mask\n\n\n"
f.write_text(head + body)
print("patched", tol)
