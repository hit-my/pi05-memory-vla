import shutil
from pathlib import Path
f = Path("xxxx/rma/trackv/tv_fuse.py")
if not f.with_suffix(".py.bak_area").exists():
    shutil.copy(f, f.with_suffix(".py.bak_area"))
s = f.read_text()
a = "            c = c_in[t, j]\n            cx, cy = int(np.clip(round(c[0]), 0, 255)), int(np.clip(round(c[1]), 0, 255))\n"
b = ("            c = c_in[t, j]\n"
     "            # area veto: a mask far larger than the object (it swallowed the table, a drawer front, the arm) is not the object\n"
     "            if area[t, j] > AREA_ABS or (AREA_MIN <= area0[j] <= AREA_ABS and area[t, j] > AREA_RATIO * area0[j]):\n"
     "                recent[j] = []\n"
     "                continue\n"
     "            cx, cy = int(np.clip(round(c[0]), 0, 255)), int(np.clip(round(c[1]), 0, 255))\n")
c1 = "    c_in, v_in = centres_from_masks(masks)\n    e = eef_pixels(ep, states)\n    closed = np.r_[False, actions[:-1, 6] > 0]\n"
d1 = c1 + "    area = masks.sum(axis=(2, 3))\n    area0 = area[0]\n"
assert s.count(a) >= 1 and c1 in s
i = s.index("def observe_only")
head, tail = s[:i], s[i:]
if "AREA_ABS" not in s:
    tail = tail.replace(a, b, 1).replace(c1, d1, 1)
    head += "AREA_ABS, AREA_RATIO, AREA_MIN = 1200, 3.0, 20  # mask-area sanity (objects are 65-375 px at 256 px)\n\n\n"
    f.write_text(head + tail)
    print("patched")
else:
    print("already patched")
