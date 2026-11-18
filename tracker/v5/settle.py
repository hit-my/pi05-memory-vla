import numpy as np
import ev5
for task in (18, 19, 1, 12, 25):
    D = []
    A = []
    for ep in ev5.task_eps(task, "A")[:40]:
        g = np.load(ev5.GT / f"{ep}.npz")
        mc = g["mask_centre"][:41]
        D.append(np.sqrt(((mc - mc[40]) ** 2).sum(-1)))
        A.append(g["mask_area"][:41])
    D = np.array(D); A = np.array(A)
    print(task, "frames 0..20 median/p90 distance of GT centre to its frame-40 place (px):")
    print("   med", np.round(np.nanmedian(D, axis=(0, 2))[:21], 1).tolist())
    print("   p90", np.round(np.nanpercentile(D, 90, axis=(0, 2))[:21], 1).tolist())
    print("   visible>=30 share", np.round((A >= 30).mean(axis=(0, 2))[:21], 2).tolist())
