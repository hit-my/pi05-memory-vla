from pathlib import Path
d = Path("xxxx/rma/trackv")
s = (d / "tv_sam.py").read_text()
a = 'first, _ = load_masks("cutie", ep)'
assert s.count(a) == 1, s.count(a)
s = s.replace(a, 'lab = np.load(ROOT / "initB" / f"{ep}.npy"); kk = load_masks("cutie", ep)[0].shape[1]; first = np.stack([lab == j + 1 for j in range(kk)])[None]')
n1 = s.count('(ROOT / "out" / method / f"{ep}.npz").exists()')
n2 = s.count("save_masks(method, ep,")
assert n1 == 1 and n2 == 1, (n1, n2)
s = s.replace('(ROOT / "out" / method / f"{ep}.npz").exists()', '(ROOT / "out" / (method + "b") / f"{ep}.npz").exists()').replace("save_masks(method, ep,", 'save_masks(method + "b", ep,')
(d / "tv_sam_b.py").write_text(s)
print("ok")
