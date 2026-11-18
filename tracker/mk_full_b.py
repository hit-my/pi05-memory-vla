from pathlib import Path
d = Path("/data/lgd/rma/trackv")
s = (d / "tv_full.py").read_text()
a = s.index('    if (ROOT / "out/cutie" / f"{ep}.npz").exists():')
b = s.index("    masks = np.zeros((n, k, 256, 256), bool)")
s = s[:a] + ('    lab = np.load(ROOT / "initB" / f"{ep}.npy")  # first frame: SAM3 generic detections assigned to the fixed clicks (method B)\n'
             '    first = np.stack([lab == j + 1 for j in range(k)])\n') + s[b:]
assert s.count('out = ROOT / "all" / "sam3veto"') == 1 and s.count("first=np.packbits(first, axis=-1),") == 1
s = s.replace('out = ROOT / "all" / "sam3veto"', 'out = ROOT / "all" / "final"')
s = s.replace("first=np.packbits(first, axis=-1),", "first=np.packbits(first, axis=-1), masks=np.packbits(masks, axis=-1),")
s = s.replace("first-frame masks as today (clicks -> SAM2.1 -> CUTIE frame 0)", "first-frame masks by method B (initB/<ep>.npy)")
(d / "tv_full_b.py").write_text(s)
(d / "run_full_b.sh").write_text('''#!/bin/bash
# run_full_b.sh : final tracker on all 2600 demos (method-B first frame -> SAM3 frame by frame -> seen-only + gripper veto)
cd /data/lgd/rma/trackv; T=/data/lgd/pi05-abc/tracker; mkdir -p logs all/final
export PYTHONPATH=$T/gpu_extra:$T/assets/deps PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
N=24; s=0
for g in 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 0 1 2 3 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup $T/venv/bin/python tv_full_b.py $s $N > logs/full_$s.log 2>&1 &
  s=$((s + 1))
done
echo started $N
''')
print("ok")
