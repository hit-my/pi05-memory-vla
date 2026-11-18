"""Per-frame class-agnostic small-object detections with SAM3 (generic text prompts), for re-detection + memory fusion.
tv_sam3det.py EVERY SHARD NSHARDS [all]  -> out/sam3det/<ep>.npz : frames[F], boxes (object array per frame: [M,5] x0,y0,x1,y1,score)"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"))
sys.path.append(str(HERE / "pydeps"))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tv_common import LR, ROOT, episodes  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

every, shard, shards = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
PROMPTS = ["box", "bottle", "carton", "small object"]
model = build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False)
proc = Sam3Processor(model, confidence_threshold=0.15)
out = ROOT / "out/sam3det"
out.mkdir(parents=True, exist_ok=True)
eps = episodes() if len(sys.argv) < 5 else sorted(p.stem for p in (LR / "meta").glob("t*_seed*.npz"))
for i, ep in enumerate(eps):
    if i % shards != shard or (out / f"{ep}.npz").exists():
        continue
    rgb = np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")
    frames = list(range(0, len(rgb), every))
    boxes = np.empty(len(frames), object)
    start = time.time()
    for n, t in enumerate(frames):
        found = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(Image.fromarray(np.ascontiguousarray(rgb[t])).resize((1008, 1008), Image.BICUBIC))
            for prompt in PROMPTS:
                o = proc.set_text_prompt(state=state, prompt=prompt)
                b, s = o["boxes"].float().cpu().numpy() / 1008 * 256, o["scores"].float().cpu().numpy()
                for bb, ss in zip(b, s):
                    if 20 <= (bb[2] - bb[0]) * (bb[3] - bb[1]) <= 900:
                        found.append([*bb, ss])
        boxes[n] = np.asarray(found, np.float32).reshape(-1, 5)
    np.savez_compressed(out / f"{ep}.npz", frames=np.asarray(frames), boxes=boxes, allow_pickle=True)
    print("sam3det", ep, len(frames), f"{(time.time() - start) / len(frames) * 1000:.0f} ms/frame", f"mean boxes {np.mean([len(b) for b in boxes]):.1f}", flush=True)
