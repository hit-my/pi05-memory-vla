"""v5 frame-0 detections for tasks 18/19: SAM3 text prompts on a ZOOMED crop of frame 0 (the cabinet-1 region, fixed per task),
so that the ~10 px objects are ~120 px for the detector. det5.py SHARD NSHARDS OUTDIR TASK[,TASK] [frame]
Output OUTDIR/<ep>.npz: masks packed [M,256,256] (in full-frame coordinates), scores, prompt (index into PROMPTS), boxes."""
import json
import sys
import time
from pathlib import Path

import numpy as np

TV = Path("xxxx/rma/trackv")
sys.path.insert(0, str(TV))
sys.path.insert(0, str(TV / "src/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"))
sys.path.append(str(TV / "pydeps"))
sys.path.append("xxxx/pi05-abc/tracker/assets/deps")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

LR = Path("xxxx/rma/lr_eps")
CROP = {18: (56, 66, 152, 162), 19: (56, 66, 152, 162)}  # x0, y0, x1, y1 (96 px square)
PROMPTS = ["box", "bottle", "carton", "small object", "food package", "chocolate box", "butter", "milk carton", "juice carton", "ketchup bottle", "chocolate bar", "brown box", "dark box"]
THR = 0.15


def detect(proc, rgb, task, crop=None):
    x0, y0, x1, y1 = crop or CROP[task]
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    w = x1 - x0
    masks, scores, pid = [], [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(Image.fromarray(crop).resize((1008, 1008), Image.BICUBIC))
        for p, prompt in enumerate(PROMPTS):
            o = proc.set_text_prompt(state=state, prompt=prompt)
            if len(o["scores"]) == 0:
                continue
            m = F.interpolate(o["masks_logits"].float(), (w, w), mode="area")[:, 0] > 0.5
            full = np.zeros((len(m), 256, 256), bool)
            full[:, y0:y1, x0:x1] = m.cpu().numpy()
            masks.append(full); scores.append(o["scores"].float().cpu().numpy()); pid.append(np.full(len(m), p))
    if not masks:
        return np.zeros((0, 256, 256), bool), np.zeros(0, np.float32), np.zeros(0, int)
    masks = np.concatenate(masks); scores = np.concatenate(scores); pid = np.concatenate(pid)
    area = masks.sum(axis=(1, 2))
    keep = (area >= 8) & (area <= 2000)
    return masks[keep], scores[keep], pid[keep]


if __name__ == "__main__":
    shard, shards = int(sys.argv[1]), int(sys.argv[2])
    OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
    tasks = [int(x) for x in sys.argv[4].split(",")]
    frame = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    model = build_sam3_image_model(checkpoint_path=str(TV / "src/sam3.pt"), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=THR)
    eps = sorted(p.stem for t in tasks for p in (LR / "meta").glob(f"t{t:02d}_seed*.npz"))
    for i, ep in enumerate(eps):
        if i % shards != shard or (OUT / f"{ep}.npz").exists():
            continue
        rgb = np.ascontiguousarray(np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")[frame])
        start = time.time()
        masks, scores, pid = detect(proc, rgb, int(ep[1:3]))
        np.savez_compressed(OUT / f"{ep}.tmp.npz", masks=np.packbits(masks, axis=-1), scores=scores, prompt=pid, prompts=json.dumps(PROMPTS))
        (OUT / f"{ep}.tmp.npz").rename(OUT / f"{ep}.npz")
        print("det5", ep, len(masks), f"{time.time() - start:.1f}s", flush=True)
    print("DET5_DONE", shard, flush=True)
