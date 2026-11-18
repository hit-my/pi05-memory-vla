"""Frame-0 localisation with SAM3 text prompts, same episodes and truth as tv_init.py.  tv_init_sam3.py PER_TASK"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, str(HERE / "src/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"))
sys.path.append(str(HERE / "pydeps"))
sys.path.append("/data/lgd/pi05-abc/tracker/assets/deps")
import rma_spec  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tv_common import LR, ROOT  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

per_task = int(sys.argv[1])
GEN = len(sys.argv) > 2  # generic prompts: is ANY detection on the object? (class-agnostic recall; identity must come from elsewhere)
GENERIC = ["box", "bottle", "carton", "small object", "food package"]
PROMPT = {"cookies": ["cookie box", "box of cookies"], "sauce": ["ketchup bottle", "tomato sauce bottle"], "butter": ["butter", "butter box"],
          "popcorn": ["popcorn box", "popcorn"], "cream": ["cream cheese", "cream cheese box"], "chocolate": ["chocolate bar", "chocolate box"],
          "bottle": ["wine bottle"], "milk": ["milk carton"], "juice": ["orange juice carton", "juice carton"]}
model = build_sam3_image_model(checkpoint_path=str(HERE / "src/sam3.pt"), load_from_HF=False)
proc = Sam3Processor(model, confidence_threshold=0.15)
clicks = json.load(open("/pfs/pfs-7jnepv/lgd/pi05-abc/rma/clicks26.json"))["episodes"]
by_task = {}
for ep in sorted(clicks):
    by_task.setdefault(int(ep[1:3]), []).append(ep)
rows = []
for task, eps in sorted(by_task.items()):
    for ep in eps[::max(1, len(eps) // per_task)][:per_task]:
        truth = [None if p is None else (p[2:] if len(p) == 4 else p[:2]) for p in clicks[ep]]
        rgb = np.ascontiguousarray(np.load(LR / "front" / f"{ep}.npy", mmap_mode="r")[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(Image.fromarray(rgb).resize((1008, 1008), Image.BICUBIC))
            for k, (noun, _, _) in enumerate(rma_spec.TASKS[task]["objects"]):
                if truth[k] is None:
                    continue
                best = None
                for prompt in (GENERIC if GEN else PROMPT[noun]):
                    out = proc.set_text_prompt(state=state, prompt=prompt)
                    boxes, scores = out["boxes"].float().cpu().numpy() / 1008 * 256, out["scores"].float().cpu().numpy()
                    for b, s in zip(boxes, scores):
                        area = (b[2] - b[0]) * (b[3] - b[1])
                        dd = float(np.hypot((b[0] + b[2]) / 2 - truth[k][0], (b[1] + b[3]) / 2 - truth[k][1]))
                        if GEN and 20 <= area <= rma_spec.MAX_AREA and (best is None or dd < best[4]):
                            best = (float(s), [float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)], prompt, len(scores), dd)
                        elif not GEN and 20 <= area <= rma_spec.MAX_AREA and (best is None or s > best[0]):
                            best = (float(s), [float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)], prompt, len(scores), dd)
                d = None if best is None else float(np.hypot(best[1][0] - truth[k][0], best[1][1] - truth[k][1]))
                rows.append(dict(ep=ep, task=task, obj=k, noun=noun, truth=truth[k], sam3=d, score=None if best is None else best[0],
                                 n=None if best is None else best[3], prompt=None if best is None else best[2]))
    part = [r for r in rows if r["task"] == task]
    print(f"task {task:2d} n={len(part):3d}  SAM3 within 12px {100 * np.mean([r['sam3'] is not None and r['sam3'] < 12 for r in part]):5.1f}%"
          f"  (no box {100 * np.mean([r['sam3'] is None for r in part]):4.1f}%)", flush=True)
    (ROOT / "init").mkdir(exist_ok=True)
    (ROOT / "init" / ("init_sam3_generic.json" if GEN else "init_sam3.json")).write_text(json.dumps(rows))
print(f"ALL SAM3 {100 * np.mean([r['sam3'] is not None and r['sam3'] < 12 for r in rows]):.1f}%")
for noun in PROMPT:
    part = [r for r in rows if r["noun"] == noun]
    if part:
        print(f"  {noun:10s} n={len(part):3d}  {100 * np.mean([r['sam3'] is not None and r['sam3'] < 12 for r in part]):5.1f}%")
