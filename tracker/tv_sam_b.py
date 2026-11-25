"""SAM2-family causal trackers started from the same frame-0 masks as the baseline (out/cutie).
Methods: sam2v (SAM2.1 video predictor), samurai, dam (DAM4SAM), long (SAM2Long). All base_plus, one object per run.
Usage: tv_sam.py METHOD SHARD NSHARDS"""
import logging
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tv_common import ROOT, episodes, load, load_masks, save_masks  # noqa: E402

method, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
GEN = Path("xxxx/pi05-abc/tracker/gen")
WEIGHTS = str(GEN / "weights/sam2.1_hiera_base_plus.pt")
SRC = ROOT / "src"
sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logging.getLogger("x")))
import torch  # noqa: E402


def patch_loader(module, size_px, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean)[:, None, None]
    std = torch.tensor(std)[:, None, None]

    def load_frames(video_path, image_size=None, offload_video_to_cpu=True, *args, **kwargs):
        chunks = []
        for a in range(0, len(video_path), 64):
            f = torch.from_numpy(video_path[a:a + 64]).permute(0, 3, 1, 2).float() / 255
            f = torch.nn.functional.interpolate(f, size=(size_px, size_px), mode="bilinear", align_corners=False)
            chunks.append((f - mean) / std)
        return torch.cat(chunks), video_path.shape[1], video_path.shape[2]
    module.load_video_frames = load_frames


class Video:
    def __init__(self, kind):
        self.kind = kind
        if kind == "sam3":  # SAM3's own video tracker (the SAM2-style interface), same first-frame mask as the others
            sys.path.insert(0, str(SRC / "sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"))
            sys.path.append(str(HERE / "pydeps"))
            sys.path.append("xxxx/pi05-abc/tracker/assets/deps")
            from sam3.model_builder import build_sam3_video_model
            import sam3.model.sam3_tracking_predictor as module
            model = build_sam3_video_model(checkpoint_path=str(SRC / "sam3.pt"), load_from_HF=False)
            self.predictor = model.tracker
            self.predictor.backbone = model.detector.backbone
            patch_loader(module, self.predictor.image_size, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            return
        if kind == "long":
            sys.path.insert(0, str(SRC / "SAM2Long-main"))
            config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            sys.path.insert(0, str(GEN / "src/samurai/sam2"))
            config = f"configs/{'samurai' if kind == 'samurai' else 'sam2.1'}/sam2.1_hiera_b+.yaml"
        from sam2.build_sam import build_sam2_video_predictor
        import sam2.sam2_video_predictor as module
        self.predictor = build_sam2_video_predictor(config, WEIGHTS, device="cuda")
        patch_loader(module, self.predictor.image_size)

    def track(self, rgb, mask):
        p = self.predictor
        if self.kind == "samurai":
            p.kf_mean, p.kf_covariance, p.stable_frames = None, None, 0
        out = np.zeros(rgb.shape[:3], bool)
        score = np.full(len(rgb), np.nan, np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = p.init_state(video_path=rgb, offload_video_to_cpu=True)
            if self.kind == "sam3":
                p.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=torch.from_numpy(mask.astype(np.float32)))
                for frame, _, _, video_res, obj_scores in p.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=len(rgb),
                                                                              reverse=False, propagate_preflight=True, tqdm_disable=True):
                    out[frame] = (video_res[0] > 0).reshape(rgb.shape[1:3]).cpu().numpy()
                    try:
                        score[frame] = float(obj_scores.float().reshape(-1)[0])
                    except Exception:
                        pass
                return out, score
            if self.kind == "long":
                state["num_pathway"], state["iou_thre"], state["uncertainty"] = 3, 0.1, 2
            p.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
            if self.kind == "long":
                # SAM2Long returns everything at the end, read along the pathway that is best at the last frame:
                # its published output is NOT causal (kept only as a reference).
                _, all_masks = p.propagate_in_video(state, start_frame_idx=0, reverse=False)
                for frame, logits in enumerate(all_masks):
                    out[frame] = (logits[0, 0] > 0).cpu().numpy()
                return out, score
            for item in p.propagate_in_video(state):
                frame, logits = item[0], item[2]
                out[frame] = (logits[0, 0] > 0).cpu().numpy()
                try:
                    per = state["output_dict_per_obj"][0]
                    rec = per["non_cond_frame_outputs"].get(frame) or per["cond_frame_outputs"].get(frame)
                    score[frame] = float(rec["object_score_logits"].float().reshape(-1)[0])
                except Exception:
                    pass
        return out, score


class Dam:
    def __init__(self):
        root = SRC / "DAM4SAM-master"
        sys.path.insert(0, str(root))
        ck = root / "checkpoints/sam2.1_hiera_base_plus.pt"
        ck.parent.mkdir(exist_ok=True)
        if not ck.exists():
            os.symlink(WEIGHTS, ck)
        # the VOT toolkit is only used for three small geometry helpers
        vot, region, raster, shapes = (types.ModuleType(n) for n in ("vot", "vot.region", "vot.region.raster", "vot.region.shapes"))

        class RegionType:
            RECTANGLE = "rectangle"

        class Mask:
            def __init__(self, m):
                self.m = np.asarray(m)

            def rasterize(self, bounds):
                return self.m

            def convert(self, kind):
                ys, xs = np.nonzero(self.m)
                return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (0, 0, 0, 0)

        def calculate_overlaps(a, b):
            res = []
            for (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) in zip(a, b):
                iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
                union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - iw * ih
                res.append(iw * ih / union if union > 0 else 0.0)
            return res
        region.RegionType, shapes.Mask, raster.calculate_overlaps = RegionType, Mask, calculate_overlaps
        vot.region, region.raster, region.shapes = region, raster, shapes
        sys.modules.update({"vot": vot, "vot.region": region, "vot.region.raster": raster, "vot.region.shapes": shapes})
        from dam4sam_tracker import DAM4SAMTracker
        self.tracker = DAM4SAMTracker("sam21pp-B")

    def track(self, rgb, mask):
        from PIL import Image
        out = np.zeros(rgb.shape[:3], bool)
        out[0] = self.tracker.initialize(Image.fromarray(rgb[0]), mask.astype(np.float32))["pred_mask"] > 0
        for t in range(1, len(rgb)):
            out[t] = self.tracker.track(Image.fromarray(rgb[t]))["pred_mask"] > 0
        return out, np.full(len(rgb), np.nan, np.float32)


tracker = Dam() if method == "dam" else Video(method)
order = list(enumerate(episodes()))
if os.environ.get("REVERSE"):
    order = order[::-1]
for i, ep in order:
    if i % shards != shard or (ROOT / "out" / (method + "b") / f"{ep}.npz").exists():
        continue
    if not (ROOT / "out/cutie" / f"{ep}.npz").exists():
        print("no init yet", ep, flush=True)
        continue
    rgb, _, _ = load(ep)
    lab = np.load(ROOT / "initB" / f"{ep}.npy"); kk = load_masks("cutie", ep)[0].shape[1]; first = np.stack([lab == j + 1 for j in range(kk)])[None]
    k = first.shape[1]
    masks = np.zeros((len(rgb), k) + rgb.shape[1:3], bool)
    scores = np.full((len(rgb), k), np.nan, np.float32)
    start = time.time()
    for j in range(k):
        if first[0, j].sum() == 0:
            continue
        masks[:, j], scores[:, j] = tracker.track(rgb, first[0, j])
    save_masks(method + "b", ep, masks, score=scores)
    print(method, ep, len(rgb), k, f"{(time.time() - start) / (len(rgb) * k) * 1000:.0f} ms/frame/object", flush=True)
