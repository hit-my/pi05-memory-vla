"""SAM3 video tracker, frame by frame (for closed-loop evaluation and for building the dataset with the same code).

reset(rgb, masks) starts one track per object from its first-frame mask; step(rgb) returns the masks for the next frame.
Each object has its own tracker state (as in the offline comparison). KEEP bounds memory: outputs older than KEEP frames
are dropped (None = keep everything, which reproduces the offline whole-video run exactly).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("xxxx/rma/trackv")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"))
sys.path.append(str(ROOT / "pydeps"))
sys.path.append("xxxx/pi05-abc/tracker/assets/deps")
import torch  # noqa: E402

NUM_FRAMES = 100000  # an upper bound; the stream has no known length


class Sam3Stream:
    def __init__(self, checkpoint=str(SRC / "sam3.pt"), keep=None):
        from sam3.model_builder import build_sam3_video_model
        model = build_sam3_video_model(checkpoint_path=checkpoint, load_from_HF=False)
        self.p = model.tracker
        self.p.backbone = model.detector.backbone
        self.size = self.p.image_size
        self.keep = keep
        self.states = []

    def _prep(self, rgb):
        f = torch.from_numpy(np.ascontiguousarray(rgb)[None]).permute(0, 3, 1, 2).float() / 255
        f = torch.nn.functional.interpolate(f, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return ((f - 0.5) / 0.5)[0]

    def _run(self, state, t, preflight=False):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame, _, _, video_res, scores in self.p.propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0, reverse=False,
                                                                           propagate_preflight=preflight, tqdm_disable=True):
                return (video_res[0] > 0).reshape(self.hw).cpu().numpy()

    def reset(self, rgb, masks):
        """masks: list of bool [H,W] (an all-False mask = object not found: it stays empty)."""
        self.hw = rgb.shape[:2]
        self.t = 0
        self.states = []
        img = self._prep(rgb)
        out = []
        for m in masks:
            if not m.any():
                self.states.append(None)
                out.append(np.zeros(self.hw, bool))
                continue
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = self.p.init_state(video_height=self.hw[0], video_width=self.hw[1], num_frames=NUM_FRAMES)
                st["images"] = {0: img}
                self.p.add_new_mask(inference_state=st, frame_idx=0, obj_id=1, mask=torch.from_numpy(m.astype(np.float32)))
            self.states.append(st)
            out.append(self._run(st, 0, preflight=True))
        return out

    def step(self, rgb):
        self.t += 1
        t = self.t
        img = self._prep(rgb)
        out = []
        for st in self.states:
            if st is None:
                out.append(np.zeros(self.hw, bool))
                continue
            st["images"] = {t: img}
            out.append(self._run(st, t))
            if self.keep is not None:
                old = t - self.keep
                for d in [st["output_dict"]] + list(st["output_dict_per_obj"].values()):
                    d["non_cond_frame_outputs"].pop(old, None)
                st["frames_already_tracked"].pop(old, None)
        return out


if __name__ == "__main__":  # equivalence with the offline whole-video run (out/sam3)
    sys.path.insert(0, str(ROOT))
    from tv_common import load, load_masks
    keep = None if sys.argv[1] == "all" else int(sys.argv[1])
    tr = Sam3Stream(keep=keep)
    import time
    for ep in sys.argv[2:]:
        rgb, _, _ = load(ep)
        ref, _ = load_masks("sam3", ep)
        first, _ = load_masks("cutie", ep)
        n, k = ref.shape[:2]
        start = time.time()
        got = np.zeros_like(ref)
        ms = tr.reset(rgb[0], [first[0, j] for j in range(k)])
        for j in range(k):
            got[0, j] = ms[j]
        for t in range(1, n):
            ms = tr.step(rgb[t])
            for j in range(k):
                got[t, j] = ms[j]
        diff = (got != ref).reshape(n, k, -1).sum(-1)
        union = (got | ref).reshape(n, k, -1).sum(-1)
        inter = (got & ref).reshape(n, k, -1).sum(-1)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        print(f"{ep} keep={keep} frames {n} objects {k}: frames with any differing pixel {int((diff.sum(1) > 0).sum())}, differing pixels total {int(diff.sum())}, "
              f"min IoU {iou.min():.3f}, mean IoU {iou.mean():.4f}, {1000 * (time.time() - start) / (n * k):.0f} ms/frame/object, "
              f"GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
