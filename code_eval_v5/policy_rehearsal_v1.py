"""REHEARSAL ONLY: serve a v1-format checkpoint (two images, e.g. C_c100k/checkpoint_050000 trained with
code_causal_b_fix) behind the three-image request of rma_adapter_v5, to check the evaluation pipeline end to end.

Run with PYTHONPATH = the checkpoint's code dir (the adapter sets it from code_dir). Each request must carry the three
images main_png_b64 / wrist_png_b64 / raw_png_b64 (decoded and checked: RGB uint8 224x224); raw_png_b64 is then dropped
(the v1 model has no clean-front slot) and the rest is answered by that code's policy_ens / policy.JointMemoryPolicy
exactly as its own --serve-json loop does. Stats of the received images go to stderr (policy_stderr.log) every request.
Not for scoring a v2r checkpoint: those are served by code_causal_v2/policy_ens.py directly.
"""
import base64
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]   # never import code_eval_v5's policy.py

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import policy_ens  # noqa: E402,F401  (patches policy.JointMemoryPolicy: ACTION_SAMPLES / ACTION_SEED)
import policy as base  # noqa: E402


def main():
    path = sys.argv[sys.argv.index("--serve-json") + 1]
    transport = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model = base.JointMemoryPolicy(path)
    print(f"REHEARSAL policy from {base.__file__} checkpoint {path}", file=sys.stderr, flush=True)
    required = {"main_png_b64", "wrist_png_b64", "raw_png_b64", "state", "instruction", "previous_memory_text"}
    n = 0
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not required <= set(request) <= required | {"previous_latents"}:
                raise ValueError(f"request must carry exactly {sorted(required)} (+ previous_latents)")
            images = {}
            for key in ("main_png_b64", "wrist_png_b64", "raw_png_b64"):
                with Image.open(io.BytesIO(base64.b64decode(request[key], validate=True))) as image:
                    images[key] = np.asarray(image.convert("RGB"))
                if images[key].shape != (224, 224, 3):
                    raise ValueError(f"{key} has shape {images[key].shape}")
            n += 1
            diff = float(np.abs(images["main_png_b64"].astype(int) - images["raw_png_b64"].astype(int)).mean())
            print(f"REHEARSAL request {n}: three images 224x224 ok; mean|overlay-raw| {diff:.3f}; "
                  f"means {[round(float(images[k].mean()), 1) for k in images]}", file=sys.stderr, flush=True)
            with contextlib.redirect_stdout(sys.stderr):
                result = model.infer(images["main_png_b64"], images["wrist_png_b64"], np.asarray(request["state"], np.float32),
                                     request["instruction"], request["previous_memory_text"], request.get("previous_latents"))
            result["actions"] = result["actions"].tolist()
            result["latents"] = result["latents"].tolist()
        except Exception as error:
            print(f"Policy request failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            result = {"error": f"{type(error).__name__}: {error}"}
        transport.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        transport.flush()


if __name__ == "__main__":
    main()
