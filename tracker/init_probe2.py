"""Replay the real first request of an evaluation episode, swapping one input at a time. init_probe2.py CKPT REQUEST.json"""
import base64, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.home() / "pi05-abc/code_causal_b"))
from policy import JointMemoryPolicy
DS = Path("xxxx/rma/full/dataset_causal")
policy = JointMemoryPolicy(sys.argv[1])
req = json.load(open(sys.argv[2]))
png = lambda b: np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
main, wrist, state = png(req["main_png_b64"]), png(req["wrist_png_b64"]), np.asarray(req["state"], np.float32)
row = json.loads(open(DS / "rows/t12_seed100.jsonl").readline())
img = lambda p: np.asarray(Image.open(DS / p).convert("RGB").resize((224, 224), Image.BILINEAR))
tmain, twrist = img(row["image_path"]), img(row["wrist_image_path"])
tstate = np.load(DS / row["episode_npz"])["states"][0].astype(np.float32)
print("eval main", main.shape, "train main", tmain.shape, "eval state", state.round(4), "train state", tstate.round(4))
def run(name, m, w, s):
    r = policy.infer(m, w, s, req["instruction"], req["previous_memory_text"], None)
    print(f"{name:55s} valid={r['memory_valid']}  {r['memory_text'].splitlines()[1][:75]}", flush=True)
g = lambda s, v: np.r_[s[:6], v, -v].astype(np.float32)
run("eval request as is", main, wrist, state)
run("eval request, gripper -> 0.0211 (training frame 0)", main, wrist, g(state, 0.0211))
run("eval request, whole state -> training frame 0", main, wrist, tstate)
run("eval images + eval state, gripper 0.030", main, wrist, g(state, 0.030))
run("training frame 0 as is", tmain, twrist, tstate)
run("training frame 0, gripper -> 0.0387 (evaluation)", tmain, twrist, g(tstate, 0.0387))
run("training images, eval state", tmain, twrist, state)
run("eval main + training wrist, eval state", main, twrist, state)
run("training main + eval wrist, eval state", tmain, wrist, state)
print("PROBE2_DONE")
