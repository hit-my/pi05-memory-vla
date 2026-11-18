"""Why does the first checklist lose "open ..."?  Run the policy on first frames only, nothing is executed.
init_probe.py CKPT_DIR OUT.json TASK[,TASK...]
 train : dataset frame 0 (overlay image the model was trained on), 6 seeds per task
 evalraw: first frame of the saved evaluation videos (raw camera image, no overlay drawn), same state as a demo's frame 0
 train_wristswap / train_mainswap: training main image with the evaluation wrist image and vice versa (which camera matters)
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
CODE = Path.home() / "pi05-abc/code_causal_b"
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE))
from policy import JointMemoryPolicy
DS = Path("/data/lgd/rma/full/dataset_causal")
PREV = "Action:none\nSubgoal:Not initialized.\nTask:Incomplete"
ckpt, out, tasks = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
policy = JointMemoryPolicy(ckpt)
def img(path, size=224):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))
res = []
for task in tasks:
    rows = sorted(glob.glob(str(DS / "rows" / f"t{task:02d}_seed*.jsonl")))
    picks = rows[:3] + rows[-3:]
    first = None
    for path in picks:
        row = json.loads(open(path).readline())
        state = np.load(DS / row["episode_npz"])["states"][0].astype(np.float32)
        main, wrist = img(DS / row["image_path"]), img(DS / row["wrist_image_path"])
        if first is None:
            first = (main, wrist, state, row)
        r = policy.infer(main, wrist, state, row["instruction"], PREV, None)
        res.append(dict(task=task, kind="train", ep=row["episode_id"], text=r["memory_text"], valid=r["memory_valid"], label=row["memory_target"]["subgoal_delta"]))
        print(task, "train", row["episode_id"], r["memory_valid"], r["memory_text"].split("\n")[1][:90], flush=True)
    main0, wrist0, state0, row0 = first
    for f in sorted(glob.glob(f"/data/lgd/rma/trackv/evalfirst/t{task:02d}_*_front.png"))[:4]:
        em, ew = img(f), img(f.replace("_front", "_wrist"))
        for kind, a, b in (("evalraw", em, ew), ("train_main+eval_wrist", main0, ew), ("eval_main+train_wrist", em, wrist0)):
            r = policy.infer(a, b, state0, row0["instruction"], PREV, None)
            res.append(dict(task=task, kind=kind, ep=Path(f).name, text=r["memory_text"], valid=r["memory_valid"]))
            print(task, kind, Path(f).name, r["memory_valid"], r["memory_text"].split("\n")[1][:90], flush=True)
    json.dump(res, open(out, "w"))
print("PROBE_DONE")
