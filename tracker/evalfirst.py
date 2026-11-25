"""First front/wrist frames of saved evaluation videos (tasks 1-16 on this node) -> xxxx/rma/trackv/evalfirst/*.png"""
import glob, os, cv2
out = "xxxx/rma/trackv/evalfirst"; os.makedirs(out, exist_ok=True)
for task in (4, 5, 11, 12, 13, 14, 2, 8):
    vids = sorted(v for v in glob.glob(f"xxxx/rma/eval/C_c100k_040000_nogate_10/C/t{task}_s5*/videos/task{task}/*.mp4") if not v.endswith("_wrist.mp4"))[:4]
    for v in vids:
        seed = v.split("seed")[-1].split(".")[0]
        for tag, path in (("front", v), ("wrist", v.replace(".mp4", "_wrist.mp4"))):
            ok, f = cv2.VideoCapture(path).read()
            if ok:
                cv2.imwrite(f"{out}/t{task:02d}_seed{seed}_{tag}.png", f)
print(len(os.listdir(out)))
