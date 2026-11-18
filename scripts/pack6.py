"""Pack the C_v3 10k 6-task eval for a viewer page: copies videos, writes meta.json (per episode: task, seed,
result, stage Y/N, failure reason, memory timeline). Run on 9031: pack6.py EVAL_DIR OUT_DIR"""
import glob, json, os, re, shutil, sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out + "/v", exist_ok=True)
EP = re.compile(r"Episode (\d+) \(seed=(\d+)\): TSR=(\d+)% \| CSR=([\d.]+)% \| (.*)$")
meta = []
for bdir in sorted(glob.glob(src + "/t*_s*")):
    m = re.search(r"/t(\d+)_s(\d+)$", bdir)
    if not m or not os.path.isdir(bdir):
        continue
    task, s0 = int(m.group(1)), int(m.group(2))
    stages = {}
    if os.path.exists(bdir + ".log"):
        for line in open(bdir + ".log", errors="replace"):
            e = EP.search(line.strip())
            if e:
                st = [x.split("=") for x in e.group(5).split(" | ") if "=" in x]
                stages[int(e.group(1))] = [(k.strip(), v.strip()) for k, v in st]
    tsv = {}
    if os.path.exists(bdir + "/episodes.tsv"):
        L = open(bdir + "/episodes.tsv").read().splitlines()
        h = L[0].split("\t")
        for l in L[1:]:
            d = dict(zip(h, l.split("\t")))
            tsv[int(d["ep"])] = d
    mem = {}
    mf = bdir + "/adapter_log/memory.jsonl"
    if os.path.exists(mf):
        for l in open(mf, errors="replace"):
            try:
                d = json.loads(l)
            except ValueError:
                continue
            g = (d.get("generated") or "").replace("<unused0>", "")
            p = d.get("perception") or {}
            mem.setdefault(d["episode"], []).append(
                [d.get("frame", 0), g, bool(d.get("memory_valid")), d.get("lenient"), d.get("stopped"),
                 (p.get("checklist_item") or [None, None])[1]])
    for f in sorted(glob.glob(bdir + f"/videos/task{task}/task{task}_*_ep*_seed*.mp4")):
        if f.endswith("_wrist.mp4"):
            continue
        b = os.path.basename(f)[:-4]
        r = re.search(r"_(success|failure)_ep(\d+)_seed(\d+)$", b)
        ep, seed = int(r.group(2)), int(r.group(3))
        name = f"t{task}_seed{seed}"
        shutil.copy(f, f"{out}/v/{name}.mp4")
        w = f[:-4] + "_wrist.mp4"
        if os.path.exists(w):
            shutil.copy(w, f"{out}/v/{name}_w.mp4")
        t = tsv.get(ep, {})
        meta.append({"task": task, "seed": seed, "ep": ep, "name": name, "success": r.group(1) == "success",
                     "TSR": float(t.get("TSR") or 0), "CSR": float(t.get("CSR") or 0),
                     "reason": t.get("failure_reason", ""), "prompt": t.get("prompt", ""),
                     "stages": stages.get(ep, []), "memory": mem.get(ep + 1, [])})
json.dump(meta, open(out + "/meta.json", "w"), ensure_ascii=False)
print(len(meta), "episodes", sum(m["success"] for m in meta), "successes",
      "mem-matched", sum(bool(m["memory"]) for m in meta))
