"""memstat.py EVAL_DIR... : per task, how the adapter's memory log ends in each episode."""
import collections, glob, json, os, sys
for root in sys.argv[1:]:
    print("=====", root)
    per = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "*", "t*_s*", "adapter_log", "memory.jsonl"))):
        task = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[0][1:])
        eps = collections.defaultdict(list)
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            eps[r["episode"]].append(r)
        tsr = {}
        tsv = os.path.join(os.path.dirname(os.path.dirname(f)), "episodes.tsv")
        if os.path.exists(tsv):
            rows = [l.rstrip("\n").split("\t") for l in open(tsv)][1:]
            tsr = {i + 1: (float(r[3]), float(r[4])) for i, r in enumerate(rows)}
        for e, rs in sorted(eps.items()):
            stops = [r for r in rs if r.get("stopped")]
            per[task].append((len(rs), stops[0]["call"] if stops else None, stops[0]["stopped"] if stops else "-", stops[0].get("frame") if stops else None, tsr.get(e)))
    for task, rows in sorted(per.items()):
        print(f"task {task:2d}:", " ".join(f"[calls {n} stop@{c} {s} f{fr} {t}]" if c else f"[calls {n} ok {t}]" for n, c, s, fr, t in rows))
