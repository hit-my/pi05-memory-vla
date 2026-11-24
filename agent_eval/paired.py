"""paired.py BASE_DIRS(comma) ARM_DIR: per (task, seed) success of an arm vs the standard evaluation, plus invalid-memory calls."""
import glob, json, re, sys, collections
def load(dirs):
    res, inv, calls = {}, collections.Counter(), collections.Counter()
    for d in dirs:
        for log in glob.glob(d + "/*/t*_s*.log"):
            t = int(re.search(r"/t(\d+)_s", log).group(1)); eps = []
            if "ERROR:root:Episode failed" in open(log, errors="replace").read(): continue
            for ln in open(log, errors="replace"):
                b = re.match(r"INFO:root:Episode (\d+) \(seed=(\d+)\): TSR=(\d+)% \| CSR=([\d.]+)%", ln)
                if b: res[(t, int(b.group(2)))] = (int(b.group(3)) == 100, float(b.group(4))); eps.append((int(b.group(1)), int(b.group(2))))
            try:
                m = {e + 1: s for e, s in eps}
                for l in open(log[:-4] + "/adapter_log/memory.jsonl"):
                    r = json.loads(l)
                    if r["episode"] in m:
                        calls[(t, m[r["episode"]])] += 1
                        if r.get("memory_valid") is not True: inv[(t, m[r["episode"]])] += 1
            except OSError: pass
    return {k: v for k, v in res.items() if calls[k] >= 3}, inv  # episodes that never ran (tracker/policy crash) are dropped
base, binv = load(sys.argv[1].split(",")); arm, ainv = load([sys.argv[2]])
per = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0.0, 0.0, 0, 0])
for k in sorted(set(base) & set(arm)):
    p = per[k[0]]; p[0] += 1; p[1] += base[k][0]; p[2] += arm[k][0]; p[3] += (arm[k][0] and not base[k][0]); p[4] += (base[k][0] and not arm[k][0])
    p[5] += base[k][1]; p[6] += arm[k][1]; p[7] += binv[k] >= 10; p[8] += ainv[k] >= 10
tot = [0] * 9
for t, p in sorted(per.items()):
    print(f"T{t}: n {p[0]} base {p[1]} arm {p[2]} | gained {p[3]} lost {p[4]} | CSR {p[5]/p[0]:.1f} -> {p[6]/p[0]:.1f} | frozen eps (>=10 invalid) {p[7]} -> {p[8]}")
    tot = [a + b for a, b in zip(tot, p)]
if tot[0]: print(f"ALL: n {tot[0]} base {tot[1]} arm {tot[2]} | gained {tot[3]} lost {tot[4]} | CSR {tot[5]/tot[0]:.1f} -> {tot[6]/tot[0]:.1f} | frozen {tot[7]} -> {tot[8]}")
