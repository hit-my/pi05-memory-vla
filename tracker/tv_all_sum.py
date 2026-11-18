import glob, json, collections
rows = []
for f in glob.glob("/data/lgd/rma/trackv/all/part_*.json"):
    rows += json.load(open(f))
T = collections.defaultdict(list)
for r in rows:
    T[r["task"]].append(r)
def agg(rs, name):
    bad = sum(r[name]["bad"] for r in rs); seen = sum(r[name]["seen"] for r in rs)
    hb = sum(r[name]["held_bad"] for r in rs); hs = sum(r[name]["held_seen"] for r in rs)
    eps = sum(r[name]["bad"] >= 10 for r in rs)
    return 100 * bad / max(seen, 1), 100 * hb / max(hs, 1), 100 * eps / len(rs), 100 * sum(r[name]["vis"] for r in rs) / len(rs)
print(len(rows), "episodes")
print("task | current: off-ref% (all) off-ref% (while carried) eps>=10 bad frames% vis% | fused: same four | grasps right/all, extra, rejected")
for task, rs in sorted(T.items()):
    a, b = agg(rs, "cur"), agg(rs, "fuse")
    g = (sum(r["grasps_right"] for r in rs), sum(r["grasps"] for r in rs), sum(r["grasps_extra"] for r in rs), sum(r["not_a_grasp"] for r in rs))
    print(f"{task:3d} | {a[0]:5.1f} {a[1]:5.1f} {a[2]:5.0f} {a[3]:4.0f} | {b[0]:5.1f} {b[1]:5.1f} {b[2]:5.0f} {b[3]:4.0f} | {g[0]}/{g[1]} +{g[2]} rej {g[3]}")
a, b = agg(rows, "cur"), agg(rows, "fuse")
g = (sum(r["grasps_right"] for r in rows), sum(r["grasps"] for r in rows), sum(r["grasps_extra"] for r in rows), sum(r["not_a_grasp"] for r in rows))
print(f"ALL | {a[0]:5.1f} {a[1]:5.1f} {a[2]:5.0f} {a[3]:4.0f} | {b[0]:5.1f} {b[1]:5.1f} {b[2]:5.0f} {b[3]:4.0f} | {g[0]}/{g[1]} +{g[2]} rej {g[3]}")
