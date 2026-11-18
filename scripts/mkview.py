
import json, hashlib, os, sys
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f not in ("metadata.json", "COMPLETE.json", "train_state"):
        p = os.path.join(dst, f)
        if not os.path.lexists(p): os.symlink(os.path.join(src, f), p)
m = json.load(open(os.path.join(src, "metadata.json")))
m["openpi_root"] = "/pfs/pfs-7jnepv/lgd/pi05-abc/openpi"
m["lgd_eval_view"] = {"source": src, "changed": ["openpi_root"]}
b = json.dumps(m, indent=2, ensure_ascii=False).encode()
open(os.path.join(dst, "metadata.json"), "wb").write(b)
cm = json.load(open(os.path.join(src, "COMPLETE.json")))
cm["metadata_sha256"] = hashlib.sha256(b).hexdigest()
json.dump(cm, open(os.path.join(dst, "COMPLETE.json"), "w"), indent=2)
print(sorted(os.listdir(dst)))

