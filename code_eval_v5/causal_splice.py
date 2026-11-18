"""Move causal labels to a cluster that has the original full dataset but not the causal one.

  causal_splice.py make  NEW_DATASET DELTA.jsonl.xz   # rows of NEW whose text differs from the original (tasks 4/5)
  causal_splice.py apply OLD_DATASET DELTA.jsonl.xz NEW_MANIFEST OUT_DIR
      writes OUT_DIR/{train,val}.jsonl = OLD index with the delta rows swapped in by sample_id, then refuses unless
      both sha256 equal the ones in NEW_MANIFEST (the causal split_manifest.json from lgd), and copies the manifest.
Everything else (episodes, images, norm_stats) is the unchanged original data.
"""
import hashlib
import json
import lzma
import shutil
import sys
from pathlib import Path


def make(new, out):
    n = 0
    with lzma.open(out, "wt") as w:
        for part in ("train", "val"):
            for line in open(Path(new) / f"{part}.jsonl"):
                if '"task_id": 4,' in line or '"task_id": 5,' in line:
                    r = json.loads(line)
                    if r["task_id"] in (4, 5):
                        w.write(line)
                        n += 1
    print("delta rows", n, out)


def apply(old, delta, manifest, out):
    new_rows = {}
    with lzma.open(delta, "rt") as f:
        for line in f:
            new_rows[json.loads(line)["sample_id"]] = line
    m = json.load(open(manifest))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    used = 0
    for part in ("train", "val"):
        h = hashlib.sha256()
        with open(Path(old) / f"{part}.jsonl") as src, open(out / f"{part}.jsonl", "w") as dst:
            for line in src:
                sid = json.loads(line)["sample_id"]
                if sid in new_rows:
                    line = new_rows[sid]
                    used += 1
                dst.write(line)
                h.update(line.encode())
        want = m["splits"][part]["index_sha256"]
        if h.hexdigest() != want:
            raise SystemExit(f"{part}: sha256 {h.hexdigest()} != causal manifest {want}; refusing")
        print(part, "sha256 matches the causal manifest")
    if used != len(new_rows):
        raise SystemExit(f"only {used} of {len(new_rows)} delta rows found in the old index")
    shutil.copy(manifest, out / "split_manifest.json")
    print("OK", out)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "apply":
        apply(*sys.argv[2:6])
    else:
        raise SystemExit(__doc__)
