"""Install the A-v3 label indexes into ~/pi05-abc/dataset and patch its split manifest.

Run on the cluster with the project venv python, after copy_to_pfs.sh has copied
norm_stats.json, split_manifest.json, episodes/ and images/ into that directory.

Copies ~/pi05-ledger-A/dataset_A3/{train,val}.jsonl to the paths the manifest names,
then rewrites only each split's index_sha256. FullCorpus compares those hashes with
the index files. Group B and C text is rendered from the same A-v3 rows at load time,
so all three groups share this one dataset directory.

Idempotent: the first run saves split_manifest.orig.json, and every run patches from
that backup, so rerunning gives the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

SOURCE = Path.home() / "pi05-ledger-A" / "dataset_A3"
ROOT = Path.home() / "pi05-abc" / "dataset"
SPLITS = ("train", "val")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_atomically(path, write):
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    write(temporary)
    temporary.replace(path)


def count_rows(path):
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def main():
    manifest_path = ROOT / "split_manifest.json"
    backup_path = ROOT / "split_manifest.orig.json"
    if not backup_path.exists():
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} is missing; has copy_to_pfs.sh copied it yet?")
        shutil.copy2(manifest_path, backup_path)
    manifest = json.loads(backup_path.read_text(encoding="utf-8"))
    if set(manifest["splits"]) != set(SPLITS):
        raise ValueError(f"Expected splits {SPLITS}, got {sorted(manifest['splits'])}")

    # Labels change, normalization must not: the A-v3 rows reuse the 10k statistics.
    norm_sha = sha256_file(ROOT / "norm_stats.json")
    if norm_sha != manifest["normalization"]["sha256"]:
        raise ValueError("norm_stats.json differs from the manifest; refusing to patch around it")

    report = {}
    for name in SPLITS:
        split = manifest["splits"][name]
        source = SOURCE / f"{name}.jsonl"
        destination = (ROOT / split["index"]).resolve()
        if not destination.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Manifest index escapes the dataset root: {split['index']}")
        rows = count_rows(source)
        if rows != int(split["queries"]):
            raise ValueError(f"{source} has {rows} rows, but the manifest expects {split['queries']}")
        digest = sha256_file(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha256_file(destination) != digest:
            replace_atomically(destination, lambda path, src=source: shutil.copyfile(src, path))
        if sha256_file(destination) != digest:
            raise RuntimeError(f"Copy verification failed: {destination}")
        report[name] = {"index": split["index"], "rows": rows, "old_sha256": split["index_sha256"], "new_sha256": digest}
        split["index_sha256"] = digest

    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    replace_atomically(manifest_path, lambda path: path.write_text(text, encoding="utf-8"))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
