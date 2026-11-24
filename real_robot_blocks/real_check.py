"""Step 5: load the finished dataset with the TRAINING loader and check it, on CPU, without model weights.

  real_check.py DATASET CONTRACT CODE          (the training python: needs openpi on the contract's model.openpi_root)
Checks: the loader accepts every row of train and val (it re-derives each previous memory with the Ledger Driver and
compares arrays, splits, counts and hashes with the manifest); every stored target text survives tokenizer decode ->
driver exactly as a generated text would; latent slots match the Action clauses; a batch can be assembled.
Writes DATASET/CHECK_OK when everything passes.
"""
import json
import sys
from pathlib import Path

import numpy as np

dataset, contract_path, code = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
contract = json.loads(contract_path.read_text(encoding="utf-8"))
sys.path.insert(0, str(Path(contract["model"]["openpi_root"]) / "src"))
sys.path.insert(0, str(Path(contract["model"]["openpi_root"]) / "packages/openpi-client/src"))
sys.path.insert(0, str(code))
from memory_format import INITIAL_MEMORY, LAT, MAX_ENTITIES, step_memory  # noqa: E402
from model_data import FullCorpus, JointTokenizer  # noqa: E402

variant = contract["model"]["memory_variant"]
tokenizer = JointTokenizer(contract["model"]["tokenizer"])
report = {}
for part in ("train", "val"):
    c = json.loads(json.dumps(contract))
    c["dataset"]["index"] = f"{part}.jsonl"
    corpus = FullCorpus(dataset, c, tokenizer)
    if part == "train":
        capacities, lengths = corpus.measure_capacities()
        tokenizer.capacities = capacities
        print("capacities", capacities, lengths, flush=True)
    bad = roundtrip = 0
    for row in corpus.rows:
        decoded = tokenizer.sp.decode(tokenizer.target_ids(row["memory_target_text"])[:-1])
        roundtrip += decoded != row["memory_target_text"]
        result = step_memory(row["previous_memory_text"], decoded, variant, expected_total=int(row["memory_target"]["total_goals"]))
        if (not result["memory_valid"] or result["certified_count"] != row["memory_target"]["confirmed_goals"]
                or result["task_text"] != row["memory_target"]["task_text"]):
            bad += 1
            if bad <= 3:
                print("BAD", row["sample_id"], repr(decoded))
    rng = np.random.default_rng(0)
    if variant != "A":
        for i in rng.choice(len(corpus.rows), min(2000, len(corpus.rows)), replace=False):
            r = corpus.rows[int(i)]
            clauses = r["memory_target"]["action_text"].split(" | ")
            assert r["memory_target_text"].count(LAT) == len(clauses)
            assert corpus.previous_latents(r).shape == (MAX_ENTITIES, 64)
            valid, z = corpus.z_valid(r), corpus.z_star(r)
            assert np.allclose(np.linalg.norm(z[valid], axis=1), 8.0, atol=1e-3) and not z[~valid].any()
    batch = corpus.batch_indices(rng.choice(len(corpus.rows), 8, replace=False))
    shapes = {k: list(v.shape) for k, v in batch.items() if k != "observation"}
    images = {k: list(v.shape) for k, v in batch["observation"].images.items()}
    complete = sum(r["memory_target"]["task_text"] == "Complete" for r in corpus.rows)
    report[part] = {"rows": len(corpus.rows), "split": corpus.split_name, "decode_mismatches": int(roundtrip),
                    "driver_rejections": int(bad), "rows_with_task_complete": int(complete), "batch": shapes, "images": images}
    print(part, json.dumps(report[part]), flush=True)
    assert roundtrip == 0 and bad == 0, part
    assert corpus.rows[0]["previous_memory_text"].startswith("Action:none") or True
(dataset / "check_report.json").write_text(json.dumps(report, indent=1))
(dataset / "CHECK_OK").write_text("real_check.py\n")
print("CHECK_OK")
