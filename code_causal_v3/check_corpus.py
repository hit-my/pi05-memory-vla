"""CPU checks of one group's data path on real rows, without weights. Usage: check_corpus.py A|B|C"""
import json
import sys
from pathlib import Path

import numpy as np

code = Path(__file__).parent
group = sys.argv[1]
contract_file = sys.argv[2] if len(sys.argv) > 2 else f"contract_{group}.json"
contract = json.loads((code / contract_file).read_text())
sys.path.insert(0, str(Path(contract["model"]["openpi_root"]) / "src"))
sys.path.insert(0, str(Path(contract["model"]["openpi_root"]) / "packages/openpi-client/src"))

from memory_format import INITIAL_MEMORY, LAT, MAX_ENTITIES, step_memory  # noqa: E402
from model_data import FullCorpus, JointTokenizer  # noqa: E402

tokenizer = JointTokenizer(contract["model"]["tokenizer"])
root = Path.home() / "pi05-abc/dataset"
corpus = FullCorpus(root, contract, tokenizer)
capacities, lengths = corpus.measure_capacities()
tokenizer.capacities = capacities
print(group, "rows", len(corpus.rows), "split", corpus.split_name, capacities, lengths, flush=True)

rng = np.random.default_rng(0)
indices = rng.choice(len(corpus.rows), 64, replace=False)
batch = corpus.batch_indices(np.r_[indices[:60], [-1, -1, -1, -1]])
print({k: (v.shape, str(v.dtype)) for k, v in batch.items() if k != "observation"})
print("observation prompt", batch["observation"].tokenized_prompt.shape, "images",
      {k: v.shape for k, v in batch["observation"].images.items()})

# 1. Every stored target survives decode -> driver, as a generated text would.
bad = roundtrip = 0
previous = {}
for row in corpus.rows:
    ids = tokenizer.target_ids(row["memory_target_text"])
    decoded = tokenizer.sp.decode(ids[:-1])
    roundtrip += decoded != row["memory_target_text"]
    total = int(row["memory_target"]["total_goals"])
    result = step_memory(row["previous_memory_text"], decoded, group, expected_total=total)
    if not result["memory_valid"] or result["certified_count"] != row["memory_target"]["confirmed_goals"]:
        bad += 1
        if bad <= 3:
            print("BAD", row["sample_id"], repr(decoded), repr(row["previous_memory_text"]))
    elif result["task_text"] != row["memory_target"]["task_text"]:
        bad += 1
print("decode mismatches", roundtrip, "driver rejections", bad, "of", len(corpus.rows))
assert roundtrip == 0 and bad == 0

# 2. Masks behave as designed.
unreliable = [i for i, r in enumerate(corpus.rows) if not r["memory_target"]["action_reliable"]]
reliable_row = corpus.batch(0)
assert np.array_equal(reliable_row["target_mask"], reliable_row["target_loss_mask"])
row = corpus.batch(unreliable[0])
masked = int(row["target_mask"].sum() - row["target_loss_mask"].sum())
print("unreliable rows", len(unreliable), "masked tokens in one of them", masked)
assert (masked == 0) == (group == "B")
if masked:
    ids = row["target_ids"][0]
    kept = tokenizer.sp.decode([int(x) for x, m in zip(ids, row["target_loss_mask"][0]) if m])
    print("still supervised:", repr(kept))
    assert "Subgoal:" in kept and "Task:" in kept and "Action" not in kept

# 3. Latent slots match entities, and z*(t-20) really is the earlier window.
if group != "A":
    for i in rng.choice(len(corpus.rows), 2000, replace=False):
        r = corpus.rows[int(i)]
        clauses = r["memory_target"]["action_text"].split(" | ")
        assert r["memory_target_text"].count(LAT) == len(clauses)
        assert tokenizer.target_ids(r["memory_target_text"]).count(tokenizer.lat_id) == len(clauses)
        t = int(r["observation_index"])
        expected = 0 if t < 20 else len(clauses)
        assert r["previous_memory_text"].count(LAT) == expected
        prev = corpus.previous_latents(r)
        assert prev.shape == (MAX_ENTITIES, 64)
        if t >= 20:
            earlier = corpus.latents[r["episode_npz"]][0][t - 20]
            assert np.array_equal(prev[:len(clauses)], earlier)
        valid = corpus.z_valid(r)
        z = corpus.z_star(r)
        assert np.allclose(np.linalg.norm(z[valid], axis=1), 8.0, atol=1e-3) and not z[~valid].any()
        for k, clause in enumerate(clauses):
            assert ("occluded" in clause) == (not valid[k]), (r["sample_id"], clause, valid[k])
    print("latent slots ok; example:", repr(r["memory_target_text"][:120]))
    print("first-window memory:", INITIAL_MEMORY == corpus.rows[0]["previous_memory_text"])
if group == "C":
    row = corpus.batch(int(rng.integers(len(corpus.rows))))
    print("verb_targets", row["verb_targets"].shape, "verb_mask", row["verb_mask"])
failure = [i for i, r in enumerate(corpus.rows) if "failures/" in r["episode_npz"]]
if failure:
    print("failure rows", len(failure), "of", len(corpus.rows), f"= {len(failure) / len(corpus.rows):.3f}")
    for i in rng.choice(failure, 200, replace=False):
        row = corpus.batch(int(i))
        assert not row["action_mask"].any() and row["target_mask"].any(), corpus.rows[int(i)]["sample_id"]
    print("failure rows never supervise actions: ok")
print("CHECK_OK", group)
