"""dataset_v3 = dataset_v2r with the scheme-B accumulated Action line (user-approved 11-15).
Only train.jsonl / val.jsonl / split_manifest.json / contract change; images, episodes (incl. the per-window
action_text used by the latents), latents, norm_stats, priors are hard-linked from v2r (never modified).
Per row (all rows of an episode, every frame t):
  window = v2r memory_target.action_text            (last 20 frames, unchanged rule)
  reset  = t >= 20 and confirmed[t] > confirmed[t-20] (a subgoal was confirmed in this round)
  acc[t] = memory_format.accumulate_action(acc[t-20] if t >= 20 else None, window, reset)
  memory_target.action_text = acc[t]; memory_target.window_action_text = window
  memory_target_text line 1 = "Action:" + acc[t]
  previous_memory_text line 1 = "Action:" + cap_action(acc[t-20])   (t >= 20; INITIAL_MEMORY otherwise)
usage: relabel_v3.py V2R_DATASET V3_DATASET CODE_V3"""
import hashlib, json, os, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path

SRC, DST, CODE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE))
from memory_format import INITIAL_MEMORY, accumulate_action, cap_action, action_entities, MAX_CHAIN  # noqa: E402

assert (SRC / "CHECK_OK").is_file(), "source dataset has no CHECK_OK"
assert not DST.exists(), f"{DST} exists"
DST.mkdir(parents=True)
for d in ("images", "episodes", "latents"):
    subprocess.run(["cp", "-al", str(SRC / d), str(DST / d)], check=True)
for f in ("norm_stats.json", "priors.json"):
    shutil.copy2(SRC / f, DST / f)
print("hard links done", flush=True)

stats = {"rows": 0, "resets": 0, "max_words": 0, "max_chars": 0, "changed_rows": 0}
sha = {}
for part in ("train", "val"):
    lines = (SRC / f"{part}.jsonl").read_text().splitlines()
    by_ep = defaultdict(list)
    for i, line in enumerate(lines):
        j = line.find('"episode_id": "') + 15
        by_ep[line[j:line.index('"', j)]].append(i)
    out = [None] * len(lines)
    for ep, idx in by_ep.items():
        rows = {}
        for i in idx:
            r = json.loads(lines[i])
            rows[r["observation_index"]] = (i, r)
        n = len(rows)
        assert sorted(rows) == list(range(n)), ep
        acc = [None] * n
        for t in range(n):
            i, r = rows[t]
            mt = r["memory_target"]
            window = mt["action_text"]
            conf = mt["confirmed_goals"]
            reset = t >= 20 and conf > rows[t - 20][1]["memory_target"]["confirmed_goals"]
            acc[t] = accumulate_action(acc[t - 20] if t >= 20 else None, window, reset)
            assert action_entities(acc[t], "A") is not None, (ep, t, acc[t])
            stats["resets"] += int(reset)
            words = max(len(c.split(": ", 1)[1].split(" -> ")) for c in acc[t].split(" | "))
            stats["max_words"] = max(stats["max_words"], words)
            stats["max_chars"] = max(stats["max_chars"], len(acc[t]))
            stats["changed_rows"] += int(acc[t] != window)
            new_mt = {}
            for k, v in mt.items():
                new_mt[k] = acc[t] if k == "action_text" else v
                if k == "action_text":
                    new_mt["window_action_text"] = window
            r["memory_target"] = new_mt
            tl = r["memory_target_text"].split("\n")
            assert tl[0] == "Action:" + window, (ep, t)
            r["memory_target_text"] = "\n".join(["Action:" + acc[t]] + tl[1:])
            if t >= 20:
                pl = r["previous_memory_text"].split("\n")
                assert pl[0] == "Action:" + rows[t - 20][1]["memory_target"]["window_action_text"], (ep, t)
                r["previous_memory_text"] = "\n".join(["Action:" + cap_action(acc[t - 20])] + pl[1:])
            else:
                assert r["previous_memory_text"] == INITIAL_MEMORY, (ep, t)
            out[i] = json.dumps(r)
            stats["rows"] += 1
    data = ("\n".join(out) + "\n").encode()
    (DST / f"{part}.jsonl").write_bytes(data)
    sha[part] = hashlib.sha256(data).hexdigest()
    print(part, "rows", len(out), "sha", sha[part], flush=True)

man = json.loads((SRC / "split_manifest.json").read_text())
for part in ("train", "val"):
    man["splits"][part]["index_sha256"] = sha[part]
man["labels_v3"] = {"from": str(SRC), "rule": "scheme B: Action line accumulates direction/event words per object within "
                    "the current subgoal, restarts in the round that confirms a subgoal; window line kept as "
                    "memory_target.window_action_text (verb head, latents unchanged)", "max_chain": MAX_CHAIN, "stats": stats}
(DST / "split_manifest.json").write_text(json.dumps(man, indent=1))
c = json.loads((SRC / "contract_C_v2r.json").read_text())
c["dataset"]["latents_dir"] = str(DST / "latents" / "C")
if "memory_target_fields" in c["dataset"] and "window_action_text" not in c["dataset"]["memory_target_fields"]:
    c["dataset"]["memory_target_fields"].append("window_action_text")
(DST / "contract_C_v3.json").write_text(json.dumps(c, indent=2))
print("RELABEL_DONE", json.dumps(stats))
