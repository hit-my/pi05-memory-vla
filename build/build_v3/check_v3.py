"""Integrity check of dataset_v3 against dataset_v2r. Writes CHECK_OK only if every check passes.
  H1 same sample ids per split; every field identical to v2r except the Action line in memory_target_text,
     previous_memory_text, memory_target.action_text; new memory_target.window_action_text == v2r action_text
  H2 Action line recomputed by an independent implementation of the scheme-B rule (not memory_format's) == stored
  H3 grammar: every target / previous Action line valid in variants A and C (code_causal_v3 memory_format)
  H4 eval-driver replay: step_memory(previous[t-20] (C), target[t-20] (C)) -> next_memory_text == previous[t] (C)
     for every row of 300 episodes (both splits) = training input and deployment cannot diverge
  H5 model_data.FullCorpus (code_causal_v3, variant C) loads train and val; measured token capacities reported
usage: check_v3.py V2R V3 CODE_V3"""
import json, random, re, sys, hashlib
from collections import defaultdict
from pathlib import Path

V2, V3, CODE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE))
import memory_format as mf  # noqa: E402
import model_data  # noqa: E402

fails = []
def check(name, ok, note=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {note}" if note else ""), flush=True)
    if not ok:
        fails.append(name)

DIRS = {"up", "down", "forward", "backward", "left", "right"}
EVENTS = {"picked", "placed", "put-in", "taken-out", "opened", "closed", "pressed"}

def indep(prev, window, reset):
    """Independent scheme-B rule: per object list of words; window words appended unless equal to the last one."""
    state = {}
    if prev is not None and not reset:
        for cl in prev.split(" | "):
            nm, bd = cl.split(": ", 1)
            state[nm] = [] if bd in ("still", "occluded") else [w for w in bd.replace(" ; ", " -> ").split(" -> ")]
    res = []
    for cl in window.split(" | "):
        nm, bd = cl.split(": ", 1)
        ws = state.get(nm, [])[:]
        if bd not in ("still", "occluded"):
            for w in bd.replace(" ; ", " -> ").split(" -> "):
                assert w in DIRS or w in EVENTS, w
                if len(ws) == 0 or ws[-1] != w:
                    ws += [w]
        res.append(nm + ": " + (" -> ".join(ws[-12:]) if ws else bd))
    return " | ".join(res)

def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["sample_id"]] = r
    return rows

report = {}
for part in ("train", "val"):
    a, b = load(V2 / f"{part}.jsonl"), load(V3 / f"{part}.jsonl")
    check(f"H1 {part}: same sample ids", set(a) == set(b), f"{len(a)} vs {len(b)}")
    bad_fields, bad_rule, bad_gram = [], [], []
    eps = defaultdict(dict)
    for sid, r3 in b.items():
        r2 = a[sid]
        eps[r3["episode_id"]][r3["observation_index"]] = r3
        x2, x3 = json.loads(json.dumps(r2)), json.loads(json.dumps(r3))
        if x3["memory_target"].pop("window_action_text", None) != r2["memory_target"]["action_text"]:
            bad_fields.append((sid, "window"))
        for x in (x2, x3):
            x["memory_target"].pop("action_text")
            for k in ("memory_target_text", "previous_memory_text"):
                x[k] = x[k].split("\n", 1)[1] if x[k].startswith("Action:") else x[k]
        if x2 != x3:
            bad_fields.append((sid, [k for k in x2 if x2[k] != x3.get(k)]))
        for text in (r3["memory_target_text"], r3["previous_memory_text"]):
            act = text.split("\n")[0][len("Action:"):]
            if act != "none" and (mf.action_entities(act, "A") is None or mf.action_entities(mf.render_action(act, "C"), "C") is None):
                bad_gram.append(sid)
    for ep, rows in eps.items():
        n = len(rows); acc = [None] * n
        for t in range(n):
            w = rows[t]["memory_target"]["window_action_text"]
            reset = t >= 20 and rows[t]["memory_target"]["confirmed_goals"] > rows[t - 20]["memory_target"]["confirmed_goals"]
            acc[t] = indep(acc[t - 20] if t >= 20 else None, w, reset)
            if acc[t] != rows[t]["memory_target"]["action_text"] or rows[t]["memory_target_text"].split("\n")[0] != "Action:" + acc[t]:
                bad_rule.append((ep, t))
            if t >= 20 and rows[t]["previous_memory_text"].split("\n")[0] != "Action:" + acc[t - 20]:
                bad_rule.append((ep, t, "prev"))
    check(f"H1 {part}: only the Action line differs from v2r; window_action_text == v2r action_text", not bad_fields, str(bad_fields[:3]))
    check(f"H2 {part}: independent scheme-B recomputation == stored labels", not bad_rule, str(bad_rule[:3]))
    check(f"H3 {part}: Action grammar valid (A and C) for targets and previous memories", not bad_gram, str(bad_gram[:3]))
    # H4 eval-driver replay
    random.seed(0)
    sample = random.sample(sorted(eps), min(150, len(eps)))
    bad_replay = []; nrep = 0
    for ep in sample:
        rows = eps[ep]
        for t in range(20, len(rows)):
            prev_c = mf.render_memory_text(rows[t - 20]["previous_memory_text"], "C")
            gen_c = mf.render_memory_text(rows[t - 20]["memory_target_text"], "C")
            res = mf.step_memory(prev_c, gen_c, "C", round_index=t - 20)
            nrep += 1
            if not res["memory_valid"] or res["next_memory_text"] != mf.render_memory_text(rows[t]["previous_memory_text"], "C"):
                bad_replay.append((ep, t, res.get("next_memory_text")))
    check(f"H4 {part}: eval-driver replay reproduces every previous memory ({nrep} rounds, {len(sample)} episodes)", not bad_replay, str(bad_replay[:2]))
    lens = [len(r["memory_target"]["action_text"]) for r in b.values()]
    words = [len(c.split(": ", 1)[1].split(" -> ")) for r in b.values() for c in r["memory_target"]["action_text"].split(" | ")]
    report[part] = {"action_chars_max": max(lens), "chain_words_max": max(words)}
    print(f"   {part}: Action line chars max {max(lens)}, chain words max {max(words)}", flush=True)
    del a, b, eps

try:
    contract = json.loads((V3 / "contract_C_v3.json").read_text())
    tok = model_data.JointTokenizer(contract["model"]["tokenizer"])
    for part in ("train", "val"):
        c = json.loads(json.dumps(contract)); c["dataset"]["index"] = f"{part}.jsonl"
        corpus = model_data.FullCorpus(V3, c, tok)
        caps, lengths = corpus.measure_capacities()
        report[f"loader_{part}"] = {"rows": len(corpus.rows), "capacities": caps, "observed": lengths}
        check(f"H5 model_data.FullCorpus (v3, {model_data.PROMPT_VERSION}) accepts {part}.jsonl", True,
              f"rows {len(corpus.rows)} capacities {caps} observed {lengths}")
except Exception as exc:
    import traceback; traceback.print_exc()
    check("H5 model_data.FullCorpus accepts the dataset", False, f"{type(exc).__name__}: {exc}")

(V3 / "check_v3_report.json").write_text(json.dumps({"failures": fails, **report}, indent=1))
if not fails:
    with open(V3 / "CHECK_OK", "w") as f:
        for p in ("train.jsonl", "val.jsonl", "norm_stats.json", "latents/C/standardize.json"):
            f.write(hashlib.sha256((V3 / p).read_bytes()).hexdigest() + "  " + str(V3 / p) + "\n")
print("CHECK_RESULT", "OK" if not fails else f"FAIL {fails}")
