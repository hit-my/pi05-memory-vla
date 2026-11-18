"""Side-car W&B logger for the C_v3 run: only the key metrics (user: "wandb上放关键的就行").
Reads OUT/training.jsonl (mean of every 50 optimizer steps) and OUT/validation.jsonl (per checkpoint summary),
never touches the training process. Credentials: the existing ~/.netrc login (nothing is printed or stored).
  train : wandb_sync_v3.py OUT [--dry-run]
  eval  : wandb_sync_v3.py OUT --eval STEP SUCCESS_RATE      (simulator success rate of one checkpoint, 0..1)
Resumes the same W&B run (id in OUT/wandb_sidecar.json) and continues after the last logged step."""
import json, os, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); DRY = "--dry-run" in sys.argv
ENTITY, PROJECT, NAME = "YOUR_WANDB_ENTITY", "pi05-memory-training-review", "C_v3-accB"
TRAIN = {"loss": "train/loss", "flow32": "train/action_loss_32d", "flow7": "train/action_loss_7d",
         "text_ce": "train/memory_text_loss", "align": "train/latent_align_loss", "verb": "train/verb_loss",
         "learning_rate": "train/learning_rate", "gradient_norm": "train/grad_norm", "step_seconds": "train/seconds_per_step"}
VAL = {"action/normalized_mse7": "val/action_mse7", "action/gripper_sign_accuracy": "val/gripper_accuracy",
       "text/action_exact_accuracy": "val/action_line_exact", "text/subgoal_delta_accuracy": "val/subgoal_accuracy",
       "text/done_recall": "val/done_recall", "text/false_done_rate": "val/false_done_rate",
       "text/latent_cosine_mean": "val/latent_cosine", "text/format_valid_rate": "val/format_valid"}
EVERY = 50
state_path = OUT / "wandb_sidecar.json"
state = json.loads(state_path.read_text()) if state_path.is_file() else {}


def run_handle():
    if DRY:
        class R:
            def log(self, d): print("LOG", json.dumps(d))
        return R()
    import wandb
    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME, id=state.get("id"), resume="allow",
                     dir=str(OUT), config={"run": "C_v3", "labels": "scheme-B accumulated Action line",
                                           "steps": 50000, "batch": 128, "dataset": "dataset_v3", "code": "code_causal_v3"})
    run.define_metric("step"); run.define_metric("*", step_metric="step")
    state["id"] = run.id; state["url"] = run.url
    save()
    return run


def save():
    if not DRY:
        state_path.write_text(json.dumps(state))


run = run_handle()
if "--eval" in sys.argv:
    i = sys.argv.index("--eval")
    step, rate = int(sys.argv[i + 1]), float(sys.argv[i + 2])
    run.log({"step": step, "eval/success_rate": rate})
    state.setdefault("eval", {})[str(step)] = rate; save()
    sys.exit(0)

last = state.get("last_step", 0); vdone = set(state.get("val_steps", []))
offset, buf, idle_since = 0, [], time.time()
while True:
    progressed = False
    tj = OUT / "training.jsonl"
    if tj.is_file():
        with open(tj) as f:
            f.seek(offset)
            for line in f:
                if not line.endswith("\n"):
                    break
                offset += len(line.encode())
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("event") != "optimizer_update" or e["optimizer_step"] <= last:
                    continue
                buf.append(e); progressed = True
                if e["optimizer_step"] % EVERY == 0:
                    d = {"step": e["optimizer_step"]}
                    for k, name in TRAIN.items():
                        vals = [x[k] for x in buf if isinstance(x.get(k), (int, float))]
                        if vals:
                            d[name] = sum(vals) / len(vals)
                    run.log(d); last = e["optimizer_step"]; buf = []
                    state["last_step"] = last; save()
    vj = OUT / "validation.jsonl"
    if vj.is_file():
        for line in open(vj):
            try:
                v = json.loads(line)
                s = int(v["optimizer_step"])
                if s in vdone:
                    continue
                m = json.loads(Path(v["summary"]).read_text()).get("wandb_metrics", {})
            except (ValueError, KeyError, OSError):
                continue
            d = {"step": s}
            for k, name in VAL.items():
                val = m.get("validation/generated_history/" + k)
                if isinstance(val, (int, float)):
                    d[name] = val
            run.log(d); vdone.add(s); state["val_steps"] = sorted(vdone); save(); progressed = True
    if DRY:
        break
    if progressed:
        idle_since = time.time()
    elif time.time() - idle_since > 4 * 3600 or (50000 in vdone):
        break
    time.sleep(60)
if not DRY:
    import wandb
    wandb.finish()
