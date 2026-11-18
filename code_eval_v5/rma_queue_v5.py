#!/usr/bin/env python3
"""Queue the v5 RoboMemArena evaluation on this node: GROUPS x TASKS x seed blocks (= rma/rma_queue_fix.py with the
v5 adapter and runner wrapper).

Usage: rma_queue_v5.py OUT_ROOT GROUP=CHECKPOINT[,GROUP=CHECKPOINT...] [--tasks 1,...,26] [--trials 10] [--block 5]
         [--seed 50] [--gpus 0,1,...] [--max-running 10] [--free-mb 40000] [--code-dir ~/pi05-abc/code_causal_v2]
         [--max-steps 2500] [--replan 20]
Each job = one runner process (code_eval_v5/rma_eval_v5.sh: scripts_g0, RMA_DUMMY_GRIP=0, 10 warm-up steps) for one
task and the seeds seed+b*block .. +block-1; adapter code_eval_v5/rma_adapter_v5.py (v5 perception, three images,
lenient memory). --code-dir is the CHECKPOINT's code (policy server + memory format), code_causal_v2 for v2r runs.
Finished jobs (aggregate.json and every episode seen by the adapter) are skipped, so the queue can be restarted. A failed
job is retried once after moving its folder aside.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("out_root")
p.add_argument("models")
p.add_argument("--tasks", default=",".join(str(t) for t in range(1, 27)))
p.add_argument("--trials", type=int, default=10)
p.add_argument("--block", type=int, default=5)
p.add_argument("--seed", type=int, default=50)
p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
p.add_argument("--max-running", type=int, default=10)
p.add_argument("--free-mb", type=int, default=40000)
p.add_argument("--code-dir", default=str(HOME / "pi05-abc/code_causal_v2"))
p.add_argument("--max-steps", type=int, default=2500)
p.add_argument("--replan", type=int, default=20)
args = p.parse_args()

out_root = Path(args.out_root)
models = dict(item.split("=", 1) for item in args.models.split(","))
gpus = [int(g) for g in args.gpus.split(",")]
jobs = [(group, int(task), args.seed + b * args.block)
        for task in args.tasks.split(",") for b in range(args.trials // args.block) for group in models]
adapter = HERE / "rma_adapter_v5.py"
runner = HERE / "rma_eval_v5.sh"


def folder(job):
    group, task, seed = job
    return out_root / group / f"t{task}_s{seed}"


def done(job):
    """Finished AND the adapter actually ran every episode (a crashed adapter still yields an aggregate of zeros)."""
    if not (folder(job) / "aggregate.json").is_file():
        return False
    memory = folder(job) / "adapter_log" / "memory.jsonl"
    try:
        seen = {json.loads(line)["episode"] for line in memory.open()}
    except (OSError, ValueError, KeyError):
        return False
    return len(seen) >= args.block


def gpu_free():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}


running, attempts, launched_on = {}, {}, {}
while True:
    for job, proc in list(running.items()):
        if proc.poll() is not None:
            del running[job]
            print(time.strftime("%H:%M:%S"), "finished", job, "ok" if done(job) else f"FAILED rc={proc.returncode}", flush=True)
    waiting = [j for j in jobs if not done(j) and j not in running and attempts.get(j, 0) < 2]
    if not waiting and not running:
        break
    if waiting and len(running) < args.max_running:
        free = gpu_free()
        load = {g: sum(1 for j in running if launched_on.get(j) == g) for g in gpus}
        candidates = [g for g in gpus if free.get(g, 0) >= args.free_mb]
        if candidates:
            gpu = min(candidates, key=lambda g: (load[g], -free[g]))
            job = waiting[0]
            group, task, seed = job
            target = folder(job)
            if target.exists():
                target.rename(target.with_name(target.name + f".failed{int(time.time())}"))
            target.parent.mkdir(parents=True, exist_ok=True)
            kwargs = {"checkpoint_dir": models[group], "gpu": str(gpu), "code_dir": args.code_dir,
                      "log_dir": str(target / "adapter_log"), "action_seed": seed}
            command = ["bash", str(runner), str(task), str(seed), str(args.block), str(gpu),
                       str(target), str(adapter), json.dumps(kwargs), "--max-steps", str(args.max_steps),
                       "--replan-steps", str(args.replan), "--num-steps-wait", "10"]
            log = open(str(target) + ".log", "w")
            running[job] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            start_new_session=True)
            attempts[job] = attempts.get(job, 0) + 1
            launched_on[job] = gpu
            print(time.strftime("%H:%M:%S"), "start", job, "gpu", gpu, "attempt", attempts[job], flush=True)
            time.sleep(60)  # let the policy server and the SAM3 tracker claim their memory before measuring again
            continue
    time.sleep(30)
failed = [j for j in jobs if not done(j)]
print("QUEUE_DONE", len(jobs) - len(failed), "of", len(jobs), "failed", failed, flush=True)
