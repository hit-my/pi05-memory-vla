#!/bin/bash
# eval_v5_full.sh CHECKPOINT_DIR GROUP GPUS [MAX_RUNNING] [CODE_DIR]
#   Full RoboMemArena evaluation of one scheme-C v2r checkpoint with the v5 evaluation perception (code_eval_v5):
#   26 tasks x 10 trials (seeds 50-59, two blocks of 5 per task = 52 runner jobs), official runner copy scripts_g0 with
#   10 warm-up steps at gripper command 0 (RMA_DUMMY_GRIP=0), replan every 20 steps, max 2500 steps, no evidence gate,
#   lenient memory (wrong-length checklist / unparsable memory never ends the episode).
#   Example (NOT started by its author):
#     nohup bash ~/pi05-abc/code_eval_v5/eval_v5_full.sh ~/pi05-abc-ckpt/C_v2r/checkpoint_050000 C_v2r_050000 0,1,2,3 10 \
#        > xxxx/rma/eval/C_v2r_050000_v5.log 2>&1 &
#   Output: xxxx/rma/eval/<GROUP>_v5_<host>/<GROUP>/t<task>_s<seed>/{aggregate.json,episodes.tsv,adapter_log/,videos/}
#           and DONE when every job finished; summary: python3 ~/pi05-abc/rma_summary.py <that dir> (as for earlier runs).
#   Restartable: finished jobs are skipped.
set -u
CKPT=${1:?checkpoint dir}; GROUP=${2:?group name}; GPUS=${3:?comma-separated GPU list}; MAXR=${4:-10}
CODE_DIR=${5:-$HOME/pi05-abc/code_causal_v2}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$CKPT/COMPLETE.json" ] || { echo "no COMPLETE.json in $CKPT"; exit 2; }
H=$(hostname | sed 's/.*-//')
O=xxxx/rma/eval/${GROUP}_v5_$H
mkdir -p $O
echo "EVAL_V5_START $GROUP $CKPT gpus $GPUS max_running $MAXR code $CODE_DIR $(date)"
$HOME/SANE/.venv/bin/python -u $HERE/rma_queue_v5.py $O "$GROUP=$CKPT" --tasks $(seq -s, 1 26) --trials 10 --block 5 --seed 50 \
  --gpus $GPUS --max-running $MAXR --free-mb 40000 --code-dir $CODE_DIR --max-steps 2500 --replan 20 && touch $O/DONE
echo "EVAL_V5_DONE rc=$? $(date)"
