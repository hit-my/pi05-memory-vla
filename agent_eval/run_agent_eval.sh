#!/bin/bash
# run_agent_eval.sh NAME TASKS [ENV=VALUE...]: the standard closed-loop evaluation (26 tasks x 10 episodes, seeds 50-59,
# replan 20) of one checkpoint, plus the external ledger checker of policy_agent.py. The checker service (checker_omp.py
# or any service with the same HTTP interface) must be listening at STUDY_AGENT_URL.
#   CODE  = a copy of code_causal_v3 with policy_agent.py added and "import policy_agent" in policy_ens.py (see README)
#   EVAL  = code_eval_v5 (rma_queue_v5.py);  CKPT = exported checkpoint;  OUT = result root
NAME=${1:?}; TASKS=${2:?}; shift 2
: "${CODE:?}" "${EVAL:?}" "${CKPT:?}" "${OUT:?}" "${PY:=python}"
O=$OUT/arm_$NAME; mkdir -p $O; export STUDY_AGENT=1 STUDY_AGENT_LOG=$O/agentlog
for kv in "$@"; do export "$kv"; done
echo "$(date) env: $(echo "$*" | sed 's#\(STUDY_AGENT_URL=[^ ]*/\)[^ ]*#\1***#')" >> $O/arm.txt
$PY -u $EVAL/rma_queue_v5.py $O "model=$CKPT" --tasks $TASKS --trials ${TRIALS:-10} --block 5 --seed 50 \
  --gpus ${GPUS:-0,1,2,3,4,5,6,7} --max-running ${MAXRUN:-4} --free-mb 40000 --code-dir $CODE --max-steps 2500 --replan 20 >> $O/queue.log 2>&1 && touch $O/DONE