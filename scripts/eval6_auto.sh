#!/bin/bash
# eval6_auto.sh STEP (9031 only: the one node with the full v5 eval environment). User 2026-11-17: 6-task eval at each
# milestone may start directly. Waits for the C_v3 checkpoint pushed from L20X, builds the lgd view (weights symlinked,
# only openpi_root rewritten), runs tasks 1,4,8,10,20,25 x 10 (seeds 50-59), replan 20, no gate, code_causal_v3.
set -u
STEP=${1:?step}; CK=$HOME/pi05-abc-ckpt/C_v3/checkpoint_$STEP; V=${CK}_lgdview
CL=$HOME/pi05-abc/rma/stage/C_v3_eval6_$STEP.claim; O=/data/lgd/rma/eval/C_v3_${STEP}_v5_6t_10
PY=$HOME/SANE/.venv/bin/python
log(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
log "waiting for $CK"
until [ -f $CK.ready ] && [ -f $CK/COMPLETE.json ]; do sleep 120; done
mkdir $CL 2>/dev/null || { log "already claimed"; exit 0; }
echo 10 > $CL/host; chmod -R a+rwx $CL
$PY /data/lgd/rma/mkview.py $CK $V || { log "view failed"; exit 1; }
mkdir -p $O; log "start"
$PY -u $HOME/pi05-abc/code_eval_v5/rma_queue_v5.py $O "C_v3_$STEP=$V" --tasks 1,4,8,10,20,25 --trials 10 --block 5 --seed 50 \
  --gpus 0,1,2,3,4,5,6,7 --max-running 8 --free-mb 40000 --code-dir $HOME/pi05-abc/code_causal_v3 \
  --max-steps 2500 --replan 20 >> $O/queue.log 2>&1 && { touch $O/DONE $CL/DONE; }
log "queue finished rc=$?"

