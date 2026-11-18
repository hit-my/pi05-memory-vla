#!/bin/bash
# rehearse_v5.sh TASK GPU OUT [MAX_STEPS] : ONE closed-loop episode (seed 50) with the real runner (scripts_g0, RMA_DUMMY_GRIP=0)
# and the v5 adapter, to check the pipeline end to end. Policy: the previous scheme-C checkpoint C_c100k/checkpoint_050000
# (v1 format, two images) behind policy_rehearsal_v1.py, which checks and logs the three images and drops the third.
# Overlay / raw / wrist frames are saved every 10 frames under OUT/frames. Not a score.
TASK=$1; GPU=$2; OUT=$3; STEPS=${4:-300}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CK=$HOME/pi05-abc-ckpt/C_c100k/checkpoint_050000
mkdir -p $OUT
KW="{\"checkpoint_dir\": \"$CK\", \"gpu\": \"$GPU\", \"code_dir\": \"$HOME/pi05-abc/code_causal_b_fix\", \"policy_script\": \"$HERE/policy_rehearsal_v1.py\", \"log_dir\": \"$OUT/adapter_log\", \"action_seed\": 50}"
START=$(date +%s)
RMA_SAVE_FRAMES=$OUT/frames RMA_SAVE_EVERY=10 timeout 3000 bash $HERE/rma_eval_v5.sh $TASK 50 1 $GPU $OUT $HERE/rma_adapter_v5.py "$KW" \
  --max-steps $STEPS --replan-steps 20 --num-steps-wait 10 > $OUT/runner.log 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - START ))" | tee $OUT/rc.txt
