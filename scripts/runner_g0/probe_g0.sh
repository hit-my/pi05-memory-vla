#!/bin/bash
# probe_g0.sh GRIP GPU : one short episode of task 2; the 10 warm-up steps use gripper command GRIP (-1 official, 0 = hold).
# Only reads the first state the policy is given (adapter_log/first_state.jsonl). Not a score.
GRIP=$1; GPU=$2
RMA=xxxx/rma/eval/RoboMemArena-main/evaluation_benchmark
P=xxxx/rma/probe_g0; mkdir -p $P
if [ ! -d $RMA/scripts_g0 ]; then
  cp -r $RMA/scripts $RMA/scripts_g0
  python3 - <<'EOF'
from pathlib import Path
f = Path("xxxx/rma/eval/RoboMemArena-main/evaluation_benchmark/scripts_g0/eval_common.py")
s = f.read_text()
old = "LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]"
assert s.count(old) == 1
f.write_text(s.replace(old, 'LIBERO_DUMMY_ACTION = [0.0] * 6 + [float(__import__("os").environ.get("RMA_DUMMY_GRIP", "-1"))]'))
print("patched copy scripts_g0")
EOF
fi
sed -e "s#scripts/run_all_tasks1_26.py#scripts_g0/run_all_tasks1_26.py#; s#\$RMA/scripts:#\$RMA/scripts_g0:#" $HOME/pi05-abc/rma_eval.sh > $P/rma_eval_g0.sh
grep -c scripts_g0 $P/rma_eval_g0.sh
O=$P/grip_$GRIP; rm -rf $O
KW="{\"checkpoint_dir\": \"$HOME/pi05-abc-ckpt/C_c100k/checkpoint_050000\", \"gpu\": \"$GPU\", \"code_dir\": \"$HOME/pi05-abc/code_causal_b\", \"log_dir\": \"$O/adapter_log\", \"action_seed\": 50}"
RMA_DUMMY_GRIP=$GRIP RMA_OLD=xxxx/mirror/old PLAYWRIGHT_DIR=xxxx/mirror/pw/playwright_browsers LIBERO_SIMULATOR_DRIVER_ROOT=xxxx/mirror/old RMA_EVIDENCE=0 \
  timeout 900 bash $P/rma_eval_g0.sh 2 50 1 $GPU $O $HOME/pi05-abc/code/rma_adapter_fs.py "$KW" --max-steps 60 --replan-steps 20 --num-steps-wait 10 > $P/grip_$GRIP.log 2>&1
echo "rc=$?"; cat $O/adapter_log/first_state.jsonl 2>/dev/null | head -2; tail -n 3 $P/grip_$GRIP.log | cut -c1-200
