#!/bin/bash
# Runs on 9031 and on 9022 (both hold xxxx/rma/dataset_v2r/dataset with CHECK_OK). Every 5 min: sample all GPUs
# 10 times over ~1 min; a GPU counts as free when it is healthy, has >= 60000 MiB free and its max sampled utilization
# is <= 15 %. When 4 GPUs are free on 3 consecutive rounds (~15 min), take the shared claim (mkdir on /pfs, so only one
# node launches) and run train_C_v2r.sh on them. If the training dies before its first checkpoint within 1 h, release
# the claim and keep watching (max 3 attempts). User-approved 11-14.
D=xxxx/rma/dataset_v2r/dataset; L=xxxx/rma/logs; B=$HOME/pi05-abc/build_v2r
CLAIM=$HOME/pi05-abc/rma/stage/C_v2r.claim; H=$(hostname)
log(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
[ -f $D/CHECK_OK ] || { log "no CHECK_OK dataset on $H"; exit 2; }
streak=0; prev=""; attempts=0
log "watch start on $H"
while true; do
  if [ -d $CLAIM ] && [ "$(cat $CLAIM/host 2>/dev/null)" != "$H" ]; then
    log "claimed by $(cat $CLAIM/host 2>/dev/null), exit"; exit 0
  fi
  for i in $(seq 1 10); do
    nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 6
  done > /tmp/gw_$$.txt 2>/dev/null
  free=$(awk -F', ' '{ if ($4 !~ /^[0-9]+$/) bad[$1]=1; else { if ($4>mx[$1]) mx[$1]=$4; if (!($1 in mn) || $2-$3<mn[$1]) mn[$1]=$2-$3 }; seen[$1]=1 }
       END { for (g in seen) if (!(g in bad) && mn[g]>=60000 && mx[g]<=15) print g }' /tmp/gw_$$.txt | sort -n | head -4 | paste -sd,)
  rm -f /tmp/gw_$$.txt
  n=$(echo "$free" | tr ',' '\n' | grep -c .)
  if [ "$n" -eq 4 ]; then
    [ "$free" = "$prev" ] && streak=$((streak+1)) || streak=1; prev=$free
  else streak=0; prev=""; fi
  [ $(( $(date +%M) % 30 )) -lt 5 ] && log "free GPUs [$free] streak $streak"
  if [ $streak -ge 3 ]; then
    if ! mkdir $CLAIM 2>/dev/null; then log "claim taken by $(cat $CLAIM/host 2>/dev/null), exit"; exit 0; fi
    echo $H > $CLAIM/host; chmod -R a+rwx $CLAIM
    if ! CUDA_VISIBLE_DEVICES=$free XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 180 $HOME/SANE/.venv/bin/python -c "import jax; assert len(jax.devices('gpu'))==4" 2>/dev/null; then
      log "GPUs $free failed the CUDA test, release"; rm -rf $CLAIM; streak=0; sleep 300; continue
    fi
    attempts=$((attempts+1)); t0=$(date +%s)
    log "LAUNCH attempt $attempts on $H GPUs $free"
    bash $B/train_C_v2r.sh $free
    log "train_C_v2r.sh returned after $(( ($(date +%s)-t0)/60 )) min"
    if ! ls -d xxxx/rma/C_v2r/checkpoint_* >/dev/null 2>&1 && [ $(( $(date +%s)-t0 )) -lt 3600 ] && [ $attempts -lt 3 ]; then
      log "died early without checkpoint, release claim and keep watching"; tail -3 $L/C_v2r.log; rm -rf $CLAIM; streak=0; continue
    fi
    log "done (claim kept)"; exit 0
  fi
  sleep 240
done
