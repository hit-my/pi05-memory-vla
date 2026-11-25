#!/bin/bash
# GPU watcher for the C_v3 training (scheme-B labels), one copy per lgd node (9022 9024 9025 9027 9028 9031 9032 9033). User-approved 11-15.
# Every ~5 min: sample all GPUs 10x over ~1 min; a GPU is free when healthy, >= 60000 MiB free, max util <= 15 %.
# 4 free GPUs (same set) on 3 consecutive rounds -> take the shared claim (atomic mkdir on /pfs; one node only).
# If this node has no checked dataset yet: unpack the staged tar from /pfs, verify the index sha256, then re-check
# that the GPUs are still free. CUDA test, then train_C_v3.sh. Early death (<1 h, no checkpoint): release, keep watching
# (max 3 launches). 1 h after a healthy launch the staged tar is deleted.
D=xxxx/rma/dataset_v3; L=xxxx/rma/logs; B=$HOME/pi05-abc/build_v3; S=$HOME/pi05-abc/rma/stage
CLAIM=$S/C_v3.claim; H=$(hostname); PY=$HOME/SANE/.venv/bin/python
export NCCL_NVLS_ENABLE=0
mkdir -p $L
log(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
sample(){  # prints the free GPU list (up to 4)
  for i in $(seq 1 10); do nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 6; done 2>/dev/null |
  awk -F', ' '{ if ($4 !~ /^[0-9]+$/) bad[$1]=1; else { if ($4>mx[$1]) mx[$1]=$4; if (!($1 in mn) || $2-$3<mn[$1]) mn[$1]=$2-$3 }; seen[$1]=1 }
       END { for (g in seen) if (!(g in bad) && mn[g]>=60000 && mx[g]<=15) print g }' | sort -n | head -4 | paste -sd,
}
release(){ rm -rf $CLAIM; streak=0; prev=""; }
unpack(){
  [ -f $D/dataset/CHECK_OK ] && return 0
  [ -f $S/dataset_v3.tar.done ] || { log "no staged tar yet"; return 1; }
  if [ -e $D/dataset ]; then
    [ -f $D/.unpacking ] || { log "$D/dataset exists without CHECK_OK and not ours; not touching it"; return 1; }
    rm -rf $D/dataset
  fi
  mkdir -p $D && touch $D/.unpacking
  log "unpack start"; tar xf $S/dataset_v3.tar -C $D || { log "untar failed"; return 1; }
  ( cd $D/dataset && $PY -c "
import json, hashlib
m = json.load(open('split_manifest.json'))
for p in ('train', 'val'):
    h = hashlib.sha256()
    with open(p + '.jsonl', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    assert h.hexdigest() == m['splits'][p]['index_sha256'], p
" ) || { log "sha256 mismatch"; return 1; }
  [ -f $D/dataset/CHECK_OK ] || { log "tar had no CHECK_OK"; return 1; }
  rm -f $D/.unpacking; log "unpack ok: $(ls $D/dataset/episodes | wc -l) episodes"; return 0
}
streak=0; prev=""; attempts=0
log "watch start on $H (local dataset: $([ -f $D/dataset/CHECK_OK ] && echo yes || echo no))"
while true; do
  if [ -d $CLAIM ] && [ "$(cat $CLAIM/host 2>/dev/null)" != "$H" ]; then
    # another node holds the claim: keep waiting (it may release); exit once its training has run for 2 h
    if [ -f $CLAIM/running ] && [ $(( $(date +%s) - $(stat -c %Y $CLAIM/running) )) -gt 7200 ]; then
      log "training running on $(cat $CLAIM/host) for > 2 h, exit"; exit 0
    fi
    streak=0; prev=""; sleep 300; continue
  fi
  free=$(sample); n=$(echo "$free" | tr ',' '\n' | grep -c .)
  if [ "$n" -eq 4 ]; then [ "$free" = "$prev" ] && streak=$((streak+1)) || streak=1; prev=$free; else streak=0; prev=""; fi
  [ $(( 10#$(date +%M) % 30 )) -lt 5 ] && log "free GPUs [$free] streak $streak"
  if [ $streak -ge 3 ]; then
    if ! mkdir $CLAIM 2>/dev/null; then log "claim taken by $(cat $CLAIM/host 2>/dev/null)"; streak=0; prev=""; continue; fi
    echo $H > $CLAIM/host; chmod -R a+rwx $CLAIM
    log "claimed with GPUs $free"
    if ! unpack; then release; sleep 240; continue; fi
    free2=$(sample)
    if [ "$(echo "$free2" | tr ',' '\n' | grep -c .)" -ne 4 ]; then log "GPUs no longer free after unpack ([$free2]), release"; release; sleep 240; continue; fi
    free=$free2
    if ! CUDA_VISIBLE_DEVICES=$free XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 180 $PY -c "import jax; assert len(jax.devices('gpu'))==4" 2>/dev/null; then
      log "GPUs $free failed the CUDA test, release"; release; sleep 240; continue
    fi
    attempts=$((attempts+1)); t0=$(date +%s)
    log "LAUNCH attempt $attempts on $H GPUs $free"; touch $CLAIM/running; chmod a+rw $CLAIM/running
    ( sleep 3600; pgrep -u $(id -u) -f "[c]ode_causal_v2/train.py" >/dev/null && rm -f $S/dataset_v3.tar $S/dataset_v3.tar.done && log "staged tar deleted" ) &
    bash $B/train_C_v3.sh $free
    log "train_C_v3.sh returned after $(( ($(date +%s)-t0)/60 )) min"
    if ! ls -d xxxx/rma/C_v3/checkpoint_* >/dev/null 2>&1 && [ $(( $(date +%s)-t0 )) -lt 3600 ] && [ $attempts -lt 3 ]; then
      log "died early without checkpoint, release and keep watching"; tail -3 $L/C_v3.log; release; continue
    fi
    log "done (claim kept)"; exit 0
  fi
  sleep 240
done
