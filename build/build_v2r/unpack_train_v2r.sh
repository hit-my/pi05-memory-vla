#!/bin/bash
# on 9022: wait for the staged v2r dataset, unpack it to the same path as on 9031 (the contract's latents_dir is
# absolute), verify the index sha256 against split_manifest.json, delete the staging tar, then start the 50k-step
# scheme-C training on 4 GPUs as soon as 4 GPUs have enough free memory (idle GPUs preferred). Approved by the user.
set -e
S=$HOME/pi05-abc/rma/stage; L=/data/lgd/rma/logs; D=/data/lgd/rma/dataset_v2r
mkdir -p $L
log(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
log "wait for stage"
until [ -f $S/dataset_v2r.tar.done ]; do sleep 60; done
if [ ! -f $D/dataset/CHECK_OK ]; then
  log "unpack"
  test ! -e $D/dataset
  mkdir -p $D
  tar xf $S/dataset_v2r.tar -C $D
fi
cd $D/dataset
$HOME/SANE/.venv/bin/python -c "
import json, hashlib
m = json.load(open('split_manifest.json'))
for p in ('train', 'val'):
    h = hashlib.sha256()
    with open(p + '.jsonl', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    assert h.hexdigest() == m['splits'][p]['index_sha256'], p
print('index sha256 ok')"
log "$(ls episodes | wc -l) episodes, $(ls images | wc -l) image dirs, latents: $(ls latents/C | wc -l) files"
grep -o '"latents_dir": *"[^"]*"' contract_C_v2r.json
rm -f $S/dataset_v2r.tar $S/dataset_v2r.tar.done
log "READY"
NEED=60000
while true; do
  idle=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}' | head -4 | paste -sd,)
  roomy=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' -v n=$NEED '($2-$3)>=n{print $1}' | head -4 | paste -sd,)
  G=""
  [ "$(echo $idle | tr ',' '\n' | grep -c .)" -eq 4 ] && G=$idle
  [ -z "$G" ] && [ "$(echo $roomy | tr ',' '\n' | grep -c .)" -eq 4 ] && G=$roomy
  if [ -n "$G" ]; then
    log "launch on GPUs $G"
    bash $HOME/pi05-abc/build_v2r/train_C_v2r.sh $G
    log "train_C_v2r.sh returned rc=$?"
    exit 0
  fi
  log "no 4 GPUs free (idle=$idle roomy=$roomy), retry in 10 min"
  sleep 600
done
