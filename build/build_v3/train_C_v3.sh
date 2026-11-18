#!/bin/bash
# v3 (scheme-B accumulated Action line, code_causal_v3, dataset_v3). Launch the round-2 scheme-C run: 50000 steps, batch 128 (4 GPUs x 32), checkpoints/validation every 10000,
# from pi05_base, code ~/pi05-abc/code_causal_v2, contract = contract_C_v2.json with dataset.latents_dir of the v2r
# dataset (written by the dataset build as <dataset>/contract_C_v2r.json; nothing else differs).
# Modelled on rma/r93_f32.sh and rma/r88_wait_train.sh (same environment variables, same export watcher).
#
#   setsid nohup bash ~/pi05-abc/build_v2r/train_C_v2r.sh 0,1,2,3 > /data/lgd/rma/logs/C_v2r.launch.log 2>&1 < /dev/null &
#
# usage: train_C_v2r.sh GPUS [RUN=C_v2r] [DATASET=/data/lgd/rma/dataset_v2r/dataset]
# Run it on the node that holds the dataset (built on 9031; /data is node-local). Output /data/lgd/rma/$RUN (node-local,
# five ~35 GB checkpoints), log /data/lgd/rma/logs/$RUN.log, inference files of each milestone exported to
# ~/pi05-abc-ckpt/$RUN/checkpoint_0X0000 by rma/r64_export.sh. Running the same command again resumes from the latest
# retained checkpoint (train.py does that itself). NEVER edit ~/pi05-abc/code_causal_v2 while this runs (source hash check).
set -u
GPUS=${1:?usage: train_C_v2r.sh GPU,GPU,GPU,GPU [RUN] [DATASET]}
RUN=${2:-C_v3}
DATASET=${3:-/data/lgd/rma/dataset_v3/dataset}
CODE=$HOME/pi05-abc/code_causal_v3
CONTRACT=$DATASET/contract_C_v3.json
NEED=${NEED:-56000}   # MiB free per GPU: 30 % of 183 GB is preallocated (measured peak ~34 GB per GPU at 4 x 32)
L=/data/lgd/rma/logs; OUT=/data/lgd/rma/$RUN
mkdir -p $L

[ "$(echo $GPUS | tr ',' '\n' | grep -c .)" -eq 4 ] || { echo "need exactly 4 GPU indices"; exit 2; }
[ -f "$DATASET/CHECK_OK" ] || { echo "$DATASET has no CHECK_OK: the dataset build did not pass its integrity check"; exit 2; }
[ -f "$CONTRACT" ] || { echo "missing $CONTRACT"; exit 2; }
grep -q '"partial": true' "$DATASET/split_manifest.json" && { echo "$DATASET is a partial (rehearsal) build"; exit 2; }
if pgrep -u "$(id -u)" -f "[t]rain.py .*--output $OUT( |\$)" > /dev/null; then echo "a train.py with --output $OUT is already running"; exit 2; fi
for g in $(echo $GPUS | tr ',' ' '); do
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | awk -F', ' '{print $1-$2}')
  [ "${free:-0}" -ge $NEED ] || { echo "GPU $g has only ${free:-?} MiB free (< $NEED)"; exit 2; }
done
ls -d $OUT/checkpoint_* > /dev/null 2>&1 || rm -rf $OUT   # nothing retained yet: start clean (as r88)
mkdir -p $OUT
for s in 10000 20000 30000 40000 50000; do
  pgrep -u "$(id -u)" -f "[r]64_export.sh $RUN $s\$" > /dev/null || \
    setsid nohup bash $HOME/pi05-abc/rma/r64_export.sh $RUN $s >> $L/${RUN}_export.log 2>&1 < /dev/null &
done
echo "$(date '+%m-%d %H:%M:%S') $(hostname) train $RUN on GPUs $GPUS dataset $DATASET" >> $L/$RUN.log
cd $CODE && unset FINETUNE_FROM TRACKER_IMAGES && umask 022 && \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$GPUS PI05_DEVICES=4 PI05_PER_DEVICE_BATCH=32 PI05_TOTAL_UPDATES=50000 \
PI05_CHECKPOINT_INTERVAL=10000 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
$HOME/SANE/.venv/bin/python -u train.py --contract $CONTRACT \
  --dataset $DATASET --output $OUT --no-wandb >> $L/$RUN.log 2>&1
echo "train exited rc=$? $(date)" >> $L/$RUN.log
