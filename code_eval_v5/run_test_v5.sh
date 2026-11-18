#!/bin/bash
# run_test_v5.sh OUT NPROC PER_TASK MODE GPU [ep ...] : test_perception_v5.py in NPROC shards (simulator venv + tracker subprocess)
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; N=$2; PER=$3; MODE=$4; GPU=$5; shift 5
SIM=/data/lgd/mirror/old
export LIBERO_SIMULATOR_DRIVER_ROOT=$SIM PLAYWRIGHT_BROWSERS_PATH=/data/lgd/mirror/pw/playwright_browsers PYTHONDONTWRITEBYTECODE=1
[ -d /data/lgd/mirror/syslib ] && export LD_LIBRARY_PATH=/data/lgd/mirror/syslib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TRACKER_V5_THREADS=2 TRACKER_THREADS=2
unset CHECKS
mkdir -p $OUT/logs
pids=()
for s in $(seq 0 $((N-1))); do
  TRACKER_V5_LOG=$OUT/logs/tracker_$s.log $SIM/sim-venv/bin/python -u $CODE/test_perception_v5.py $OUT $s $N $PER $MODE "$@" > $OUT/logs/test_$s.log 2>&1 &
  pids+=($!)
done
echo "${pids[*]}" > $OUT/logs/pids
for p in "${pids[@]}"; do wait $p; done
echo ALLDONE > $OUT/logs/done
