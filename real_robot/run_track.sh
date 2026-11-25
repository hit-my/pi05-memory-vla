#!/bin/bash
# run_track.sh LEROBOT_DIR RAW "GPU GPU ..."   -- real_extract.py, then real_track.py split over the given GPUs.
# Paths below are the ones of the node this was built on (lgd 9031); change them for another machine.
set -u
SRC=${1:?lerobot dir}; RAW=${2:?raw out}; GPUS=(${3:?gpus})
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/SANE/.venv/bin/python}                       # any python with pandas + pyarrow
T=${TRACKER_HOME:-xxxx/pi05-abc/tracker}               # tracker venv (torch + SAM3 deps)
mkdir -p "$RAW/logs"
PYTHONDONTWRITEBYTECODE=1 $PY "$HERE/real_extract.py" "$SRC" "$RAW" || exit 1
N=$($PY -c "import json;print(json.load(open('$SRC/meta/info.json'))['total_episodes'])")
K=${#GPUS[@]}; per=$(( (N + K - 1) / K ))
for i in "${!GPUS[@]}"; do
  a=$(( i * per )); b=$(( a + per - 1 )); [ $b -ge $N ] && b=$(( N - 1 ))
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} PYTHONPATH=$T/gpu_extra:$T/assets/deps HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    setsid nohup $T/venv/bin/python -u "$HERE/real_track.py" "$SRC" "$RAW" $a $b > "$RAW/logs/track_$a.log" 2>&1 < /dev/null &
done
wait
grep -h -c TRACKED "$RAW"/logs/track_*.log | paste -sd+ | bc
