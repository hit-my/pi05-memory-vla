#!/bin/bash
# run_track.sh HDF5_DIR RAW "GPU GPU ..."   -- real_extract.py, then real_track.py split over the given GPUs (resumable).
# Paths below are the ones of the node this was built on; change them for another machine.
set -u
SRC=${1:?hdf5 dir}; RAW=${2:?raw out}; GPUS=(${3:?gpus})
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}                       # any python with h5py + Pillow
T=${TRACKER_HOME:-/path/to/tracker}               # tracker venv (torch + SAM3 deps + h5py)
mkdir -p "$RAW/logs"
[ -f "$RAW/extract.json" ] || PYTHONDONTWRITEBYTECODE=1 $PY "$HERE/real_extract.py" "$SRC" "$RAW" > "$RAW/logs/extract.log" || exit 1
N=$(ls "$SRC"/episode_init_*.hdf5 | wc -l); K=${#GPUS[@]}; per=$(( (N + K - 1) / K ))
for i in "${!GPUS[@]}"; do
  a=$(( i * per )); b=$(( a + per - 1 )); [ $b -ge $N ] && b=$(( N - 1 )); [ $a -gt $b ] && continue
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} PYTHONPATH=$T/gpu_extra:$T/assets/deps HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    $T/venv/bin/python -u "$HERE/real_track.py" "$SRC" "$RAW" $a $b > "$RAW/logs/track_$a.log" 2>&1 < /dev/null &
done
wait
echo "tracked $(ls "$RAW/tracks" | wc -l) of $(ls "$RAW/meta" | wc -l)"