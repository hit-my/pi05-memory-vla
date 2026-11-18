#!/bin/bash
# usage: rma_eval.sh TASK SEED TRIALS GPU OUT ADAPTER_PY ADAPTER_KWARGS_JSON [extra runner args]
# One RoboMemArena runner process for one task and a contiguous block of seeds (seed..seed+TRIALS-1).
OLD=${RMA_OLD:-/cpfs01/USER/pi05-full-t06-t08}  # idle nodes without /cpfs01 use a local mirror (RMA_OLD, PLAYWRIGHT_DIR)
RMA=/data/lgd/rma/eval/RoboMemArena-main/evaluation_benchmark
TASK=$1; SEED=$2; TRIALS=$3; GPU=$4; OUT=$5; ADAPTER=$6; KW=$7; shift 7
export PYTHONPATH=$RMA/scripts_g0:$RMA/libero_fork:$OLD/vendor/libero-mem/thirdparty/robomimic:$OLD/vendor/libero-mem/thirdparty/robosuite:$HOME/pi05-abc/code
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1
export SIMULATOR_TORCH_SITE=$HOME/pi05-abc/torch_site
export LIBERO_CONFIG_PATH=/data/lgd/rma/eval/libero_config
export NUMBA_CACHE_DIR=/data/lgd/pi05-abc/cache/numba MPLCONFIGDIR=/data/lgd/pi05-abc/cache/mpl XDG_CACHE_HOME=/data/lgd/pi05-abc/cache/xdg PYTHONDONTWRITEBYTECODE=1
export RMA_LOG_DIR=$OUT/adapter_log
export LIBERO_SIMULATOR_DRIVER_ROOT=$OLD
export PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_DIR:-/cpfs01/USER/pi05-english-10k/evaluation_formal_10k/playwright_browsers}
[ -d /data/lgd/mirror/syslib ] && export LD_LIBRARY_PATH=/data/lgd/mirror/syslib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # libasound for chromium on nodes without it
T=/data/lgd/pi05-abc/tracker
export TRACKER_PYTHON=$T/venv/bin/python TRACKER_ROOT=$T/assets TRACKER_PYTHONPATH=$T/gpu_extra:$HOME/pi05-abc/code TRACKER_THREADS=${TRACKER_THREADS:-2}
mkdir -p $OUT
cd $RMA
exec $OLD/sim-venv/bin/python scripts_g0/run_all_tasks1_26.py --adapter-spec $ADAPTER:build_adapter --adapter-kwargs "$KW" \
  --task-start $TASK --task-end $TASK --num-trials-per-task $TRIALS --seed $SEED --out-root $OUT "$@"
