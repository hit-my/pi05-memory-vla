#!/bin/bash
# usage: rma_eval_v5.sh TASK SEED TRIALS GPU OUT ADAPTER_PY ADAPTER_KWARGS_JSON [extra runner args]
# = xxxx/rma/probe_g0/rma_eval_g0.sh (official runner copy scripts_g0) with
#   RMA_DUMMY_GRIP=0 (the 10 warm-up steps send gripper command 0 instead of -1), the local mirrors of this node,
#   and the v5 tracker subprocess settings (rma_perception_v5.TrackerProcessV5; same GPU as the policy server).
OLD=${RMA_OLD:-xxxx/mirror/old}
RMA=xxxx/rma/eval/RoboMemArena-main/evaluation_benchmark
TASK=$1; SEED=$2; TRIALS=$3; GPU=$4; OUT=$5; ADAPTER=$6; KW=$7; shift 7
export RMA_DUMMY_GRIP=${RMA_DUMMY_GRIP:-0} RMA_EVIDENCE=0
export PYTHONPATH=$RMA/scripts_g0:$RMA/libero_fork:$OLD/vendor/libero-mem/thirdparty/robomimic:$OLD/vendor/libero-mem/thirdparty/robosuite
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1
export SIMULATOR_TORCH_SITE=$HOME/pi05-abc/torch_site
export LIBERO_CONFIG_PATH=xxxx/rma/eval/libero_config
export NUMBA_CACHE_DIR=xxxx/pi05-abc/cache/numba MPLCONFIGDIR=xxxx/pi05-abc/cache/mpl XDG_CACHE_HOME=xxxx/pi05-abc/cache/xdg PYTHONDONTWRITEBYTECODE=1
export RMA_LOG_DIR=$OUT/adapter_log
export LIBERO_SIMULATOR_DRIVER_ROOT=$OLD
export PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_DIR:-xxxx/mirror/pw/playwright_browsers}
[ -d xxxx/mirror/syslib ] && export LD_LIBRARY_PATH=xxxx/mirror/syslib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
T=xxxx/pi05-abc/tracker
export TRACKER_PYTHON=$T/venv/bin/python TRACKER_ROOT=$T/assets TRACKER_V5_PYTHONPATH=$T/gpu_extra:$T/assets/deps
export TRACKER_THREADS=${TRACKER_THREADS:-2} TRACKER_V5_THREADS=${TRACKER_V5_THREADS:-2} TRACKER_V5_LOG=$OUT/adapter_log/tracker_stderr.log
unset CHECKS
mkdir -p $OUT/adapter_log
cd $RMA
exec $OLD/sim-venv/bin/python scripts_g0/run_all_tasks1_26.py --adapter-spec $ADAPTER:build_adapter --adapter-kwargs "$KW" \
  --task-start $TASK --task-end $TASK --num-trials-per-task $TRIALS --seed $SEED --out-root $OUT "$@"
