#!/bin/bash
# run_build.sh RAW DATASET CODE CONTRACT_IN [WORKERS=8]
#   RAW          output of run_track.sh
#   DATASET      output: the trainer-ready dataset
#   CODE         training code directory (code_causal_v3)
#   CONTRACT_IN  a scheme-C contract whose model.* paths are valid on this machine (contracts/contract_C_v3.json edited,
#                or the simulator dataset's contract)
# Steps: 2 images + rows (real_build.py, simulator venv with the Chromium renderer, CPU only)
#        3 split / normalization / manifest / contract (real_assemble.py)
#        4 scheme-C latent targets (CODE/rma_latents.py, unchanged)
#        5 check with the training loader (real_check.py)
# Paths below are those of the node this was built on (lgd 9031); set the variables for another machine.
set -u
RAW=${1:?}; D=${2:?}; CODE=${3:?}; CIN=${4:?}; WORKERS=${5:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
SIM=${LIBERO_SIMULATOR_DRIVER_ROOT:-/data/lgd/mirror/old}        # data_common.py, render_adapter.py
SIMPY=${SIMPY:-$SIM/sim-venv/bin/python}                          # python with playwright
PY=${PY:-$HOME/SANE/.venv/bin/python}                             # the training python (openpi, jax, torch)
export LIBERO_SIMULATOR_DRIVER_ROOT=$SIM PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH:-/data/lgd/mirror/pw/playwright_browsers}
[ -d /data/lgd/mirror/syslib ] && export LD_LIBRARY_PATH=/data/lgd/mirror/syslib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
L=$D/logs; mkdir -p "$L"
say(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $SIMPY -u "$HERE/real_build.py" --raw "$RAW" --out "$D" --code "$CODE" \
    --shard $i --shards $WORKERS >> "$L/build_$i.log" 2>&1 < /dev/null &
  pids+=($!)
done
fail=0; for pid in "${pids[@]}"; do wait $pid || fail=1; done
[ $fail = 0 ] || { say "BUILD FAILED"; tail -3 "$L"/build_*.log; exit 4; }
say "rows + images done: $(ls "$D/rows" | wc -l) episodes"

$PY -u "$HERE/real_assemble.py" --out "$D" --contract-in "$CIN" --contract-out "$D/contract_C_real.json" ${ASSEMBLE_ARGS:-} > "$L/assemble.log" 2>&1 \
  || { say "ASSEMBLE FAILED"; tail -5 "$L/assemble.log"; exit 5; }
tail -2 "$L/assemble.log"

rm -rf "$D/latents.tmp"
if [ -n "${LATENTS_GPU:-}" ]; then LAT_ENV="CUDA_VISIBLE_DEVICES=$LATENTS_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false"; else LAT_ENV="CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu"; fi
( cd "$L" && env $LAT_ENV PYTHONPATH="$CODE" $PY -u "$CODE/rma_latents.py" "$D" "$D/latents.tmp" ) > "$L/latents.log" 2>&1 \
  || { say "LATENTS FAILED"; tail -5 "$L/latents.log"; exit 6; }
grep -q LATENTS_DONE "$L/latents.log" || { say "LATENTS FAILED (no LATENTS_DONE)"; exit 6; }
rm -rf "$D/latents" && mv "$D/latents.tmp" "$D/latents"
grep -E "knn5|recon_rmse|majority|LATENTS_DONE" "$L/latents.log" | tail -4

CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY -u "$HERE/real_check.py" "$D" "$D/contract_C_real.json" "$CODE" > "$L/check.log" 2>&1
rc=$?; tail -4 "$L/check.log"
[ $rc = 0 ] && say "DATASET READY: $D" || { say "CHECK FAILED"; exit 7; }
