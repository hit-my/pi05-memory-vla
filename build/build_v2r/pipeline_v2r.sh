#!/bin/bash
# pipeline_v2r.sh TRACKS_DIR OUT_ROOT     (generic; use build_full_v2r.sh or rehearse_v2r.sh)
#   OUT_ROOT/dataset            the training dataset (images, episodes, rows, train/val.jsonl, norm_stats, manifest, priors)
#   OUT_ROOT/dataset/latents/C  scheme-C latent targets z* + standardize.json  (contract dataset.latents_dir)
#   OUT_ROOT/dataset/contract_C_v2r.json   = contract_C_v2.json with dataset.latents_dir pointing here
#   OUT_ROOT/logs               build_<shard>.log, assemble.log, latents.log, check.log, check_report.json
#   OUT_ROOT/dataset/CHECK_OK   written only when the final integrity check passed
# env: WORKERS (default 8, hard cap 8), EPISODES_FILE (subset; implies a partial build), PARTIAL=1,
#      PREV (previous dataset for the diff, default /data/lgd/rma/full/dataset_causal), CODE, RAW,
#      LATENTS_GPU (GPU index for the small target encoder; default: CPU), IMAGE_SAMPLES, SKIP_LOADER=1
# Resumable: run the same command again; finished episodes are skipped, later stages are redone (they are cheap).
set -u
TRACKS=${1:?tracks dir}; ROOT=${2:?output root}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAW=${RAW:-/data/lgd/rma/lr_eps}
CODE=${CODE:-$HOME/pi05-abc/code_causal_v2}
PREV=${PREV:-/data/lgd/rma/full/dataset_causal}
WORKERS=${WORKERS:-8}; [ "$WORKERS" -gt 8 ] && WORKERS=8
PARTIAL=${PARTIAL:-0}; [ -n "${EPISODES_FILE:-}" ] && PARTIAL=1
D=$ROOT/dataset; L=$ROOT/logs
SIM=/data/lgd/mirror/old
SIMPY=$SIM/sim-venv/bin/python
PY=$HOME/SANE/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

[ -d "$TRACKS" ] || { echo "no tracks dir $TRACKS"; exit 2; }
mkdir -p "$D" "$L"
if ! mkdir "$ROOT/.lock" 2>/dev/null; then echo "another pipeline_v2r.sh holds $ROOT/.lock (remove it if that run is dead)"; exit 2; fi
trap 'rmdir "$ROOT/.lock" 2>/dev/null' EXIT
rm -f "$D/CHECK_OK"
say "START tracks=$TRACKS out=$ROOT workers=$WORKERS partial=$PARTIAL host=$(hostname)"

# ---- 0. preflight: one track file per episode, with the three arrays
$PY - "$TRACKS" "$RAW" "${EPISODES_FILE:-}" <<'EOF' || { echo "PREFLIGHT FAILED"; exit 3; }
import sys, zipfile
from pathlib import Path
tracks, raw, subset = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
eps = sorted(p.stem for p in (raw / "meta").glob("t*_seed*.npz"))
if subset:
    eps = [x for x in Path(subset).read_text().split() if x]
missing = [e for e in eps if not (tracks / f"{e}.npz").is_file()]
bad = []
for e in eps:
    if e in missing:
        continue
    try:
        names = set(zipfile.ZipFile(tracks / f"{e}.npz").namelist())
    except Exception as exc:  # truncated file still being written
        bad.append(f"{e}: {exc}"); continue
    if not {"centres.npy", "visibility.npy", "retired.npy"} <= names:
        bad.append(f"{e}: keys {sorted(names)}")
print(f"preflight: {len(eps)} episodes, {len(missing)} without a track file, {len(bad)} unreadable/incomplete")
if missing or bad:
    print("  missing:", missing[:5], " bad:", bad[:3]); sys.exit(1)
EOF

# ---- 1. row writer + images, WORKERS shards (simulator venv: headless Chromium renderer; no GPU)
export LIBERO_SIMULATOR_DRIVER_ROOT=$SIM PLAYWRIGHT_BROWSERS_PATH=/data/lgd/mirror/pw/playwright_browsers
[ -d /data/lgd/mirror/syslib ] && export LD_LIBRARY_PATH=/data/lgd/mirror/syslib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
pids=()
for i in $(seq 0 $((WORKERS - 1))); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $SIMPY -u "$HERE/rma_build_v2r.py" --raw "$RAW" --tracks "$TRACKS" --out "$D" \
     --shard $i --shards $WORKERS --code "$CODE" ${EPISODES_FILE:+--episodes-file "$EPISODES_FILE"} ${REBUILD_STALE:+--rebuild-stale} \
     >> "$L/build_$i.log" 2>&1 < /dev/null &
  pids+=($!)
done
say "row writers started: ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do wait $pid || fail=1; done
for i in $(seq 0 $((WORKERS - 1))); do tail -1 "$L/build_$i.log" | grep -q "^BUILD_DONE $i" || { fail=1; echo "shard $i did not finish:"; tail -5 "$L/build_$i.log"; }; done
[ $fail = 0 ] || { say "ROW WRITER FAILED (fix and run the same command again; finished episodes are kept)"; exit 4; }
say "row writers done: $(ls "$D/rows_base" | wc -l) episodes"

# ---- 2. causal labels, split, normalization, manifest, contract
PF=""; [ "$PARTIAL" = 1 ] && PF="--partial"
$PY -u "$HERE/rma_assemble_v2r.py" --out "$D" --raw "$RAW" --tracks "$TRACKS" --code "$CODE" \
    --latents-dir "$D/latents/C" --contract-out "$D/contract_C_v2r.json" $PF > "$L/assemble.log" 2>&1 \
  || { say "ASSEMBLE FAILED"; tail -5 "$L/assemble.log"; exit 5; }
tail -3 "$L/assemble.log"

# ---- 3. scheme-C latent targets: the unchanged rma_latents.py (target encoder: rec 30 + cons + SupCon 3, 3000 steps,
#         then train-only verb-balanced standardization) on the NEW centres / visibility / Action verbs
rm -rf "$D/latents.tmp"
if [ -n "${LATENTS_GPU:-}" ]; then
  LAT_ENV="CUDA_VISIBLE_DEVICES=$LATENTS_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false"; PIN=""
else
  LAT_ENV="CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu"
  CPUS=$($PY - <<'EOF'
import time
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); v = list(map(int, f[1:])); out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out
a = snap(); time.sleep(1.0); b = snap()
busy = sorted((1 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1), c) for c in a)
print(",".join(str(c) for _, c in busy[:8]))
EOF
)
  PIN="taskset -c $CPUS"
fi
( cd "$L" && env $LAT_ENV PYTHONPATH="$CODE" $PIN $PY -u "$CODE/rma_latents.py" "$D" "$D/latents.tmp" ) > "$L/latents.log" 2>&1 \
  || { say "LATENTS FAILED"; tail -5 "$L/latents.log"; exit 6; }
grep -q LATENTS_DONE "$L/latents.log" || { say "LATENTS FAILED (no LATENTS_DONE)"; exit 6; }
rm -rf "$D/latents" && mv "$D/latents.tmp" "$D/latents"
grep -E "knn5|recon_rmse|majority|LATENTS_DONE" "$L/latents.log"

# ---- 4. integrity check (includes the training loader and the diff against the previous dataset)
$PY -u "$HERE/rma_check_v2r.py" --dataset "$D" --raw "$RAW" --tracks "$TRACKS" --contract "$D/contract_C_v2r.json" --code "$CODE" \
    --prev "$PREV" $PF --image-samples ${IMAGE_SAMPLES:-400} --report "$L/check_report.json" ${SKIP_LOADER:+--skip-loader} > "$L/check.log" 2>&1
rc=$?
grep -E "^(PASS|FAIL|CHECK_RESULT)|^     " "$L/check.log"
if [ $rc = 0 ]; then
  { echo "tracks=$TRACKS"; date; sha256sum "$D/train.jsonl" "$D/val.jsonl" "$D/norm_stats.json" "$D/latents/C/standardize.json"; } > "$D/CHECK_OK"
  say "PIPELINE_OK $D  ($(du -sh "$D" | cut -f1))"
else
  say "CHECK FAILED (see $L/check.log); dataset left in place, CHECK_OK not written"; exit 7
fi
