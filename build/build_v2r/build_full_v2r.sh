#!/bin/bash
# THE ONE COMMAND: build the full round-2 training dataset (scheme C, 2600 episodes) from a tracks directory.
#
#   setsid nohup bash ~/pi05-abc/build_v2r/build_full_v2r.sh xxxx/rma/trackv/all/final_v5 \
#        > xxxx/rma/dataset_v2r.build.log 2>&1 < /dev/null &
#
# Output: xxxx/rma/dataset_v2r/dataset (+ latents/C, contract_C_v2r.json, CHECK_OK) and xxxx/rma/dataset_v2r/logs.
# Run on node 9031 (the LeRobot arrays xxxx/rma/lr_eps, the renderer environment xxxx/mirror and the previous
# dataset for the diff are node-local there). 8 CPU workers, no GPU. Resumable: run the same command again.
# It stops (nothing is half-accepted) when: a track file is missing/unreadable, an episode was already built from a
# different track file (set REBUILD_STALE=1 to rebuild those), or the final integrity check fails.
# Progress:  ls xxxx/rma/dataset_v2r/dataset/rows_base | wc -l      (of 2600)
set -u
TRACKS=${1:?usage: build_full_v2r.sh TRACKS_DIR [OUT_ROOT]}
ROOT=${2:-xxxx/rma/dataset_v2r}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
unset EPISODES_FILE; export PARTIAL=0 WORKERS=${WORKERS:-8}
n=$(ls "$TRACKS" | grep -c '^t[0-9][0-9]_seed[0-9]*\.npz$')
[ "$n" -eq 2600 ] || { echo "expected 2600 track files in $TRACKS, found $n (is the tracking still running?)"; exit 2; }
exec bash "$HERE/pipeline_v2r.sh" "$TRACKS" "$ROOT"
