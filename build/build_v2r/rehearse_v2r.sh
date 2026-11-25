#!/bin/bash
# Rehearsal: 2 episodes per task (sorted positions 0 and 50 of each task's seeds) -> xxxx/rma/rehearsal_v2r
#   bash rehearse_v2r.sh [TRACKS_DIR=xxxx/rma/trackv/all/final_v4] [OUT_ROOT=xxxx/rma/rehearsal_v2r]
set -u
TRACKS=${1:-xxxx/rma/trackv/all/final_v4}
ROOT=${2:-xxxx/rma/rehearsal_v2r}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$ROOT"
$HOME/SANE/.venv/bin/python - > "$ROOT/episodes.txt" <<'EOF'
from collections import defaultdict
from pathlib import Path
seeds = defaultdict(list)
for p in Path("xxxx/rma/lr_eps/meta").glob("t*_seed*.npz"):
    seeds[int(p.stem[1:3])].append(int(p.stem.split("seed")[1]))
for task in sorted(seeds):
    s = sorted(seeds[task])
    for position in (0, 50):
        print(f"t{task:02d}_seed{s[position]}")
EOF
echo "rehearsal episodes: $(wc -l < "$ROOT/episodes.txt")"
export EPISODES_FILE="$ROOT/episodes.txt" PARTIAL=1 WORKERS=${WORKERS:-8}
exec bash "$HERE/pipeline_v2r.sh" "$TRACKS" "$ROOT"
