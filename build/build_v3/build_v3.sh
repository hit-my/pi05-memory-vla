#!/bin/bash
# on 9031: code_causal_v3 (copy of code_causal_v2 + scheme-B patch), dataset_v3 (relabel of dataset_v2r), check.
set -eu
B=$HOME/pi05-abc/build_v3; C2=$HOME/pi05-abc/code_causal_v2; C3=$HOME/pi05-abc/code_causal_v3
PY=$HOME/SANE/.venv/bin/python
D2=/data/lgd/rma/dataset_v2r/dataset; D3=/data/lgd/rma/dataset_v3/dataset
log(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
if [ ! -d $C3 ]; then
  cp -a $C2 $C3; rm -rf $C3/__pycache__
  $PY $B/patch_v3.py $C3
fi
cd $C3 && PYTHONDONTWRITEBYTECODE=1 $PY -c "
from memory_format import accumulate_action as A, cap_action, action_entities, render_action, step_memory
w1='blue cookies: up ; picked | orange sauce: still'
a1=A(None,w1,False); assert a1=='blue cookies: up -> picked | orange sauce: still', a1
a2=A(a1,'blue cookies: up -> left | orange sauce: occluded',False); assert a2=='blue cookies: up -> picked -> up -> left | orange sauce: occluded', a2
a3=A(a2,'blue cookies: still | orange sauce: still',False); assert a3=='blue cookies: up -> picked -> up -> left | orange sauce: still', a3
a4=A(a3,'blue cookies: down ; placed | orange sauce: still',True); assert a4=='blue cookies: down -> placed | orange sauce: still', a4
long='blue cookies: '+' -> '.join(['up','down']*8)
assert action_entities(long,'A') and len(cap_action(long).split(' -> '))==12
assert action_entities(render_action(a2,'C'),'C')
print('unit tests ok')"
log "relabel"
cd /tmp && PYTHONDONTWRITEBYTECODE=1 $PY -u $B/relabel_v3.py $D2 $D3 $C3
log "check"
cd /tmp && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY -u $B/check_v3.py $D2 $D3 $C3
log "BUILD_V3_END"
