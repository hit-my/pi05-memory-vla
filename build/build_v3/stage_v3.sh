#!/bin/bash
# on 9031: replace the stale v2r staging tar with the checked dataset_v3 tar (for nodes without a local dataset_v3)
set -e
D=/data/lgd/rma/dataset_v3; S=$HOME/pi05-abc/rma/stage
[ -f $D/dataset/CHECK_OK ]
rm -f $S/dataset_v2r.tar $S/dataset_v2r.tar.done
test ! -e $S/dataset_v3.tar
echo "STAGE3 start $(date)"
tar cf $S/dataset_v3.tar.part -C $D dataset
chmod a+r $S/dataset_v3.tar.part; mv $S/dataset_v3.tar.part $S/dataset_v3.tar
touch $S/dataset_v3.tar.done; chmod a+rw $S/dataset_v3.tar.done
echo "STAGE3 done $(date) $(du -sh $S/dataset_v3.tar | cut -f1)"
