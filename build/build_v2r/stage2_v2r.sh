#!/bin/bash
# on 9031: stage the checked v2r dataset as one tar on /pfs for the other nodes' watchers (readable by every uid)
set -e
D=/data/lgd/rma/dataset_v2r; S=$HOME/pi05-abc/rma/stage
[ -f $D/dataset/CHECK_OK ]
test ! -e $S/dataset_v2r.tar
echo "STAGE2 tar start $(date)"
tar cf $S/dataset_v2r.tar.part -C $D dataset
chmod a+r $S/dataset_v2r.tar.part
mv $S/dataset_v2r.tar.part $S/dataset_v2r.tar
touch $S/dataset_v2r.tar.done; chmod a+rw $S/dataset_v2r.tar.done
echo "STAGE2 done $(date) $(du -sh $S/dataset_v2r.tar | cut -f1)"
