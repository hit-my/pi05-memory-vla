#!/bin/bash
# on 9031: wait until the v2r dataset build passed its check (CHECK_OK), then stage it as one tar on the shared /pfs
# home for the training node (9022). Writes stage/dataset_v2r.tar and then stage/dataset_v2r.tar.done.
set -e
D=/data/lgd/rma/dataset_v2r; S=$HOME/pi05-abc/rma/stage; L=/data/lgd/rma/logs
mkdir -p $S $L
echo "STAGE wait $(date)"
until [ -f $D/dataset/CHECK_OK ]; do
  if ! pgrep -u "$(id -u)" -f "[p]ipeline_v2r.sh|[b]uild_full_v2r.sh|[g]uard_v2r.sh" > /dev/null && [ ! -f $D/dataset/CHECK_OK ]; then
    sleep 300
    if ! pgrep -u "$(id -u)" -f "[p]ipeline_v2r.sh|[b]uild_full_v2r.sh|[g]uard_v2r.sh" > /dev/null && [ ! -f $D/dataset/CHECK_OK ]; then
      echo "BUILD ENDED WITHOUT CHECK_OK $(date)"; exit 3
    fi
  fi
  sleep 120
done
echo "STAGE tar start $(date)"
test ! -e $S/dataset_v2r.tar
tar cf $S/dataset_v2r.tar.part -C $D dataset
mv $S/dataset_v2r.tar.part $S/dataset_v2r.tar
touch $S/dataset_v2r.tar.done
echo "STAGE done $(date) $(du -sh $S/dataset_v2r.tar | cut -f1)"
