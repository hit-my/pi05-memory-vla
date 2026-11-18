#!/bin/bash
# on 9031: keep the resumable v2r build alive. If no new episode is finished for 30 min while the row writers run
# (headless renderer hang, seen 11-14 16:40), kill ONLY this user's row writers + their renderer children by PID,
# wait for pipeline_v2r.sh to exit, and run build_full_v2r.sh again (finished episodes are skipped). Max 3 restarts.
# Exits when CHECK_OK exists, or when the pipeline ended on its own in a later stage (assemble/latents/check).
D=/data/lgd/rma/dataset_v2r; L=/data/lgd/rma/logs; B=$HOME/pi05-abc/build_v2r; U=$(id -u)
log(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
start_build(){
  setsid nohup bash $B/build_full_v2r.sh /data/lgd/rma/trackv/all/final_v5 >> /data/lgd/rma/dataset_v2r.build.log 2>&1 < /dev/null &
  log "build started pid $!"
}
restarts=0; last=-1; since=$(date +%s)
log "guard start"
while true; do
  [ -f $D/dataset/CHECK_OK ] && { log "CHECK_OK, guard exit"; exit 0; }
  n=$(ls $D/dataset/rows_base 2>/dev/null | wc -l)
  writers=$(pgrep -u $U -f "[r]ma_build_v2r.py" | paste -sd' ')
  pipe=$(pgrep -u $U -f "[p]ipeline_v2r.sh" | paste -sd' ')
  if [ "$n" != "$last" ]; then last=$n; since=$(date +%s); fi
  if [ -z "$pipe" ]; then
    if [ "$n" -lt 2600 ] && [ $restarts -lt 3 ]; then
      log "pipeline not running at $n/2600, restart $((restarts+1))"; tail -3 /data/lgd/rma/dataset_v2r.build.log
      rmdir $D/.lock 2>/dev/null; restarts=$((restarts+1)); start_build; since=$(date +%s)
    else
      log "pipeline ended at $n/2600 without CHECK_OK (restarts=$restarts); guard exit"; tail -5 /data/lgd/rma/dataset_v2r.build.log; exit 3
    fi
  elif [ -n "$writers" ] && [ $(( $(date +%s) - since )) -ge 1800 ]; then
    if [ $restarts -ge 3 ]; then log "stalled again at $n after 3 restarts; guard exit, leaving it"; exit 4; fi
    kids=""
    for w in $writers; do kids="$kids $(pgrep -P $w | paste -sd' ')"; done
    grand=""
    for k in $kids; do grand="$grand $(pgrep -P $k | paste -sd' ')"; done
    log "stalled 30 min at $n/2600: kill writers [$writers] node [$kids] browsers [$grand]"
    kill $writers $kids $grand 2>/dev/null; sleep 20; kill -9 $writers $kids $grand 2>/dev/null
    for i in $(seq 1 30); do [ -z "$(pgrep -u $U -f "[p]ipeline_v2r.sh")" ] && break; sleep 10; done
    pkill -9 -u $U -f "[c]hromium_headless_shell-1187" 2>/dev/null
    rmdir $D/.lock 2>/dev/null; restarts=$((restarts+1)); log "restart $restarts"; start_build; since=$(date +%s)
  fi
  sleep 120
done
