#!/bin/bash
# one-off: stop the stalled v2r row writers (own PIDs only), install guard + new stage watcher, restart via the guard
B=$HOME/pi05-abc/build_v2r; L=/data/lgd/rma/logs; U=$(id -u)
cd $B
for f in guard_v2r.sh stage_v2r.sh; do sed -i 's/\r$//' $f.new; bash -n $f.new || { echo "syntax $f"; exit 1; }; done
kill 3353580 3353577 2>/dev/null; sleep 1
mv guard_v2r.sh.new guard_v2r.sh; mv stage_v2r.sh.new stage_v2r.sh
writers=$(pgrep -u $U -f "[r]ma_build_v2r.py" | paste -sd' ')
kids=""; for w in $writers; do kids="$kids $(pgrep -P $w | paste -sd' ')"; done
grand=""; for k in $kids; do grand="$grand $(pgrep -P $k | paste -sd' ')"; done
echo "kill writers [$writers] node [$kids] browsers [$grand]"
kill $writers $kids $grand 2>/dev/null; sleep 15; kill -9 $writers $kids $grand 2>/dev/null
for i in $(seq 1 30); do [ -z "$(pgrep -u $U -f "[p]ipeline_v2r.sh")" ] && break; sleep 5; done
pkill -9 -u $U -f "[c]hromium_headless_shell-1187" 2>/dev/null
echo "pipeline left: [$(pgrep -u $U -f "[p]ipeline_v2r.sh")]  chrome left: $(pgrep -u $U -c -f "[c]hrom")"
tail -4 /data/lgd/rma/dataset_v2r.build.log
setsid nohup bash $B/guard_v2r.sh >> $L/guard_v2r.log 2>&1 < /dev/null &
sleep 3
setsid nohup bash $B/stage_v2r.sh >> $L/stage_v2r.log 2>&1 < /dev/null &
sleep 60
echo "--- guard"; cat $L/guard_v2r.log; echo "--- procs"; pgrep -u $U -af "[g]uard_v2r|[s]tage_v2r|[p]ipeline_v2r" | cut -c1-120; pgrep -u $U -c -f "[r]ma_build_v2r.py"
tail -3 /data/lgd/rma/dataset_v2r.build.log; tail -2 /data/lgd/rma/dataset_v2r/logs/build_0.log
