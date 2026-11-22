#!/bin/bash
# Run everything for one scenario server, in order: scripted reference episodes (paused + live, for videos and the
# stale-memory baseline), then the model evaluation jobs. Usage: run_server_queue.sh <server> <port> "<sids>" <jobs.jsonl>
set -u
S=$1; PORT=$2; SIDS=$3; JOBS=$4
R=${RLM_ROOT:?set RLM_ROOT to your working folder}
cd $R
mkdir -p evals/scripted logs
declare -A VSEED=( [s01_put_away]=3 [s02_annas_mug]=1 [s03_vacuum_ball]=3 [s04_salted_soup]=3 [s05_rotary_reject]=3
                   [s06_conveyor_lanes]=3 [s07_press_lamp]=9 [s08_parcel_locker]=3 [s09_kitting_partner]=3 [s10_cloche_pass]=1 )
for sid in $SIDS; do
  seed=${VSEED[$sid]:-3}
  for clock in paused live; do
    f=evals/scripted/${sid}_${clock}_stale.json
    [ -s $f ] && continue
    python3 bench/harness/scripted_agent.py --url http://127.0.0.1:$PORT --scenario $sid --seed $seed --clock $clock \
      --policy stale --think 25 --think2 8 --out /out/$S/videos/${sid}_${clock}_stale > $f.tmp 2> logs/scripted_${sid}_${clock}.err && mv $f.tmp $f
    echo "$(date +%H:%M:%S) scripted $sid $clock: $(cat $f 2>/dev/null)"
  done
done
python3 bench/harness/run_agent.py --jobs $JOBS --root $R/evals/models_$S \
  --claude ${CLAUDE_BIN:-claude} --codex ${CODEX_BIN:-codex} \
  --secret ${CLAUDE_SECRET_FILE:?file exporting CLAUDE_CODE_OAUTH_TOKEN} --codex-home ${CODEX_HOME:?} --timeout 1800
echo "$(date +%H:%M:%S) queue done"
