#!/bin/bash
# v3_worker.sh <server session> <port> <jobs.jsonl> <eval root>: wait for the server, then work through the shared queue
R=${RLM_ROOT:?set RLM_ROOT to your working folder}; cd $R
for i in $(seq 1 240); do curl -s -m 5 -X POST http://127.0.0.1:$2/status | grep -q "\"ok\": true" && break; sleep 15; done
echo "$(date +%H:%M:%S) worker $1 start"
python3 bench/harness/run_agent.py --jobs $3 --root $4 --url http://127.0.0.1:$2 --server-out /out/$1/episodes \
  --claude ${CLAUDE_BIN:-claude} --codex ${CODEX_BIN:-codex} \
  --secret ${CLAUDE_SECRET_FILE:?file exporting CLAUDE_CODE_OAUTH_TOKEN} --codex-home ${CODEX_HOME:?} --timeout 1800
echo "$(date +%H:%M:%S) worker $1 done"
