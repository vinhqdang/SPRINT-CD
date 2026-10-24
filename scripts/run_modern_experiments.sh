#!/bin/sh
cd /home/user/SPRINT-CD
export PYTHONPATH=.
( while true; do sleep 240; scripts/commit_partials.sh; done ) &
[ -f results/exp38_pairs_modern.json ] || python3 experiments/exp38_pairs_modern.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e38.log 2>&1
[ -f results/exp39_graphs_modern.json ] || python3 experiments/exp39_graphs_modern.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e39.log 2>&1
