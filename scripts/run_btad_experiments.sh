#!/bin/sh
cd /home/user/SPRINT-CD
export PYTHONPATH=.
( while true; do sleep 240; scripts/commit_partials.sh; done ) &
while pgrep -f "experiments/exp29_[b]tad" >/dev/null; do sleep 15; done
[ -f results/exp29_btad_sim.json ] || python3 experiments/exp29_btad_sim.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e29.log 2>&1
[ -f results/exp30_btad_chambers.json ] || python3 experiments/exp30_btad_chambers.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e30.log 2>&1
