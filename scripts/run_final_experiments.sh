#!/bin/sh
cd /home/user/SPRINT-CD
export PYTHONPATH=.
( while true; do sleep 240; scripts/commit_partials.sh; done ) &
while pgrep -f "experiments/exp26_[p]cgcm" >/dev/null; do sleep 15; done
[ -f results/exp26_pcgcm_monitoring.json ] || python3 experiments/exp26_pcgcm_monitoring.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e26.log 2>&1
[ -f results/exp27_chamber_fresh.json ] || python3 experiments/exp27_chamber_fresh.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e27.log 2>&1
[ -f results/exp28_selective_orientation.json ] || python3 experiments/exp28_selective_orientation.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e28.log 2>&1
