#!/bin/sh
cd /home/user/SPRINT-CD
export PYTHONPATH=.
( while true; do sleep 300; scripts/commit_partials.sh; done ) &
[ -f results/exp19_tuebingen_seeds.json ] || python3 experiments/exp19_tuebingen_seeds.py > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e19.log 2>&1
[ -f results/exp18_sim.json ] || python3 experiments/exp18_pairs_heldout.py --suite sim --reps 40 > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e18s.log 2>&1
[ -f results/exp17_ablation.json ] || python3 experiments/exp17_scaling_ablation.py --suite ablation --R 30 --workers 4 > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/exp17a.log 2>&1
[ -f results/exp18_sweep.json ] || python3 experiments/exp18_pairs_heldout.py --suite sweep --reps 25 > /tmp/claude-0/-home-user-SPRINT-CD/2b2cb543-ae06-5cf9-a3cb-55cfc41331f4/scratchpad/e18w.log 2>&1
