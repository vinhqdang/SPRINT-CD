#!/bin/sh
# Relaunch-safe: every step resumes from its partial file.  Run with nohup.
cd /home/user/SPRINT-CD
export PYTHONPATH=.
L=${LOGDIR:-/tmp}
( while true; do sleep 240; scripts/commit_partials.sh; done ) &
[ -f results/exp18_sim.json ] || python3 experiments/exp18_pairs_heldout.py --suite sim --reps 30 > $L/e18s.log 2>&1
[ -f results/exp17_ablation.json ] || python3 experiments/exp17_scaling_ablation.py --suite ablation --R 20 --workers 4 > $L/exp17a.log 2>&1
[ -f results/exp19_tuebingen_seeds.json ] || python3 experiments/exp19_tuebingen_seeds.py > $L/e19.log 2>&1
[ -f results/exp18_sweep.json ] || python3 experiments/exp18_pairs_heldout.py --suite sweep --reps 20 > $L/e18w.log 2>&1
