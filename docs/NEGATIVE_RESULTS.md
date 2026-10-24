# Negative results of the algorithm search (real-data evaluation)

All numbers are strict skeleton F1 or forced accuracy against the reference structure; scripts and
JSON are in `experiments/` and `results/`.  Evaluation recordings: Causal Chambers (2 i.i.d. + 19
new recordings), Sachs, Tuebingen pairs.

| Attempt | Idea | Outcome on real data |
|---|---|---|
| CERT-CD (sequential certificates) | assert edges only by rejecting; anytime-valid | false-edge control holds in simulation (2.7% vs 27-91%), not on the chambers |
| BTAD | block-replicated additive gains, grow-shrink blankets | highest precision (0.60) but recall 0.16, F1 0.25-0.28 on walk recordings |
| LRG | gradient boosting on block-gain features trained on simulations | F1 0.36 (chambers), 0.67 (Sachs); within-recording AUC 0.55 |
| PDM | learned pair-direction classifier trained on 40000 simulated pairs | 95% on simulated pairs, 40.2% on Tuebingen |
| Block-ICP | invariant causal prediction with time blocks as environments | empty graphs, F1 <= 0.07 on development recordings |
| LOCI, ANM, DAGMA, PCMCI+ | modern baselines | no better than RECI / DirectLiNGAM / GES |

The recordings with random-walk actuators have an effective sample size of about three observations
out of 4000; no method exceeds strict F1 0.42 on them.
