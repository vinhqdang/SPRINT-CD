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
| Sorted-conditional compression, TRA, BF-BIC+BOSS (2025-2026, exp42-43) | re-implemented published methods, settings frozen before running | Tuebingen 60.8-64.7% (RECI 60.8%, LOCI 64.7%); chamber i.i.d. pairs TRA 86.2% vs RECI 97.9%; compression 29.8%; Sachs F1 0.67 (ties LRG, PC-GCM 0.62-0.64); chambers F1 0.49 i.i.d., 0.44 on walks with 36 false edges per run |
| Interventional route (lt_interventions_standard_v1, pilot, not pre-registered) | two-sample shift of each sensor under each intervened actuator; conditional (cross-fitted residual) version | marginal KS: precision 1.00, recall 0.72 (28/39 arcs), AUC 0.93; the 11 missed arcs (l_ij -> ir/vis) show no shift (p 0.06-0.86); the conditional version raises recall to 0.79 at precision 0.51. No headroom for a new method here |
| Published methods with public code (exp45-46): AVICI, GOLEM, NOTEARS-MLP, GraN-DAG, DAG-GNN, GRaSP, BOSS, RESIT, CAM-UV | library defaults | best: BOSS F1 0.55 on i.i.d. chambers (ties DirectLiNGAM), GRaSP 0.68 / BOSS 0.65 on large-ESS (GES 0.70), Sachs <= 0.56, walks <= 0.37; AVICI pairs 90.4% chambers, 58.8% Tuebingen (RECI 97.9%, 60.8%); many neural timeouts at d=16 |
| GATE: ANM-HSIC when a symmetric heteroscedasticity diagnostic is low, else RECI (exp51, frozen on simulation) | assumption-regime gating | Tuebingen 65.7% (weighted 77.1%) vs RECI 60.8% (70.9%), McNemar p=0.30; chambers 75.5% / 77.0% vs RECI 97.9% / 94.3% (p<1e-6). Predictions >=0.70 and within 0.05 of RECI failed |

The recordings with random-walk actuators have an effective sample size of about three observations
out of 4000; no method exceeds strict F1 0.42 on them.
