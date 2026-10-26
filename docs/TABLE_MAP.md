# Map from tables and figures of `paper/resolution_jci` to scripts and stored results

Run order: experiments (`experiments/expNN_*.py`) write `results/*.json`; scripts in `scripts/` build tables and figures.

| Item | Script that builds it | Stored results |
|---|---|---|
| Collections table, diagnostics, example pairs | `scripts/build_collection_diagnostics.py` (+ exp49 loaders, exp53) | `results/collection_diagnostics.json` |
| Rule x collection accuracy table, heat map, Kendall matrix, collection CIs | `scripts/build_resolution_assets.py`, `scripts/build_resolution_assets4.py` | `results/resolution_assets.json`, `results/cluster_bootstrap.json` |
| Multiverse, Holm families, subsets, detectable difference | `experiments/exp47_multiverse.py` | `results/exp47_multiverse.json` |
| Common-pair family, tie-break variance, Holm for Kendall / Mann-Whitney, cluster sign-flip tests, analytic power | `scripts/final_checks.py` | `results/final_checks.json` |
| Protocol multiverse | `experiments/exp61_protocol_multiverse.py` | `results/exp61_protocol_multiverse.json` |
| Marginal-only classifier (first protocol, heteroscedasticity) | `experiments/exp49_explain.py`, `scripts/analyse_exp49_strata.py` | `results/exp49_explain.json`, `results/exp49_strata.json` |
| Leakage controls, matched dose-response | `experiments/exp52_controls.py`, `scripts/build_resolution_assets2.py` (figures) | `results/exp52_controls.json` |
| Factorial simulation (appendix) | `experiments/exp56_factorial.py`, `scripts/build_resolution_assets2.py`, `scripts/build_resolution_assets3.py` | `results/exp56_factorial.json` |
| GATE (appendix) | `experiments/exp51_gate.py` | `results/exp51_gate.json` |
| Predictive validation (appendix) | `experiments/exp57_predictive_validation.py` | `results/exp57_predictive_validation.json` (run 2: `exp57_run2.json`) |
| Compression audit, 18-cell grid | `scripts/audit_compression_conventions.py`, `experiments/exp54_compression_faithful.py` | `results/exp54_audit_conventions.json`, `results/exp54_compression_faithful.json` |
| Tie handling, cdt reference check | `experiments/exp59_ties.py`, `experiments/exp55_cdt_loci_new.py`, `scripts/igci_reference_check.py` | `results/exp59_ties.json`, `results/exp55_cdt_loci_new.json`, `results/igci_reference_check.json` |
| All-recordings walk pairs, TRA on raw arrays | `experiments/exp58_walks_all.py`, `experiments/exp60_tra_raw.py` | `results/exp58_walks_all.json`, `results/exp60_tra_raw.json` |
| AVICI and RESIT pairs | `experiments/exp46_pairs_sota.py` | `results/exp46_pairs_sota.json` |
| TRA, compression (first runs) | `experiments/exp42_pairs_2026.py` | `results/exp42_pairs_2026.json` |
| LOCI, Gaussian-process ANM (original collections) | `experiments/exp38_pairs_modern.py` | `results/exp38_pairs_modern.json` |
| Graph table (Table of graph methods) | `scripts/build_benchmark_tables.py`, `scripts/analyse_exp45.py` | `results/benchmark_graph_summary.json`, `results/exp45_summary.json` |
| Sachs resolution (bootstrap) | `experiments/exp48_sachs_resolution.py` | `results/exp48_sachs_resolution.json` |
| Interventional pilot | `scripts/lt_interventions_check.py` | `results/lt_interventions_check.json` |
| Pre-registration (commit) log | `git log` of `experiments/` | appendix table |

Numbers quoted in the text for the graph experiments of Section "Graph benchmarks" (exp 36, 39, 43) come from the analysis scripts `scripts/analyse_exp36.py`, `scripts/build_benchmark_tables.py`.
