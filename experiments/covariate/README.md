# Covariate experiment

Tests Q-Ising when block adoption is driven by a hidden covariate (SBM with K blocks, ten block covariates per period, one effective), against DQN, DQN on all covariates, topological, random and oracle policies, over N, K and T_train.

- Run from this folder: `python run_cluster_sweep.py` (environment `covariate_group_env.py`, policy `covariate_group_policy.py`, Ising fit `covariate_group_ising.py`).
- Environment variables (optional): `N_LEVELS`, `K_LEVELS`, `T_TRAIN_LEVELS` (comma lists), `N_REPLICATES`, `T_TEST`, `N_EVAL_EPS`, `DATA_SEED`, `NET_SEED`, `P_Z`, `P_BASE`, `P_ESCALATED`, `P_IN`, `P_OUT`, `ALPHA_CQL`, `N_STEPS_MAX`, `N_STEPS_PER_EPOCH`, `PATIENCE`, `HIDDEN_UNITS`, `FORCE_RERUN`.
- Output: `cluster_sweep_all_results.csv`, `cluster_sweep_all_baselines.csv`, `cluster_sweep_all_curves.csv`, `cluster_sweep_all_baseline_curves.csv`, `cluster_sweep_summary.csv`.

Saved results: the combined `cluster_sweep_all_*.csv` files and `cluster_sweep_summary.csv`; `cache/` holds the same results split by (N, K). The script reuses `cache/` and only rebuilds the combined files and the summary; set `FORCE_RERUN=1` to retrain everything.
