# Misspecification experiment (E2)

Tests Q-Ising when the Ising model is misspecified: a Barabasi-Albert network (100 nodes, 10 degree bins) with a second-order saturation response, against DQN, topological and random baselines over several T_train.

- Run from this folder: `python run_qising_sweep.py` (environment `saturation_env.py`; it imports the single-T_train pass `run_qising_quick.py`).
- Environment variables (optional): `T_TRAIN_LEVELS` (comma list), `N_REPLICATES`, `T_TEST`, `N_EVAL_EPS`, `DATA_SEED`, `ALPHA_CQL`, `N_STEPS_MAX`, `N_STEPS_PER_EPOCH`, `PATIENCE`, `HIDDEN_UNITS`, `FORCE_RERUN`; `run_qising_quick.py` reads `T_TRAIN`.
- Output: `qising_sweep_all_results.csv`, `qising_sweep_summary.csv`, `qising_sweep_baselines.csv`, `qising_sweep_ising_fit.csv`, `qising_quick_summary_T400.csv`; figure in `figures/`.

Saved results resume per (T_train, replicate). Baselines are recomputed each run; the saved file uses 250 episodes, so set `N_EVAL_EPS=250` to reproduce it (default 50 overwrites it).
