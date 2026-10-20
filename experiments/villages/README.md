# Microfinance villages

Evaluates Q-Ising on the 43 Karnataka villages (reads `data/villages/adjacency_all.json`): SIS dynamics, EMVS and MCMC Ising fits, offline CQL, against random, degree, LIR and degree-bin policies.

- Notebooks (run from this folder): `policy_eval.ipynb` (all villages, writes `village_policy_results.csv`), `mcmc_ensemble.ipynb` (village 50 posterior and ensemble).
- `python uq_calibration_villages.py`; optional env vars `VILLAGE_IDS` (comma list), `T_TRAIN`, `T_TEST`, `N_TEST_EPS`, `N_MC`, `ROLLOUT_H`, `N_ENSEMBLE`, `N_STEPS_MAX`, `N_STEPS_PER_EPOCH`, `PATIENCE`, `FORCE_RERUN`. Writes `uq_calibration_village<ID>_decisions.csv` per village and `uq_calibration_villages_all_decisions.csv`.
- Re-runs (from the repository root, each writes its own `out/`): `emvs_roc/run_emvs_roc.py`; `ensemble_v50/run_v50.py mcmc` and `ensemble` (macOS: see the SDKROOT note in `requirements.txt`); `seeding_example/run_fig3_once.py` (see `seeding_example/README_once.md`).
- Output: `village_policy_results.csv`, `village_modularity.csv`, `uq_calibration_villages_all_decisions.csv`.

`uq_calibration_villages_all_decisions.csv` pools the decisions of all villages; the per-village files it was built from are not included, except villages 6 and 42. Paper statistics: `python scripts/paper_numbers.py`.

Running `policy_eval.ipynb` rewrites `village_policy_results.csv`, which holds the saved results of the paper; work on a copy to avoid overwriting it.
