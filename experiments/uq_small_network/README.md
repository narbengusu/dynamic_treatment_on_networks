# Uncertainty calibration in a small network (E3)

Checks ensemble uncertainty where true decision values are computable: a 10-node SIS network, ensemble of CQL agents trained on Ising posterior draws; vote entropy is compared with the true value gap and redeployed under majority-vote dynamics (`tiny_env.py`, `majority_vote_env.py`).

- Run from this folder: `python confirm_instability_sis.py` (writes the network parameters), then `python uq_calibration_sis.py`.
- Environment variables (optional): `DATA_SEED`, `SKIP_PROB`, `COOLDOWN`, `T_TRAIN`, `WARM_START_T`; the rest are listed at the top of `uq_calibration_sis.py`.
- Output: `sis_network_params.npz`, `majority_network_params.npz`, `sis_instability_summary.csv`, `uq_calibration_sis_decisions.csv`, `uq_entropy_sis.csv`, `uq_entropy_majority_misspecified.csv`; figures in `figures/`.
