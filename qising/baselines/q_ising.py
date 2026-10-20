"""
Q-Ising policy: offline CQL agent on group-Ising state representations. Modes:
ising_augmentation=True/ensemble=False (Q-Ising EMVS MAP), True/True (MCMC posterior
ensemble), False/False (plain DQN on adoption rates).
"""
import numpy as np

from ..ising.group_ising import (
    fit_group_ising_emvs, predict_group_ising_emvs,
    fit_group_ising_mcmc, predict_group_ising_mcmc,
)
from ..utils.auxiliary import build_dataset, make_cql


class QIsingPolicy:
    """
    Offline RL policy on group Ising state representations.

    ising_augmentation : True -> state = [ising_pred_mean, adoption_rate] per bin; False -> [adoption_rate].
    ensemble : True -> MCMC posterior ensemble with majority vote; False -> single EMVS MAP agent (argmax Q).
    After fit(), _ensemble_models holds (idx, draw_params, model, so_fn) tuples.
    """

    def __init__(
        self,
        ensemble=False,
        ising_augmentation=True,
        # EMVS hyperparams
        v0=0.01, v1=10.0, c=1.0, sigma_sq=10.0,
        # CQL hyperparams
        discount=0.8, alpha_cql=0.1, hidden=[256, 256], batch_size=64,
        n_steps_max=30_000, n_steps_per_epoch=1_000,
        patience=10, min_delta=1e-4,
        # MCMC hyperparams (used only when ensemble=True)
        v0_mcmc=0.01, v1_mcmc=10.0, c_prior=1.0, sigma_sq_mcmc=10.0,
        n_mcmc_draws=200, n_mcmc_tune=300, n_mcmc_chains=1, mcmc_seed=42,
        max_mcmc_rows=8_000, n_ensemble=20, ensemble_seed=1234,
        label="q_ising",
        verbose=False,
    ):
        if ensemble and not ising_augmentation:
            raise ValueError(
                "ensemble=True requires ising_augmentation=True. "
                "The MCMC ensemble is defined over Ising state representations."
            )

        self.ensemble           = ensemble
        self.ising_augmentation = ising_augmentation
        self.v0                 = v0
        self.v1                 = v1
        self.c                  = c
        self.sigma_sq           = sigma_sq
        self.discount           = discount
        self.alpha_cql          = alpha_cql
        self.hidden             = hidden or [256, 256]
        self.batch_size         = batch_size
        self.n_steps_max        = n_steps_max
        self.n_steps_per_epoch  = n_steps_per_epoch
        self.patience           = patience
        self.min_delta          = min_delta
        self.v0_mcmc            = v0_mcmc
        self.v1_mcmc            = v1_mcmc
        self.c_prior            = c_prior
        self.sigma_sq_mcmc      = sigma_sq_mcmc
        self.n_mcmc_draws       = n_mcmc_draws
        self.n_mcmc_tune        = n_mcmc_tune
        self.n_mcmc_chains      = n_mcmc_chains
        self.mcmc_seed          = mcmc_seed
        self.max_mcmc_rows      = max_mcmc_rows
        self.n_ensemble         = n_ensemble
        self.ensemble_seed      = ensemble_seed
        self.label              = label
        self.verbose            = verbose

        self._emvs_result     = None
        self._mcmc_result     = None
        self._model           = None
        self._ensemble_models = []
        self._bins            = None
        self._n_bins          = None

    # Public interface

    def fit(self, trajs, adj, block_labels,
            _emvs_result=None, _mcmc_draws=None):
        """
        Train from behavioral trajectories (collect_behavioral_episode dicts); adj (N, N);
        block_labels (N,) in 0..k-1. _emvs_result / _mcmc_draws: precomputed fits to skip
        EMVS / MCMC. Returns self.
        """
        n_bins = int(block_labels.max()) + 1
        self._n_bins = n_bins
        self._bins   = [np.where(block_labels == b)[0] for b in range(n_bins)]

        X_arr = np.stack([tr["adopted"][:-1].astype(float) for tr in trajs])
        y_arr = np.stack([tr["adopted"][1:].astype(float)  for tr in trajs])
        t_arr = np.stack([self._extract_treated(tr)[:-1]   for tr in trajs])

        if _emvs_result is not None:
            self._emvs_result = _emvs_result
        elif self.ising_augmentation:
            self._emvs_result = fit_group_ising_emvs(
                X_arr, y_arr, t_arr, adj, block_labels,
                v0=self.v0, v1=self.v1, c=self.c, sigma_sq=self.sigma_sq,
            )

        if self.ensemble:
            if _mcmc_draws is not None:
                self._mcmc_result = {
                    "g_idx":          self._emvs_result["g_idx"],
                    "n_groups":       n_bins,
                    "normalize":      self._emvs_result.get("normalize", False),
                    "neighbor_lists": self._emvs_result["neighbor_lists"],
                    "neigh_g_idx":    self._emvs_result["neigh_g_idx"],
                    "degrees":        self._emvs_result["degrees"],
                    "mcmc_draws":     _mcmc_draws,
                    "n_draws":        self.n_mcmc_draws * self.n_mcmc_chains,
                }
            else:
                self._mcmc_result = fit_group_ising_mcmc(
                    X_arr, y_arr, t_arr, adj, block_labels,
                    v0=self.v0_mcmc, v1=self.v1_mcmc, c=self.c_prior,
                    sigma_sq=self.sigma_sq_mcmc, n_draws=self.n_mcmc_draws,
                    n_tune=self.n_mcmc_tune, n_chains=self.n_mcmc_chains,
                    seed=self.mcmc_seed, max_rows=self.max_mcmc_rows,
                    verbose=self.verbose,
                )
            self._build_ensemble(trajs)
        else:
            dataset = build_dataset(trajs, self._obs, n_bins)
            self._model = make_cql(
                dataset, label=self.label,
                discount=self.discount, alpha_cql=self.alpha_cql,
                hidden=self.hidden, batch_size=self.batch_size,
                n_steps_max=self.n_steps_max, n_steps_per_epoch=self.n_steps_per_epoch,
                patience=self.patience, min_delta=self.min_delta,
                verbose=self.verbose,
            )

        return self

    def __call__(self, obs, env):
        ba       = obs["adopted"]
        n        = env.n_bins
        avl      = set(env.get_available_actions())
        all_acts = np.arange(n, dtype=np.int32)

        if self.ensemble:
            votes = np.zeros(n, dtype=int)
            for _, _, mdl, so_fn in self._ensemble_models:
                ob = so_fn(ba).reshape(1, -1)
                qv = mdl.predict_value(np.repeat(ob, n, axis=0), all_acts)
                for b in np.argsort(qv)[::-1]:
                    if int(b) in avl:
                        votes[int(b)] += 1
                        break
            return int(np.argmax(votes))

        ob = self._obs(ba).reshape(1, -1)
        qv = self._model.predict_value(np.repeat(ob, n, axis=0), all_acts)
        for b in np.argsort(qv)[::-1]:
            if int(b) in avl:
                return int(b)
        return int(np.random.randint(0, n))

    def reset(self):
        pass

    # Internal helpers

    @staticmethod
    def _extract_treated(traj):
        if "seeded_nodes" in traj:
            return np.asarray(traj["seeded_nodes"], dtype=int)
        return np.array(
            [info["seeded_node"] if info.get("seeded_node") is not None else -1
             for info in traj["infos"]],
            dtype=int,
        )

    def _obs(self, ba):
        if self.ising_augmentation:
            iv = predict_group_ising_emvs(
                ba.astype(float), action=-1, result=self._emvs_result, A=None
            ).astype(np.float32)
            parts = []
            for b in range(self._n_bins):
                m = self._bins[b]
                parts.append(float(iv[m].mean())             if len(m) > 0 else 0.0)
                parts.append(float(np.asarray(ba)[m].mean()) if len(m) > 0 else 0.0)
            return np.array(parts, dtype=np.float32)
        return np.array(
            [float(np.asarray(ba)[self._bins[b]].mean()) if len(self._bins[b]) > 0 else 0.0
             for b in range(self._n_bins)],
            dtype=np.float32,
        )

    def _mcmc_obs(self, ba, draw_idx):
        iv = predict_group_ising_mcmc(
            ba.astype(float), action=-1,
            result=self._mcmc_result, draw_idx=draw_idx,
        ).astype(np.float32)
        parts = []
        for b in range(self._n_bins):
            m = self._bins[b]
            parts.append(float(iv[m].mean())             if len(m) > 0 else 0.0)
            parts.append(float(np.asarray(ba)[m].mean()) if len(m) > 0 else 0.0)
        return np.array(parts, dtype=np.float32)

    def _build_ensemble(self, trajs):
        n_total = self._mcmc_result["n_draws"]
        rng     = np.random.default_rng(self.ensemble_seed)
        indices = rng.integers(0, n_total, size=self.n_ensemble)
        self._ensemble_models = []

        for di, draw_idx in enumerate(indices):
            _di       = int(draw_idx)
            draw_params = {gi: self._mcmc_result["mcmc_draws"][gi][_di]
                           for gi in range(self._n_bins)}
            so_fn     = lambda ba, d=_di: self._mcmc_obs(ba, d)
            dataset   = build_dataset(trajs, so_fn, self._n_bins)
            model     = make_cql(
                dataset, label=f"{self.label}_ens_{di}",
                discount=self.discount, alpha_cql=self.alpha_cql,
                hidden=self.hidden, batch_size=self.batch_size,
                n_steps_max=self.n_steps_max, n_steps_per_epoch=self.n_steps_per_epoch,
                patience=self.patience, min_delta=self.min_delta,
                verbose=self.verbose,
            )
            self._ensemble_models.append((di, draw_params, model, so_fn))
            if self.verbose and (di + 1) % 5 == 0:
                print(f"  [Ensemble] {di + 1}/{self.n_ensemble} members trained")
