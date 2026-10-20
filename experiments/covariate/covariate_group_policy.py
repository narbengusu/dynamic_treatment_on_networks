"""
Offline CQL policy for the covariate environment. State modes: 'plain' = adoption rate per
bin (dqn-plain); 'all' = raw covariates + adoption rate per bin (dqn-all); 'ising' =
covariate-Ising forecast + adoption rate per bin (q-ising).
"""
import numpy as np

from qising.utils.auxiliary import make_cql
from covariate_group_ising import (
    fit_group_ising_logistic_controllable, predict_group_ising_logistic_controllable,
)


def build_dataset_with_cgroup(trajs, obs_fn, n_bins):
    import d3rlpy
    obs_l, acts_l, rews_l, terms_l = [], [], [], []
    for traj in trajs:
        ep_obs, ep_acts, ep_rews = [], [], []
        for t in range(traj["T"] - 1):
            if int(traj["actions"][t]) >= n_bins:
                continue
            ep_obs.append(obs_fn(traj["adopted"][t], traj["C_group"][t]))
            ep_acts.append(int(traj["actions"][t]))
            ep_rews.append(float(traj["rewards"][t]))
        if not ep_obs:
            continue
        ep_terms = [False] * len(ep_obs)
        ep_terms[-1] = True
        obs_l.extend(ep_obs); acts_l.extend(ep_acts)
        rews_l.extend(ep_rews); terms_l.extend(ep_terms)
    return d3rlpy.dataset.MDPDataset(
        observations=np.stack(obs_l).astype(np.float32),
        actions=np.array(acts_l, dtype=np.int32),
        rewards=np.array(rews_l, dtype=np.float32),
        terminals=np.array(terms_l, dtype=bool),
        action_size=n_bins,
    )


class CovariateGroupPolicy:
    def __init__(
        self,
        mode,  # 'plain' | 'all' | 'ising'
        discount=0.8, alpha_cql=0.05, hidden=None, batch_size=64,
        n_steps_max=5_000, n_steps_per_epoch=250,
        patience=6, min_delta=1e-4,
        label="covariate_group",
        verbose=False,
    ):
        assert mode in ('plain', 'all', 'ising')
        self.mode = mode
        self.discount = discount
        self.alpha_cql = alpha_cql
        self.hidden = hidden or [64, 64]
        self.batch_size = batch_size
        self.n_steps_max = n_steps_max
        self.n_steps_per_epoch = n_steps_per_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.label = label
        self.verbose = verbose

        self._ising_result = None
        self._model = None
        self._n_bins = None
        self._bins = None
        self._block_labels = None

    def fit(self, trajs, adj, block_labels):
        n_bins = int(block_labels.max()) + 1
        self._n_bins = n_bins
        self._block_labels = np.asarray(block_labels)
        self._bins = [np.where(self._block_labels == b)[0] for b in range(n_bins)]

        if self.mode == 'ising':
            X_arr = np.stack([tr["adopted"][:-1].astype(int) for tr in trajs])
            y_arr = np.stack([tr["adopted"][1:].astype(int) for tr in trajs])
            treated_arr = np.stack([tr["seeded_nodes"][:-1] for tr in trajs])
            Cg_arr = np.stack([tr["C_group"][:-1] for tr in trajs])  # (m, T-1, n_bins, n_cov)
            C_own_group_arr = Cg_arr[:, :, self._block_labels, :]   # (m, T-1, N, n_cov)
            self._ising_result = fit_group_ising_logistic_controllable(
                X_arr, y_arr, treated_arr, C_own_group_arr, self._block_labels)

        dataset = build_dataset_with_cgroup(trajs, self._obs, n_bins)
        self._model = make_cql(
            dataset, label=self.label,
            discount=self.discount, alpha_cql=self.alpha_cql,
            hidden=self.hidden, batch_size=self.batch_size,
            n_steps_max=self.n_steps_max, n_steps_per_epoch=self.n_steps_per_epoch,
            patience=self.patience, min_delta=self.min_delta,
            verbose=self.verbose,
        )
        return self

    def _obs(self, ba, c_group):
        ba = np.asarray(ba)
        if self.mode == 'plain':
            return np.array(
                [float(ba[self._bins[b]].mean()) for b in range(self._n_bins)], dtype=np.float32)
        if self.mode == 'all':
            c_group = np.asarray(c_group)  # (n_bins, n_cov)
            parts = []
            for b in range(self._n_bins):
                parts.extend(float(v) for v in c_group[b])
                parts.append(float(ba[self._bins[b]].mean()))
            return np.array(parts, dtype=np.float32)
        # mode == 'ising'
        c_own_group = np.asarray(c_group)[self._block_labels]  # (N, n_cov)
        iv = predict_group_ising_logistic_controllable(
            ba.astype(float), None, c_own_group.astype(float), self._ising_result)
        parts = []
        for b in range(self._n_bins):
            m = self._bins[b]
            parts.append(float(iv[m].mean()))
            parts.append(float(ba[m].mean()))
        return np.array(parts, dtype=np.float32)

    def __call__(self, obs, env):
        ba = obs["adopted"]
        cg = obs["C_group"]
        n = env.n_bins
        avl = set(env.get_available_actions())
        all_acts = np.arange(n, dtype=np.int32)

        ob = self._obs(ba, cg).reshape(1, -1)
        qv = self._model.predict_value(np.repeat(ob, n, axis=0), all_acts)
        for b in np.argsort(qv)[::-1]:
            if int(b) in avl:
                return int(b)
        return int(np.random.randint(0, n))

    def reset(self):
        pass
