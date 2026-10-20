"""
Majority-vote (conformity) contagion on the tiny network: a node adopts w.p. p_high_j if the
fraction of adopted neighbors is >= theta_j, else p_low_j (all three heterogeneous per node);
seeding forces adoption. Not representable by the linear-logistic Ising form. Same
bypass-construction pattern as tiny_env.py.
"""

import numpy as np

from qising.environments.sbm_env import SBMCovariateEnv
from tiny_env import (
    build_tiny_network, run_episode, random_skip_action_fn,  # re-exported for convenience
)


class MajorityVoteEnv(SBMCovariateEnv):

    def step(self, action):
        # seed first: forced adoption counts toward this period's neighbor fractions
        seeded_node = None
        if action is not None:
            a = int(action)
            if a < 0:
                seeded_node = self._seed_node(-(a + 1))
            elif a < self.n_bins:
                seeded_node = self._seed_bin(a)

        # majority vote, one rule for all nodes
        n_adopted_neighbors = self.adj @ self.adopted.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(self.degree > 0, n_adopted_neighbors / np.maximum(self.degree, 1), 0.0)
        majority = ratio >= self.theta_vec
        p_vote = np.where(majority, self.p_high_vec, self.p_low_vec)
        draws = self.rng.random(self.N)
        new_adopted = draws < p_vote

        # direct treatment guarantees adoption
        if seeded_node is not None:
            new_adopted[seeded_node] = True

        newly_adopted = new_adopted & ~self.adopted
        churned       = self.adopted & ~new_adopted
        self.adopted = new_adopted

        self.current_time += 1
        done = self.current_time >= self.T
        n_adopted_after = int(self.adopted.sum())
        reward = float(n_adopted_after)

        self.reward_history.append(reward)
        if seeded_node is not None:
            self.seeded_history.append((self.current_time, seeded_node))

        info = {
            'n_adopted': n_adopted_after,
            'n_susceptible': self.N - n_adopted_after,
            'n_newly_adopted_spread': int(newly_adopted.sum()),
            'n_churned': int(churned.sum()),
            'adoption_rate': float(self.adopted.mean()),
            'seeded_node': seeded_node,
            'seeded_bin': action,
            'action_wasted': (action is not None and seeded_node is None),
            't': self.current_time,
        }
        return self._get_obs(), reward, done, info


def draw_majority_params(N, seed, p_high_range=(0.5, 0.9), p_low_range=(0.05, 0.3),
                          theta_range=(0.3, 0.7)):
    rng = np.random.default_rng(seed)
    p_high_vec = rng.uniform(*p_high_range, size=N)
    p_low_vec  = rng.uniform(*p_low_range, size=N)
    theta_vec  = rng.uniform(*theta_range, size=N)
    return dict(p_high_vec=p_high_vec, p_low_vec=p_low_vec, theta_vec=theta_vec)


def make_majority_env(adj, params, T, seed, init_rate=0.0, p_high_scale=1.0):
    """Bypass-construct a MajorityVoteEnv with one bin per node (K=N).
    p_high_scale multiplies p_high_vec (clipped to [0,1]) — the calibration lever."""
    N = adj.shape[0]
    block_labels = np.arange(N)

    env = MajorityVoteEnv.__new__(MajorityVoteEnv)
    env.rng             = np.random.default_rng(seed)
    env.adj             = adj.copy().astype(float)
    np.fill_diagonal(env.adj, 0)
    env.N               = N
    env.covariates      = np.zeros((N, 1), dtype=int)
    env.block_labels    = block_labels.copy()
    env.beta            = float(params['p_high_vec'].mean() * p_high_scale)  # interface parity
    env.T               = T
    env.reward_type     = 'current'
    env.init_rate       = init_rate
    env.n_bins          = N
    env.n_actions       = N
    env.degree          = adj.sum(axis=1)
    env.delta_vec       = params['p_low_vec'].copy()   # interface parity, unused by step()
    env.beta_vec        = params['p_high_vec'].copy() * p_high_scale  # interface parity, unused
    env.p_high_vec      = np.clip(params['p_high_vec'].copy() * p_high_scale, 0.0, 1.0)
    env.p_low_vec       = params['p_low_vec'].copy()
    env.theta_vec       = params['theta_vec'].copy()
    env.beta_college    = None
    env.beta_noncollege = None
    env.beta_young_mult = 1.0
    env.bins            = [np.array([i]) for i in range(N)]
    env.bin_labels       = block_labels.copy()
    env.bin_names        = [f"node_{i}" for i in range(N)]
    env.adopted          = None
    env.current_time      = None
    env.seeded_history    = None
    env.reward_history    = None
    env.reset()
    return env


def make_majority_env_clustered(adj, params, block_labels, T, seed, init_rate=0.0, p_high_scale=1.0):
    """Like make_majority_env, but one bin per cluster of block_labels (K < N); each action
    seeds a random susceptible node in the cluster."""
    N = adj.shape[0]
    n_bins = int(block_labels.max()) + 1

    env = MajorityVoteEnv.__new__(MajorityVoteEnv)
    env.rng             = np.random.default_rng(seed)
    env.adj             = adj.copy().astype(float)
    np.fill_diagonal(env.adj, 0)
    env.N               = N
    env.covariates      = np.zeros((N, 1), dtype=int)
    env.block_labels    = block_labels.copy()
    env.beta            = float(params['p_high_vec'].mean() * p_high_scale)
    env.T               = T
    env.reward_type     = 'current'
    env.init_rate       = init_rate
    env.n_bins          = n_bins
    env.n_actions       = n_bins
    env.degree          = adj.sum(axis=1)
    env.delta_vec       = params['p_low_vec'].copy()
    env.beta_vec        = params['p_high_vec'].copy() * p_high_scale
    env.p_high_vec      = np.clip(params['p_high_vec'].copy() * p_high_scale, 0.0, 1.0)
    env.p_low_vec       = params['p_low_vec'].copy()
    env.theta_vec       = params['theta_vec'].copy()
    env.beta_college    = None
    env.beta_noncollege = None
    env.beta_young_mult = 1.0
    env.bins            = [np.where(block_labels == b)[0] for b in range(n_bins)]
    env.bin_labels       = block_labels.copy()
    env.bin_names        = [f"cluster_{b}" for b in range(n_bins)]
    env.adopted          = None
    env.current_time      = None
    env.seeded_history    = None
    env.reward_history    = None
    env.reset()
    return env


def calibrate_majority_scale(adj, params, skip_prob, cooldown=0, target_mean_rate=0.25,
                              T=100, n_reps=150, seed=0, lo=0.05, hi=2.0, tol=0.02, max_iter=25,
                              env_factory=None):
    """Bisect p_high_scale so the mean adoption rate under the random-policy/skip_prob/cooldown
    loop lands near target_mean_rate (hi=2.0: p_high is clipped to [0,1])."""
    if env_factory is None:
        env_factory = lambda T_, seed_, scale_: make_majority_env(
            adj, params, T=T_, seed=seed_, init_rate=0.0, p_high_scale=scale_)

    def mean_rate(scale):
        rng = np.random.default_rng(seed)
        totals = []
        for _ in range(n_reps):
            env = env_factory(T, int(rng.integers(0, 1_000_000)), scale)
            action_fn = random_skip_action_fn(env.n_bins, skip_prob, rng)
            traj = run_episode(env, action_fn, T=T, cooldown=cooldown)
            totals.append(traj['adopted'].mean())
        return float(np.mean(totals))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate = mean_rate(mid)
        if abs(rate - target_mean_rate) < tol:
            return mid, rate
        if rate > target_mean_rate:
            hi = mid
        else:
            lo = mid
    if abs(rate - target_mean_rate) >= tol:
        raise RuntimeError(
            f'calibrate_majority_scale did not converge: achieved={rate:.3f} vs '
            f'target={target_mean_rate:.3f} after {max_iter} iterations (scale={mid:.3f}, '
            f'search bounds=[{lo:.3f},{hi:.3f}]).'
        )
    return mid, rate
