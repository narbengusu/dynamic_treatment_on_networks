"""
Tiny node-resolved SIS environment (N=10, one bin per node): SBMCovariateEnv mechanics with
block_labels = arange(N), heterogeneous (beta_i, delta_i) calibrated by `calibrate_beta_scale`.
"""

import numpy as np
import networkx as nx

from qising.environments.sbm_env import SBMCovariateEnv


def build_tiny_network(N=10, p=0.35, seed=0):
    """Small connected random graph. Isolated nodes get one forced edge."""
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
    adj = nx.to_numpy_array(G)
    degree = adj.sum(axis=1)
    isolated = np.where(degree == 0)[0]
    for node in isolated:
        others = [j for j in range(N) if j != node]
        partner = int(rng.choice(others))
        adj[node, partner] = 1.0
        adj[partner, node] = 1.0
    np.fill_diagonal(adj, 0)
    return adj


def draw_heterogeneous_params(N, seed, beta_range=(0.05, 0.25), delta_range=(0.5, 0.9)):
    """Heterogeneous (beta_i, delta_i) drawn uniformly from the given ranges."""
    rng = np.random.default_rng(seed)
    beta_vec  = rng.uniform(*beta_range, size=N)
    delta_vec = rng.uniform(*delta_range, size=N)
    return beta_vec, delta_vec


def make_tiny_env(adj, beta_vec, delta_vec, T, seed, init_rate=0.0):
    """Bypass-construct an SBMCovariateEnv with one bin per node (K=N)."""
    N = adj.shape[0]
    block_labels = np.arange(N)

    env = SBMCovariateEnv.__new__(SBMCovariateEnv)
    env.rng             = np.random.default_rng(seed)
    env.adj             = adj.copy().astype(float)
    np.fill_diagonal(env.adj, 0)
    env.N               = N
    env.covariates      = np.zeros((N, 1), dtype=int)
    env.block_labels     = block_labels.copy()
    env.beta            = float(beta_vec.mean())
    env.T               = T
    env.reward_type      = 'current'
    env.init_rate        = init_rate
    env.n_bins           = N
    env.n_actions         = N
    env.degree           = adj.sum(axis=1)
    env.delta_vec        = delta_vec.copy()
    env.beta_vec         = beta_vec.copy()
    env.beta_college      = None
    env.beta_noncollege   = None
    env.beta_young_mult   = 1.0
    env.bins             = [np.array([i]) for i in range(N)]
    env.bin_labels        = block_labels.copy()
    env.bin_names         = [f"node_{i}" for i in range(N)]
    env.adopted          = None
    env.current_time      = None
    env.seeded_history    = None
    env.reward_history    = None
    env.reset()
    return env


def run_episode(env, action_fn, T, cooldown=0, reset=True, initial_cooldown=0):
    """
    Episode runner with an optional post-treatment cooldown: after any attempted action,
    `cooldown` periods of forced inaction follow (counted from the attempt).
    action_fn(t, obs, env) -> action (int) or None; not called during cooldown.
    reset=False continues from the env's current state (initial_cooldown carries the counter).
    Returns a trajectory dict (adopted, actions, rewards, infos, T, N, n_bins,
    cooldown_left_at_end) usable by QIsingPolicy.fit / build_dataset.
    """
    if reset:
        env.reset()
    N, n_bins = env.N, env.n_bins
    SKIP_CODE = n_bins

    adopted_traj = np.zeros((T, N), dtype=bool)
    action_traj  = np.full(T, SKIP_CODE, dtype=int)
    reward_traj  = np.zeros(T)
    infos = []

    cooldown_left = initial_cooldown
    for t in range(T):
        adopted_traj[t] = env.adopted.copy()
        if cooldown_left > 0:
            action = None
            cooldown_left -= 1
        else:
            obs = env._get_obs()
            action = action_fn(t, obs, env)
            if action is not None:
                cooldown_left = cooldown

        action_traj[t] = action if action is not None else SKIP_CODE
        _, r, _, info = env.step(action)
        reward_traj[t] = r
        infos.append(info)

    return {
        'adopted': adopted_traj, 'actions': action_traj, 'rewards': reward_traj,
        'infos': infos, 'T': T, 'N': N, 'n_bins': n_bins,
        'cooldown_left_at_end': cooldown_left,
    }


def random_skip_action_fn(n_bins, skip_prob, rng):
    """Random-bin behavioral policy: do nothing w.p. skip_prob, else a uniform random bin."""
    def fn(t, obs, env):
        if rng.random() < skip_prob:
            return None
        return int(rng.integers(0, n_bins))
    return fn


def per_node_R(adj, beta_vec, delta_vec):
    """Naive per-node effective reproduction number beta_i * degree_i / delta_i."""
    degree = adj.sum(axis=1)
    return beta_vec * degree / delta_vec


def calibrate_beta_scale(adj, base_beta_vec, delta_vec, skip_prob, cooldown=0,
                          target_mean_rate=0.25,
                          T=100, n_reps=150, seed=0, lo=0.05, hi=8.0, tol=0.02, max_iter=25):
    """
    Bisect a global multiplier on base_beta_vec so the mean adoption rate under the actual
    data-collection loop (random policy, skip_prob, cooldown, empty start) is near
    target_mean_rate; cooldown must match the downstream loop.
    """
    def mean_rate(scale):
        beta_vec = base_beta_vec * scale
        rng = np.random.default_rng(seed)
        totals = []
        for _ in range(n_reps):
            env = make_tiny_env(adj, beta_vec, delta_vec, T=T,
                                 seed=int(rng.integers(0, 1_000_000)), init_rate=0.0)
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
            f'calibrate_beta_scale did not converge: achieved={rate:.3f} vs '
            f'target={target_mean_rate:.3f} after {max_iter} iterations (scale={mid:.3f}, '
            f'search bounds=[{lo:.3f},{hi:.3f}]). Likely saturated at a search bound — '
            'widen lo/hi rather than silently returning an uncalibrated network.'
        )
    return mid, rate
