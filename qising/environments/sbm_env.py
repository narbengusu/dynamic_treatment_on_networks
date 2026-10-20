"""
SBM covariate environment: SIS diffusion on a stochastic block model whose 4 blocks are
defined by node covariates (college x age_young; block_id = college*2 + age_young).
Churn is lower for college-educated nodes. Action = which covariate bin to seed (uniform
over its susceptible nodes). Same interface as SISDiffusionEnv.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple


# Block name lookup

BIN_NAMES = [
    "Older, Non-col, F",   # block 0: gender=0, college=0, age_young=0
    "Young, Non-col, F",   # block 1: gender=0, college=0, age_young=1
    "Older, College, F",   # block 2: gender=0, college=1, age_young=0
    "Young, College, F",   # block 3: gender=0, college=1, age_young=1
]

DELTA_COLLEGE    = 0.10   # sticky adoption for college-educated nodes
DELTA_NON_COLLEGE = 0.50  # faster churn for non-college nodes


# Network generation

def generate_sbm(
    N: int = 300,
    p_in: float = 0.15,
    p_out: float = 0.02,
    seed: int = 42,
    block_weights: list = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 4-block SBM (block_id = college*2 + age_young).

    block_weights : relative block sizes (default [1,1,1,1]; [3,3,1,1] = 75% non-college).
    Returns adj (N, N), covariates (N, 2) [college, age_young], block_labels (N,).
    """
    rng = np.random.default_rng(seed)
    n_blocks = 4

    # nodes per block proportional to block_weights
    if block_weights is None:
        block_weights = [1, 1, 1, 1]
    total_w = sum(block_weights)
    block_sizes = [int(N * w / total_w) for w in block_weights]
    remainder = N - sum(block_sizes)
    for i in range(remainder):
        block_sizes[i] += 1

    # covariates and block labels
    covariates   = np.zeros((N, 2), dtype=int)
    block_labels = np.zeros(N,      dtype=int)
    idx = 0
    for b, sz in enumerate(block_sizes):
        college   = (b >> 1) & 1
        age_young = (b >> 0) & 1
        for _ in range(sz):
            covariates[idx]   = [college, age_young]
            block_labels[idx] = b
            idx += 1

    # adjacency
    adj = np.zeros((N, N), dtype=float)
    # first node index of each block
    starts = np.zeros(n_blocks + 1, dtype=int)
    for b, sz in enumerate(block_sizes):
        starts[b + 1] = starts[b] + sz

    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if block_labels[i] == block_labels[j] else p_out
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    np.fill_diagonal(adj, 0)

    # connect isolated nodes to a random block-mate
    degree = adj.sum(axis=1)
    isolated = np.where(degree == 0)[0]
    for node in isolated:
        b = int(block_labels[node])
        block_members = np.where(block_labels == b)[0]
        candidates = block_members[block_members != node]
        if len(candidates) > 0:
            partner = int(rng.choice(candidates))
            adj[node, partner] = 1.0
            adj[partner, node] = 1.0

    print(f"SBM: N={N}, blocks={n_blocks}, p_in={p_in}, p_out={p_out}")
    print(f"  Block sizes: {block_sizes}")
    mean_deg = adj.sum(axis=1).mean()
    print(f"  Mean degree: {mean_deg:.1f}")
    return adj, covariates, block_labels


# Covariate bin construction

def make_covariate_bins(
    block_labels: np.ndarray,
    n_blocks: int = 8,
) -> List[np.ndarray]:
    """Node-index arrays, one per covariate block."""
    return [np.where(block_labels == b)[0] for b in range(n_blocks)]


# SBM covariate SIS environment

class SBMCovariateEnv:
    """
    SIS influence maximisation on an SBM with covariate-defined action bins.

    Extra attributes: covariates (N, 2), block_labels (N,), delta_vec (N,) churn rates,
    bin_names (see BIN_NAMES). beta is the scalar per-edge adoption probability.
    reward_type: 'current' (total adopted, recommended) | 'new' (newly adopted by spread)
    | 'final' (0 each step, total adopted at the end).
    """

    def __init__(
        self,
        adj: np.ndarray,
        covariates: np.ndarray,
        block_labels: np.ndarray,
        beta: float = 0.02,
        T: int = 300,
        seed: int = 42,
        reward_type: str = 'current',
        init_rate: float = 0.05,
        beta_college: float = None,
        beta_noncollege: float = None,
        beta_young_mult: float = 1.0,
    ):
        self.rng = np.random.default_rng(seed)

        self.adj = adj.copy().astype(float)
        np.fill_diagonal(self.adj, 0)
        self.N = self.adj.shape[0]

        self.covariates   = covariates.copy()
        self.block_labels = block_labels.copy()

        self.beta         = beta
        self.T            = T
        self.reward_type  = reward_type
        self.init_rate    = init_rate
        self.n_bins       = 4   # one per covariate block
        self.n_actions    = self.n_bins + 1  # action = which bin to seed

        # degree (used by baselines)
        self.degree = self.adj.sum(axis=1)

        # per-node churn rate, by college status and age
        self.delta_vec = 0.7 * np.where(
            covariates[:, 0] == 1,
            DELTA_COLLEGE,
            DELTA_NON_COLLEGE,
        ).astype(float) + 0.3 * np.where(
            covariates[:, 1] == 1,
            DELTA_COLLEGE,
            DELTA_NON_COLLEGE,
        ).astype(float)

        # per-node infection probability by college status (default: scalar beta)
        _bc  = beta_college    if beta_college    is not None else beta
        _bnc = beta_noncollege if beta_noncollege is not None else beta
        self.beta_vec = np.where(
            covariates[:, 0] == 1, _bc, _bnc,
        ).astype(float)
        # young nodes scale their sender beta
        if beta_young_mult != 1.0:
            self.beta_vec[covariates[:, 1] == 1] *= beta_young_mult
        self.beta_college    = _bc
        self.beta_noncollege = _bnc
        self.beta_young_mult = beta_young_mult

        # bins: one per covariate block
        self.bins       = make_covariate_bins(block_labels, self.n_bins)
        self.bin_labels = block_labels.copy()
        self.bin_names  = BIN_NAMES

        bin_sizes = [len(b) for b in self.bins]
        print(f"Covariate bins (block sizes): {bin_sizes}")
        self._print_calibration_info()

        self.adopted          = None
        self.current_time     = None
        self.seeded_history   = None
        self.reward_history   = None
        self.reset()

    def _print_calibration_info(self):
        print(f"\nSBM calibration:")
        print(f"  beta_college={self.beta_college:.3f}  beta_noncollege={self.beta_noncollege:.3f}"
              f"  beta_young_mult={self.beta_young_mult:.2f}")
        print(f"  delta_college={DELTA_COLLEGE}  delta_non_college={DELTA_NON_COLLEGE}")
        print(f"  Per-block effective R = beta_b * mean_degree_b / delta_b:")
        for b, members in enumerate(self.bins):
            if len(members) == 0:
                continue
            beta_b  = float(self.beta_vec[members[0]])
            delta_b = float(self.delta_vec[members].mean())
            deg_b   = float(self.degree[members].mean())
            R_b     = beta_b * deg_b / delta_b
            status  = "SUPERCRIT" if R_b > 1 else "subcrit"
            bname   = self.bin_names[b] if b < len(self.bin_names) else f"bin_{b}"
            print(f"    [{b}] {bname:<26} beta={beta_b:.3f}  delta={delta_b:.3f}"
                  f"  k={deg_b:.1f}  R={R_b:.2f}  {status}")
        n_supercrit = sum(
            1 for m in self.bins
            if len(m) > 0 and
               self.beta_vec[m[0]] * self.degree[m].mean() / self.delta_vec[m].mean() > 1
        )
        if n_supercrit == 0:
            print("  WARNING: all blocks subcritical — epidemic dies out everywhere.")
        elif n_supercrit == len(self.bins):
            print("  WARNING: all blocks supercritical — targeting may not help much.")
        else:
            print(f"  STATUS: {n_supercrit}/{len(self.bins)} blocks supercritical "
                  f"— heterogeneous R creates learnable targeting advantage.")

    # Core API

    def reset(self) -> dict:
        """Reset episode. init_rate fraction randomly infected."""
        self.adopted = np.zeros(self.N, dtype=bool)
        n_init = int(self.init_rate * self.N)
        if n_init > 0:
            init_nodes = self.rng.choice(self.N, size=n_init, replace=False)
            self.adopted[init_nodes] = True
        self.current_time   = 0
        self.seeded_history = []
        self.reward_history = []
        return self._get_obs()

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """One period: churn, seed, spread. action: bin index, -(node+1), or None."""
        n_adopted_before = int(self.adopted.sum())

        # churn (per-node delta)
        adopted_indices = np.where(self.adopted)[0]
        if len(adopted_indices) > 0:
            churn_draws     = self.rng.random(len(adopted_indices))
            per_node_delta  = self.delta_vec[adopted_indices]
            churned         = adopted_indices[churn_draws < per_node_delta]
            self.adopted[churned] = False
        else:
            churned = np.array([], dtype=int)

        # seed: bin (action >= 0) or exact node (action < 0, encoded as -(node+1))
        seeded_node = None
        if action is not None:
            a = int(action)
            if a < 0:
                seeded_node = self._seed_node(-(a + 1))
            elif a < self.n_bins:
                seeded_node = self._seed_bin(a)

        # spread: each infected node i spreads at its own beta_i;
        # log p_escape_j = sum_{i infected} adj[j,i] * log(1 - beta_i)
        if self.adopted.any():
            safe_beta    = np.minimum(self.beta_vec, 1.0 - 1e-15)
            log_p_escape = self.adj @ (self.adopted.astype(float) * np.log(1.0 - safe_beta))
            p_escape     = np.exp(log_p_escape)
            p_adopt      = 1.0 - p_escape
            susceptible         = ~self.adopted
            spread_draws        = self.rng.random(self.N)
            newly_adopted       = susceptible & (spread_draws < p_adopt)
            self.adopted[newly_adopted] = True
        else:
            newly_adopted = np.zeros(self.N, dtype=bool)

        self.current_time += 1
        done = self.current_time >= self.T
        n_adopted_after = int(self.adopted.sum())

        if self.reward_type == 'current':
            reward = float(n_adopted_after)
        elif self.reward_type == 'new':
            reward = float(int(newly_adopted.sum()))
        elif self.reward_type == 'final':
            reward = float(n_adopted_after) if done else 0.0
        else:
            reward = float(n_adopted_after)

        self.reward_history.append(reward)
        if seeded_node is not None:
            self.seeded_history.append((self.current_time, seeded_node))

        info = {
            'n_adopted':               n_adopted_after,
            'n_susceptible':           self.N - n_adopted_after,
            'n_newly_adopted_spread':  int(newly_adopted.sum()),
            'n_churned':               int(len(churned)),
            'adoption_rate':           float(self.adopted.mean()),
            'seeded_node':             seeded_node,
            'seeded_bin':              action,
            'action_wasted':           (action is not None and seeded_node is None),
            't':                       self.current_time,
        }
        return self._get_obs(), reward, done, info

    # Observation

    def _get_obs(self) -> dict:
        return {
            'adopted':    self.adopted.copy(),
            'bin_states': self._bin_adoption_rates(),
            't':          self.current_time,
        }

    def obs_flat(self) -> np.ndarray:
        """Flat obs for Q-network. Shape: (N + n_bins + 1,)."""
        return np.concatenate([
            self.adopted.astype(float),
            self._bin_adoption_rates(),
            [self.current_time / self.T],
        ])

    def _bin_adoption_rates(self) -> np.ndarray:
        """Mean adoption rate per covariate bin. Shape: (n_bins,)."""
        return np.array([
            float(self.adopted[members].mean()) if len(members) > 0 else 0.0
            for members in self.bins
        ])

    # Seeding

    def _seed_bin(self, bin_idx: int) -> Optional[int]:
        """Seed a uniformly random susceptible node in the chosen bin."""
        members = self.bins[bin_idx]
        susceptible_in_bin = members[~self.adopted[members]]
        if len(susceptible_in_bin) == 0:
            return None
        best = int(self.rng.choice(susceptible_in_bin))
        self.adopted[best] = True
        return best

    def _seed_node(self, node_idx: int) -> Optional[int]:
        """Seed a specific node (node-specific policies). Returns None if already adopted."""
        if self.adopted[node_idx]:
            return None
        self.adopted[node_idx] = True
        return node_idx

    # Interface compatibility with SISDiffusionEnv

    def get_available_actions(self) -> list:
        """Bin indices with at least one susceptible node, plus skip (always available)."""
        seeding = [
            b for b in range(self.n_bins)
            if len(self.bins[b]) > 0 and (~self.adopted[self.bins[b]]).any()
        ]
        return seeding + [self.n_bins]   # skip action is always available

    def get_taker_rate(self) -> float:
        return float(self.adopted.mean())

    def get_informed_rate(self) -> float:
        return float(self.adopted.mean())

    def degree_centrality(self) -> np.ndarray:
        return self.degree / self.degree.max()

    def bin_of_node(self, node_idx: int) -> int:
        return int(self.block_labels[node_idx])

    def add_centrality_bin(self, threshold: float = 0.8) -> np.ndarray:
        """
        Append a 'High Centrality' bin (index n_bins) of nodes whose normalized degree exceeds
        `threshold`, in place. block_labels and bin_of_node are unchanged.
        Returns the node indices of the new bin.
        """
        norm_deg = self.degree / self.degree.max()
        central_nodes = np.where(norm_deg > threshold)[0]
        self.bins.append(central_nodes)
        self.n_bins += 1
        self.bin_names = self.bin_names + [f'High Centrality (deg>{threshold})']
        print(f'Added centrality bin {self.n_bins - 1}: '
              f'{len(central_nodes)} nodes with norm_degree > {threshold}')
        return central_nodes

    def get_bin_summary(self) -> dict:
        return {
            b: {
                'size':         len(members),
                'adoption_rate': float(self.adopted[members].mean()) if len(members) > 0 else 0.0,
                'n_adopted':    int(self.adopted[members].sum()),
                'n_susceptible': int((~self.adopted[members]).sum()),
                'mean_degree':  float(self.degree[members].mean()) if len(members) > 0 else 0.0,
                'mean_delta':   float(self.delta_vec[members].mean()) if len(members) > 0 else 0.0,
                'bin_name':     self.bin_names[b] if b < len(self.bin_names) else f'bin_{b}',
            }
            for b, members in enumerate(self.bins)
        }

    # Diagnostics

    def run_endemic_diagnostic(
        self,
        n_periods: int = 500,
        verbose: bool = True,
    ) -> np.ndarray:
        """Simulate without seeding to check endemic dynamics (state is restored)."""
        saved_adopted = self.adopted.copy()
        saved_time    = self.current_time

        self.reset()
        rates = np.zeros(n_periods)
        for t in range(n_periods):
            _, _, _, info = self.step(None)
            rates[t] = info['adoption_rate']

        if verbose:
            burn = n_periods // 3
            m, s = rates[burn:].mean(), rates[burn:].std()
            print(f"\nEndemic diagnostic (no seeding, {n_periods} periods):")
            print(f"  Burn-in discarded:    first {burn} periods")
            print(f"  Endemic mean:         {m:.3f}")
            print(f"  Endemic std:          {s:.3f}")
            print(f"  Range: [{rates[burn:].min():.3f}, {rates[burn:].max():.3f}]")

            # per-block rates
            print("  Block-level endemic rates (last 2/3 of run):")
            for b, members in enumerate(self.bins):
                if len(members) > 0:
                    block_rates = np.array([
                        self.adopted[members].mean()
                        for _ in range(1)
                    ])
            block_means = []
            for b, members in enumerate(self.bins):
                if len(members) > 0:
                    block_means.append(float(self.adopted[members].mean()))
                else:
                    block_means.append(0.0)
            for b, bm in enumerate(block_means):
                bname = self.bin_names[b] if b < len(self.bin_names) else f'bin_{b}'
                print(f"    block {b} [{bname}]: {bm:.3f}")

            if m < 0.10:
                print("  STATUS: dies out. Raise beta.")
            elif m > 0.75:
                print("  STATUS: too saturated. Lower beta.")
            elif s < 0.01:
                print("  STATUS: very low variance. Dynamics may be too stable.")
            else:
                print("  STATUS: good endemic dynamics.")

        self.adopted      = saved_adopted
        self.current_time = saved_time
        return rates

    # Data collection

    def collect_behavioral_episode(
        self,
        behavioral_policy: str = 'random',
        burn_in: int = 50,
        verbose: bool = False,
        skip_prob = 0.5,
    ) -> dict:
        """
        Collect one long trajectory under a behavioral policy: 'random' (uniform random bin),
        'no_treat', 'centrality' (bin with highest mean delta) or 'greedy' (lowest adoption).
        """
        self.reset()
        for _ in range(burn_in):
            self.step(None)

        self.current_time   = 0
        self.seeded_history = []
        self.reward_history = []

        adopted_traj = np.zeros((self.T, self.N), dtype=bool)
        bin_traj     = np.zeros((self.T, self.n_bins), dtype=float)
        action_traj  = np.full(self.T, -1, dtype=int)
        reward_traj  = np.zeros(self.T, dtype=float)
        infos        = []

        SKIP_CODE = self.n_bins

        for t in range(self.T):
            adopted_traj[t] = self.adopted.copy()
            bin_traj[t]     = self._bin_adoption_rates()

            available = self.get_available_actions()
            if len(available) == 0:
                action = None
            elif behavioral_policy == 'random':
                action = int(self.rng.integers(0, self.n_bins))
            elif behavioral_policy == 'no_treat':
                action = None
            elif behavioral_policy == 'centrality':
                action = 0   # bin 0 is oldest, non-college — lowest delta
            elif behavioral_policy == 'greedy':
                rates  = self._bin_adoption_rates()
                masked = np.where(
                    [b in available for b in range(self.n_bins)],
                    rates, np.inf,
                )
                action = int(np.argmin(masked))
            else:
                action = int(self.rng.integers(0, self.n_bins))

            if self.rng.random() < skip_prob:
                action = None  # skip seeding with skip_prob probability

            action_traj[t] = action if action is not None else SKIP_CODE
            obs, reward, done, info = self.step(action)
            reward_traj[t]  = reward
            infos.append(info)

            if verbose and t % 50 == 0:
                print(f"  t={t:4d} | adoption={info['adoption_rate']:.3f} "
                      f"| churned={info['n_churned']:3d} "
                      f"| spread={info['n_newly_adopted_spread']:3d} "
                      f"| bin={action} | r={reward:.1f}")

        print(f"\nTrajectory: T={self.T}, N={self.N}, "
              f"mean adoption={adopted_traj.mean():.3f}")

        return {
            'adopted':    adopted_traj,
            'bin_states': bin_traj,
            'actions':    action_traj,
            'rewards':    reward_traj,
            'infos':      infos,
            'beta':       self.beta,
            'T':          self.T,
            'N':          self.N,
            'n_bins':     self.n_bins,
        }

    def collect_test_episodes(
        self,
        policy_fn,
        n_episodes: int = 100,
        burn_in: int = 0,
    ) -> dict:
        """Evaluate a policy over independent episodes."""
        total_rewards   = np.zeros(n_episodes)
        mean_adoptions  = np.zeros(n_episodes)
        adoption_curves = np.zeros((n_episodes, self.T))

        for ep in range(n_episodes):
            self.reset()
            if burn_in > 0:
                for _ in range(burn_in):
                    self.step(None)
                self.current_time   = 0
                self.seeded_history = []
                self.reward_history = []

            ep_reward = 0.0
            ep_rates  = np.zeros(self.T)
            for t in range(self.T):
                obs = self._get_obs()
                action = policy_fn(obs, self)
                _, reward, done, info = self.step(action)
                ep_reward  += reward
                ep_rates[t] = info['adoption_rate']

            total_rewards[ep]    = ep_reward
            mean_adoptions[ep]   = ep_rates.mean()
            adoption_curves[ep]  = ep_rates

        print(f"\nTest ({n_episodes} episodes, T={self.T}):")
        print(f"  Mean total reward: {total_rewards.mean():.2f} "
              f"+/- {total_rewards.std():.2f}")
        print(f"  Mean adoption:     {mean_adoptions.mean():.3f} "
              f"+/- {mean_adoptions.std():.3f}")

        return {
            'total_rewards':      total_rewards,
            'mean_adoption':      mean_adoptions,
            'adoption_curves':    adoption_curves,
            'mean_total_reward':  float(total_rewards.mean()),
            'std_total_reward':   float(total_rewards.std()),
        }


# One-call setup

def make_sbm_env(
    N: int = 300,
    p_in: float = 0.15,
    p_out: float = 0.02,
    beta: float = 0.02,
    T: int = 300,
    seed: int = 42,
    reward_type: str = 'current',
    init_rate: float = 0.05,
    block_weights: list = None,
    beta_college: float = None,
    beta_noncollege: float = None,
    beta_young_mult: float = 1.0,
) -> SBMCovariateEnv:
    """Generate the SBM network and build the covariate SIS environment."""
    adj, covariates, block_labels = generate_sbm(N=N, p_in=p_in, p_out=p_out,
                                                  seed=seed,
                                                  block_weights=block_weights)
    return SBMCovariateEnv(
        adj=adj,
        covariates=covariates,
        block_labels=block_labels,
        beta=beta,
        T=T,
        seed=seed,
        reward_type=reward_type,
        init_rate=init_rate,
        beta_college=beta_college,
        beta_noncollege=beta_noncollege,
        beta_young_mult=beta_young_mult,
    )
