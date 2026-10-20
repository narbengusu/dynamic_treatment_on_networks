"""
Second-order saturation environment for the Ising-misspecification experiment. The dynamics
are not representable by the first-order group Ising model: a seed reaches first and second
neighbors, and churn is a threshold on 2-hop saturation. Step order:
  1. churn: adopted nodes revert w.p. 0.9 if >50% of their 1+2-hop neighbors are adopted
     (pre-churn state), else 0.2
  2. seed: the chosen node adopts
  3. spread: each node within two hops of the seed adopts w.p. P_SPREAD
"""

import numpy as np
import networkx as nx


def build_ba_graph_with_bins(N, n_bins, m, seed=0):
    """Barabasi-Albert graph (heavy-tailed degrees) with bins assigned by degree rank
    round-robin (rank r -> bin r mod n_bins), so the bin representatives are exactly the
    global top-n_bins hubs. Returns (adj, block_labels)."""
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)
    np.fill_diagonal(adj, 0)

    degree = adj.sum(axis=1)
    order = np.argsort(-degree)  # descending degree: order[r] = node at rank r
    block_labels = np.zeros(N, dtype=int)
    for rank, node in enumerate(order):
        block_labels[node] = rank % n_bins

    return adj, block_labels


def build_er_graph_with_bins(N, n_bins, target_avg_degree, seed=0):
    """Erdos-Renyi graph at about `target_avg_degree`, split into n_bins equal-ish bins
    (bins only define the action space, no causal meaning)."""
    rng = np.random.default_rng(seed)
    p = target_avg_degree / (N - 1)

    adj = np.zeros((N, N), dtype=float)
    iu = np.triu_indices(N, k=1)
    draws = rng.random(len(iu[0]))
    hit = draws < p
    adj[iu[0][hit], iu[1][hit]] = 1.0
    adj[iu[1][hit], iu[0][hit]] = 1.0

    degree = adj.sum(axis=1)
    isolated = np.where(degree == 0)[0]
    for node in isolated:
        candidates = [j for j in range(N) if j != node]
        partner = int(rng.choice(candidates))
        adj[node, partner] = 1.0
        adj[partner, node] = 1.0

    base = N // n_bins
    sizes = [base] * n_bins
    for i in range(N - base * n_bins):
        sizes[i] += 1
    block_labels = np.zeros(N, dtype=int)
    idx = 0
    for b, sz in enumerate(sizes):
        block_labels[idx:idx + sz] = b
        idx += sz

    return adj, block_labels


def compute_two_hop_reach(adj):
    """Returns (reach_mask, reach_size, mean_first_degree, mean_combined_degree);
    reach_mask[i,j] is True iff j is a first- or second-order neighbor of i (j != i)."""
    N = adj.shape[0]
    adj_bool = adj > 0
    two_hop = (adj_bool.astype(float) @ adj_bool.astype(float)) > 0
    reach_mask = adj_bool | two_hop
    np.fill_diagonal(reach_mask, False)
    reach_size = reach_mask.sum(axis=1)
    mean_first_degree = float(adj_bool.sum(axis=1).mean())
    mean_combined_degree = float(reach_size.mean())
    return reach_mask, reach_size, mean_first_degree, mean_combined_degree


class RandomNodePolicy:
    """Seeds a uniformly random NODE each period (action = -(node+1)); unlike
    qising.baselines.RandomPolicy, which picks a bin (always a real hub here)."""

    def __init__(self, N, seed=None):
        self.N = N
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs, env=None):
        node = int(self._rng.integers(0, self.N))
        return -(node + 1)

    def reset(self):
        pass


class SaturationSpilloverEnv:
    def __init__(self, adj, block_labels, p_spread=0.7,
                 churn_high=0.9, churn_low=0.2, saturation_threshold=0.5, seed=0):
        self.adj = adj.copy().astype(float)
        np.fill_diagonal(self.adj, 0)
        self.N = adj.shape[0]
        self.degree = self.adj.sum(axis=1)
        self.block_labels = np.asarray(block_labels, dtype=int)
        self.n_bins = int(self.block_labels.max()) + 1
        self.bins = [np.where(self.block_labels == b)[0] for b in range(self.n_bins)]
        self._bin_rep = np.array([
            int(self.bins[b][np.argmax(self.degree[self.bins[b]])]) for b in range(self.n_bins)
        ])

        self.reach_mask, self.reach_size, self.mean_first_degree, self.mean_combined_degree = \
            compute_two_hop_reach(self.adj)

        self.p_spread = float(p_spread)
        self.churn_high = float(churn_high)
        self.churn_low = float(churn_low)
        self.saturation_threshold = float(saturation_threshold)

        self.rng = np.random.default_rng(seed)
        self.adopted = None
        self.current_time = 0
        self.reset()

    def reset(self, init_adopted=None):
        if init_adopted is not None:
            self.adopted = np.asarray(init_adopted, dtype=bool).copy()
        else:
            self.adopted = np.zeros(self.N, dtype=bool)
        self.current_time = 0
        return self._get_obs()

    def _get_obs(self):
        return {'adopted': self.adopted.copy(), 'adoption_rate': float(self.adopted.mean())}

    def get_available_actions(self):
        return list(range(self.n_bins))  # every bin always seedable (direct-treat convention)

    def _decode_action(self, action):
        a = int(action)
        if a < 0:
            return -(a + 1)
        return int(self._bin_rep[a])

    def saturation(self, adopted=None):
        """Fraction of each node's 1+2-hop neighbor set that is adopted."""
        state = self.adopted if adopted is None else adopted
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = (self.reach_mask.astype(float) @ state.astype(float)) / np.maximum(self.reach_size, 1)
        return frac

    def step(self, action):
        # 1. churn on the pre-churn state
        pre_state = self.adopted.copy()
        sat = self.saturation(pre_state)
        churn_prob = np.where(sat > self.saturation_threshold, self.churn_high, self.churn_low)
        idx = np.where(pre_state)[0]
        if len(idx) > 0:
            churned = idx[self.rng.random(len(idx)) < churn_prob[idx]]
            self.adopted[churned] = False

        # 2. seed
        seeded_node = None
        if action is not None:
            seeded_node = self._decode_action(action)
            self.adopted[seeded_node] = True

        # 3. two-hop spread from the seeded node only
        newly_adopted = np.zeros(self.N, dtype=bool)
        if seeded_node is not None:
            reach = self.reach_mask[seeded_node]
            draws = self.rng.random(self.N) < self.p_spread
            flip = reach & draws & ~self.adopted
            newly_adopted[flip] = True
            self.adopted[flip] = True

        self.current_time += 1
        reward = float(self.adopted.sum())
        info = {'seeded_node': seeded_node, 'n_newly_adopted': int(newly_adopted.sum()),
                'saturation_at_seed': float(sat[seeded_node]) if seeded_node is not None else None}
        return self._get_obs(), reward, False, info
