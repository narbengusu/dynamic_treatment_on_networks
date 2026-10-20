"""
Memoryless SBM environment for the hidden-covariate sweep. Each block has N_COVARIATES
covariates redrawn every period; only EFFECTIVE_IDX (binary, Bernoulli(p_z)) matters.
If it is "hot", block members adopt w.p. P_BASE, or P_ESCALATED if the block is seeded;
the seeded node always adopts. Actions: bin b treats the block's highest-degree node;
a negative action -(node+1) treats that node directly (topological baselines).
"""

import numpy as np

N_COVARIATES = 10
EFFECTIVE_IDX = 0


def build_sbm_network(N, n_blocks, p_in, p_out, seed=0, block_weights=None):
    """General n_blocks-block SBM. Returns (adj, block_labels)."""
    rng = np.random.default_rng(seed)
    if block_weights is None:
        block_weights = [1] * n_blocks
    total_w = sum(block_weights)
    block_sizes = [int(N * w / total_w) for w in block_weights]
    remainder = N - sum(block_sizes)
    for i in range(remainder):
        block_sizes[i] += 1

    block_labels = np.zeros(N, dtype=int)
    idx = 0
    for b, sz in enumerate(block_sizes):
        block_labels[idx:idx + sz] = b
        idx += sz

    adj = np.zeros((N, N), dtype=float)
    for i in range(N):
        same_block = block_labels[i + 1:] == block_labels[i]
        js = np.arange(i + 1, N)
        p = np.where(same_block, p_in, p_out)
        draws = rng.random(len(js))
        hit = js[draws < p]
        adj[i, hit] = 1.0
        adj[hit, i] = 1.0
    np.fill_diagonal(adj, 0)

    degree = adj.sum(axis=1)
    isolated = np.where(degree == 0)[0]
    for node in isolated:
        b = int(block_labels[node])
        members = np.where(block_labels == b)[0]
        candidates = members[members != node]
        if len(candidates) > 0:
            partner = int(rng.choice(candidates))
            adj[node, partner] = 1.0
            adj[partner, node] = 1.0

    return adj, block_labels


class CovariateGroupControllableEnv:
    def __init__(self, adj, block_labels, p_z, p_base=0.3, p_escalated=0.6,
                 n_covariates=N_COVARIATES, effective_idx=EFFECTIVE_IDX, seed=0):
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
        self.p_z = float(p_z)
        self.p_base = float(p_base)
        self.p_escalated = float(p_escalated)
        self.n_covariates = n_covariates
        self.effective_idx = effective_idx
        self.rng = np.random.default_rng(seed)
        self.adopted = None
        self.C_group = None  # (n_bins, n_covariates); column effective_idx is binary {0,1}
        self.current_time = 0
        self.reset()

    def _draw_covariates(self):
        C = self.rng.random((self.n_bins, self.n_covariates))
        C[:, self.effective_idx] = (self.rng.random(self.n_bins) < self.p_z).astype(float)
        return C

    def reset(self, init_adopted=None):
        if init_adopted is not None:
            self.adopted = np.asarray(init_adopted, dtype=bool).copy()
        else:
            self.adopted = np.zeros(self.N, dtype=bool)
        self.C_group = self._draw_covariates()
        self.current_time = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            'adopted': self.adopted.copy(),
            'C_group': self.C_group.copy(),
            'adoption_rate': float(self.adopted.mean()),
        }

    def get_available_actions(self):
        return list(range(self.n_bins))  # memoryless -- always all valid

    def _decode_action(self, action):
        a = int(action)
        if a < 0:
            return -(a + 1)
        return int(self._bin_rep[a])

    def step(self, action):
        c_group_this_period = self.C_group.copy()
        z_hot = c_group_this_period[:, self.effective_idx] > 0.5  # (n_bins,) binary

        # reset: no persistence
        self.adopted[:] = False

        seeded_node = None
        seeded_group = None
        if action is not None:
            seeded_node = self._decode_action(action)
            seeded_group = int(self.block_labels[seeded_node])

        # per-node adoption probability (escalated for the whole seeded block if hot)
        p_per_node = np.zeros(self.N)
        hot_mask = z_hot[self.block_labels]
        p_per_node[hot_mask] = self.p_base
        if seeded_group is not None and z_hot[seeded_group]:
            p_per_node[self.bins[seeded_group]] = self.p_escalated

        newly_adopted = self.rng.random(self.N) < p_per_node
        if seeded_node is not None:
            newly_adopted[seeded_node] = True  # deterministic direct treatment

        self.adopted[newly_adopted] = True
        reward = float(newly_adopted.sum())

        # covariate draw for the next period
        self.C_group = self._draw_covariates()
        self.current_time += 1

        info = {'seeded_node': seeded_node, 'seeded_group': seeded_group,
                'n_newly_adopted': int(newly_adopted.sum()), 'c_group_used': c_group_this_period}
        return self._get_obs(), reward, False, info
