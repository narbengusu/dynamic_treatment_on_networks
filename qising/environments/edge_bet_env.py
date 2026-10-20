import numpy as np


class EdgeBetCommunitySISEnv:
    """
    SIS diffusion with igraph edge-betweenness community-based bins.
    Variable n_bins (one per detected community). No skip action.
    """

    def __init__(self, adj, beta_vec, delta_vec, block_labels,
                 T=300, seed=42, reward_type="current", init_rate=0.0):
        self.rng          = np.random.default_rng(seed)
        self.adj          = adj.copy().astype(float)
        np.fill_diagonal(self.adj, 0)
        self.N            = adj.shape[0]
        self.T            = T
        self.reward_type  = reward_type
        self.init_rate    = init_rate
        self.beta_vec     = np.asarray(beta_vec,  dtype=float).copy()
        self.delta_vec    = np.asarray(delta_vec, dtype=float).copy()
        self.degree       = self.adj.sum(axis=1)
        block_labels      = np.asarray(block_labels, dtype=int)
        self.block_labels = block_labels
        self.n_bins       = int(block_labels.max()) + 1
        self.n_actions    = self.n_bins
        self.bins         = [np.where(block_labels == b)[0] for b in range(self.n_bins)]
        self.bin_sizes    = [len(m) for m in self.bins]
        self.bin_names    = [f"Comm {b} ({len(self.bins[b])} nodes)" for b in range(self.n_bins)]
        self.adopted = None; self.current_time = None
        self.seeded_history = None; self.reward_history = None
        self.reset()

    def bin_of_node(self, node):
        return int(self.block_labels[node])

    def reset(self, init_state=None):
        if init_state is not None:
            self.adopted = np.asarray(init_state, dtype=bool).copy()
        else:
            self.adopted = np.zeros(self.N, dtype=bool)
            n_init = int(self.init_rate * self.N)
            if n_init > 0:
                self.adopted[self.rng.choice(self.N, size=n_init, replace=False)] = True
        self.current_time = 0
        self.seeded_history = []; self.reward_history = []
        return self._get_obs()

    def step(self, action):
        idx = np.where(self.adopted)[0]
        if len(idx) > 0:
            churned = idx[self.rng.random(len(idx)) < self.delta_vec[idx]]
            self.adopted[churned] = False

        seeded_node = None
        if action is not None:
            a = int(action)
            if a < 0:
                seeded_node = self._seed_node(-(a + 1))
            elif a < self.n_bins:
                seeded_node = self._seed_bin(a)

        newly_adopted = np.zeros(self.N, dtype=bool)
        if self.adopted.any():
            log_p_esc = self.adj @ (self.adopted.astype(float) * np.log1p(-self.beta_vec))
            p_adopt   = 1.0 - np.exp(log_p_esc)
            new_mask  = ~self.adopted & (self.rng.random(self.N) < p_adopt)
            self.adopted[new_mask] = True
            newly_adopted = new_mask

        self.current_time += 1
        done    = self.current_time >= self.T
        n_after = int(self.adopted.sum())
        reward  = (float(n_after) if self.reward_type in ("current", "")
                   else float(newly_adopted.sum()) if self.reward_type == "new"
                   else (float(n_after) if done else 0.0))
        self.reward_history.append(reward)
        if seeded_node is not None:
            self.seeded_history.append((self.current_time, seeded_node))
        info = {"n_adopted": n_after, "adoption_rate": float(self.adopted.mean()),
                "seeded_node": seeded_node, "seeded_bin": action, "t": self.current_time}
        return self._get_obs(), reward, done, info

    def _seed_bin(self, bin_idx):
        members = self.bins[bin_idx]
        cands   = members[~self.adopted[members]]
        if len(cands) == 0:
            return None
        node = int(self.rng.choice(cands))
        self.adopted[node] = True
        return node

    def _seed_node(self, node_idx):
        """Seed a specific node (node-specific policies). Returns None if already adopted."""
        if self.adopted[node_idx]:
            return None
        self.adopted[node_idx] = True
        return node_idx

    def _get_obs(self):
        return {"adopted": self.adopted.copy(),
                "bin_states": self._bin_adoption_rates(), "t": self.current_time}

    def _bin_adoption_rates(self):
        return np.array([float(self.adopted[m].mean()) if len(m) > 0 else 0.0
                         for m in self.bins])

    def get_available_actions(self):
        return [b for b in range(self.n_bins)
                if len(self.bins[b]) > 0 and (~self.adopted[self.bins[b]]).any()]

    def get_taker_rate(self):
        return float(self.adopted.mean())

    def collect_behavioral_episode(self, skip_prob=0.0, seed=None, verbose=False):
        """Bin-uniform behavioral policy; skips encoded as action=n_bins."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset()
        adopted_traj  = np.zeros((self.T, self.N), dtype=bool)
        action_traj   = np.full(self.T, self.n_bins, dtype=int)
        seeded_nodes  = np.full(self.T, -1, dtype=int)
        reward_traj   = np.zeros(self.T, dtype=float)
        infos         = []
        for t in range(self.T):
            adopted_traj[t] = self.adopted.copy()
            if skip_prob > 0.0 and self.rng.random() < skip_prob:
                action = None
            else:
                action = int(self.rng.integers(0, self.n_bins))
                action_traj[t] = action
            obs, reward, done, info = self.step(action)
            reward_traj[t] = reward
            infos.append(info)
            if info["seeded_node"] is not None:
                seeded_nodes[t] = info["seeded_node"]
        if verbose:
            n_seeds = (action_traj < self.n_bins).sum()
            print(f"  T={self.T}, N={self.N}, n_bins={self.n_bins}, "
                  f"seeds={n_seeds}/{self.T}, mean_adoption={adopted_traj.mean():.3f}")
        return {"adopted": adopted_traj, "actions": action_traj,
                "seeded_nodes": seeded_nodes, "rewards": reward_traj,
                "infos": infos, "T": self.T, "N": self.N}
