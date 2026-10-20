import numpy as np


class LIRPolicy:
    """
    Local Index Rank heuristic (Liu et al. 2017, Scientific Reports).

    A node is a local leader if its degree strictly exceeds that of all its neighbors
    (isolated nodes qualify). The priority order (local leaders by degree, then the rest)
    is fixed at construction; each period seeds the highest-priority susceptible node and
    returns its bin, or None if all are adopted.

    Parameters
    ----------
    env : environment with .adj, .degree, .N, .bin_of_node()
    """

    def __init__(self, env):
        degree = env.degree.copy().astype(float)
        adj    = env.adj

        local_leaders = []
        non_leaders   = []

        for i in range(env.N):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) == 0:
                local_leaders.append(i)
            elif degree[i] > degree[neighbors].max():
                local_leaders.append(i)
            else:
                non_leaders.append(i)

        # sort each group by degree descending
        local_leaders.sort(key=lambda x: -degree[x])
        non_leaders.sort(key=lambda x: -degree[x])

        # leaders first, non-leaders as fallback
        self.priority_order = local_leaders + non_leaders
        self.n_leaders = len(local_leaders)
        self._env  = env
        self._step = 0

    def __call__(self, obs, env=None) -> int:
        if self._step >= len(self.priority_order):
            return None
        node = self.priority_order[self._step]
        self._step += 1
        return -(node + 1)

    def reset(self):
        self._step = 0
