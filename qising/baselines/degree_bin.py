import numpy as np


class DegreeBinPolicy:
    """Seeds the highest-degree node of each bin in round-robin order across bins; None when exhausted."""

    def __init__(self, env):
        bin_lists = []
        for b in range(env.n_bins):
            members = sorted(env.bins[b], key=lambda x: -env.degree[x])
            bin_lists.append(members)

        self.priority_order = []
        max_len = max(len(bl) for bl in bin_lists)
        for i in range(max_len):
            for bl in bin_lists:
                if i < len(bl):
                    self.priority_order.append(bl[i])

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
