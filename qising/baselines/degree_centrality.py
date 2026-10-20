import numpy as np


class DegreeCentralityPolicy:
    """Fixed degree-descending node ranking; each call returns the bin of the next node, ignoring adoption state."""

    def __init__(self, env):
        self.seq  = sorted(range(env.N), key=lambda x: -env.degree[x])
        self._env = env
        self._step = 0

    def __call__(self, obs, env=None) -> int:
        if self._step >= len(self.seq):
            return None
        node = self.seq[self._step]
        self._step += 1
        return -(node + 1)

    def reset(self):
        self._step = 0
