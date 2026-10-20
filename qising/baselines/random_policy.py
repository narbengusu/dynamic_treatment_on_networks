import numpy as np


class RandomPolicy:
    """Uniform random bin each period (floor baseline)."""

    def __init__(self, n_bins: int, seed: int = None):
        self.n_bins = n_bins
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs, env=None) -> int:
        return int(self._rng.integers(0, self.n_bins))

    def reset(self):
        pass
