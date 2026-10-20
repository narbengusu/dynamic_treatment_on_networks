import numpy as np


class EnsemblePolicy:
    """
    Majority-vote ensemble over multiple CQL models sampled from the Ising posterior.
    models: list of (_, draw_params, mdl_draw, so_draw) tuples.
    """

    def __init__(self, models):
        self.models = models

    def __call__(self, obs, env):
        ba = obs["adopted"]
        avl = set(env.get_available_actions())
        if not avl:
            return None
        votes = np.zeros(env.n_bins, dtype=int)
        for _, _draw_params, mdl_draw, so_draw in self.models:
            ob = so_draw(ba).reshape(1, -1)
            qv = mdl_draw.predict_value(
                np.repeat(ob, env.n_bins, axis=0),
                np.arange(env.n_bins, dtype=np.int32),
            )
            for b in np.argsort(qv)[::-1]:
                if int(b) in avl:
                    votes[int(b)] += 1
                    break
        avl_arr = np.array(list(avl), dtype=int)
        return int(avl_arr[np.argmax(votes[avl_arr])])

    def reset(self):
        pass
