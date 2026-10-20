"""Network drawing shared by the seeding-sequence and community-structure figures."""
import numpy as np
from style import community_color, COMM_MARKERS


def draw_network(ax, A, pos, labels, n_bins, size=None):
    """A: adjacency, pos: (N,2) array, labels: community per node."""
    N = A.shape[0]
    iu, ju = np.nonzero(np.triu(A, 1))
    n_edges = len(iu)
    alpha = float(np.clip(8.0 / max(n_edges / N * 10, 1), 0.04, 0.45))
    segs = np.stack([pos[iu], pos[ju]], axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#888888", linewidths=0.4, alpha=alpha, zorder=1))
    s = size if size is not None else max(10, 150 / np.sqrt(N) * 1.6)
    for b in range(n_bins):
        m = np.where(labels == b)[0]
        if len(m) == 0:
            continue
        ax.scatter(pos[m, 0], pos[m, 1], marker=COMM_MARKERS[b % len(COMM_MARKERS)],
                   color=community_color(b), s=s, edgecolors="white", linewidths=0.3, zorder=3,
                   label=str(b))
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
