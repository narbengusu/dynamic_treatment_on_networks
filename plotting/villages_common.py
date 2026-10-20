"""Village adjacency data, community detection and the notebook's community-aware layout."""
import json
import numpy as np, networkx as nx
from style import ROOT
from qising.utils import build_village_envdata

BETA_LIST = [0.01, 0.5, 0.05, 0.07, 0.06, 0.02, 0.01, 0.4, 0.1, 0.3]
DELTA_LIST = [0.5, 0.9, 0.9, 0.6, 0.5, 0.50, 0.7, 0.6, 0.5, 0.8]
ADJ_JSON = ROOT / "data" / "villages" / "adjacency_all.json"


def load_villages(ids=None):
    adj_all = json.load(open(ADJ_JSON))
    ids = sorted(adj_all, key=int) if ids is None else [str(i) for i in ids]
    out = {}
    for v in ids:
        A = np.array(adj_all[v], dtype=float)
        np.fill_diagonal(A, 0)
        out[v] = build_village_envdata(A, BETA_LIST, DELTA_LIST,
                                       comm_method="edge_betweenness", min_comm_size=10)
    return out


def community_layout(G, block_labels, n_bins, seed=42):
    """policy_eval.ipynb cell 17, unchanged."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    centers = {b: (np.cos(a), np.sin(a)) for b, a in enumerate(angles)}
    init_pos = {}
    for node in G.nodes():
        b = int(block_labels[node])
        cx, cy = centers[b]
        init_pos[node] = (cx + rng.normal(0, 0.25), cy + rng.normal(0, 0.25))
    return nx.spring_layout(G, pos=init_pos, seed=seed, k=1.2 / np.sqrt(len(G)), iterations=60)
