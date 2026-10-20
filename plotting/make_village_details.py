"""figures/output.pdf (community histograms) and figures/structure.pdf (village examples)."""
import numpy as np, matplotlib.pyplot as plt
from style import *
from network_draw import draw_network
from villages_common import load_villages, community_layout
import networkx as nx

apply_style()
V = load_villages()
viable = {v: d for v, d in V.items() if d["n_bins"] >= 2}
counts = [d["n_bins"] for d in viable.values()]
sizes = [int(s) for d in viable.values() for s in d["bin_counts"]]
print(f"villages {len(V)}, viable {len(viable)}")
print(f"communities per village: mean {np.mean(counts):.1f} median {np.median(counts):.1f} min {min(counts)} max {max(counts)}")
print(f"nodes per community: mean {np.mean(sizes):.1f} median {np.median(sizes):.1f} min {min(sizes)} max {max(sizes)}")

fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5), constrained_layout=True)
a.hist(counts, bins=np.arange(0.5, max(counts) + 1.5, 1), color=BLUE, edgecolor="white")
a.set_xticks(np.arange(1, max(counts) + 1)); a.set_xlabel("Communities"); a.set_ylabel("Villages")
a.set_title("(a) Per village"); a.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); grid(a)
b.hist(sizes, bins=20, color=BLUE, edgecolor="white")
b.set_xlabel("Nodes per community"); b.set_ylabel("Count"); b.set_title("(b) Pooled"); grid(b)
save(fig, "output", None)

ids = ["25", "65", "60", "3", "52", "64"]
fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 3.9), constrained_layout=True)
for ax, v in zip(axes.ravel(), ids):
    d = V[v]; G = nx.from_numpy_array(d["adj"])
    pos = community_layout(G, d["block_labels"], d["n_bins"])
    P = np.array([pos[i] for i in range(d["N"])])
    draw_network(ax, d["adj"], P, d["block_labels"], d["n_bins"], size=8)
    ax.set_title(f"Village {v}, $K={d['n_bins']}$")
save(fig, "structure", None)
