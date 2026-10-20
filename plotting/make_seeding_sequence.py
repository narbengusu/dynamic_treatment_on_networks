import json, numpy as np, matplotlib.pyplot as plt
from style import *
from network_draw import draw_network

apply_style()
O = EXP / "villages/seeding_example/out"
S = json.load(open(O / "fig3_once_summary.json"))
# displayed episode: test seed 0 of the trained agent; panels (a) net 59, (b) seq 59, (c) net 71, (d) seq 71
fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.3), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.0, 1.25, 1.0, 1.25]})
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02)
VILLAGES = ["59", "71"]  # same villages as the original figure
for r, v in enumerate(VILLAGES):
    z = np.load(O / f"fig3_v{v}_layout.npz"); s = S[v]
    labels = np.array(s["block_labels"]); K = s["K"]
    ax = axes[2 * r]
    draw_network(ax, z["adj"], z["pos"], labels, K)
    ax.set_title(f"({'ac'[r]}) Village {v}")
    ax = axes[2 * r + 1]
    ch = np.array(s["choices_by_test_seed"]["0"])
    ax.step(ch[:, 0], ch[:, 1], where="mid", lw=1.0, color="0.6", zorder=2)
    for b in range(K):
        t = ch[ch[:, 1] == b, 0]
        ax.scatter(t, [b] * len(t), color=community_color(b), marker=COMM_MARKERS[b % len(COMM_MARKERS)],
                   s=22, zorder=3, edgecolors="white", linewidths=0.3)
    ax.set_yticks(range(K)); ax.set_ylim(-0.6, K - 0.4); ax.set_xlim(-0.8, ch[:, 0].max() + 0.8)
    ax.set_xticks([0, 12, 24])
    ax.set_xlabel("Step")
    if r == 0:
        ax.set_ylabel("Community seeded")
    ax.set_title(f"({'bd'[r]}) Village {v}"); grid(ax)
save(fig, "network_seeding_sequence_edgebet")
