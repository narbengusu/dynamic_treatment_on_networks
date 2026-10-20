import numpy as np, pandas as pd, matplotlib.pyplot as plt
from style import *

apply_style()
D = EXP / "covariate"
res = pd.read_csv(D / "cluster_sweep_all_results.csv")
base = pd.read_csv(D / "cluster_sweep_all_baselines.csv")
TT = [100, 300, 600]
NS = [50, 100, 200, 500]
K = 5

fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.9), sharex=True, constrained_layout=True)
for i, (ax, N) in enumerate(zip(axes.ravel(), NS)):
    s = N * 20
    for key, col in [("dqn", "reward_plain_mean"), ("dqn_all", "reward_all_mean"), ("q_ising", "reward_ising_mean")]:
        g = res[(res.N == N) & (res.K == K)].groupby("T_train")[col]
        m = g.mean().reindex(TT) / s
        se = g.std(ddof=1).reindex(TT) / np.sqrt(g.count().reindex(TT)) / s
        ax.fill_between(TT, m - se, m + se, color=COLORS[key], alpha=0.2, lw=0)
        ax.plot(TT, m, color=COLORS[key], marker=MARKERS[key], ls=LINESTYLES[key], label=LABELS[key], zorder=3)
    b = base[(base.N == N) & (base.K == K)].set_index("policy").mean_deploy_reward
    ax.axhline(b[["random", "degree_centrality", "lir", "degree_bin"]].max() / s, color=COLORS["degree_bin"],
               ls="-.", label="Best topological")
    ax.axhline(b["oracle"] / s, color=COLORS["oracle"], ls="--", label="Oracle")
    ax.set_title(f"({'abcd'[i]}) $N={N}$")
    ax.set_xticks(TT)
    if i % 2 == 0: ax.set_ylabel("Reward per node-period")
    if i >= 2: ax.set_xlabel("$T_{\\mathrm{train}}$")
    grid(ax)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, columnspacing=1.2, handlelength=1.8)
save(fig, "e1_sweep")
