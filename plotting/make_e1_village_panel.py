import numpy as np, pandas as pd, matplotlib.pyplot as plt
from style import *

apply_style()
D = EXP / "covariate"
res = pd.read_csv(D / "cluster_sweep_all_results.csv")
base = pd.read_csv(D / "cluster_sweep_all_baselines.csv")
vil = pd.read_csv(EXP / "villages/village_policy_results.csv", index_col=0)
TT, N, K = [100, 300, 600], 100, 5
s = N * 20

fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), constrained_layout=True)
fig.get_layout_engine().set(wspace=0.08)

# (a) covariate experiment
for key, col in [("dqn", "reward_plain_mean"), ("dqn_all", "reward_all_mean"), ("q_ising", "reward_ising_mean")]:
    g = res[(res.N == N) & (res.K == K)].groupby("T_train")[col]
    m = g.mean().reindex(TT) / s
    se = g.std(ddof=1).reindex(TT) / np.sqrt(g.count().reindex(TT)) / s
    a.fill_between(TT, m - se, m + se, color=COLORS[key], alpha=0.2, lw=0)
    a.plot(TT, m, color=COLORS[key], marker=MARKERS[key], ls=LINESTYLES[key], label=LABELS[key], zorder=3)
    print(key, [round(v, 4) for v in m])
bb = base[(base.N == N) & (base.K == K)].set_index("policy").mean_deploy_reward
best = bb[["random", "degree_centrality", "lir", "degree_bin"]].max() / s
orc = bb["oracle"] / s
print("best baseline", round(best, 4), "oracle", round(orc, 4))
a.axhline(best, color=COLORS["degree_bin"], ls="-.", label="Best topological")
a.axhline(orc, color=COLORS["oracle"], ls="--", label="Oracle")
a.set_title("(a) Covariate experiment")
a.set_xticks(TT); a.set_xlabel("$T_{\\mathrm{train}}$"); a.set_ylabel("Reward per node-period")
grid(a)

# (b) villages
imp = ((vil.q_ising_mean - vil.degree_bin_mean) / (vil.degree_bin_mean.abs() + 1e-9) * 100).sort_values().values
col = np.where(imp < 0, VERMILLION, np.where(imp <= 5, "#808080", BLUE))
b.bar(np.arange(len(imp)), imp, color=col, edgecolor="black", linewidth=0.4, zorder=3)
for y in (5, -5):
    b.axhline(y, color=VERMILLION, ls="--", lw=1.0, zorder=2)
b.axhline(0, color="black", lw=0.8, zorder=4)
b.set_yscale("symlog", linthresh=5, linscale=0.6)
ticks = [-20, -5, 0, 5, 20, 50]
b.set_yticks(ticks); b.set_yticklabels([f"{t}" for t in ticks]); b.minorticks_off()
b.set_title("(b) Villages"); b.set_xlabel("Village (sorted)"); b.set_ylabel("Improvement (%)")
grid(b)

h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5, columnspacing=1.0, handlelength=1.6, handletextpad=0.5)
save(fig, "e1_village_panel")
