import json, numpy as np, matplotlib.pyplot as plt
from style import *

apply_style()
O = EXP / "villages/ensemble_v50/out"
draws_z = np.load(O / "v50_mcmc_draws.npz"); em = np.load(O / "v50_emvs.npz")
meta = json.load(open(O / "v50_meta.json"))
G = meta["G"]
draws = [draws_z[f"g{g}"] for g in range(G)]
mp = [em[f"map_g{g}"] for g in range(G)]
SC = ["Intercept", "Persistence", "Direct", "Spillover"]
sub = "posterior"  # figures/mcmc_ensemble


def hist(ax, d, mapv):
    lo, hi = np.percentile(d, [2.5, 97.5])
    ax.axvspan(lo, hi, color=BAND, alpha=0.7, lw=0, zorder=1)
    ax.hist(d, bins=40, color=BLUE, alpha=0.85, density=True, edgecolor="none", zorder=2)
    ax.axvline(mapv, color=VERMILLION, lw=1.5, ls="--", zorder=3)
    ax.axvline(0, color="0.5", lw=0.8, ls=":", zorder=3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", color="0.8", linewidth=0.5); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune=None))


# scalars: G x 4
fig, axes = plt.subplots(G, 4, figsize=(FULL_WIDTH, 7.0), constrained_layout=True)
for g in range(G):
    for p in range(4):
        ax = axes[g, p]
        hist(ax, draws[g][:, p], mp[g][p])
        if g == 0: ax.set_title(SC[p])
        if p == 0: ax.set_ylabel(f"Comm {g}")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Line2D([0], [0], color=VERMILLION, lw=1.5, ls="--", label="EMVS MAP"),
                    Patch(fc=BAND, label="95% interval")],
           loc="outside lower center", ncol=2)
save(fig, "posterior_scalar_v50", "mcmc_ensemble")

# couplings: G x G
fig, axes = plt.subplots(G, G, figsize=(FULL_WIDTH, 7.0), constrained_layout=True)
for g in range(G):
    for m in range(G):
        ax = axes[g, m]
        hist(ax, draws[g][:, 4 + m], mp[g][4 + m])
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        if g == 0: ax.set_title(f"$m={m}$")
        if m == 0: ax.set_ylabel(f"$k={g}$")
        if g == G - 1: ax.set_xlabel(f"$\\gamma_{{{g},{m}}}$" if False else "$\\gamma$")
fig.legend(handles=[Line2D([0], [0], color=VERMILLION, lw=1.5, ls="--", label="EMVS MAP"),
                    Patch(fc=BAND, label="95% interval")],
           loc="outside lower center", ncol=2)
save(fig, "posterior_couplings_v50", "mcmc_ensemble")

# heatmaps
V0, V1 = meta["V0"], meta["V1"]
tau = np.sqrt(2 * V0 * V1 * np.log(V1 / V0) / (V1 - V0))
mcmc_pip = np.array([[(np.abs(draws[g][:, 4 + m]) > tau).mean() for m in range(G)] for g in range(G)])
std_mat = np.array([[draws[g][:, 4 + m].std() for m in range(G)] for g in range(G)])
cm = em["coupling_matrix_mean"]; pip = em["inclusion_probs"]
vmax_map = float(np.percentile(np.abs(cm), 95))
specs = [(pip, "(a) EMVS incl. prob.", SEQ_CMAP, 0, 1),
         (mcmc_pip, "(b) MCMC incl. prob.", SEQ_CMAP, 0, 1),
         (cm, "(c) EMVS MAP", DIV_CMAP, -vmax_map, vmax_map),
         (std_mat, "(d) Posterior SD", SEQ_CMAP, 0, float(std_mat.max()))]
fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.4), constrained_layout=True)
for ax, (mat, title, cmap, lo, hi) in zip(axes.ravel(), specs):
    mesh = ax.pcolormesh(mat, cmap=cmap, vmin=lo, vmax=hi, rasterized=False, edgecolors="face", linewidth=0.3)
    ax.set_xlim(0, G); ax.set_ylim(G, 0); ax.set_aspect("equal")
    ax.set_xticks(np.arange(G) + 0.5); ax.set_xticklabels(range(G))
    ax.set_yticks(np.arange(G) + 0.5); ax.set_yticklabels(range(G))
    ax.set_title(title); ax.set_xlabel("Neighbor $m$"); ax.set_ylabel("Focal $k$")
    for s in ax.spines.values(): s.set_visible(True)
    ax.tick_params(length=0)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cb.solids.set_rasterized(False); cb.solids.set_edgecolor("face")
    cb.ax.tick_params(labelsize=TICK_SIZE)
    for i in range(G):
        for j in range(G):
            v = mat[i, j]; frac = (v - lo) / (hi - lo)
            dark = frac > 0.65 if cmap is SEQ_CMAP else (frac > 0.85 or frac < 0.15)
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=TICK_SIZE - 2 if False else TICK_SIZE,
                    color="white" if dark else "black")
save(fig, "posterior_couplings_heatmap_v50", "mcmc_ensemble")
print("tau", tau)
