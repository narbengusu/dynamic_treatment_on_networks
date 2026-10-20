import numpy as np, pandas as pd, matplotlib.pyplot as plt
from style import *

apply_style()
d = pd.read_csv(EXP / "villages/village_policy_results.csv", index_col=0)
imp = ((d.q_ising_mean - d.degree_bin_mean) / (d.degree_bin_mean.abs() + 1e-9) * 100).sort_values().values
col = np.where(imp < 0, VERMILLION, np.where(imp <= 5, "#808080", BLUE))
print(f"villages={len(imp)}  >5%: {(imp > 5).sum()}  <0: {(imp < 0).sum()}  "
      f"max abs diff to csv Improvement: {np.abs(np.sort(d.Improvement.values) - imp).max():.2e}")
fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6), constrained_layout=True)
ax.bar(np.arange(len(imp)), imp, color=col, edgecolor="black", linewidth=0.5, zorder=3)
for y in (5, -5):
    ax.axhline(y, color=VERMILLION, ls="--", lw=1.0, zorder=2)
ax.axhline(0, color="black", lw=0.8, zorder=4)
ax.set_yscale("symlog", linthresh=5, linscale=0.6)
ticks = [-20, -5, 0, 5, 20, 50]
ax.set_yticks(ticks); ax.set_yticklabels([f"{t}%" for t in ticks])
ax.set_xlabel("Villages, sorted"); ax.set_ylabel("Improvement over degree-bin")
grid(ax)
save(fig, "q_ising_improvement_hist")
