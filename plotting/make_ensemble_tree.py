"""Ensemble majority-vote tree for village 50 at 5.5 in column width: communities x time step with the majority path and offset alternatives, plus an adoption-rate panel."""
import json, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import *

rec_path = EXP / "villages/ensemble_v50/out/v50_tree_records.json"
if not rec_path.exists():
    print(f"skipped: {rec_path.relative_to(ROOT)} does not exist"); raise SystemExit(0)
apply_style()
d = json.load(open(rec_path)); meta = json.load(open(EXP / "villages/ensemble_v50/out/v50_meta.json"))
R, NE = d["records"], int(d["n_ensemble"])
nb = len(R[0]["vote_counts"]); T = len(R); cnt = meta["bin_counts"]
MAJ, ALT = COLORS["q_ising"], "#4095C5"
OFF = 0.12                                   # alternatives sit 0.12 steps to the right
D_MAJ = lambda v: 0.14 + 0.02 * v / NE       # diameters in inches (0.14-0.16)
D_ALT = lambda v: 0.12 + 0.01 * v / NE       # 0.12-0.13
DIGIT = 7
sz = lambda dd: (dd * 72) ** 2

fig, (ax, axb) = plt.subplots(2, 1, figsize=(FULL_WIDTH, 3.3), sharex=True, constrained_layout=True,
                              gridspec_kw={"height_ratios": [3.3, 1]})
fig.get_layout_engine().set(h_pad=0.04, w_pad=0.02)
ax.set_xlim(-0.55, T - 0.4); ax.set_ylim(-0.6, nb - 0.4)
ax.set_yticks(range(nb)); ax.set_yticklabels([f"{g} ({cnt[g]})" for g in range(nb)])
ax.set_ylabel("Community"); grid(ax)
ax.tick_params(axis="x", length=0)
for r in R:
    t, maj, vc = r["t"], r["majority"], r["vote_counts"]
    if t > 0:
        ax.plot([t - 1, t], [R[t - 1]["majority"], maj], "-", color=MAJ, lw=1.2, alpha=0.7, zorder=2)
    for alt in range(nb):
        if alt != maj and vc[alt] > 0:
            ax.plot([t, t + OFF], [maj, alt], ":", color=ALT, lw=0.9, zorder=2)
            ax.scatter(t + OFF, alt, s=sz(D_ALT(vc[alt])), color=ALT, zorder=3, linewidths=0)
            ax.text(t + OFF, alt, str(vc[alt]), ha="center", va="center", fontsize=DIGIT, color="white", zorder=4)
    ax.scatter(t, maj, s=sz(D_MAJ(vc[maj])), color=MAJ, zorder=3.5, linewidths=0)
    ax.text(t, maj, str(vc[maj]), ha="center", va="center", fontsize=DIGIT, color="white", zorder=5)
axb.plot([r["t"] for r in R], [r["adoption_rate"] for r in R], color="black", ls="--", lw=0.9)
axb.set_ylim(0, 0.2); axb.set_yticks([0, 0.1, 0.2]); axb.set_ylabel("Adoption")
axb.set_xticks(range(0, T, 5)); axb.set_xlabel("Time step"); grid(axb)
fig.legend(handles=[Patch(color=MAJ, label="Majority vote"), Patch(color=ALT, label="Alternative")],
           loc="outside upper center", ncol=2, columnspacing=1.5, handlelength=1.5)
# overlap check: circles of different steps in the same row must not touch
ax_w_in = ax.get_position().width * FULL_WIDTH; step = ax_w_in / (ax.get_xlim()[1] - ax.get_xlim()[0])
cs = []
for r in R:
    for g in range(nb):
        v = r["vote_counts"][g]
        if v > 0:
            m = g == r["majority"]
            cs.append((g, r["t"] + (0 if m else OFF), (D_MAJ(v) if m else D_ALT(v)) / 2))
bad = [(a, b) for i, a in enumerate(cs) for b in cs[i + 1:] if a[0] == b[0] and abs(a[1] - b[1]) * step < a[2] + b[2]]
print(f"step {step:.3f} in; overlapping circle pairs: {len(bad)}")
save(fig, "ensemble_decision_tree_v50", "mcmc_ensemble")
