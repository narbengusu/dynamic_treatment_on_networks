import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from style import *

apply_style()
O = EXP / "villages/emvs_roc/out"
m = pd.read_csv(O / "emvs_roc_macro.csv")
z = np.load(O / "emvs_roc_pernode.npz")
auc = z["auc"].mean()
print(f"mean AUC {auc:.4f} (json {json.load(open(O / 'emvs_auc_summary.json'))['mean_auc']:.4f}), nodes {len(z['auc'])}")
fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.6), constrained_layout=True)
ax.plot(m.fpr, m.mean_tpr, color=BLUE, lw=1.5, label="Mean ROC")
ax.fill_between(m.fpr, np.maximum(m.mean_tpr - m.std_tpr, 0), np.minimum(m.mean_tpr + m.std_tpr, 1),
                color=BAND, alpha=0.8, lw=0, label="$\\pm1$ SD over nodes")
ax.plot([0, 1], [0, 1], color=COLORS["random"], ls=":", label="Chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, columnspacing=1.0, handlelength=1.6)
grid(ax)
save(fig, "emvs_roc_auc")
