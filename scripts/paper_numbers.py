"""Recompute the numbers quoted in the paper from the saved results (no training).

    python scripts/paper_numbers.py

Prints: the covariate-experiment table (E1), the misspecification table (E2), the EMVS AUC,
the uncertainty-calibration statistics for the small network (E3) and the villages (E4).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def pooled_se(means, ses, n_eval):
    """Standard error of the mean over all episodes, from per-replicate means and per-replicate SEs."""
    within = np.mean((np.asarray(ses) ** 2) * n_eval)
    between = np.var(np.asarray(means))
    return float(np.sqrt((within + between) / (len(means) * n_eval)))


def e1():
    D = EXP / "covariate"
    res = pd.read_csv(D / "cluster_sweep_all_results.csv")
    base = pd.read_csv(D / "cluster_sweep_all_baselines.csv")
    T_TEST, N_EVAL = 20, 50
    print("E1: reward per node-period, mean (SE over replicate x episode), K as listed")
    rows = []
    for (N, K, T), g in res.groupby(["N", "K", "T_train"]):
        norm = N * T_TEST
        row = dict(N=N, K=K, T_train=T)
        for key in ("plain", "all", "ising"):
            m = g[f"reward_{key}_mean"].values / norm
            s = g[f"reward_{key}_se"].values / norm
            row[key] = f"{m.mean():.4f} ({pooled_se(m, s, N_EVAL):.4f})"
        b = base[(base.N == N) & (base.K == K)]
        top = b[b.policy != "oracle"].sort_values("mean_deploy_reward").iloc[-1]
        orc = b[b.policy == "oracle"].iloc[0]
        row["best_baseline"] = f"{top.mean_deploy_reward / norm:.4f} ({top.policy})"
        row["oracle"] = f"{orc.mean_deploy_reward / norm:.4f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def e2():
    D = EXP / "misspecification"
    res = pd.read_csv(D / "qising_sweep_all_results.csv")
    base = pd.read_csv(D / "qising_sweep_baselines.csv")
    fit = pd.read_csv(D / "qising_sweep_ising_fit.csv")
    print("\nE2: reward per node-period (divided by N*T_test = 4000)")
    print(base.assign(reward=base.mean_reward / 4000, se=base.se_reward / 4000)[["policy", "reward", "se"]].round(4).to_string(index=False))
    rows = []
    for (T, pol), g in res.groupby(["T_train", "policy"]):
        m = g.mean_reward.values / 4000
        rows.append(dict(T_train=T, policy=pol, reward="%.4f (%.4f)" % (m.mean(), pooled_se(m, g.se_reward.values / 4000, 50))))
    print(pd.DataFrame(rows).pivot(index="T_train", columns="policy", values="reward").to_string())
    print("in-sample Ising AUC by T_train (mean, min, max over replicates):")
    print(fit.groupby("T_train").auc.agg(["mean", "min", "max"]).round(3).to_string())


def roc():
    s = pd.read_json(EXP / "villages" / "emvs_roc" / "out" / "emvs_auc_summary.json", typ="series")
    print("\nEMVS pooled AUC: mean %.4f, sd %.4f, nodes %d" % (s["mean_auc"], s["std_auc"], s["n_nodes"]))


def e3():
    d = pd.read_csv(EXP / "uq_small_network" / "uq_calibration_sis_decisions.csv", skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    sub = d.dropna(subset=["entropy"])
    r, p = stats.pearsonr(sub.entropy, sub.true_gap)
    diff = sub[sub.majority_action != sub.point_action]
    print("\nE3: n=%d decisions with defined entropy, Pearson r=%.4f (p=%.2g)" % (len(sub), r, p))
    print("    ensemble differs from point estimate at %.1f%% of decisions; better in %.1f%% of those; mean gain %.3f; "
          "mean gap there %.3f vs %.3f overall" % (
              100 * len(diff) / len(sub), 100 * (diff.majority_value > diff.point_value).mean(),
              (diff.majority_value - diff.point_value).mean(), diff.true_gap.mean(), sub.true_gap.mean()))


def e4():
    d = pd.read_csv(EXP / "villages" / "uq_calibration_villages_all_decisions.csv", skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["pct_gap"] = 100 * d.true_gap / d.oracle_value        # percentage value gap
    sub = d.dropna(subset=["entropy"])
    print("\nE4: %d pooled decisions, %d with a defined entropy; percentage gap = 100 * true_gap / oracle_value" % (len(d), len(sub)))
    for name, s in (("all", sub), ("entropy > 0", sub[sub.entropy > 0]), ("entropy >= 0.2", sub[sub.entropy >= 0.2])):
        r, p = stats.pearsonr(s.entropy, s.pct_gap)
        print("    %-15s n=%6d  r=%.4f  p=%.2g" % (name, len(s), r, p))


def modularity():
    """Modularity of each village (edge-betweenness communities) and its rank correlation with the improvement over degree-bin."""
    import sys
    sys.path.insert(0, str(ROOT / "plotting"))
    from villages_common import load_villages
    rows = []
    for v, vd in load_villages().items():
        if vd["n_bins"] < 2:
            continue
        adj, bl = vd["adj"], vd["block_labels"]
        deg, m = adj.sum(1), adj.sum() / 2
        same = (bl[:, None] == bl[None, :]).astype(float)
        q = ((adj - np.outer(deg, deg) / (2 * m + 1e-9)) * same).sum() / (2 * m + 1e-9)
        rows.append(dict(village=int(v), N=int(vd["N"]), n_bins=int(vd["n_bins"]), Q=float(q)))
    q = pd.DataFrame(rows)
    res = pd.read_csv(EXP / "villages" / "village_policy_results.csv", index_col=0)
    j = q.merge(res, on=["N", "n_bins"])
    rho, p = stats.spearmanr(j.Q, j.Improvement)
    print("\nModularity vs improvement over degree-bin (%d villages): Spearman %.3f (p=%.2g), Pearson %.3f" % (
        len(j), rho, p, stats.pearsonr(j.Q, j.Improvement)[0]))
    print("    village 59: Q=%.3f   village 71: Q=%.3f" % tuple(q.set_index("village").loc[[59, 71], "Q"]))


if __name__ == "__main__":
    e1(); e2(); roc(); e3(); e4()
    if (ROOT / "data" / "villages" / "adjacency_all.json").exists():
        modularity()
    else:
        print("\nModularity: skipped (needs data/villages/adjacency_all.json, see README)")
