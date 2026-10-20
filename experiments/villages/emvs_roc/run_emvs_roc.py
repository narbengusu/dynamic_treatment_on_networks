"""Re-run the EMVS ROC/AUC pass of policy_eval.ipynb (cell 20) and save per-node TPR curves, AUCs and village metadata to ./out."""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
from qising.ising.group_ising import fit_group_ising_emvs, predict_group_ising_emvs
from qising.utils import build_village_envdata
from sklearn.metrics import roc_auc_score, roc_curve

ADJ_JSON = os.path.join(ROOT, "data", "villages", "adjacency_all.json")
BETA_LIST  = [0.01, 0.5, 0.05, 0.07,  0.06, 0.02,  0.01,  0.4,  0.1, 0.3]
DELTA_LIST = [0.5, 0.9, 0.9, 0.6, 0.5,  0.50,  0.7,  0.6,  0.5, 0.8]
COMM_METHOD, MIN_COMM_SIZE = "edge_betweenness", 10
T_TRAIN, DATA_SEED, INIT_RATE, SKIP_PROB = 500, 42, 0.0, 0
V0_E, V1_E, C_E, SIGMA_E = 0.01, 10.0, 1.0, 10.0

adj_all = json.load(open(ADJ_JSON))
VILL_IDS = sorted(adj_all.keys(), key=lambda x: int(x))
village_data = {}
for v in VILL_IDS:
    A = np.array(adj_all[v], dtype=float); np.fill_diagonal(A, 0)
    try:
        village_data[v] = build_village_envdata(A, BETA_LIST, DELTA_LIST,
                                                comm_method=COMM_METHOD, min_comm_size=MIN_COMM_SIZE)
    except Exception as e:
        print("failed", v, e)
VIABLE = [v for v in village_data if village_data[v]["n_bins"] >= 2]
print("viable villages:", len(VIABLE))

fpr_grid = np.linspace(0, 1, 300)
tprs, aucs, meta = [], [], []
t0 = time.perf_counter()
for v in VIABLE:
    vd = village_data[v]
    env = EdgeBetCommunitySISEnv(vd["adj"], vd["beta_vec"], vd["delta_vec"], vd["block_labels"],
                                 T=T_TRAIN, seed=DATA_SEED, init_rate=INIT_RATE)
    tr = env.collect_behavioral_episode(skip_prob=SKIP_PROB)
    X = tr["adopted"][:-1].astype(float)[np.newaxis]
    y = tr["adopted"][1:].astype(float)[np.newaxis]
    t = tr["seeded_nodes"][:-1][np.newaxis]
    try:
        res = fit_group_ising_emvs(X, y, t, vd["adj"], vd["block_labels"],
                                   v0=V0_E, v1=V1_E, c=C_E, sigma_sq=SIGMA_E, verbose=False)
    except Exception as e:
        print("EMVS failed", v, e); continue
    Xf, yf = X[0], y[0]
    p = np.array([predict_group_ising_emvs(Xf[k], action=-1, result=res, A=None) for k in range(Xf.shape[0])])
    for j in range(vd["N"]):
        yj = yf[:, j]
        if yj.std() == 0:
            continue
        try:
            a = roc_auc_score(yj, p[:, j])
            f, tp, _ = roc_curve(yj, p[:, j])
            tprs.append(np.interp(fpr_grid, f, tp)); aucs.append(a); meta.append((int(v), j))
        except Exception:
            pass
print(f"done {time.perf_counter()-t0:.1f}s nodes={len(aucs)}")
aucs = np.array(aucs); tprs = np.array(tprs)
print(f"AUC mean={aucs.mean():.4f} std={aucs.std():.4f} min={aucs.min():.4f} max={aucs.max():.4f}")
print("paper/notebook: mean 0.7621 sd 0.1300 min 0.2008 max 0.9995, nodes 8287")
np.savez_compressed(os.path.join(OUT, "emvs_roc_pernode.npz"), fpr=fpr_grid, tpr=tprs, auc=aucs,
                    village=np.array([m[0] for m in meta]), node=np.array([m[1] for m in meta]))
pd.DataFrame({"fpr": fpr_grid, "mean_tpr": tprs.mean(0), "std_tpr": tprs.std(0)}).to_csv(
    os.path.join(OUT, "emvs_roc_macro.csv"), index=False)
json.dump({"mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()), "min_auc": float(aucs.min()),
           "max_auc": float(aucs.max()), "n_nodes": int(len(aucs))},
          open(os.path.join(OUT, "emvs_auc_summary.json"), "w"), indent=1)
