"""Fig. 3, train-once version: trains one Q-Ising agent per village (59, 71; training seed 0, settings as in run_fig3.py),
saves it to out/models/ (v{V}_cql.d3, v{V}_state.pkl), and runs test episodes with seeds 0..99.
Writes out/fig3_once_summary.json. The displayed episode has the most distinct communities seeded (ties: lowest seed).
Reload a saved agent with reload_policy() below (see README_once.md)."""
import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
MODELS = os.path.join(OUT, "models"); os.makedirs(MODELS, exist_ok=True)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import torch, d3rlpy, logging
logging.getLogger("d3rlpy").setLevel(logging.WARNING)
from qising.baselines import QIsingPolicy
from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
from qising.utils import build_village_envdata

VILLAGES = ["59", "71"]
TRAIN_SEED = 0
TEST_SEEDS = list(range(100))
ADJ_JSON = os.path.join(ROOT, "data", "villages", "adjacency_all.json")
BETA_LIST  = [0.01, 0.5, 0.05, 0.07,  0.06, 0.02,  0.01,  0.4,  0.1, 0.3]
DELTA_LIST = [0.5, 0.9, 0.9, 0.6, 0.5,  0.50,  0.7,  0.6,  0.5, 0.8]
T_TRAIN, N_TRAIN_EPS, DATA_SEED, INIT_RATE, SKIP_PROB = 500, 1, 42, 0.0, 0
T_TEST, K_SEEDS, TEST_INIT_RATE = 25, 25, 0.0
qi_kwargs = dict(discount=0.8, alpha_cql=0.1)

def record_choices(policy_fn, vd, seed=99):  # identical to run_fig3.py
    env = EdgeBetCommunitySISEnv(vd["adj"], vd["beta_vec"], vd["delta_vec"], vd["block_labels"],
                                 T=T_TEST, seed=seed, init_rate=TEST_INIT_RATE)
    obs = env.reset(); done = False; used = 0; step = 0; choices = []
    while not done:
        a = policy_fn(obs, env) if used < K_SEEDS else None
        obs, _, done, _ = env.step(a)
        if a is not None and 0 <= int(a) < env.n_bins:
            choices.append((step, int(a))); used += 1
        step += 1
    return choices

def reload_policy(v):
    st = pickle.load(open(os.path.join(MODELS, f"v{v}_state.pkl"), "rb"))
    pol = QIsingPolicy(ising_augmentation=True, **st["qi_kwargs"])
    pol._emvs_result, pol._bins, pol._n_bins = st["emvs_result"], st["bins"], st["n_bins"]
    pol._model = d3rlpy.load_learnable(os.path.join(MODELS, f"v{v}_cql.d3"), device="cpu")
    return pol

def stats(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}

if __name__ == "__main__":
    adj_all = json.load(open(ADJ_JSON))
    res = pd.read_csv(os.path.join(ROOT, "experiments", "villages", "village_policy_results.csv"))
    summary = {}
    for v in VILLAGES:
        A = np.array(adj_all[v], dtype=float); np.fill_diagonal(A, 0)
        vd = build_village_envdata(A, BETA_LIST, DELTA_LIST, comm_method="edge_betweenness", min_comm_size=10)
        bl, adj = vd["block_labels"], vd["adj"]
        deg = adj.sum(1); m = adj.sum() / 2
        same = (bl[:, None] == bl[None, :]).astype(float)
        Q = float(((adj - np.outer(deg, deg) / (2 * m + 1e-9)) * same).sum() / (2 * m + 1e-9))
        row = res[(res["N"] == vd["N"]) & (res["n_bins"] == vd["n_bins"])]
        imp = float(row["Improvement"].iloc[0]) if len(row) == 1 else float("nan")

        torch.manual_seed(TRAIN_SEED); np.random.seed(TRAIN_SEED); d3rlpy.seed(TRAIN_SEED)
        trajs = []
        for ep in range(N_TRAIN_EPS):
            env = EdgeBetCommunitySISEnv(adj, vd["beta_vec"], vd["delta_vec"], bl, T=T_TRAIN,
                                         seed=DATA_SEED + ep, init_rate=INIT_RATE)
            trajs.append(env.collect_behavioral_episode(skip_prob=SKIP_PROB))
        pol = QIsingPolicy(ising_augmentation=True, label=f"once_{v}", **qi_kwargs).fit(trajs, adj, bl)

        # save model and everything needed to use it
        pol._model.save(os.path.join(MODELS, f"v{v}_cql.d3"))
        pickle.dump(dict(emvs_result=pol._emvs_result, bins=pol._bins, n_bins=pol._n_bins, block_labels=bl,
                         adj=adj, beta_vec=vd["beta_vec"], delta_vec=vd["delta_vec"],
                         qi_kwargs=qi_kwargs, train_seed=TRAIN_SEED),
                    open(os.path.join(MODELS, f"v{v}_state.pkl"), "wb"))

        eps = {}
        for ts in TEST_SEEDS:
            eps[ts] = record_choices(pol, vd, seed=ts)
        comms = {ts: [c[1] for c in ch] for ts, ch in eps.items()}
        n_distinct = {ts: len(set(c)) for ts, c in comms.items()}
        n_switch = {ts: int(sum(a != b for a, b in zip(c[:-1], c[1:]))) for ts, c in comms.items()}
        best = max(n_distinct.values())
        sel = min(ts for ts in TEST_SEEDS if n_distinct[ts] == best)
        print(f"village {v}: N={vd['N']} K={vd['n_bins']} Q={Q:.3f} imp={imp:+.1f}% selected seed {sel} "
              f"(distinct={best}) choices={comms[sel]}", flush=True)
        summary[v] = {"N": int(vd["N"]), "K": int(vd["n_bins"]), "Q_mod": Q, "imp_pct_vs_degree_bin": imp,
                      "avg_degree": float(deg.mean()), "bin_counts": vd["bin_counts"].tolist(),
                      "block_labels": bl.tolist(), "train_seed": TRAIN_SEED, "test_seeds": TEST_SEEDS,
                      "choices_by_test_seed": {str(k): val for k, val in eps.items()},
                      "n_distinct": {str(k): val for k, val in n_distinct.items()},
                      "n_switches": {str(k): val for k, val in n_switch.items()},
                      "distinct_stats": stats(list(n_distinct.values())),
                      "switch_stats": stats(list(n_switch.values())),
                      "selection_rule": "max number of distinct communities seeded among test seeds 0..99; ties -> lowest seed",
                      "selected_seed": int(sel), "choices": eps[sel]}
    json.dump(summary, open(os.path.join(OUT, "fig3_once_summary.json"), "w"), indent=1)
