"""Fig. 3 data (logic from policy_eval.ipynb): layouts, communities and Q-Ising seeding sequences for villages 59 and 71; the seed is set and saved to ./out."""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import networkx as nx
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import torch, d3rlpy, logging
logging.getLogger("d3rlpy").setLevel(logging.WARNING)
from qising.baselines import QIsingPolicy
from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
from qising.utils import build_village_envdata

VILLAGES = ["59", "71"]
TRAIN_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ADJ_JSON = os.path.join(ROOT, "data", "villages", "adjacency_all.json")
BETA_LIST  = [0.01, 0.5, 0.05, 0.07,  0.06, 0.02,  0.01,  0.4,  0.1, 0.3]
DELTA_LIST = [0.5, 0.9, 0.9, 0.6, 0.5,  0.50,  0.7,  0.6,  0.5, 0.8]
T_TRAIN, N_TRAIN_EPS, DATA_SEED, INIT_RATE, SKIP_PROB = 500, 1, 42, 0.0, 0
T_TEST, K_SEEDS, TEST_INIT_RATE = 25, 25, 0.0
qi_kwargs = dict(discount=0.8, alpha_cql=0.1)

def community_layout(G, block_labels, n_bins, seed=42):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    centers = {b: (np.cos(a), np.sin(a)) for b, a in enumerate(angles)}
    init_pos = {}
    for node in G.nodes():
        cx, cy = centers[int(block_labels[node])]
        init_pos[node] = (cx + rng.normal(0, 0.25), cy + rng.normal(0, 0.25))
    return nx.spring_layout(G, pos=init_pos, seed=seed, k=1.2 / np.sqrt(len(G)), iterations=60)

def record_choices(policy_fn, vd, seed=99):
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

adj_all = json.load(open(ADJ_JSON))
res = pd.read_csv(os.path.join(ROOT, "experiments", "villages", "village_policy_results.csv"))
mod = pd.read_csv(os.path.join(ROOT, "experiments", "villages", "village_modularity.csv"))
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
    G = nx.from_numpy_array(adj)
    pos = community_layout(G, bl, vd["n_bins"])

    torch.manual_seed(TRAIN_SEED); np.random.seed(TRAIN_SEED); d3rlpy.seed(TRAIN_SEED)
    trajs = []
    for ep in range(N_TRAIN_EPS):
        env = EdgeBetCommunitySISEnv(adj, vd["beta_vec"], vd["delta_vec"], bl, T=T_TRAIN,
                                     seed=DATA_SEED + ep, init_rate=INIT_RATE)
        trajs.append(env.collect_behavioral_episode(skip_prob=SKIP_PROB))
    pol = QIsingPolicy(ising_augmentation=True, label=f"showcase_{v}", **qi_kwargs).fit(trajs, adj, bl)
    choices = record_choices(pol, vd, seed=99)
    print(f"village {v}: N={vd['N']} K={vd['n_bins']} Q={Q:.3f} imp={imp:+.1f}% choices={[c[1] for c in choices]}", flush=True)
    summary[v] = {"N": int(vd["N"]), "K": int(vd["n_bins"]), "Q_mod": Q, "imp_pct_vs_degree_bin": imp,
                  "avg_degree": float(deg.mean()), "bin_counts": vd["bin_counts"].tolist(),
                  "block_labels": bl.tolist(), "beta_by_bin": [float(vd["beta_vec"][bl == b][0]) for b in range(vd["n_bins"])],
                  "choices": choices, "train_seed": TRAIN_SEED}
    np.savez_compressed(os.path.join(OUT, f"fig3_v{v}_layout.npz"), pos=np.array([pos[i] for i in range(vd["N"])]), adj=adj.astype(np.int8))
json.dump(summary, open(os.path.join(OUT, f"fig3_summary_seed{TRAIN_SEED}.json"), "w"), indent=1)
