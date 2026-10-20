"""Village 50 MCMC posterior and ensemble decision tree (logic from mcmc_ensemble.ipynb); saves to ./out.

From the repo root:
    python experiments/villages/ensemble_v50/run_v50.py mcmc       # EMVS + MCMC (minutes)
    python experiments/villages/ensemble_v50/run_v50.py ensemble   # after mcmc; trains 20 CQL agents (~20 min+)
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

STAGE = sys.argv[1] if len(sys.argv) > 1 else "mcmc"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

import d3rlpy, logging
logging.getLogger("d3rlpy").setLevel(logging.WARNING)
from qising.baselines import QIsingPolicy
from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
from qising.ising.group_ising import fit_group_ising_emvs
from qising.utils import build_village_envdata

ADJ_JSON = os.path.join(ROOT, "data", "villages", "adjacency_all.json")
DEV_VILLAGE = "50"
BETA_LIST  = [0.01, 0.5, 0.05, 0.07, 0.06, 0.02, 0.01, 0.4, 0.1, 0.3]
DELTA_LIST = [0.5,  0.9, 0.9,  0.6,  0.5,  0.50, 0.7,  0.6, 0.5, 0.8]
COMM_METHOD, MIN_COMM_SIZE = "edge_betweenness", 10
T_TRAIN, N_TRAIN_EPS, DATA_SEED = 500, 1, 42
T_TEST, SKIP_PROB, INIT_RATE, TEST_INIT_RATE = 25, 0, 0.0, 0.0
DISCOUNT, ALPHA_CQL = 0.8, 0.1
V0_E, V1_E, C_E, SIGMA_E = 0.01, 10.0, 1.0, 10.0
V0_MCMC, V1_MCMC, C_PRIOR, SIGMA_SQ_MCMC = 0.01, 10.0, 1.0, 10.0
N_MCMC_DRAWS, N_MCMC_TUNE, N_MCMC_CHAINS, MCMC_SEED, MAX_MCMC_ROWS = 200, 300, 1, 42, 8_000
N_ENSEMBLE, ENSEMBLE_SEED = 20, 1234

adj_all = json.load(open(ADJ_JSON))
A = np.array(adj_all[DEV_VILLAGE], dtype=float); np.fill_diagonal(A, 0)
dvd = build_village_envdata(A, BETA_LIST, DELTA_LIST, comm_method=COMM_METHOD, min_comm_size=MIN_COMM_SIZE)
G = dvd["n_bins"]
print(f"village {DEV_VILLAGE}: N={dvd['N']} G={G} bins={dvd['bin_counts'].tolist()}")

dev_trajs = []
for ep in range(N_TRAIN_EPS):
    env = EdgeBetCommunitySISEnv(dvd["adj"], dvd["beta_vec"], dvd["delta_vec"], dvd["block_labels"],
                                 T=T_TRAIN, seed=DATA_SEED + ep, init_rate=INIT_RATE)
    dev_trajs.append(env.collect_behavioral_episode(skip_prob=SKIP_PROB, verbose=False))
X = np.stack([t["adopted"][:-1].astype(float) for t in dev_trajs])
Y = np.stack([t["adopted"][1:].astype(float) for t in dev_trajs])
TR = np.stack([t["seeded_nodes"][:-1] for t in dev_trajs])

emvs = fit_group_ising_emvs(X, Y, TR, dvd["adj"], dvd["block_labels"],
                            v0=V0_E, v1=V1_E, c=C_E, sigma_sq=SIGMA_E, verbose=False)
ext = emvs["extracted"]
map_params = {gi: emvs["group_params"][gi] for gi in range(G)}
n_base = 4

if STAGE == "mcmc":
    import pymc as pm
    g_idx, nl, ng = emvs["g_idx"], emvs["neighbor_lists"], emvs["neigh_g_idx"]
    n_feat = n_base + G
    Xm = [[] for _ in range(G)]; ym = [[] for _ in range(G)]
    for traj in dev_trajs:
        Xa = traj["adopted"][:-1].astype(float); Ya = traj["adopted"][1:].astype(float)
        seeded = traj["seeded_nodes"][:-1]; T_ep, N_v = Xa.shape
        ti = np.zeros((T_ep, N_v), dtype=bool)
        for t in range(T_ep):
            idx = int(seeded[t])
            if 0 <= idx < N_v: ti[t, idx] = True
        for j in range(N_v):
            gi = int(g_idx[j]); neigh = nl[j]; ngj = ng[j]; d = len(neigh)
            for t in range(T_ep):
                row = np.zeros(n_feat); row[0] = 1.0; row[1] = Xa[t, j]; row[2] = float(ti[t, j])
                if d > 0:
                    row[3] = ti[t, neigh].sum(); ns = Xa[t, neigh]
                    for gm in range(G): row[n_base + gm] = ns[ngj == gm].sum()
                Xm[gi].append(row); ym[gi].append(int(Ya[t, j]))
    Xm = [np.array(x) for x in Xm]; ym = [np.array(y) for y in ym]
    pi_coup = min(C_PRIOR / G, 0.99)
    traces = {}
    for gi in range(G):
        Xg, yg = Xm[gi], ym[gi]
        if len(np.unique(yg)) < 2:
            traces[gi] = None; print(f"group {gi}: single class, EMVS fallback"); continue
        if len(yg) > MAX_MCMC_ROWS:
            sub = np.random.default_rng(MCMC_SEED + gi + 100).choice(len(yg), size=MAX_MCMC_ROWS, replace=False)
            Xg, yg = Xg[sub], yg[sub]
        t0 = time.perf_counter()
        with pm.Model():
            baseline = pm.Normal("baseline", mu=0.0, sigma=np.sqrt(SIGMA_SQ_MCMC), shape=n_base)
            z = pm.NormalMixture("z_coup", w=np.array([1.0 - pi_coup, pi_coup]), mu=np.zeros(2),
                                 sigma=np.array([np.sqrt(V0_MCMC), np.sqrt(V1_MCMC)]), shape=G)
            theta = pm.math.concatenate([baseline, z])
            pm.Bernoulli("y_obs", logit_p=pm.math.dot(Xg, theta), observed=yg.astype(int))
            traces[gi] = pm.sample(N_MCMC_DRAWS, tune=N_MCMC_TUNE, chains=N_MCMC_CHAINS,
                                   random_seed=MCMC_SEED + gi, progressbar=False, target_accept=0.85)
        print(f"group {gi} ({len(yg)} rows) done {time.perf_counter()-t0:.1f}s", flush=True)
    n_total = N_MCMC_DRAWS * N_MCMC_CHAINS
    draws = {}
    for gi in range(G):
        if traces[gi] is None:
            draws[gi] = np.tile(map_params[gi], (n_total, 1)); continue
        b = traces[gi].posterior["baseline"].values.reshape(-1, n_base)
        c = traces[gi].posterior["z_coup"].values.reshape(-1, G)
        draws[gi] = np.concatenate([b, c], axis=1)
    np.savez_compressed(os.path.join(OUT, "v50_mcmc_draws.npz"), **{f"g{gi}": draws[gi] for gi in range(G)})
    np.savez_compressed(os.path.join(OUT, "v50_emvs.npz"),
                        coupling_matrix_mean=ext["coupling_matrix_mean"],
                        inclusion_probs=emvs["inclusion_probs"],
                        **{f"map_g{gi}": map_params[gi] for gi in range(G)})
    json.dump({"G": G, "bin_counts": dvd["bin_counts"].tolist(), "n_base": n_base,
               "n_draws": n_total, "V0": V0_MCMC, "V1": V1_MCMC},
              open(os.path.join(OUT, "v50_meta.json"), "w"), indent=1)
    print("saved draws, EMVS arrays, meta ->", OUT)

elif STAGE == "ensemble":
    draws = {gi: v for gi, v in enumerate(
        [np.load(os.path.join(OUT, "v50_mcmc_draws.npz"))[f"g{gi}"] for gi in range(G)])}
    kw = dict(v0=V0_E, v1=V1_E, c=C_E, sigma_sq=SIGMA_E, discount=DISCOUNT, alpha_cql=ALPHA_CQL,
              v0_mcmc=V0_MCMC, v1_mcmc=V1_MCMC, c_prior=C_PRIOR, sigma_sq_mcmc=SIGMA_SQ_MCMC,
              n_mcmc_draws=N_MCMC_DRAWS, n_mcmc_tune=N_MCMC_TUNE, n_mcmc_chains=N_MCMC_CHAINS,
              mcmc_seed=MCMC_SEED, max_mcmc_rows=MAX_MCMC_ROWS, n_ensemble=N_ENSEMBLE, ensemble_seed=ENSEMBLE_SEED)
    t0 = time.perf_counter()
    pol = QIsingPolicy(ensemble=True, ising_augmentation=True, label="qi_ens", verbose=True, **kw).fit(
        dev_trajs, dvd["adj"], dvd["block_labels"], _emvs_result=emvs, _mcmc_draws=draws)
    print(f"ensemble trained in {time.perf_counter()-t0:.0f}s")
    models = pol._ensemble_models
    def make_te(seed):
        return EdgeBetCommunitySISEnv(dvd["adj"], dvd["beta_vec"], dvd["delta_vec"], dvd["block_labels"],
                                      T=T_TEST, seed=seed, init_rate=TEST_INIT_RATE)
    te = make_te(2026); obs = te.reset(); done = False; step = 0; recs = []
    all_b = np.arange(G, dtype=np.int32)
    while not done and step < T_TEST:
        avl = set(te.get_available_actions()); votes = np.zeros(G, dtype=int)
        for _, dp, mdl, so in models:
            ob = so(obs["adopted"]).reshape(1, -1)
            qv = mdl.predict_value(np.repeat(ob, G, axis=0), all_b)
            for b in np.argsort(qv)[::-1]:
                if int(b) in avl:
                    votes[int(b)] += 1; break
        maj = int(np.argmax(votes))
        recs.append({"t": step, "majority": maj, "vote_counts": votes.tolist(),
                     "adoption_rate": float(te.adopted.mean())})
        obs, _, done, _ = te.step(maj); step += 1
    json.dump({"village": DEV_VILLAGE, "n_ensemble": N_ENSEMBLE, "tree_seed": 2026, "records": recs},
              open(os.path.join(OUT, "v50_tree_records.json"), "w"), indent=1)
    print("saved tree records ->", os.path.join(OUT, "v50_tree_records.json"))
