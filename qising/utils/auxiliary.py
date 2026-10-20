import os
import shutil
import tempfile
import glob as _glob
import numpy as np
import igraph as ig


# Community detection
def detect_communities_igraph(adj, comm_method="edge_betweenness", min_comm_size=10):
    """Communities via igraph ('edge_betweenness' or 'louvain'); bin 0 is the largest, bins smaller than min_comm_size merge into bin 0. Returns (block_labels, n_communities)."""
    g = ig.Graph.Adjacency((adj > 0).tolist(), mode="undirected")
    if comm_method == "louvain":
        communities = g.community_multilevel(weights=None)
    elif comm_method == "edge_betweenness":
        communities = g.community_edge_betweenness(directed=False).as_clustering()
    else:
        raise ValueError(f"Unknown comm_method: {comm_method!r}")

    raw_membership = np.array(communities.membership, dtype=int)
    unique_comms, counts = np.unique(raw_membership, return_counts=True)
    order    = np.argsort(-counts)
    rank_map = {int(old): int(new) for new, old in enumerate(unique_comms[order])}
    ranked   = np.array([rank_map[c] for c in raw_membership], dtype=int)

    new_counts = np.bincount(ranked)
    for sb in np.where(new_counts < min_comm_size)[0]:
        if sb != 0:
            ranked[ranked == sb] = 0
    survivors = sorted(set(ranked.tolist()))
    compact   = {old: new for new, old in enumerate(survivors)}
    ranked    = np.array([compact[c] for c in ranked], dtype=int)
    return ranked, int(ranked.max()) + 1


def build_village_envdata(adj, beta_list, delta_list,
                          comm_method="edge_betweenness", min_comm_size=10):
    """Detect communities and assign per-node beta/delta by community rank. Returns a dict (adj, N, beta_vec, delta_vec, block_labels, n_bins, bin_counts)."""
    N = adj.shape[0]
    block_labels, n_comms = detect_communities_igraph(
        adj, comm_method=comm_method, min_comm_size=min_comm_size)
    beta_vec  = np.zeros(N, dtype=float)
    delta_vec = np.zeros(N, dtype=float)
    for b in range(n_comms):
        mask = block_labels == b
        beta_vec[mask]  = beta_list[min(b, len(beta_list)  - 1)]
        delta_vec[mask] = delta_list[min(b, len(delta_list) - 1)]
    bin_counts = np.bincount(block_labels, minlength=n_comms)
    return dict(adj=adj, N=N, beta_vec=beta_vec, delta_vec=delta_vec,
                block_labels=block_labels, n_bins=n_comms, bin_counts=bin_counts)


# CQL training
def make_cql(dataset, label="last", alpha=None,
             discount=0.8, alpha_cql=0.1, hidden=None, batch_size=64,
             n_steps_max=30_000, n_steps_per_epoch=1_000,
             patience=10, min_delta=1e-4, verbose=False):
    """Train a discrete CQL agent with early stopping on TD loss; returns the model."""
    import d3rlpy
    if hidden is None:
        hidden = [256, 256]
    _alpha = alpha if alpha is not None else alpha_cql
    enc = d3rlpy.models.VectorEncoderFactory(
        hidden_units=hidden, use_batch_norm=True, dropout_rate=0.3)
    cfg = d3rlpy.algos.DiscreteCQLConfig(
        encoder_factory=enc,
        q_func_factory=d3rlpy.models.MeanQFunctionFactory(),
        gamma=discount, alpha=_alpha,
        learning_rate=3e-4, batch_size=batch_size,
    )
    mdl = cfg.create(device="cpu")
    max_epochs = n_steps_max // n_steps_per_epoch
    best_loss, no_improve = float("inf"), 0
    log_dir = tempfile.mkdtemp()
    try:
        for epoch in range(1, max_epochs + 1):
            mdl.fit(dataset, n_steps=n_steps_per_epoch,
                    n_steps_per_epoch=n_steps_per_epoch, show_progress=False,
                    logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir=log_dir))
            td_loss = None
            for csv_path in _glob.glob(os.path.join(log_dir, "**", "*.csv"), recursive=True):
                if not any(k in os.path.splitext(os.path.basename(csv_path))[0].lower()
                           for k in ("loss", "error", "td")):
                    continue
                try:
                    lines = open(csv_path).read().strip().splitlines()
                    if lines:
                        td_loss = float(lines[-1].split(",")[2])
                except Exception:
                    pass
            if td_loss is not None:
                if best_loss - td_loss > min_delta:
                    best_loss, no_improve = td_loss, 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"  [{label}] Converged epoch {epoch}  loss={best_loss:.5f}")
                    break
                if verbose and epoch % 2 == 0:
                    print(f"  [{label}] epoch {epoch:3d}/{max_epochs}"
                          f" | loss={td_loss:.5f} | patience {no_improve}/{patience}")
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)
    return mdl


# Dataset construction

def build_dataset(trajs, so_fn, n_bins):
    """Build a d3rlpy MDPDataset from behavioral trajectories (skip transitions, action == n_bins, excluded)."""
    import d3rlpy
    obs_l, acts_l, rews_l, terms_l = [], [], [], []
    for traj in trajs:
        ep_obs, ep_acts, ep_rews = [], [], []
        for t in range(traj["T"] - 1):
            if int(traj["actions"][t]) >= n_bins:
                continue
            ep_obs.append(so_fn(traj["adopted"][t]))
            ep_acts.append(int(traj["actions"][t]))
            ep_rews.append(float(traj["rewards"][t]))
        if not ep_obs:
            continue
        ep_terms = [False] * len(ep_obs)
        ep_terms[-1] = True
        obs_l.extend(ep_obs); acts_l.extend(ep_acts)
        rews_l.extend(ep_rews); terms_l.extend(ep_terms)
    return d3rlpy.dataset.MDPDataset(
        observations=np.stack(obs_l).astype(np.float32),
        actions=np.array(acts_l, dtype=np.int32),
        rewards=np.array(rews_l, dtype=np.float32),
        terminals=np.array(terms_l, dtype=bool),
        action_size=n_bins,
    )


# Policy evaluation

def run_policy_episodes(policy_fn, env_factory, n_eps, k_seeds, norm,
                        rng_seed=99, reset_fn=None, init_state=None):
    """Run n_eps test episodes; returns (totals normalised by norm, curves, bin_counts)."""
    rng_ev = np.random.default_rng(rng_seed)
    totals, curves = [], []
    n_bins = env_factory(0).n_bins
    bin_counts = np.zeros(n_bins, dtype=int)
    for _ in range(n_eps):
        te = env_factory(int(rng_ev.integers(0, 1_000_000)))
        if reset_fn:
            reset_fn()
        obs = te.reset(init_state=init_state)
        done = False; tr = 0.0; rates = []; seeds_used = 0
        while not done:
            a = policy_fn(obs, te) if seeds_used < k_seeds else None
            obs, r, done, info = te.step(a)
            tr += r; rates.append(info["adoption_rate"])
            if a is not None and 0 <= int(a) < te.n_bins:
                bin_counts[int(a)] += 1; seeds_used += 1
        totals.append(tr / norm); curves.append(rates)
    return np.array(totals), np.array(curves), bin_counts
