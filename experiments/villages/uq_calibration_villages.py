#!/usr/bin/env python3
"""
UQ calibration on the village networks (real-network version of
experiments/uq_small_network/uq_calibration_sis.py), with the settings of mcmc_ensemble.ipynb
except a smaller CQL budget and N_TEST_EPS=20. Checks calibration (vote entropy vs true value
gap, as a per-node rate so villages are comparable) and decision change; the misspecification
check is not available for villages. Writes per-village and pooled decision CSVs to this folder.

Env vars (defaults): VILLAGE_IDS "6,42,9,32,60", T_TRAIN 500, N_TRAIN_EPS 1, DATA_SEED 42,
T_TEST 25, N_TEST_EPS 20, SKIP_PROB 0, INIT_RATE 0, TEST_INIT_RATE 0, DISCOUNT 0.8, ALPHA_CQL 0.1,
N_STEPS_MAX 10000, N_STEPS_PER_EPOCH 500, PATIENCE 8, N_ENSEMBLE 20, ROLLOUT_H 12, N_MC 200, FORCE_RERUN.
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch as _torch
import d3rlpy

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(THIS_DIR, "figures", "uq_calibration")
os.makedirs(FIGURES_DIR, exist_ok=True)

from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
from qising.utils import build_village_envdata
from qising.baselines import QIsingPolicy

logging.getLogger('d3rlpy').setLevel(logging.WARNING)

def _env(name, default, cast=int):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


ADJ_JSON = os.path.join(THIS_DIR, "..", "..", "data", "villages", "adjacency_all.json")

# default villages span the available N range (107-341)
VILLAGE_IDS = os.environ.get('VILLAGE_IDS', '6,42,9,32,60').split(',')

BETA_LIST  = [0.01, 0.5, 0.05, 0.07, 0.06, 0.02, 0.01, 0.4, 0.1, 0.3]
DELTA_LIST = [0.5,  0.9, 0.9,  0.6,  0.5,  0.50, 0.7,  0.6, 0.5, 0.8]
COMM_METHOD   = "edge_betweenness"
MIN_COMM_SIZE = 10

T_TRAIN     = _env('T_TRAIN', 500)
N_TRAIN_EPS = _env('N_TRAIN_EPS', 1)
DATA_SEED   = _env('DATA_SEED', 42)
T_TEST      = _env('T_TEST', 25)
N_TEST_EPS  = _env('N_TEST_EPS', 20)
SKIP_PROB   = _env('SKIP_PROB', 0, float)
INIT_RATE      = _env('INIT_RATE', 0.0, float)
TEST_INIT_RATE = _env('TEST_INIT_RATE', 0.0, float)

DISCOUNT  = _env('DISCOUNT', 0.8, float)
ALPHA_CQL = _env('ALPHA_CQL', 0.1, float)
CQL_KWARGS = dict(discount=DISCOUNT, alpha_cql=ALPHA_CQL, verbose=False,
                   n_steps_max=_env('N_STEPS_MAX', 10_000),
                   n_steps_per_epoch=_env('N_STEPS_PER_EPOCH', 500),
                   patience=_env('PATIENCE', 8))

N_ENSEMBLE = _env('N_ENSEMBLE', 20)
ROLLOUT_H  = _env('ROLLOUT_H', 12)
N_MC       = _env('N_MC', 200)
GAMMA      = DISCOUNT


def _seed_all(s):
    np.random.seed(s); _torch.manual_seed(s); d3rlpy.seed(s)


# Known-dynamics oracle rollout (matches EdgeBetCommunitySISEnv.step)

def _true_step_batch(states, actions, beta_vec, delta_vec, adj, log1mb, rng):
    M, N = states.shape
    churn_draws = rng.random((M, N))
    churned = states & (churn_draws < delta_vec[None, :])
    new_states = states & ~churned
    for m in range(M):
        a = actions[m]
        if a is not None:
            new_states[m, a] = True
    weighted = new_states.astype(float) * log1mb[None, :]
    log_p_escape = weighted @ adj
    p_adopt = 1.0 - np.exp(log_p_escape)
    susceptible = ~new_states
    spread_draws = rng.random((M, N))
    newly = susceptible & (spread_draws < p_adopt)
    return new_states | newly


def oracle_value(state0, action0, beta_vec, delta_vec, adj, seed):
    """Mean discounted adoption rate (divided by N) over ROLLOUT_H periods after seeding
    action0, then no treatment; seed gives common random numbers across actions."""
    N = len(state0)
    log1mb = np.log(1.0 - np.minimum(beta_vec, 1.0 - 1e-15))
    rng = np.random.default_rng(seed)
    states = np.tile(state0, (N_MC, 1)).copy()
    actions = [action0] * N_MC
    cum = np.zeros(N_MC)
    disc = 1.0
    for _h in range(ROLLOUT_H):
        states = _true_step_batch(states, actions, beta_vec, delta_vec, adj, log1mb, rng)
        cum += disc * (states.sum(axis=1) / N)
        disc *= GAMMA
        actions = [None] * N_MC
    return float(cum.mean())


def ensemble_votes(ensemble_models, ba, n_bins, avl):
    votes = np.zeros(n_bins, dtype=int)
    avl_set = set(avl)
    for _, _, mdl, so_fn in ensemble_models:
        ob = so_fn(ba).reshape(1, -1)
        qv = mdl.predict_value(np.repeat(ob, n_bins, axis=0), np.arange(n_bins, dtype=np.int32))
        for b in np.argsort(qv)[::-1]:
            if int(b) in avl_set:
                votes[int(b)] += 1
                break
    return votes


def normalized_entropy(votes, avl):
    if len(avl) < 2:
        return np.nan
    p = votes[list(avl)].astype(float)
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p / total
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(avl)))


def collect_decision_points(env, ensemble_models, point_policy, T, beta_vec, delta_vec, adj,
                             rollout_seed_base):
    records = []
    for t in range(T):
        obs = env._get_obs()
        avl = env.get_available_actions()
        if len(avl) == 0:
            env.step(None)
            continue

        ba = obs['adopted']
        votes = ensemble_votes(ensemble_models, ba, env.n_bins, avl)
        majority_action = int(avl[np.argmax(votes[list(avl)])])
        point_action = point_policy(obs, env)
        entropy = normalized_entropy(votes, avl)

        seed = rollout_seed_base + t * 7919
        values = {a: oracle_value(ba, a, beta_vec, delta_vec, adj, seed) for a in avl}
        oracle_best_action = max(values, key=values.get)

        records.append(dict(
            t=t, n_avl=len(avl), entropy=entropy,
            majority_action=majority_action, point_action=point_action,
            oracle_best_action=oracle_best_action,
            majority_value=values[majority_action],
            point_value=values[point_action] if point_action in values else np.nan,
            oracle_value=values[oracle_best_action],
            true_gap=values[oracle_best_action] - values[majority_action],
        ))
        env.step(majority_action)
    return records


def run_village(village_id, adj_all):
    print('\n' + '=' * 70)
    print(f'Village {village_id}')
    print('=' * 70)

    A = np.array(adj_all[village_id], dtype=float)
    np.fill_diagonal(A, 0)
    dvd = build_village_envdata(A, BETA_LIST, DELTA_LIST,
                                 comm_method=COMM_METHOD, min_comm_size=MIN_COMM_SIZE)
    N, n_bins = dvd['N'], dvd['n_bins']
    print(f'  N={N}, n_bins={n_bins}, bin_counts={dvd["bin_counts"].tolist()}')

    print(f'  [1] Training data: T_train={T_TRAIN}, {N_TRAIN_EPS} trajectory, seed={DATA_SEED}')
    env = EdgeBetCommunitySISEnv(dvd['adj'], dvd['beta_vec'], dvd['delta_vec'], dvd['block_labels'],
                                  T=T_TRAIN, seed=DATA_SEED, init_rate=INIT_RATE)
    traj = env.collect_behavioral_episode(skip_prob=SKIP_PROB, verbose=True)

    print('  Fitting Q-Ising ensemble (n_ensemble=20) + point-estimate MAP ...')
    _seed_all(DATA_SEED + 2)
    pol_ensemble = QIsingPolicy(
        ising_augmentation=True, ensemble=True, n_ensemble=N_ENSEMBLE,
        label=f'v{village_id}-uq-ensemble', **CQL_KWARGS,
    ).fit([traj], dvd['adj'], dvd['block_labels'])

    _seed_all(DATA_SEED + 3)
    pol_point = QIsingPolicy(
        ising_augmentation=True, ensemble=False,
        label=f'v{village_id}-uq-point', **CQL_KWARGS,
    ).fit([traj], dvd['adj'], dvd['block_labels'])

    ensemble_models = pol_ensemble._ensemble_models
    assert len(ensemble_models) == N_ENSEMBLE

    print(f'  [2] Evaluation: {N_TEST_EPS} episodes x T_test={T_TEST}, '
          f'oracle rollout H={ROLLOUT_H}, N_MC={N_MC} ...')
    all_records = []
    rng = np.random.default_rng(DATA_SEED + 100)
    for ep in range(N_TEST_EPS):
        ep_seed = int(rng.integers(0, 1_000_000))
        eval_env = EdgeBetCommunitySISEnv(dvd['adj'], dvd['beta_vec'], dvd['delta_vec'],
                                          dvd['block_labels'], T=T_TEST, seed=ep_seed,
                                          init_rate=TEST_INIT_RATE)
        recs = collect_decision_points(
            eval_env, ensemble_models, pol_point, T_TEST, dvd['beta_vec'], dvd['delta_vec'],
            dvd['adj'], rollout_seed_base=DATA_SEED + 1000 * (ep + 1),
        )
        for r in recs:
            r['episode'] = ep
        all_records.extend(recs)
        if (ep + 1) % 5 == 0:
            print(f'    episode {ep + 1}/{N_TEST_EPS}: {len(all_records)} decision points so far')

    df = pd.DataFrame(all_records)
    df['village'] = village_id
    df['N'] = N
    df['n_bins'] = n_bins
    return df


def main():
    print('=' * 70)
    print('UQ calibration — microfinance villages, N=5 (spanning N=107-341)')
    print('=' * 70)

    adj_all = json.load(open(ADJ_JSON))
    all_dfs = []
    for vid in VILLAGE_IDS:
        out_csv = os.path.join(THIS_DIR, f'uq_calibration_village{vid}_decisions.csv')
        if os.path.exists(out_csv) and _env('FORCE_RERUN', 0) == 0:
            print(f'\nSkipping village {vid} (already have {out_csv}; '
                  f'set FORCE_RERUN=1 to redo)')
            all_dfs.append(pd.read_csv(out_csv))
            continue
        df = run_village(vid, adj_all)
        all_dfs.append(df)
        df.to_csv(out_csv, index=False)
        print(f'  Saved -> {out_csv}')

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(os.path.join(THIS_DIR, 'uq_calibration_villages_all_decisions.csv'), index=False)

    print(f'\n{"=" * 70}\nPER-VILLAGE RESULTS\n{"=" * 70}')
    per_village_rows = []
    for vid in VILLAGE_IDS:
        sub = full[(full['village'] == vid) & (full['n_avl'] >= 2)].copy()
        if len(sub) < 5:
            print(f'  Village {vid}: too few multi-option decision points (n={len(sub)}), skipping stats')
            continue
        r, p = stats.pearsonr(sub['entropy'], sub['true_gap'])
        rho, ps = stats.spearmanr(sub['entropy'], sub['true_gap'])
        differs = sub['majority_action'] != sub['point_action']
        frac_differs = float(differs.mean())
        N_v = int(sub['N'].iloc[0])
        print(f'  Village {vid} (N={N_v}, n={len(sub)}): '
              f'Pearson r={r:+.3f} (p={p:.2e})  Spearman rho={rho:+.3f} (p={ps:.2e})  '
              f'ensemble!=point-est: {frac_differs:.3f}')
        if differs.sum() > 0:
            vdiff = (sub.loc[differs, 'majority_value'] - sub.loc[differs, 'point_value'])
            print(f'    when differs: mean value diff={vdiff.mean():+.4f}  '
                  f'ensemble better in {(vdiff > 0).mean():.1%} of cases')
        per_village_rows.append(dict(village=vid, N=N_v, n=len(sub), pearson_r=r, pearson_p=p,
                                      spearman_rho=rho, spearman_p=ps, frac_decision_change=frac_differs))

    pd.DataFrame(per_village_rows).to_csv(
        os.path.join(THIS_DIR, 'uq_calibration_villages_summary.csv'), index=False)

    print(f'\n{"=" * 70}\nPOOLED ACROSS ALL 5 VILLAGES\n{"=" * 70}')
    sub_all = full[full['n_avl'] >= 2].copy()
    r, p = stats.pearsonr(sub_all['entropy'], sub_all['true_gap'])
    rho, ps = stats.spearmanr(sub_all['entropy'], sub_all['true_gap'])
    print(f'  n={len(sub_all)}  Pearson r={r:+.4f} (p={p:.2e})  Spearman rho={rho:+.4f} (p={ps:.2e})')
    differs = sub_all['majority_action'] != sub_all['point_action']
    print(f'  Fraction ensemble != point-estimate: {differs.mean():.3f} ({int(differs.sum())}/{len(sub_all)})')
    if differs.sum() > 0:
        vdiff = sub_all.loc[differs, 'majority_value'] - sub_all.loc[differs, 'point_value']
        print(f'  When differs: mean value diff={vdiff.mean():+.4f}  '
              f'ensemble better in {(vdiff > 0).mean():.1%} of cases')
    if r < -0.3 and p < 0.01:
        print('  VERDICT: supports the Fig.6 claim on real village networks too.')
    else:
        print('  VERDICT: correlation too weak/wrong-signed pooled across villages — report honestly.')

    print('\n  NOTE: Check [3] (misspecification entropy comparison) was not run -- no '
          'misspecified dynamic exists yet for the village networks. See docstring.')

    _plot_pooled(sub_all, FIGURES_DIR)
    _plot_per_village(full, FIGURES_DIR)
    print('\nDone.')


def _plot_pooled(sub_all, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(VILLAGE_IDS)))
    for vid, c in zip(VILLAGE_IDS, colors):
        s = sub_all[sub_all['village'] == vid]
        ax.scatter(s['entropy'], s['true_gap'], s=14, alpha=0.5, color=c, label=f'village {vid} (N={int(s["N"].iloc[0]) if len(s) else "-"})')
    if len(sub_all) > 1:
        m, b = np.polyfit(sub_all['entropy'], sub_all['true_gap'], 1)
        xs = np.linspace(sub_all['entropy'].min(), sub_all['entropy'].max(), 50)
        r, _ = stats.pearsonr(sub_all['entropy'], sub_all['true_gap'])
        ax.plot(xs, m * xs + b, color='k', lw=1.5, ls='--', label=f'pooled fit (r={r:+.3f})')
    ax.set_xlabel('Vote dispersion (normalized entropy, 20 agents)')
    ax.set_ylabel('True value gap (oracle-best − majority-vote, adoption-rate units)')
    ax.set_title('UQ calibration: 5 microfinance villages, pooled', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'uq_calibration_villages_pooled_scatter.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'uq_calibration_villages_pooled_scatter.pdf'), bbox_inches='tight')
    plt.close(fig)


def _plot_per_village(full, out_dir):
    fig, axes = plt.subplots(1, len(VILLAGE_IDS), figsize=(4 * len(VILLAGE_IDS), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, vid in zip(axes, VILLAGE_IDS):
        sub = full[(full['village'] == vid) & (full['n_avl'] >= 2)]
        if len(sub) < 2:
            ax.set_title(f'village {vid}\n(insufficient data)')
            continue
        ax.scatter(sub['entropy'], sub['true_gap'], s=10, alpha=0.4, color='#2ca02c')
        r, _ = stats.pearsonr(sub['entropy'], sub['true_gap'])
        N_v = int(sub['N'].iloc[0])
        ax.set_title(f'village {vid} (N={N_v})\nr={r:+.3f}', fontsize=10)
        ax.set_xlabel('entropy')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('true value gap')
    plt.suptitle('UQ calibration per village', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'uq_calibration_villages_per_village.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'uq_calibration_villages_per_village.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
