#!/usr/bin/env python3
"""
Quick Q-Ising pass at one T_train on the saturation environment (saturation_env.py): logs one
random-policy behavioral trajectory of T_train periods, fits the group Ising model and reports
coefficients with in-sample AUC/accuracy, then trains Q-Ising and plain DQN on that trajectory
and deploys them against the topological baselines. Env var: T_TRAIN (default 400).
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch as _torch
import d3rlpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from saturation_env import (
    build_ba_graph_with_bins, SaturationSpilloverEnv, RandomNodePolicy,
)
from qising.ising.group_ising import (
    fit_group_ising_emvs, predict_group_ising_emvs,
)
from qising.baselines import (
    DegreeCentralityPolicy, LIRPolicy, DegreeBinPolicy, QIsingPolicy,
)

logging.getLogger('d3rlpy').setLevel(logging.WARNING)

NET_SEED = 0
N = 100
N_BINS = 10
BA_M = 5  # Barabasi-Albert attachment param -- mean degree ~= 2*m = 10
P_SPREAD = 0.5   # lowered from 0.7 -- see saturation_env.py / smoke_test.py note
CHURN_HIGH = 0.9
CHURN_LOW = 0.2
SAT_THRESHOLD = 0.5

T_TRAIN = int(os.environ.get('T_TRAIN', 400))
SKIP_PROB = 0.15
DATA_SEED = 42

T_TEST = 40  # >> n_bins=10 hubs, so non-repeating DegreeCentrality/LIR run out of hubs
N_EVAL_EPS = 50

CQL_KWARGS = dict(discount=0.8, alpha_cql=0.05, hidden=[64, 64], batch_size=64,
                   n_steps_max=6_000, n_steps_per_epoch=300, patience=6, verbose=False)


def _seed_all(s):
    np.random.seed(s); _torch.manual_seed(s); d3rlpy.seed(s)


def build_network():
    return build_ba_graph_with_bins(N, N_BINS, BA_M, seed=NET_SEED)


def collect_behavioral_trajectory(adj, block_labels, T, seed):
    n_bins = int(block_labels.max()) + 1
    env = SaturationSpilloverEnv(adj, block_labels, p_spread=P_SPREAD, churn_high=CHURN_HIGH,
                                  churn_low=CHURN_LOW, saturation_threshold=SAT_THRESHOLD, seed=seed)
    rng = np.random.default_rng(seed + 1)
    N_ = env.N
    adopted_traj = np.zeros((T, N_), dtype=bool)
    action_traj = np.full(T, n_bins, dtype=int)   # n_bins == "skip this period" sentinel
    seeded_nodes = np.full(T, -1, dtype=int)
    reward_traj = np.zeros(T)
    infos = []
    for t in range(T):
        adopted_traj[t] = env.adopted.copy()
        action = None
        if rng.random() >= SKIP_PROB:
            action = int(rng.integers(0, n_bins))
            action_traj[t] = action
        _, r, _, info = env.step(action)
        reward_traj[t] = r
        infos.append(info)
        if info['seeded_node'] is not None:
            seeded_nodes[t] = info['seeded_node']
    return dict(adopted=adopted_traj, actions=action_traj, seeded_nodes=seeded_nodes,
                rewards=reward_traj, infos=infos, T=T, N=N_, n_bins=n_bins)


def report_ising_fit(traj, adj, block_labels):
    print('\n' + '=' * 70)
    print(f'Group Ising fit on T_train={T_TRAIN} periods (first-order-neighbor-count model)')
    print('=' * 70)

    X_arr = traj["adopted"][:-1].astype(float)[None, ...]     # (1, T-1, N)
    y_arr = traj["adopted"][1:].astype(float)[None, ...]
    t_arr = traj["seeded_nodes"][:-1][None, ...]              # (1, T-1)

    result = fit_group_ising_emvs(X_arr, y_arr, t_arr, adj, block_labels)
    ex = result['extracted']
    G = result['n_groups']
    bin_sizes = result['hyperparameters']['bin_sizes'].astype(int)

    print(f'  n_groups(bins)={G}  bin_sizes={bin_sizes.tolist()}')
    print(f'  em_iterations per bin: {result["em_iterations"]}  converged: {result["converged"]}')
    print(f'\n  Per-bin coefficients:')
    print(f'  {"bin":<5}{"intercept":>12}{"rho":>10}{"beta_treat":>12}{"spillover":>12}')
    for g in range(G):
        print(f'  {g:<5}{ex["intercept_mean"][g]:>12.3f}{ex["rho_mean"][g]:>10.3f}'
              f'{ex["beta_treat_mean"][g]:>12.3f}{ex["spillover_mean"][g]:>12.3f}')
    print(f'\n  Means across bins: intercept={ex["intercept_mean"].mean():.3f}  '
          f'rho={ex["rho_mean"].mean():.3f}  beta_treat={ex["beta_treat_mean"].mean():.3f}  '
          f'spillover={ex["spillover_mean"].mean():.3f}')

    coupling = ex['coupling_matrix_mean']  # (G, G): own-bin diag + cross-bin off-diag counts
    print(f'\n  Group-to-group coupling matrix (gamma, row=own bin, col=neighbor-bin count coeff):')
    with np.printoptions(precision=3, suppress=True, linewidth=200):
        print('  ' + str(coupling).replace('\n', '\n  '))
    diag_mean = float(np.mean(np.diag(coupling)))
    offdiag_mean = float((coupling.sum() - np.trace(coupling)) / (G * G - G))
    print(f'\n  mean diagonal (within-bin) coupling: {diag_mean:.3f}   '
          f'mean off-diagonal (cross-bin) coupling: {offdiag_mean:.3f}')

    # in-sample AUC/accuracy on the fitting features
    T_minus1, N_ = X_arr.shape[1], X_arr.shape[2]
    y_pred = np.zeros((T_minus1, N_))
    for t in range(T_minus1):
        action = int(t_arr[0, t]) if t_arr[0, t] >= 0 else -1
        y_pred[t] = predict_group_ising_emvs(X_arr[0, t], action, result, adj)

    y_true_flat = y_arr[0].flatten()
    y_pred_flat = y_pred.flatten()
    acc = accuracy_score(y_true_flat, y_pred_flat > 0.5)
    base_rate_acc = max(y_true_flat.mean(), 1 - y_true_flat.mean())
    auc = roc_auc_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else float('nan')

    print('\n' + '=' * 70)
    print('In-sample (training-data) predictive performance')
    print('=' * 70)
    print(f'  n_rows={len(y_true_flat)}  P(adopt next)_true={y_true_flat.mean():.3f}  '
          f'base_rate_acc={base_rate_acc:.3f}')
    print(f'  model_acc={acc:.3f}   AUC={auc:.3f}')
    print(f'  (AUC close to base-rate-accuracy-implied ~0.5-0.6 territory would confirm the '
          f'misspecification story: the model cannot represent the 2-hop reach or the churn '
          f'saturation threshold, so it is only picking up a weak, linear, first-order-only '
          f'approximation of the true process.)')

    return result, dict(auc=float(auc), acc=float(acc), base_rate_acc=float(base_rate_acc),
                         n_rows=len(y_true_flat))


def deploy(policy, adj, block_labels, T, n_eps, base_seed):
    rng = np.random.default_rng(base_seed)
    totals = []
    for ep in range(n_eps):
        env = SaturationSpilloverEnv(adj, block_labels, p_spread=P_SPREAD, churn_high=CHURN_HIGH,
                                      churn_low=CHURN_LOW, saturation_threshold=SAT_THRESHOLD,
                                      seed=int(rng.integers(0, 1_000_000)))
        env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
        cum_reward = 0.0
        for t in range(T):
            obs = env._get_obs()
            a = policy(obs, env)
            _, r, _, _ = env.step(a)
            cum_reward += r
        totals.append(cum_reward)
    return np.array(totals)


def main():
    print('=' * 70)
    print('Quick Q-Ising Pass — Second-Order Saturation/Crowding Environment')
    print(f'T_train={T_TRAIN}  N={N}  N_BINS={N_BINS}')
    print('=' * 70)

    adj, block_labels = build_network()
    rep_seed = DATA_SEED + T_TRAIN * 10
    traj = collect_behavioral_trajectory(adj, block_labels, T_TRAIN, seed=rep_seed)
    n_seed_events = int((traj['seeded_nodes'] >= 0).sum())
    print(f'  Collected {T_TRAIN} periods, {n_seed_events} seed events '
          f'({100*n_seed_events/T_TRAIN:.1f}% of periods, skip_prob={SKIP_PROB}).')

    ising_result, fit_stats = report_ising_fit(traj, adj, block_labels)

    print('\n' + '=' * 70)
    print(f'Training Q-Ising and DQN-plain on the SAME T_train={T_TRAIN} trajectory (quick CQL settings)')
    print('=' * 70)
    _seed_all(rep_seed + 2)
    pol_qising = QIsingPolicy(ising_augmentation=True, ensemble=False, label='q_ising_quick',
                               **CQL_KWARGS).fit([traj], adj, block_labels, _emvs_result=ising_result)
    _seed_all(rep_seed + 3)
    pol_dqn_plain = QIsingPolicy(ising_augmentation=False, ensemble=False, label='dqn_plain_quick',
                                  **CQL_KWARGS).fit([traj], adj, block_labels)

    ref_env = SaturationSpilloverEnv(adj, block_labels, seed=0)
    policies = {
        'random': RandomNodePolicy(N, seed=DATA_SEED),
        'degree_centrality': DegreeCentralityPolicy(ref_env),
        'lir': LIRPolicy(ref_env),
        'degree_bin': DegreeBinPolicy(ref_env),
        'dqn_plain': pol_dqn_plain,
        'q_ising': pol_qising,
    }

    print(f'\n  Deployment ({N_EVAL_EPS} eval episodes x {T_TEST} periods, fresh warm start):')
    results = {}
    for name, pol in policies.items():
        rewards = deploy(pol, adj, block_labels, T_TEST, N_EVAL_EPS, base_seed=555)
        se = rewards.std() / np.sqrt(N_EVAL_EPS)
        results[name] = dict(mean=float(rewards.mean()), se=float(se))
        print(f'    {name:<18} mean_cum_reward={rewards.mean():9.2f}  se={se:.2f}')

    pd.DataFrame([{
        'T_train': T_TRAIN,
        'ising_auc': fit_stats['auc'], 'ising_acc': fit_stats['acc'],
        'ising_base_rate_acc': fit_stats['base_rate_acc'],
        **{f'deploy_mean_{k}': v['mean'] for k, v in results.items()},
        **{f'deploy_se_{k}': v['se'] for k, v in results.items()},
    }]).to_csv(os.path.join(THIS_DIR, f'qising_quick_summary_T{T_TRAIN}.csv'), index=False)
    print(f'\nSummary saved -> {os.path.join(THIS_DIR, f"qising_quick_summary_T{T_TRAIN}.csv")}')


if __name__ == '__main__':
    main()
