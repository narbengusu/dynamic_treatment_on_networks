#!/usr/bin/env python3
"""
T_train sweep (levels x replicates) for Q-Ising vs DQN-plain vs topological/random baselines
on the fixed saturation network (saturation_env.py). Each replicate logs a fresh behavioral
trajectory, fits the Ising model and trains the CQL agents. Resumable from
qising_sweep_all_results.csv (FORCE_RERUN=1 redoes everything); writes qising_sweep_*.csv.

Env vars (optional; defaults): T_TRAIN_LEVELS "200,400,800", N_REPLICATES 5, T_TEST 40,
N_EVAL_EPS 50, ALPHA_CQL, N_STEPS_MAX, N_STEPS_PER_EPOCH, PATIENCE, HIDDEN_UNITS, FORCE_RERUN 0.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import torch as _torch
import d3rlpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(THIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from saturation_env import (
    SaturationSpilloverEnv, RandomNodePolicy,
)
from run_qising_quick import (
    build_network, collect_behavioral_trajectory, deploy,
    N, N_BINS, P_SPREAD, CHURN_HIGH, CHURN_LOW, SAT_THRESHOLD,
)
from qising.ising.group_ising import (
    fit_group_ising_emvs, predict_group_ising_emvs,
)
from qising.baselines import (
    DegreeCentralityPolicy, LIRPolicy, DegreeBinPolicy, QIsingPolicy,
)

logging.getLogger('d3rlpy').setLevel(logging.WARNING)


def _env(name, default, cast=int):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


def _env_list(name, default, cast=int):
    v = os.environ.get(name)
    return [cast(x) for x in v.split(',')] if v is not None else default


T_TRAIN_LEVELS = _env_list('T_TRAIN_LEVELS', [200, 400, 800])
N_REPLICATES   = _env('N_REPLICATES', 5)
DATA_SEED      = _env('DATA_SEED', 42)

T_TEST     = _env('T_TEST', 40)
N_EVAL_EPS = _env('N_EVAL_EPS', 50)

CQL_KWARGS = dict(
    discount=0.8,
    alpha_cql=_env('ALPHA_CQL', 0.05, float),
    hidden=[_env('HIDDEN_UNITS', 64), _env('HIDDEN_UNITS', 64)],
    batch_size=64,
    n_steps_max=_env('N_STEPS_MAX', 6_000),
    n_steps_per_epoch=_env('N_STEPS_PER_EPOCH', 300),
    patience=_env('PATIENCE', 6),
    verbose=False,
)

FORCE_RERUN = _env('FORCE_RERUN', 0)

RESULTS_CSV   = os.path.join(THIS_DIR, 'qising_sweep_all_results.csv')
ISING_CSV     = os.path.join(THIS_DIR, 'qising_sweep_ising_fit.csv')
BASELINES_CSV = os.path.join(THIS_DIR, 'qising_sweep_baselines.csv')
SUMMARY_CSV   = os.path.join(THIS_DIR, 'qising_sweep_summary.csv')


def _seed_all(s):
    np.random.seed(s); _torch.manual_seed(s); d3rlpy.seed(s)


def fit_and_score_ising(traj, adj, block_labels):
    """Quiet report_ising_fit: returns (result, summary_dict) without printing coefficients."""
    X_arr = traj["adopted"][:-1].astype(float)[None, ...]
    y_arr = traj["adopted"][1:].astype(float)[None, ...]
    t_arr = traj["seeded_nodes"][:-1][None, ...]

    result = fit_group_ising_emvs(X_arr, y_arr, t_arr, adj, block_labels)
    ex = result['extracted']

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

    coupling = ex['coupling_matrix_mean']
    G = result['n_groups']
    offdiag_mean = float((coupling.sum() - np.trace(coupling)) / (G * G - G)) if G > 1 else float('nan')

    summary = dict(
        auc=float(auc), acc=float(acc), base_rate_acc=float(base_rate_acc), n_rows=len(y_true_flat),
        mean_intercept=float(ex['intercept_mean'].mean()), mean_rho=float(ex['rho_mean'].mean()),
        mean_beta_treat=float(ex['beta_treat_mean'].mean()), mean_spillover=float(ex['spillover_mean'].mean()),
        mean_diag_coupling=float(np.mean(np.diag(coupling))), mean_offdiag_coupling=offdiag_mean,
    )
    return result, summary


def run_baselines(adj, block_labels):
    print(f'\n{"=" * 70}\nBaselines (data-independent, evaluated once)\n{"=" * 70}')
    ref_env = SaturationSpilloverEnv(adj, block_labels, p_spread=P_SPREAD, churn_high=CHURN_HIGH,
                                      churn_low=CHURN_LOW, saturation_threshold=SAT_THRESHOLD, seed=0)
    baseline_policies = {
        'random': RandomNodePolicy(N, seed=DATA_SEED),
        'degree_centrality': DegreeCentralityPolicy(ref_env),
        'lir': LIRPolicy(ref_env),
        'degree_bin': DegreeBinPolicy(ref_env),
    }
    rows = []
    for name, pol in baseline_policies.items():
        rewards = deploy(pol, adj, block_labels, T_TEST, N_EVAL_EPS, base_seed=555)
        se = rewards.std() / np.sqrt(N_EVAL_EPS)
        print(f'  {name:<18} mean_reward={rewards.mean():9.2f}  se={se:.2f}')
        rows.append(dict(policy=name, mean_reward=float(rewards.mean()), se_reward=float(se)))
    df = pd.DataFrame(rows)
    df.to_csv(BASELINES_CSV, index=False)
    return df


def _load_cached(path):
    if os.path.exists(path) and not FORCE_RERUN:
        return pd.read_csv(path)
    return None


def run_sweep(adj, block_labels):
    cached = _load_cached(RESULTS_CSV)
    cached_ising = _load_cached(ISING_CSV)
    result_rows = list(cached.to_dict('records')) if cached is not None else []
    ising_rows = list(cached_ising.to_dict('records')) if cached_ising is not None else []
    done = {(int(r['T_train']), int(r['replicate'])) for r in result_rows}

    for T_train in T_TRAIN_LEVELS:
        print(f'\n{"#" * 70}\n# T_train = {T_train}  ({N_REPLICATES} replicates)\n{"#" * 70}')
        for rep in range(N_REPLICATES):
            if (T_train, rep) in done:
                print(f'  [T_train={T_train} rep={rep}] cached, skipping.')
                continue

            rep_seed = DATA_SEED + T_train * 1_000 + rep
            traj = collect_behavioral_trajectory(adj, block_labels, T_train, seed=rep_seed)
            n_seed_events = int((traj['seeded_nodes'] >= 0).sum())

            ising_result, fit_stats = fit_and_score_ising(traj, adj, block_labels)
            ising_rows.append(dict(T_train=T_train, replicate=rep, n_seed_events=n_seed_events, **fit_stats))

            _seed_all(rep_seed + 2)
            pol_qising = QIsingPolicy(ising_augmentation=True, ensemble=False,
                                       label=f'q_ising-T{T_train}-r{rep}',
                                       **CQL_KWARGS).fit([traj], adj, block_labels, _emvs_result=ising_result)
            _seed_all(rep_seed + 3)
            pol_dqn_plain = QIsingPolicy(ising_augmentation=False, ensemble=False,
                                          label=f'dqn_plain-T{T_train}-r{rep}',
                                          **CQL_KWARGS).fit([traj], adj, block_labels)

            print(f'  [T_train={T_train} rep={rep}] ising_auc={fit_stats["auc"]:.3f}  ', end='')
            for name, pol in [('dqn_plain', pol_dqn_plain), ('q_ising', pol_qising)]:
                rewards = deploy(pol, adj, block_labels, T_TEST, N_EVAL_EPS, base_seed=555 + rep)
                result_rows.append(dict(T_train=T_train, replicate=rep, policy=name,
                                         mean_reward=float(rewards.mean()),
                                         se_reward=float(rewards.std() / np.sqrt(N_EVAL_EPS))))
                print(f'{name}={rewards.mean():.1f}  ', end='')
            print()

            pd.DataFrame(result_rows).to_csv(RESULTS_CSV, index=False)
            pd.DataFrame(ising_rows).to_csv(ISING_CSV, index=False)

    return pd.DataFrame(result_rows), pd.DataFrame(ising_rows)


def summarize_and_plot(results_df, ising_df, baselines_df):
    print(f'\n{"=" * 70}\nSUMMARY, per T_train\n{"=" * 70}')
    summary = results_df.groupby(['T_train', 'policy']).agg(
        mean_reward=('mean_reward', 'mean'), sem_reward=('mean_reward', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        n=('mean_reward', 'count'),
    ).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    for T_train in sorted(results_df['T_train'].unique()):
        sub = summary[summary['T_train'] == T_train]
        print(f'  T_train={T_train}')
        for _, row in sub.iterrows():
            print(f'    {row["policy"]:<15} mean={row["mean_reward"]:9.2f}  sem={row["sem_reward"]:.2f}  (n={int(row["n"])})')

    ising_summary = ising_df.groupby('T_train').agg(
        mean_auc=('auc', 'mean'), sem_auc=('auc', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        mean_acc=('acc', 'mean'), mean_beta_treat=('mean_beta_treat', 'mean'), mean_spillover=('mean_spillover', 'mean'),
    ).reset_index()
    print(f'\n  Ising in-sample fit quality by T_train:')
    for _, row in ising_summary.iterrows():
        print(f'    T_train={int(row["T_train"]):<5} AUC={row["mean_auc"]:.3f}+-{row["sem_auc"]:.3f}  '
              f'acc={row["mean_acc"]:.3f}  beta_treat={row["mean_beta_treat"]:.2f}  spillover={row["mean_spillover"]:.2f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'dqn_plain': '#2ca02c', 'q_ising': '#9467bd'}
    for name in ['dqn_plain', 'q_ising']:
        sub = summary[summary['policy'] == name].sort_values('T_train')
        ax.errorbar(sub['T_train'], sub['mean_reward'], yerr=sub['sem_reward'],
                    marker='o', capsize=3, color=colors[name], label=name)
    baseline_colors = {'random': '#7f7f7f', 'degree_centrality': '#1f77b4', 'lir': '#ff7f0e', 'degree_bin': '#d62728'}
    for _, row in baselines_df.iterrows():
        c = baseline_colors.get(row['policy'], None)
        ax.axhline(row['mean_reward'], color=c, ls='--', lw=1.5, alpha=0.8, label=row['policy'])
        ax.fill_between(ax.get_xlim(), row['mean_reward'] - row['se_reward'], row['mean_reward'] + row['se_reward'],
                         color=c, alpha=0.1)
    ax.set_xlabel('T_train')
    ax.set_ylabel(f'Cumulative reward ({T_TEST} periods)')
    ax.set_title('Q-Ising / DQN-plain vs baselines, across T_train\n'
                  '(saturation/crowding env -- Ising misspecification robustness check)', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'qising_sweep_summary.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'qising_sweep_summary.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved -> {os.path.join(FIGURES_DIR, "qising_sweep_summary.png")}')


def main():
    print('=' * 70)
    print('T_train Sweep — Second-Order Saturation/Crowding Environment')
    print(f'N={N}  N_BINS={N_BINS}  T_TRAIN_LEVELS={T_TRAIN_LEVELS}  N_REPLICATES={N_REPLICATES}  '
          f'T_TEST={T_TEST}  N_EVAL_EPS={N_EVAL_EPS}')
    print('=' * 70)

    adj, block_labels = build_network()
    mean_deg = adj.sum(axis=1).mean()
    print(f'  mean_degree={mean_deg:.2f}')

    baselines_df = run_baselines(adj, block_labels)
    results_df, ising_df = run_sweep(adj, block_labels)
    summarize_and_plot(results_df, ising_df, baselines_df)

    print('\ndone -- check:')
    print(f'  {SUMMARY_CSV}')
    print(f'  {RESULTS_CSV}   (raw per-T_train x replicate x policy cumulative reward)')
    print(f'  {ISING_CSV}     (per-replicate Ising coefficients + in-sample AUC)')
    print(f'  {BASELINES_CSV} (data-independent baselines, evaluated once)')
    print(f'  figures/qising_sweep_summary.png')


if __name__ == '__main__':
    main()
