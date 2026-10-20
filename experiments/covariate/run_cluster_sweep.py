#!/usr/bin/env python3
"""
Cluster sweep: dqn-plain vs dqn-all vs q-ising (plus topological/random baselines and a
full-information oracle) on the covariate environment, across N, K and T_train.
Configured by environment variables (see README.md); resumable per (N, K) from cached CSVs;
writes cluster_sweep_*.csv and a summary figure.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as _torch
import d3rlpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(THIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from covariate_group_env import (
    build_sbm_network, CovariateGroupControllableEnv, N_COVARIATES, EFFECTIVE_IDX,
)
from covariate_group_policy import (
    CovariateGroupPolicy,
)
from qising.baselines import (
    RandomPolicy, DegreeCentralityPolicy, LIRPolicy, DegreeBinPolicy,
)

logging.getLogger('d3rlpy').setLevel(logging.WARNING)


def _env(name, default, cast=int):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


def _env_list(name, default, cast=int):
    v = os.environ.get(name)
    return [cast(x) for x in v.split(',')] if v is not None else default


NET_SEED   = _env('NET_SEED', 11)
P_IN       = _env('P_IN', 0.1, float)
P_OUT      = _env('P_OUT', 0.03, float)
P_Z        = _env('P_Z', 0.15, float)
P_BASE     = _env('P_BASE', 0.3, float)
P_ESCALATED = _env('P_ESCALATED', 0.6, float)

N_LEVELS   = _env_list('N_LEVELS', [50, 100, 200, 500])
K_LEVELS   = _env_list('K_LEVELS', [5, 10])
T_TRAIN_LEVELS = _env_list('T_TRAIN_LEVELS', [100, 300, 600])
N_REPLICATES   = _env('N_REPLICATES', 5)
DATA_SEED      = _env('DATA_SEED', 42)

T_TEST     = _env('T_TEST', 20)
N_EVAL_EPS = _env('N_EVAL_EPS', 50)

CQL_KWARGS = dict(
    discount=_env('DISCOUNT', 0.8, float),
    alpha_cql=_env('ALPHA_CQL', 0.05, float),
    hidden=[_env('HIDDEN_UNITS', 64), _env('HIDDEN_UNITS', 64)],
    verbose=False,
    n_steps_max=_env('N_STEPS_MAX', 5_000),
    n_steps_per_epoch=_env('N_STEPS_PER_EPOCH', 250),
    patience=_env('PATIENCE', 6),
)

FORCE_RERUN = _env('FORCE_RERUN', 0)


def _seed_all(s):
    np.random.seed(s); _torch.manual_seed(s); d3rlpy.seed(s)


class OraclePolicy:
    """Full-information ceiling: seed the LARGEST currently-hot block (any block if none)."""

    def __call__(self, obs, env):
        z = obs['C_group'][:, EFFECTIVE_IDX] > 0.5
        hot = np.where(z)[0]
        if len(hot) == 0:
            return 0
        sizes = np.array([len(env.bins[b]) for b in hot])
        return int(hot[np.argmax(sizes)])

    def reset(self):
        pass


def collect_behavioral_trajectory(adj, block_labels, T, seed):
    n_bins = int(block_labels.max()) + 1
    env = CovariateGroupControllableEnv(adj, block_labels, P_Z, P_BASE, P_ESCALATED, seed=seed)
    rng = np.random.default_rng(seed + 1)
    N_ = env.N
    adopted_traj = np.zeros((T, N_), dtype=bool)
    cg_traj = np.zeros((T, n_bins, N_COVARIATES))
    action_traj = np.full(T, n_bins, dtype=int)
    seeded_nodes = np.full(T, -1, dtype=int)
    reward_traj = np.zeros(T)
    infos = []
    for t in range(T):
        adopted_traj[t] = env.adopted.copy()
        cg_traj[t] = env.C_group.copy()
        action = int(rng.integers(0, n_bins))
        action_traj[t] = action
        _, r, _, info = env.step(action)
        reward_traj[t] = r
        infos.append(info)
        if info['seeded_node'] is not None:
            seeded_nodes[t] = info['seeded_node']
    return dict(adopted=adopted_traj, C_group=cg_traj, actions=action_traj, seeded_nodes=seeded_nodes,
                rewards=reward_traj, infos=infos, T=T, N=N_, n_bins=n_bins)


def deploy_from_warm_start(policy, adj, block_labels, warm_state, T, n_eps, base_seed):
    """Returns (totals (n_eps,), curves (n_eps, T) per-period rewards)."""
    rng = np.random.default_rng(base_seed)
    totals = []
    curves = np.zeros((n_eps, T))
    for ep in range(n_eps):
        env = CovariateGroupControllableEnv(adj, block_labels, P_Z, P_BASE, P_ESCALATED,
                                             seed=int(rng.integers(0, 1_000_000)))
        env.reset(init_adopted=warm_state.copy())
        if hasattr(policy, 'reset'):
            policy.reset()
        cum_reward = 0.0
        for t in range(T):
            obs = env._get_obs()
            a = policy(obs, env)
            _, r, _, _ = env.step(a)
            cum_reward += r
            curves[ep, t] = r
        totals.append(cum_reward)
    return np.array(totals), curves


BASELINE_COLORS = {
    'random': '#7f7f7f', 'degree_centrality': '#1f77b4',
    'lir': '#ff7f0e', 'degree_bin': '#d62728', 'oracle': '#17becf',
}
BASELINE_DISPLAY = {
    'random': 'Random', 'degree_centrality': 'Degree', 'lir': 'LIR',
    'degree_bin': 'Degree-Bin', 'oracle': 'Oracle (full info)',
}


def run_one_config(N, K):
    print(f'\n{"#" * 70}\n# N = {N}   K = {K}\n{"#" * 70}')
    adj, block_labels = build_sbm_network(N=N, n_blocks=K, p_in=P_IN, p_out=P_OUT, seed=NET_SEED)
    mean_deg = adj.sum(axis=1).mean()
    print(f'  mean_degree={mean_deg:.2f}  block_sizes={[int((block_labels == b).sum()) for b in range(K)]}')
    warm_state = np.zeros(N, dtype=bool)
    n_bins = K

    print(f'  Baselines (data-independent, evaluated once)...')
    ref_env = CovariateGroupControllableEnv(adj, block_labels, P_Z, P_BASE, P_ESCALATED, seed=0)
    baseline_policies = {
        'random': RandomPolicy(n_bins, seed=DATA_SEED),
        'degree_centrality': DegreeCentralityPolicy(ref_env),
        'lir': LIRPolicy(ref_env),
        'degree_bin': DegreeBinPolicy(ref_env),
        'oracle': OraclePolicy(),
    }
    baseline_rows = []
    baseline_curve_rows = []
    for name, pol in baseline_policies.items():
        reward, curves = deploy_from_warm_start(pol, adj, block_labels, warm_state, T_TEST, N_EVAL_EPS, base_seed=555)
        print(f'    {name:<18} mean_reward={reward.mean():9.2f}  se={reward.std() / np.sqrt(N_EVAL_EPS):.2f}')
        baseline_rows.append(dict(N=N, K=K, policy=name, mean_deploy_reward=reward.mean(),
                                   se_deploy_reward=reward.std() / np.sqrt(N_EVAL_EPS)))
        mean_curve = curves.mean(axis=0)
        for t in range(T_TEST):
            baseline_curve_rows.append(dict(N=N, K=K, policy=name, period=t, reward=mean_curve[t]))

    rows = []
    curve_rows = []
    for T_train in T_TRAIN_LEVELS:
        print(f'\n  {"=" * 60}\n  T_train = {T_train}  ({N_REPLICATES} replicates)\n  {"=" * 60}')
        for rep in range(N_REPLICATES):
            rep_seed = DATA_SEED + N * 1_000_000 + K * 10_000 + T_train * 10 + rep
            traj = collect_behavioral_trajectory(adj, block_labels, T_train, seed=rep_seed)

            _seed_all(rep_seed + 2)
            pol_plain = CovariateGroupPolicy(mode='plain', label=f'plain-N{N}-K{K}-T{T_train}-r{rep}',
                                              **CQL_KWARGS).fit([traj], adj, block_labels)
            _seed_all(rep_seed + 3)
            pol_all = CovariateGroupPolicy(mode='all', label=f'all-N{N}-K{K}-T{T_train}-r{rep}',
                                            **CQL_KWARGS).fit([traj], adj, block_labels)
            _seed_all(rep_seed + 4)
            pol_ising = CovariateGroupPolicy(mode='ising', label=f'ising-N{N}-K{K}-T{T_train}-r{rep}',
                                              **CQL_KWARGS).fit([traj], adj, block_labels)

            reward_plain, curves_plain = deploy_from_warm_start(pol_plain, adj, block_labels, warm_state,
                                                                  T_TEST, N_EVAL_EPS, base_seed=777 + rep)
            reward_all, curves_all = deploy_from_warm_start(pol_all, adj, block_labels, warm_state,
                                                              T_TEST, N_EVAL_EPS, base_seed=888 + rep)
            reward_ising, curves_ising = deploy_from_warm_start(pol_ising, adj, block_labels, warm_state,
                                                                  T_TEST, N_EVAL_EPS, base_seed=999 + rep)

            gap_plain = reward_ising.mean() - reward_plain.mean()
            gap_all = reward_ising.mean() - reward_all.mean()
            print(f'    rep {rep}: dqn-plain={reward_plain.mean():.2f}  dqn-all={reward_all.mean():.2f}  '
                  f'q-ising={reward_ising.mean():.2f}  (ising-plain={gap_plain:+.2f}, ising-all={gap_all:+.2f})')

            rows.append(dict(
                N=N, K=K, T_train=T_train, replicate=rep,
                reward_plain_mean=reward_plain.mean(), reward_plain_se=reward_plain.std() / np.sqrt(N_EVAL_EPS),
                reward_all_mean=reward_all.mean(), reward_all_se=reward_all.std() / np.sqrt(N_EVAL_EPS),
                reward_ising_mean=reward_ising.mean(), reward_ising_se=reward_ising.std() / np.sqrt(N_EVAL_EPS),
                gap_ising_minus_plain=gap_plain, gap_ising_minus_all=gap_all,
            ))

            mean_curve_plain = curves_plain.mean(axis=0)
            mean_curve_all = curves_all.mean(axis=0)
            mean_curve_ising = curves_ising.mean(axis=0)
            for t in range(T_TEST):
                curve_rows.append(dict(
                    N=N, K=K, T_train=T_train, replicate=rep, period=t,
                    reward_plain=mean_curve_plain[t], reward_all=mean_curve_all[t],
                    reward_ising=mean_curve_ising[t],
                ))

    return pd.DataFrame(rows), pd.DataFrame(baseline_rows), pd.DataFrame(curve_rows), pd.DataFrame(baseline_curve_rows)


REQUIRED_RESULT_COLUMNS = {
    'N', 'K', 'T_train', 'replicate', 'reward_plain_mean', 'reward_all_mean',
    'reward_ising_mean', 'gap_ising_minus_plain', 'gap_ising_minus_all',
}


def _cache_is_valid(out_csv):
    if not os.path.exists(out_csv):
        return False
    # caches written before the curves files existed are stale
    curves_csv = out_csv.replace('_results.csv', '_curves.csv')
    baseline_curves_csv = out_csv.replace('_results.csv', '_baseline_curves.csv')
    if not (os.path.exists(curves_csv) and os.path.exists(baseline_curves_csv)):
        print(f'  Cache at {out_csv} is STALE (missing curves/baseline_curves companion file(s)) -- re-running.')
        return False
    try:
        cached = pd.read_csv(out_csv)
    except Exception:
        return False
    if not REQUIRED_RESULT_COLUMNS.issubset(cached.columns):
        missing = REQUIRED_RESULT_COLUMNS - set(cached.columns)
        print(f'  Cache at {out_csv} is STALE (missing columns {sorted(missing)}) -- re-running.')
        return False
    if set(cached['T_train'].unique()) != set(T_TRAIN_LEVELS):
        print(f'  Cache at {out_csv} is STALE (T_train levels differ from current config) -- re-running.')
        return False
    if len(cached[cached['T_train'] == T_TRAIN_LEVELS[0]]['replicate'].unique()) != N_REPLICATES:
        print(f'  Cache at {out_csv} is STALE (N_REPLICATES differs from current config) -- re-running.')
        return False
    return True


def main():
    print('=' * 70)
    print('Cluster sweep: dqn-plain vs dqn-all vs q-ising, CovariateGroupControllableEnv')
    print(f'N_LEVELS={N_LEVELS}  K_LEVELS={K_LEVELS}  T_TRAIN_LEVELS={T_TRAIN_LEVELS}  '
          f'N_REPLICATES={N_REPLICATES}  T_TEST={T_TEST}  N_EVAL_EPS={N_EVAL_EPS}')
    print(f'P_Z={P_Z}  P_BASE={P_BASE}  P_ESCALATED={P_ESCALATED}  P_IN={P_IN}  P_OUT={P_OUT}')
    print('=' * 70)

    all_dfs, all_baseline_dfs, all_curve_dfs, all_baseline_curve_dfs = [], [], [], []
    for N in N_LEVELS:
        for K in K_LEVELS:
            os.makedirs(os.path.join(THIS_DIR, 'cache'), exist_ok=True)
            out_csv = os.path.join(THIS_DIR, 'cache', f'cluster_sweep_N{N}_K{K}_results.csv')
            curves_csv = out_csv.replace('_results.csv', '_curves.csv')
            baselines_csv = out_csv.replace('_results.csv', '_baselines.csv')
            baseline_curves_csv = out_csv.replace('_results.csv', '_baseline_curves.csv')
            if not FORCE_RERUN and _cache_is_valid(out_csv):
                print(f'\nSkipping N={N} K={K} (already have valid {out_csv}; set FORCE_RERUN=1 to redo)')
                all_dfs.append(pd.read_csv(out_csv))
                all_baseline_dfs.append(pd.read_csv(baselines_csv))
                all_curve_dfs.append(pd.read_csv(curves_csv))
                all_baseline_curve_dfs.append(pd.read_csv(baseline_curves_csv))
                continue
            df, baseline_df, curve_df, baseline_curve_df = run_one_config(N, K)
            df.to_csv(out_csv, index=False)
            baseline_df.to_csv(baselines_csv, index=False)
            curve_df.to_csv(curves_csv, index=False)
            baseline_curve_df.to_csv(baseline_curves_csv, index=False)
            print(f'  Saved -> {out_csv} (+ _curves.csv, _baselines.csv, _baseline_curves.csv)')
            all_dfs.append(df)
            all_baseline_dfs.append(baseline_df)
            all_curve_dfs.append(curve_df)
            all_baseline_curve_dfs.append(baseline_curve_df)

    df = pd.concat(all_dfs, ignore_index=True)
    baseline_df = pd.concat(all_baseline_dfs, ignore_index=True)
    curve_df = pd.concat(all_curve_dfs, ignore_index=True)
    baseline_curve_df = pd.concat(all_baseline_curve_dfs, ignore_index=True)
    df.to_csv(os.path.join(THIS_DIR, 'cluster_sweep_all_results.csv'), index=False)
    baseline_df.to_csv(os.path.join(THIS_DIR, 'cluster_sweep_all_baselines.csv'), index=False)
    curve_df.to_csv(os.path.join(THIS_DIR, 'cluster_sweep_all_curves.csv'), index=False)
    baseline_curve_df.to_csv(os.path.join(THIS_DIR, 'cluster_sweep_all_baseline_curves.csv'), index=False)

    print(f'\n{"=" * 70}\nSUMMARY, per (N, K, T_train)\n{"=" * 70}')
    # mean and std/sem across replicates (not the within-replicate SE)
    summary = df.groupby(['N', 'K', 'T_train']).agg(
        mean_reward_plain=('reward_plain_mean', 'mean'),
        std_reward_plain=('reward_plain_mean', 'std'),
        mean_reward_all=('reward_all_mean', 'mean'),
        std_reward_all=('reward_all_mean', 'std'),
        mean_reward_ising=('reward_ising_mean', 'mean'),
        std_reward_ising=('reward_ising_mean', 'std'),
        mean_gap_ising_minus_plain=('gap_ising_minus_plain', 'mean'),
        std_gap_ising_minus_plain=('gap_ising_minus_plain', 'std'),
        mean_gap_ising_minus_all=('gap_ising_minus_all', 'mean'),
        std_gap_ising_minus_all=('gap_ising_minus_all', 'std'),
        n_replicates=('replicate', 'nunique'),
    ).reset_index()
    for col in ['reward_plain', 'reward_all', 'reward_ising', 'gap_ising_minus_plain', 'gap_ising_minus_all']:
        summary[f'sem_{col}'] = summary[f'std_{col}'] / np.sqrt(summary['n_replicates'])
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(THIS_DIR, 'cluster_sweep_summary.csv'), index=False)

    print(f'\n{"=" * 70}\nvs. BEST NON-ORACLE BASELINE, per (N, K)\n{"=" * 70}')
    for N in N_LEVELS:
        for K in K_LEVELS:
            bsub = baseline_df[(baseline_df['N'] == N) & (baseline_df['K'] == K) & (baseline_df['policy'] != 'oracle')]
            if len(bsub) == 0:
                continue
            best_name = bsub.loc[bsub['mean_deploy_reward'].idxmax(), 'policy']
            best_reward = bsub['mean_deploy_reward'].max()
            oracle_row = baseline_df[(baseline_df['N'] == N) & (baseline_df['K'] == K) & (baseline_df['policy'] == 'oracle')]
            oracle_reward = oracle_row['mean_deploy_reward'].iloc[0] if len(oracle_row) else float('nan')
            print(f'  N={N} K={K}  best baseline = "{best_name}" ({best_reward:.2f})  oracle={oracle_reward:.2f}')
            ssub = summary[(summary['N'] == N) & (summary['K'] == K)]
            for _, row in ssub.iterrows():
                print(f'    T_train={int(row["T_train"]):<5} plain={row["mean_reward_plain"]:.2f} '
                      f'({row["mean_reward_plain"] - best_reward:+.2f})   '
                      f'all={row["mean_reward_all"]:.2f} ({row["mean_reward_all"] - best_reward:+.2f})   '
                      f'ising={row["mean_reward_ising"]:.2f} ({row["mean_reward_ising"] - best_reward:+.2f})')

    _plot_summary(summary, baseline_df, FIGURES_DIR)
    _plot_curves(curve_df, baseline_curve_df, FIGURES_DIR)
    print(f'\nFigure -> {os.path.join(FIGURES_DIR, "cluster_sweep_summary.png")}, '
          f'{os.path.join(FIGURES_DIR, "cluster_sweep_curves.png")}')
    print('\nDone.')


def _plot_summary(summary, baseline_df, out_dir):
    Ks = sorted(summary['K'].unique())
    Ns = sorted(summary['N'].unique())
    fig, axes = plt.subplots(len(Ns), len(Ks), figsize=(4.5 * len(Ks), 4 * len(Ns)),
                              sharex=True, squeeze=False)

    for i, N in enumerate(Ns):
        for j, K in enumerate(Ks):
            ax = axes[i, j]
            ssub = summary[(summary['N'] == N) & (summary['K'] == K)]
            bsub = baseline_df[(baseline_df['N'] == N) & (baseline_df['K'] == K)]
            for _, brow in bsub.iterrows():
                ax.axhline(brow['mean_deploy_reward'], color=BASELINE_COLORS[brow['policy']], ls='--',
                           lw=1.1, alpha=0.7, label=BASELINE_DISPLAY[brow['policy']])
            # error bars: SEM across replicates
            ax.errorbar(ssub['T_train'], ssub['mean_reward_plain'], yerr=ssub['sem_reward_plain'],
                        fmt='o-', color='#9467bd', label='dqn-plain', ms=4, capsize=3)
            ax.errorbar(ssub['T_train'], ssub['mean_reward_all'], yerr=ssub['sem_reward_all'],
                        fmt='o-', color='#e377c2', label='dqn-all', ms=4, capsize=3)
            ax.errorbar(ssub['T_train'], ssub['mean_reward_ising'], yerr=ssub['sem_reward_ising'],
                        fmt='o-', color='#2ca02c', label='q-ising', ms=4, capsize=3)
            ax.set_xscale('log')
            ax.set_title(f'N={N}, K={K}', fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(Ns) - 1:
                ax.set_xlabel('T_train')
            if j == 0:
                ax.set_ylabel('Mean deploy reward')
            if i == 0 and j == len(Ks) - 1:
                ax.legend(fontsize=6, loc='best')

    plt.suptitle('dqn-plain / dqn-all / q-ising vs baselines, across N x K x T_train', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cluster_sweep_summary.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'cluster_sweep_summary.pdf'), bbox_inches='tight')
    plt.close(fig)


def _plot_curves(curve_df, baseline_curve_df, out_dir):
    """Per-period reward (mean +/- std across replicates) at the largest T_train, one panel per (N, K)."""
    Ks = sorted(curve_df['K'].unique())
    Ns = sorted(curve_df['N'].unique())
    T_train_max = curve_df['T_train'].max()
    fig, axes = plt.subplots(len(Ns), len(Ks), figsize=(4.5 * len(Ks), 4 * len(Ns)),
                              sharex=True, squeeze=False)

    for i, N in enumerate(Ns):
        for j, K in enumerate(Ks):
            ax = axes[i, j]
            csub = curve_df[(curve_df['N'] == N) & (curve_df['K'] == K) & (curve_df['T_train'] == T_train_max)]
            bsub = baseline_curve_df[(baseline_curve_df['N'] == N) & (baseline_curve_df['K'] == K)]
            for name in bsub['policy'].unique():
                brow = bsub[bsub['policy'] == name].sort_values('period')
                ax.plot(brow['period'], brow['reward'], ls='--', lw=1.1, alpha=0.7,
                        color=BASELINE_COLORS[name], label=BASELINE_DISPLAY[name])
            for col, color, label in [('reward_plain', '#9467bd', 'dqn-plain'),
                                       ('reward_all', '#e377c2', 'dqn-all'),
                                       ('reward_ising', '#2ca02c', 'q-ising')]:
                agg = csub.groupby('period')[col].agg(['mean', 'std']).reset_index()
                ax.plot(agg['period'], agg['mean'], 'o-', color=color, label=label, ms=3)
                ax.fill_between(agg['period'], agg['mean'] - agg['std'].fillna(0),
                                 agg['mean'] + agg['std'].fillna(0), color=color, alpha=0.15)
            ax.set_title(f'N={N}, K={K}', fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(Ns) - 1:
                ax.set_xlabel('Period')
            if j == 0:
                ax.set_ylabel('Reward per period')
            if i == 0 and j == len(Ks) - 1:
                ax.legend(fontsize=6, loc='best')

    plt.suptitle(f'Per-period reward trajectory at T_train={T_train_max} '
                 f'(shaded = std across replicates)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cluster_sweep_curves.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'cluster_sweep_curves.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
