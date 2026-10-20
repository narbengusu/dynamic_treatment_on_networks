#!/usr/bin/env python3
"""
Instability check for the plain SIS network (N=10, one bin per node) using tiny_env.make_tiny_env;
writes sis_network_params.npz and sis_instability_summary.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from tiny_env import (
    build_tiny_network, make_tiny_env, run_episode, random_skip_action_fn,
    draw_heterogeneous_params, calibrate_beta_scale, per_node_R,
)

N_NODES   = 10
GRAPH_P   = 0.35
DATA_SEED = 42

T_LEVELS   = [10, 20, 50, 100, 500]
N_REPS     = 300
SKIP_PROB  = 0.6
COOLDOWN   = 2
BURN_IN    = 5
TARGET_MEAN_RATE = 0.25


def main():
    print('=' * 70)
    print(f'Plain SIS ("unstable") network: N={N_NODES}, one bin per node')
    print('=' * 70)

    adj = build_tiny_network(N=N_NODES, p=GRAPH_P, seed=DATA_SEED)
    degree = adj.sum(axis=1)
    print(f'\nDegree sequence: {degree.astype(int).tolist()}')

    beta_vec, delta_vec = draw_heterogeneous_params(N_NODES, seed=DATA_SEED)

    print('\n[1] Calibrating beta_scale against the ACTUAL random-policy loop...')
    scale, achieved_rate = calibrate_beta_scale(
        adj, beta_vec, delta_vec, skip_prob=SKIP_PROB, cooldown=COOLDOWN,
        target_mean_rate=TARGET_MEAN_RATE, T=100, n_reps=150, seed=DATA_SEED,
    )
    beta_eff = np.clip(beta_vec * scale, 0.0, 1.0 - 1e-9)
    print(f'  scale={scale:.3f}  ->  mean adoption rate = {achieved_rate:.3f}')

    R = per_node_R(adj, beta_eff, delta_vec)
    print('\n  Per-node calibration:')
    print(f'  {"node":>4} {"deg":>4} {"beta":>7} {"delta":>7} {"R_i":>7}')
    for i in range(N_NODES):
        print(f'  {i:>4} {int(degree[i]):>4} {beta_eff[i]:>7.3f} {delta_vec[i]:>7.3f} {R[i]:>7.2f}')

    params_path = os.path.join(THIS_DIR, 'sis_network_params.npz')
    np.savez(params_path, adj=adj, beta_scale=scale, beta_vec=beta_eff, delta_vec=delta_vec,
             natural_mean_rate=achieved_rate, skip_prob=SKIP_PROB, cooldown=COOLDOWN,
             N=N_NODES, graph_p=GRAPH_P, data_seed=DATA_SEED)
    print(f'\nCalibrated network saved -> {params_path}')

    print(f'\n[2] Instability sweep (random behavioral policy, skip_prob={SKIP_PROB}, '
          f'cooldown={COOLDOWN}, {N_REPS} reps/T)...')
    rows = []
    example_curves = {}
    rng = np.random.default_rng(DATA_SEED + 999)
    for T in T_LEVELS:
        curves = np.zeros((N_REPS, T))
        final_adopted = np.zeros(N_REPS, dtype=int)
        ever_crashed  = np.zeros(N_REPS, dtype=bool)
        for rep in range(N_REPS):
            env = make_tiny_env(adj, beta_eff, delta_vec, T=T,
                                 seed=int(rng.integers(0, 1_000_000)), init_rate=0.0)
            ep_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
            action_fn = random_skip_action_fn(env.n_bins, SKIP_PROB, ep_rng)
            traj = run_episode(env, action_fn, T=T, cooldown=COOLDOWN)
            rates = traj['adopted'].mean(axis=1)
            curves[rep] = rates
            final_adopted[rep] = int(traj['adopted'][-1].sum())
            if T > BURN_IN:
                ever_crashed[rep] = (traj['adopted'][BURN_IN:].sum(axis=1) == 0).any()

        example_curves[T] = curves
        extinct_at_end = float((final_adopted == 0).mean())
        crash_rate = float(ever_crashed.mean()) if T > BURN_IN else float('nan')
        rows.append({
            'T': T,
            'mean_final_adoption_rate': float((final_adopted / N_NODES).mean()),
            'std_final_adoption_rate':  float((final_adopted / N_NODES).std()),
            'p_extinct_at_end': extinct_at_end,
            'p_ever_crashed_post_burnin': crash_rate,
        })
        print(f'  T={T:4d}  mean_final_rate={rows[-1]["mean_final_adoption_rate"]:.3f}'
              f'  std={rows[-1]["std_final_adoption_rate"]:.3f}'
              f'  P(extinct@T)={extinct_at_end:.3f}'
              f'  P(ever crashed)={crash_rate:.3f}')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(THIS_DIR, 'sis_instability_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nSummary saved -> {out_csv}')
    print(df.to_string(index=False))

    longest = df.iloc[-1]
    print('\n' + '=' * 70)
    if longest['std_final_adoption_rate'] > 0.12 and longest['p_ever_crashed_post_burnin'] > 0.3:
        print(f'VERDICT: consistent with instability at T={int(longest["T"])} — '
              f'std={longest["std_final_adoption_rate"]:.3f}, '
              f'P(ever crashed)={longest["p_ever_crashed_post_burnin"]:.2f}.')
    else:
        print(f'VERDICT: NOT clearly unstable — recalibrate before proceeding.')
    print('=' * 70)

    _plot_sample_trajectories(example_curves[T_LEVELS[-1]], T_LEVELS[-1], FIGURES_DIR)


def _plot_sample_trajectories(curves, T, out_dir, n_show=8):
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.linspace(0, curves.shape[0] - 1, n_show).astype(int)
    for i in idx:
        ax.plot(np.arange(T), curves[i], lw=1.2, alpha=0.8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Adoption rate')
    ax.set_title(f'Plain SIS sample trajectories (T={T}, N={N_NODES})', fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sis_sample_trajectories.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'sis_sample_trajectories.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
