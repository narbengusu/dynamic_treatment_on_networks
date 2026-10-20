#!/usr/bin/env python3
"""
UQ calibration on the plain-SIS N=10 network, from one trained Q-Ising ensemble
(N_ENSEMBLE CQL models on MCMC posterior draws). Run confirm_instability_sis.py first.
Checks: [1] corr(vote entropy, true value gap from known-SIS Monte Carlo rollouts);
[2] how often the ensemble majority action differs from the point-estimate (EMVS MAP) action;
[3] vote entropy of the same ensemble under majority-vote dynamics (misspecified) vs SIS.

Env vars (defaults): DATA_SEED 42, SKIP_PROB 0.6, COOLDOWN 2, T_TRAIN 100, WARM_START_T 30,
T_EVAL 30, N_EVAL_EPS 40, DISCOUNT 0.8, ALPHA_CQL 0.1, N_STEPS_MAX 10000,
N_STEPS_PER_EPOCH 500, PATIENCE 8, ROLLOUT_H 12, N_MC 300, N_ENSEMBLE 20.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch as _torch
import d3rlpy

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from tiny_env import (
    make_tiny_env, run_episode, random_skip_action_fn,
)
from majority_vote_env import (
    make_majority_env,
)
from qising.baselines import QIsingPolicy

logging.getLogger('d3rlpy').setLevel(logging.WARNING)

def _env(name, default, cast=int):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


SIS_PARAMS_PATH      = os.path.join(THIS_DIR, 'sis_network_params.npz')
MAJORITY_PARAMS_PATH = os.path.join(THIS_DIR, 'majority_network_params.npz')
DATA_SEED  = _env('DATA_SEED', 42)
SKIP_PROB  = _env('SKIP_PROB', 0.6, float)
COOLDOWN   = _env('COOLDOWN', 2)

T_TRAIN      = _env('T_TRAIN', 100)
WARM_START_T = _env('WARM_START_T', 30)
T_EVAL       = _env('T_EVAL', 30)
N_EVAL_EPS   = _env('N_EVAL_EPS', 40)

CQL_KWARGS = dict(
    discount=_env('DISCOUNT', 0.8, float), alpha_cql=_env('ALPHA_CQL', 0.1, float), verbose=False,
    n_steps_max=_env('N_STEPS_MAX', 10_000), n_steps_per_epoch=_env('N_STEPS_PER_EPOCH', 500),
    patience=_env('PATIENCE', 8),
)
GAMMA        = CQL_KWARGS['discount']
ROLLOUT_H    = _env('ROLLOUT_H', 12)
N_MC         = _env('N_MC', 300)
N_ENSEMBLE   = _env('N_ENSEMBLE', 20)


def _seed_all(s):
    np.random.seed(s); _torch.manual_seed(s); d3rlpy.seed(s)


# Known-dynamics oracle rollout (churn -> seed -> spread)

def _true_step_batch(states, actions, beta_vec, delta_vec, adj, log1mb, rng):
    """states: (M, N) bool. actions: (M,) int or None (list). Returns new (M, N) bool."""
    M, N = states.shape
    churn_draws = rng.random((M, N))
    churned = states & (churn_draws < delta_vec[None, :])
    new_states = states & ~churned

    for m in range(M):
        a = actions[m]
        if a is not None:
            new_states[m, a] = True

    weighted = new_states.astype(float) * log1mb[None, :]
    log_p_escape = weighted @ adj          # adj symmetric -> adj == adj.T
    p_escape = np.exp(log_p_escape)
    p_adopt = 1.0 - p_escape
    susceptible = ~new_states
    spread_draws = rng.random((M, N))
    newly = susceptible & (spread_draws < p_adopt)
    return new_states | newly


def oracle_value(state0, action0, beta_vec, delta_vec, adj, seed):
    """Mean discounted adopted-count over ROLLOUT_H periods after seeding `action0`, then no
    treatment; `seed` gives common random numbers across actions."""
    N = len(state0)
    log1mb = np.log(1.0 - np.minimum(beta_vec, 1.0 - 1e-15))
    rng = np.random.default_rng(seed)
    states = np.tile(state0, (N_MC, 1)).copy()
    actions = [action0] * N_MC
    cum = np.zeros(N_MC)
    disc = 1.0
    for _h in range(ROLLOUT_H):
        states = _true_step_batch(states, actions, beta_vec, delta_vec, adj, log1mb, rng)
        cum += disc * states.sum(axis=1)
        disc *= GAMMA
        actions = [None] * N_MC
    return float(cum.mean())


# Full ensemble vote tally

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


# Decision-point collection under a running policy, with cooldown

def collect_decision_points(env, ensemble_models, point_policy, T, cooldown, initial_cooldown,
                             beta_vec, delta_vec, adj, rollout_seed_base):
    records = []
    cooldown_left = initial_cooldown
    for t in range(T):
        obs = env._get_obs()
        avl_all = env.get_available_actions()
        avl = [a for a in avl_all if a < env.n_bins]

        if cooldown_left > 0:
            cooldown_left -= 1
            env.step(None)
            continue
        if len(avl) == 0:
            env.step(None)
            continue

        ba = obs['adopted']
        votes = ensemble_votes(ensemble_models, ba, env.n_bins, avl)
        majority_action = int(avl[np.argmax(votes[list(avl)])])
        point_action = point_policy(obs, env)
        entropy = normalized_entropy(votes, avl)

        seed = rollout_seed_base + t * 7919  # distinct per decision point, shared across actions
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

        cooldown_left = cooldown
        env.step(majority_action)
    return records


def collect_entropy_only(env, ensemble_models, T, cooldown, initial_cooldown):
    """Same decision-point loop, but only entropy (used for the misspecified
    deployment where the known-SIS oracle rollout doesn't apply)."""
    entropies = []
    cooldown_left = initial_cooldown
    for t in range(T):
        obs = env._get_obs()
        avl_all = env.get_available_actions()
        avl = [a for a in avl_all if a < env.n_bins]
        if cooldown_left > 0:
            cooldown_left -= 1
            env.step(None)
            continue
        if len(avl) == 0:
            env.step(None)
            continue
        ba = obs['adopted']
        votes = ensemble_votes(ensemble_models, ba, env.n_bins, avl)
        majority_action = int(avl[np.argmax(votes[list(avl)])])
        entropies.append(normalized_entropy(votes, avl))
        cooldown_left = cooldown
        env.step(majority_action)
    return entropies


def main():
    print('=' * 70)
    print('UQ calibration — plain SIS, N=10')
    print('=' * 70)

    p = np.load(SIS_PARAMS_PATH)
    adj = p['adj']
    N = int(p['N'])
    beta_vec = p['beta_vec']
    delta_vec = p['delta_vec']
    block_labels = np.arange(N)
    print(f'Loaded {SIS_PARAMS_PATH}: N={N}, natural mean rate={float(p["natural_mean_rate"]):.3f}')

    # train: one behavioral trajectory -> Q-Ising ensemble + point-estimate MAP
    print(f'\n[1] Training data: T_train={T_TRAIN}, single trajectory (seed={DATA_SEED})')
    env = make_tiny_env(adj, beta_vec, delta_vec, T=T_TRAIN, seed=DATA_SEED, init_rate=0.0)
    ep_rng = np.random.default_rng(DATA_SEED + 1)
    action_fn = random_skip_action_fn(env.n_bins, SKIP_PROB, ep_rng)
    traj = run_episode(env, action_fn, T=T_TRAIN, cooldown=COOLDOWN)

    print('  Fitting Q-Ising ensemble (n_ensemble=20 CQL models over MCMC posterior)...')
    _seed_all(DATA_SEED + 2)
    pol_ensemble = QIsingPolicy(
        ising_augmentation=True, ensemble=True, n_ensemble=N_ENSEMBLE,
        label='sis-uq-ensemble', **CQL_KWARGS,
    ).fit([traj], adj, block_labels)

    print('  Fitting point-estimate Q-Ising (single EMVS MAP model)...')
    _seed_all(DATA_SEED + 3)
    pol_point = QIsingPolicy(
        ising_augmentation=True, ensemble=False,
        label='sis-uq-point', **CQL_KWARGS,
    ).fit([traj], adj, block_labels)

    ensemble_models = pol_ensemble._ensemble_models
    assert len(ensemble_models) == N_ENSEMBLE

    # evaluation rollouts under the ensemble policy
    print(f'\n[2] Evaluation: {N_EVAL_EPS} episodes x T_eval={T_EVAL} '
          f'(warm_start={WARM_START_T}), oracle rollout H={ROLLOUT_H}, N_MC={N_MC} ...')
    all_records = []
    rng = np.random.default_rng(DATA_SEED + 100)
    for ep in range(N_EVAL_EPS):
        ep_seed = int(rng.integers(0, 1_000_000))
        eval_env = make_tiny_env(adj, beta_vec, delta_vec, T=WARM_START_T + T_EVAL,
                                  seed=ep_seed, init_rate=0.0)
        warm_rng = np.random.default_rng(ep_seed + 1)
        warm_action_fn = random_skip_action_fn(eval_env.n_bins, SKIP_PROB, warm_rng)
        warm_traj = run_episode(eval_env, warm_action_fn, T=WARM_START_T, cooldown=COOLDOWN, reset=True)

        recs = collect_decision_points(
            eval_env, ensemble_models, pol_point, T_EVAL, COOLDOWN,
            warm_traj['cooldown_left_at_end'], beta_vec, delta_vec, adj,
            rollout_seed_base=DATA_SEED + 1000 * (ep + 1),
        )
        for r in recs:
            r['episode'] = ep
        all_records.extend(recs)
        if (ep + 1) % 10 == 0:
            print(f'  episode {ep + 1}/{N_EVAL_EPS}: {len(all_records)} decision points so far')

    df = pd.DataFrame(all_records)
    out_csv = os.path.join(THIS_DIR, 'uq_calibration_sis_decisions.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nCollected {len(df)} decision points ({df["n_avl"].eq(1).sum()} with a single '
          f'available bin, excluded from calibration). Saved -> {out_csv}')

    sub = df[df['n_avl'] >= 2].copy()

    # check 1: calibration
    print(f'\n{"=" * 70}\n[CHECK 1] Calibration: dispersion vs true value gap  (n={len(sub)})\n{"=" * 70}')
    pear_r, pear_p = stats.pearsonr(sub['entropy'], sub['true_gap'])
    spear_r, spear_p = stats.spearmanr(sub['entropy'], sub['true_gap'])
    print(f'  Pearson  r = {pear_r:+.4f}  (p={pear_p:.2e})')
    print(f'  Spearman rho = {spear_r:+.4f}  (p={spear_p:.2e})')
    print(f'  gap: mean={sub["true_gap"].mean():.4f}  median={sub["true_gap"].median():.4f}  '
          f'max={sub["true_gap"].max():.4f}')
    print(f'  entropy: mean={sub["entropy"].mean():.4f}  median={sub["entropy"].median():.4f}')
    if pear_r < -0.3 and pear_p < 0.01:
        print('  VERDICT: supports the Fig.6 claim — dispersion is high precisely where '
              'the decision barely matters.')
    else:
        print('  VERDICT: correlation too weak/wrong-signed to support the claim as stated — '
              'report honestly.')

    _plot_calibration(sub, FIGURES_DIR)

    # check 2: decision change
    print(f'\n{"=" * 70}\n[CHECK 2] Decision change: ensemble vs. point-estimate  (n={len(sub)})\n{"=" * 70}')
    differs = sub['majority_action'] != sub['point_action']
    frac_differs = float(differs.mean())
    print(f'  Fraction of decision points where ensemble != point-estimate: {frac_differs:.3f} '
          f'({int(differs.sum())}/{len(sub)})')
    if differs.sum() > 0:
        diff_sub = sub[differs]
        value_diff = diff_sub['majority_value'] - diff_sub['point_value']
        print(f'  True-value difference when they differ (ensemble - point-estimate):')
        print(f'    mean={value_diff.mean():+.4f}  median={value_diff.median():+.4f}  '
              f'std={value_diff.std():.4f}')
        print(f'    ensemble better in {(value_diff > 0).mean():.2%} of differing cases')
        print(f'    mean true_gap AT differing points: {diff_sub["true_gap"].mean():.4f} '
              f'vs. overall mean gap {sub["true_gap"].mean():.4f}')
    else:
        print('  Ensemble NEVER changed the decision relative to the point-estimate policy in '
              'this run. Report honestly; lean the UQ case on calibration + diagnostic value.')

    # check 3: misspecification cross-check
    print(f'\n{"=" * 70}\n[CHECK 3] Misspecification cross-check: entropy under SIS vs. majority-vote\n{"=" * 70}')
    if os.path.exists(MAJORITY_PARAMS_PATH):
        mp = np.load(MAJORITY_PARAMS_PATH)
        maj_adj = mp['adj']
        same_graph = bool(np.array_equal(maj_adj, adj))
        print(f'  Loaded {MAJORITY_PARAMS_PATH} (same adjacency as SIS network: {same_graph})')
        maj_env_params = dict(p_high_vec=mp['p_high_vec'], p_low_vec=mp['p_low_vec'],
                               theta_vec=mp['theta_vec'])

        maj_entropies = []
        rng2 = np.random.default_rng(DATA_SEED + 200)
        for ep in range(N_EVAL_EPS):
            ep_seed = int(rng2.integers(0, 1_000_000))
            maj_env = make_majority_env(maj_adj, maj_env_params, T=WARM_START_T + T_EVAL,
                                         seed=ep_seed, init_rate=0.0, p_high_scale=1.0)
            warm_rng = np.random.default_rng(ep_seed + 1)
            warm_action_fn = random_skip_action_fn(maj_env.n_bins, SKIP_PROB, warm_rng)
            warm_traj = run_episode(maj_env, warm_action_fn, T=WARM_START_T, cooldown=COOLDOWN, reset=True)
            ents = collect_entropy_only(maj_env, ensemble_models, T_EVAL, COOLDOWN,
                                         warm_traj['cooldown_left_at_end'])
            maj_entropies.extend(ents)

        maj_entropies = np.array([e for e in maj_entropies if not np.isnan(e)])
        sis_entropies = sub['entropy'].dropna().values

        print(f'  SIS (well-specified)     entropy: mean={sis_entropies.mean():.4f}  '
              f'median={np.median(sis_entropies):.4f}  n={len(sis_entropies)}')
        print(f'  Majority-vote (misspec.) entropy: mean={maj_entropies.mean():.4f}  '
              f'median={np.median(maj_entropies):.4f}  n={len(maj_entropies)}')
        u_stat, u_p = stats.mannwhitneyu(maj_entropies, sis_entropies, alternative='greater')
        print(f'  Mann-Whitney U (H1: misspecified > well-specified): U={u_stat:.1f}  p={u_p:.2e}')
        if maj_entropies.mean() > sis_entropies.mean() and u_p < 0.05:
            print('  VERDICT: supports the prediction — entropy is systematically higher under '
                  'misspecification.')
        else:
            print('  VERDICT: prediction NOT supported here — report honestly.')

        pd.DataFrame({'entropy': sis_entropies, 'condition': 'sis_well_specified'}).to_csv(
            os.path.join(THIS_DIR, 'uq_entropy_sis.csv'), index=False)
        pd.DataFrame({'entropy': maj_entropies, 'condition': 'majority_misspecified'}).to_csv(
            os.path.join(THIS_DIR, 'uq_entropy_majority_misspecified.csv'), index=False)
        _plot_entropy_comparison(sis_entropies, maj_entropies, FIGURES_DIR)
    else:
        print(f'  {MAJORITY_PARAMS_PATH} not found — skipping check 3. Run '
              'confirm_instability_majority.py first.')

    print('\nDone.')


def _plot_calibration(sub, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub['entropy'], sub['true_gap'], s=14, alpha=0.4, color='#2ca02c')
    r, _ = stats.pearsonr(sub['entropy'], sub['true_gap'])
    if len(sub) > 1:
        m, b = np.polyfit(sub['entropy'], sub['true_gap'], 1)
        xs = np.linspace(sub['entropy'].min(), sub['entropy'].max(), 50)
        ax.plot(xs, m * xs + b, color='k', lw=1.5, ls='--', label=f'fit (r={r:+.3f})')
    ax.set_xlabel('Vote dispersion (normalized entropy, 20 agents)')
    ax.set_ylabel('True value gap (oracle-best − majority-vote bin)')
    ax.set_title('UQ calibration: SIS, N=10', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'uq_calibration_sis_scatter.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'uq_calibration_sis_scatter.pdf'), bbox_inches='tight')
    plt.close(fig)


def _plot_entropy_comparison(sis_ent, maj_ent, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(sis_ent, bins=bins, alpha=0.55, density=True, label='SIS (well-specified)', color='#2ca02c')
    ax.hist(maj_ent, bins=bins, alpha=0.55, density=True, label='Majority-vote (misspecified)', color='#d62728')
    ax.axvline(sis_ent.mean(), color='#2ca02c', ls='--', lw=1.5)
    ax.axvline(maj_ent.mean(), color='#d62728', ls='--', lw=1.5)
    ax.set_xlabel('Vote dispersion (normalized entropy)')
    ax.set_ylabel('Density')
    ax.set_title('Ensemble entropy: well-specified vs. misspecified deployment', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'uq_entropy_misspecification.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'uq_entropy_misspecification.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
