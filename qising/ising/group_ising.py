"""
Group-level dynamic Ising model (one parameter set per group g(j)), fitted by logistic
regression, EMVS (spike-and-slab) or MCMC (PyMC):

    logit P(y_{t+1,j}=1) = intercept_g + rho_g*y_{t,j} + beta_treat_g*1{a_t=j}
                           + spillover_g*|{k in N(j): a_t=k}| + sum_m gamma_{g,m}*C_{j,m}

with C_{j,m} the count (normalize=False, default; natural for SIS) or degree-normalized
share (normalize=True) of adopted neighbors in group m.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any



def _compute_posterior_inclusion_prob(
    gamma_jk: float,
    pi_j: float,
    v0: float,
    v1: float
) -> float:
    """Posterior inclusion probability z_jk (E-step), computed in log space."""
    log_phi_v1 = -0.5 * gamma_jk**2 / v1 - 0.5 * np.log(2 * np.pi * v1)
    log_phi_v0 = -0.5 * gamma_jk**2 / v0 - 0.5 * np.log(2 * np.pi * v0)
    
    log_num = np.log(pi_j + 1e-10) + log_phi_v1
    log_denom_term2 = np.log(1 - pi_j + 1e-10) + log_phi_v0
    
    log_denom = np.logaddexp(log_num, log_denom_term2)
    
    z_jk = np.exp(log_num - log_denom)
    
    return np.clip(z_jk, 1e-10, 1 - 1e-10)


def _compute_effective_variance(z_jk: float, v0: float, v1: float) -> float:
    """Effective variance for the M-step penalty: (z/v1 + (1-z)/v0)^{-1}."""
    return 1.0 / (z_jk / v1 + (1 - z_jk) / v0)


def _penalized_logistic_loss(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_baseline: int,
    effective_variances: np.ndarray,
    sigma_sq: float
) -> float:
    """Penalized negative log-likelihood. theta = [baseline params (Gaussian prior sigma_sq), couplings]."""
    eta = X @ theta
    
    eta = np.clip(eta, -30, 30)
    
    p = expit(eta)
    nll = -np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    
    beta_penalty = 0.5 * np.sum(theta[:n_baseline]**2) / sigma_sq
    
    gamma = theta[n_baseline:]
    if len(gamma) > 0 and len(effective_variances) > 0:
        gamma_penalty = 0.5 * np.sum(gamma**2 / effective_variances)
    else:
        gamma_penalty = 0.0
    
    return nll + beta_penalty + gamma_penalty


def _penalized_logistic_gradient(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_baseline: int,
    effective_variances: np.ndarray,
    sigma_sq: float
) -> np.ndarray:
    """Gradient of the penalized negative log-likelihood."""
    eta = X @ theta
    eta = np.clip(eta, -30, 30)
    p = expit(eta)
    
    grad_nll = X.T @ (p - y)
    
    grad_penalty = np.zeros_like(theta)
    grad_penalty[:n_baseline] = theta[:n_baseline] / sigma_sq
    
    if len(theta) > n_baseline and len(effective_variances) > 0:
        grad_penalty[n_baseline:] = theta[n_baseline:] / effective_variances
    
    return grad_nll + grad_penalty


def _gaussian_density(x: float, mu: float, var: float) -> float:
    """Compute Gaussian density φ(x; μ, v)."""
    if var <= 0:
        raise ValueError("Variance must be positive")
    return np.exp(-0.5 * (x - mu)**2 / var) / np.sqrt(2 * np.pi * var)




class _DummyModel:
    """Single-class fallback when a group has only one outcome class."""
    def __init__(self, constant_class: int, n_features: int):
        self.constant_class = constant_class
        self.coef_ = np.zeros((1, n_features))
        self.intercept_ = np.array([10.0 if constant_class == 1 else -10.0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.constant_class == 1:
            return np.tile([0.0, 1.0], (n, 1))
        return np.tile([1.0, 0.0], (n, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.constant_class)


def fit_group_ising_logistic(
    X_array: np.ndarray,
    y_array: np.ndarray,
    treated: np.ndarray,
    A: np.ndarray,
    group_labels: np.ndarray,
    normalize: bool = False,
    **logistic_kwargs
) -> Dict[str, Any]:
    """
    Fit the group-level Ising model by logistic regression.

    X_array, y_array : (m, T, N) states at t and outcomes at t+1.
    treated : (m, T) treated node indices (-1 for none) or (m, T, N) indicator.
    A : (N, N) adjacency; group_labels : (N,) group membership (re-indexed internally).
    **logistic_kwargs go to sklearn LogisticRegression.
    Returns a dict with 'models' (one per group), group indexing ('g_idx', 'groups',
    'n_groups'), cached topology ('neighbor_lists', 'neigh_g_idx', 'degrees') and
    'extracted' group-level parameter arrays.
    """
    if X_array.ndim != 3 or y_array.ndim != 3:
        raise ValueError("X_array and y_array must be 3D arrays with shape (m, T, N).")
    m, T, N = X_array.shape
    if y_array.shape != (m, T, N):
        raise ValueError("y_array must have the same shape as X_array.")
    if group_labels.shape != (N,):
        raise ValueError(f"group_labels must have shape (N,)={N}, got {group_labels.shape}.")

    # Map group labels to contiguous indices 0..G-1
    groups = np.sort(np.unique(group_labels))
    G = len(groups)
    group_to_idx = {g: i for i, g in enumerate(groups)}
    g_idx = np.array([group_to_idx[group_labels[j]] for j in range(N)], dtype=int)

    # Standardize treated into (m, T, N) indicator
    if treated.ndim == 2:
        treated_indicator = np.zeros((m, T, N), dtype=bool)
        for ep in range(m):
            for t in range(T):
                idx = int(treated[ep, t])
                if 0 <= idx < N:
                    treated_indicator[ep, t, idx] = True
    elif treated.ndim == 3:
        treated_indicator = treated.astype(bool)
    else:
        raise ValueError("treated must be shape (m,T) or (m,T,N).")

    # Precompute per-node neighbor lists, their group indices, and degrees
    neighbor_lists = [np.nonzero(A[j] != 0)[0] for j in range(N)]
    neigh_g_idx = [g_idx[neighbor_lists[j]] for j in range(N)]
    degrees = np.array([len(neighbor_lists[j]) for j in range(N)], dtype=float)

    # Feature layout per row: [y_jt, treat_jt, spillover_jt, f_j0, ..., f_j{G-1}]
    n_features = 3 + G

    # Accumulate design rows and labels per group
    X_per_group = [[] for _ in range(G)]
    y_per_group = [[] for _ in range(G)]

    for j in range(N):
        gi = g_idx[j]
        neigh = neighbor_lists[j]
        ng = neigh_g_idx[j]
        d = len(neigh)
        Nj = degrees[j]

        for ep in range(m):
            for t in range(T):
                row = np.zeros(n_features)

                row[0] = X_array[ep, t, j]               # autoregressive
                row[1] = float(treated_indicator[ep, t, j])  # direct treatment

                if d > 0:
                    row[2] = treated_indicator[ep, t, neigh].sum()  # spillover count
                    neigh_states = X_array[ep, t, neigh]
                    for gm in range(G):
                        mask = (ng == gm)
                        row[3 + gm] = neigh_states[mask].sum() / (Nj if normalize else 1.0)

                X_per_group[gi].append(row)
                y_per_group[gi].append(int(y_array[ep, t, j]))

    # Default LR hyperparameters
    lr_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    lr_params.update(logistic_kwargs)

    models = []
    intercepts = np.zeros(G)
    rho_coeffs = np.zeros(G)
    beta_treat_coeffs = np.zeros(G)
    spillover_coeffs = np.zeros(G)
    coupling_matrix = np.zeros((G, G))  # coupling_matrix[k, m] = gamma_{k,m}

    for gi in range(G):
        X_g = np.array(X_per_group[gi])   # (n_rows_g, n_features)
        y_g = np.array(y_per_group[gi])   # (n_rows_g,)

        unique_classes = np.unique(y_g)
        if len(unique_classes) < 2:
            model = _DummyModel(int(unique_classes[0]), n_features)
        else:
            model = LogisticRegression(**lr_params)
            model.fit(X_g, y_g)

        models.append(model)

        coefs = model.coef_[0]             # (n_features,)
        intercepts[gi] = model.intercept_[0]
        rho_coeffs[gi] = coefs[0]
        beta_treat_coeffs[gi] = coefs[1]
        spillover_coeffs[gi] = coefs[2]
        coupling_matrix[gi, :] = coefs[3:3 + G]

    extracted = {
        'intercept_mean': intercepts,       # (G,)
        'rho_mean': rho_coeffs,             # (G,)
        'beta_treat_mean': beta_treat_coeffs,  # (G,)
        'spillover_mean': spillover_coeffs,    # (G,)
        'coupling_matrix_mean': coupling_matrix,  # (G, G)
    }

    return {
        'method': 'group_logistic',
        'models': models,
        'group_labels': group_labels,
        'g_idx': g_idx,
        'groups': groups,
        'n_groups': G,
        'normalize': normalize,
        'neighbor_lists': neighbor_lists,
        'neigh_g_idx': neigh_g_idx,
        'degrees': degrees,
        'extracted': extracted,
    }


def predict_group_ising_logistic(
    X_t: np.ndarray,
    action: int,
    result: Dict[str, Any],
    A: np.ndarray,
    return_proba: bool = True
) -> np.ndarray:
    """
    Next-state probabilities from a fitted group logistic model.
    X_t : (N,) state; action : treated node or -1; A : adjacency (topology is cached in
    result). return_proba=False returns binary {-1, 1}.
    """
    models = result['models']
    g_idx = result['g_idx']
    G = result['n_groups']
    normalize = result.get('normalize', False)
    neighbor_lists = result['neighbor_lists']
    neigh_g_idx = result['neigh_g_idx']
    degrees = result['degrees']
    N = len(X_t)

    n_features = 3 + G
    predictions = np.zeros(N)

    # Treated indicator for spillover computation
    treated_array = np.zeros(N)
    if 0 <= action < N:
        treated_array[action] = 1.0

    for j in range(N):
        gi = g_idx[j]
        neigh = neighbor_lists[j]
        ng = neigh_g_idx[j]
        d = len(neigh)
        Nj = degrees[j]

        features = np.zeros(n_features)
        features[0] = X_t[j]                   # autoregressive
        features[1] = float(action == j)        # direct treatment

        if d > 0:
            features[2] = treated_array[neigh].sum()  # spillover count
            neigh_states = X_t[neigh]
            for gm in range(G):
                mask = (ng == gm)
                features[3 + gm] = neigh_states[mask].sum() / (Nj if normalize else 1.0)

        model = models[gi]
        if return_proba:
            predictions[j] = model.predict_proba(features.reshape(1, -1))[0, 1]
        else:
            pred = model.predict(features.reshape(1, -1))[0]
            predictions[j] = 2 * int(pred) - 1

    return predictions


def fit_group_ising_emvs(
    X_array: np.ndarray,
    y_array: np.ndarray,
    treated: np.ndarray,
    A: np.ndarray,
    group_labels: np.ndarray,
    v0: float = 0.01,
    v1: float = 10.0,
    c: float = 1.0,
    sigma_sq: float = 10.0,
    max_em_iter: int = 50,
    em_tol: float = 1e-4,
    normalize: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit the group-level Ising model with EMVS (spike-and-slab). Each group has
    theta_g = [intercept, rho, beta_treat, spillover, gamma_{g,0..G-1}]; EM alternates
    a penalized logistic M-step and an E-step for the inclusion probabilities, with
    coupling prior pi_g = min(c / B_g, 0.99) (B_g = group size).

    X_array, y_array : (m, T, N); treated : (m, T) node indices or (m, T, N) indicator;
    A : (N, N); group_labels : (N,). v0 << v1 are spike/slab variances, sigma_sq the
    baseline prior variance, max_em_iter/em_tol the EM stopping rule.
    Returns a dict ('group_params', 'inclusion_probs', 'effective_variances', 'converged',
    'extracted', 'hyperparameters', plus the topology keys of the logistic fit).
    """
    if X_array.ndim != 3 or y_array.ndim != 3:
        raise ValueError("X_array and y_array must be 3D arrays with shape (m, T, N).")
    m, T, N = X_array.shape
    if y_array.shape != (m, T, N):
        raise ValueError("y_array must have the same shape as X_array.")
    if group_labels.shape != (N,):
        raise ValueError(f"group_labels must have shape ({N},), got {group_labels.shape}.")

    groups = np.sort(np.unique(group_labels))
    G = len(groups)
    group_to_idx = {g: i for i, g in enumerate(groups)}
    g_idx = np.array([group_to_idx[group_labels[j]] for j in range(N)], dtype=int)

    bin_sizes = np.bincount(g_idx, minlength=G).astype(float)
    pi_coup = np.minimum(c / bin_sizes, 0.99)  # shape (G,), pi_g = c / B_g

    if treated.ndim == 2:
        treated_indicator = np.zeros((m, T, N), dtype=bool)
        for ep in range(m):
            for t in range(T):
                idx = int(treated[ep, t])
                if 0 <= idx < N:
                    treated_indicator[ep, t, idx] = True
    elif treated.ndim == 3:
        treated_indicator = treated.astype(bool)
    else:
        raise ValueError("treated must be shape (m,T) or (m,T,N).")

    neighbor_lists = [np.nonzero(A[j] != 0)[0] for j in range(N)]
    neigh_g_idx = [g_idx[neighbor_lists[j]] for j in range(N)]
    degrees = np.array([len(neighbor_lists[j]) for j in range(N)], dtype=float)

    # Feature layout: [intercept, y_jt, treat_jt, spillover_jt, count_g0, ..., count_g{G-1}]
    n_baseline = 4  # intercept + rho + beta_treat + spillover
    n_features = n_baseline + G

    X_per_group = [[] for _ in range(G)]
    y_per_group = [[] for _ in range(G)]

    for j in range(N):
        gi = g_idx[j]
        neigh = neighbor_lists[j]
        ng = neigh_g_idx[j]
        d = len(neigh)
        Nj = degrees[j]

        for ep in range(m):
            for t in range(T):
                row = np.zeros(n_features)
                row[0] = 1.0
                row[1] = X_array[ep, t, j]
                row[2] = float(treated_indicator[ep, t, j])
                if d > 0:
                    row[3] = treated_indicator[ep, t, neigh].sum()
                    neigh_states = X_array[ep, t, neigh]
                    for gm in range(G):
                        mask = (ng == gm)
                        row[n_baseline + gm] = neigh_states[mask].sum() / (Nj if normalize else 1.0)
                X_per_group[gi].append(row)
                y_per_group[gi].append(int(y_array[ep, t, j]))

    group_params = {}
    inclusion_probs = np.zeros((G, G))
    effective_variances_dict = {}
    em_iterations_list = []
    converged_list = []

    intercepts = np.zeros(G)
    rho_coeffs = np.zeros(G)
    beta_treat_coeffs = np.zeros(G)
    spillover_coeffs = np.zeros(G)
    coupling_matrix = np.zeros((G, G))

    for gi in range(G):
        X_g = np.array(X_per_group[gi])
        y_g = np.array(y_per_group[gi])

        if verbose:
            print(f"  Group {gi} ({len(y_g)} rows)...")

        pi_g = pi_coup[gi]

        unique_classes = np.unique(y_g)
        if len(unique_classes) < 2:
            theta = np.zeros(n_features)
            theta[0] = 10.0 if unique_classes[0] == 1 else -10.0
            group_params[gi] = theta
            effective_variances_dict[gi] = np.full(G, v0)
            inclusion_probs[gi, :] = pi_g
            em_iterations_list.append(0)
            converged_list.append(True)
            intercepts[gi] = theta[0]
            continue

        theta = np.zeros(n_features)
        z_g = np.full(G, pi_g)
        effective_variances = np.array([_compute_effective_variance(z, v0, v1) for z in z_g])

        converged = False
        for em_iter in range(max_em_iter):
            theta_old = theta.copy()

            # M-step
            opt = minimize(
                _penalized_logistic_loss,
                theta,
                args=(X_g, y_g, n_baseline, effective_variances, sigma_sq),
                method='L-BFGS-B',
                jac=_penalized_logistic_gradient,
                options={'maxiter': 200, 'disp': False},
            )
            theta = opt.x

            # E-step
            gamma_g = theta[n_baseline:]
            for gm in range(G):
                z_g[gm] = _compute_posterior_inclusion_prob(gamma_g[gm], pi_g, v0, v1)
            effective_variances = np.array([_compute_effective_variance(z, v0, v1) for z in z_g])

            if np.max(np.abs(theta - theta_old)) < em_tol:
                converged = True
                if verbose:
                    print(f"    Converged after {em_iter + 1} iterations")
                break

        if not converged and verbose:
            print(f"    Did not converge after {max_em_iter} iterations")

        em_iterations_list.append(em_iter + 1)
        converged_list.append(converged)
        group_params[gi] = theta
        effective_variances_dict[gi] = effective_variances
        inclusion_probs[gi, :] = z_g
        intercepts[gi] = theta[0]
        rho_coeffs[gi] = theta[1]
        beta_treat_coeffs[gi] = theta[2]
        spillover_coeffs[gi] = theta[3]
        coupling_matrix[gi, :] = theta[n_baseline:]

    extracted = {
        'intercept_mean': intercepts,
        'rho_mean': rho_coeffs,
        'beta_treat_mean': beta_treat_coeffs,
        'spillover_mean': spillover_coeffs,
        'coupling_matrix_mean': coupling_matrix,
    }

    return {
        'method': 'group_emvs',
        'group_labels': group_labels,
        'g_idx': g_idx,
        'groups': groups,
        'n_groups': G,
        'normalize': normalize,
        'neighbor_lists': neighbor_lists,
        'neigh_g_idx': neigh_g_idx,
        'degrees': degrees,
        'group_params': group_params,
        'inclusion_probs': inclusion_probs,
        'effective_variances': effective_variances_dict,
        'em_iterations': em_iterations_list,
        'converged': converged_list,
        'extracted': extracted,
        'hyperparameters': {
            'v0': v0, 'v1': v1, 'c': c, 'sigma_sq': sigma_sq,
            'pi_coupling': pi_coup,   # shape (G,), pi_g = c / B_g
            'bin_sizes': bin_sizes,   # shape (G,)
        },
    }


def predict_group_ising_emvs(
    X_t: np.ndarray,
    action: int,
    result: Dict[str, Any],
    A: np.ndarray,
    return_proba: bool = True,
) -> np.ndarray:
    """
    Next-state probabilities from a fitted EMVS model (same signature as the logistic
    version). X_t : (N,) state; action : treated node or -1; return_proba=False gives {-1, 1}.
    """
    group_params = result['group_params']
    g_idx = result['g_idx']
    G = result['n_groups']
    normalize = result.get('normalize', False)
    neighbor_lists = result['neighbor_lists']
    neigh_g_idx = result['neigh_g_idx']
    degrees = result['degrees']
    N = len(X_t)
    n_baseline = 4
    n_features = n_baseline + G

    treated_array = np.zeros(N)
    if 0 <= action < N:
        treated_array[action] = 1.0

    predictions = np.zeros(N)

    for j in range(N):
        gi = g_idx[j]
        neigh = neighbor_lists[j]
        ng = neigh_g_idx[j]
        d = len(neigh)
        Nj = degrees[j]

        features = np.zeros(n_features)
        features[0] = 1.0
        features[1] = X_t[j]
        features[2] = float(action == j)
        if d > 0:
            features[3] = treated_array[neigh].sum()
            neigh_states = X_t[neigh]
            for gm in range(G):
                mask = (ng == gm)
                features[n_baseline + gm] = neigh_states[mask].sum() / (Nj if normalize else 1.0)

        eta = features @ group_params[gi]
        p = expit(eta)
        predictions[j] = p if return_proba else (2 * int(p > 0.5) - 1)

    return predictions


def fit_group_ising_mcmc(
    X_array: np.ndarray,
    y_array: np.ndarray,
    treated: np.ndarray,
    A: np.ndarray,
    group_labels: np.ndarray,
    v0: float = 0.01,
    v1: float = 10.0,
    c: float = 1.0,
    sigma_sq: float = 10.0,
    n_draws: int = 200,
    n_tune: int = 300,
    n_chains: int = 1,
    seed: int = 42,
    max_rows: int = 8_000,
    normalize: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit the group-level Ising model by MCMC (PyMC) with a spike-and-slab coupling prior
    (mixture of N(0, v0) and N(0, v1), weight pi = min(c / G, 0.99)) and N(0, sigma_sq)
    baseline priors. Same inputs and topology outputs as fit_group_ising_emvs; group gi
    uses seed + gi; max_rows subsamples large design matrices.
    Returns a dict with 'mcmc_draws' (gi -> (n_draws*n_chains, 4+G) array), 'n_draws',
    'hyperparameters' and the topology keys.
    """
    import pymc as pm

    if X_array.ndim != 3 or y_array.ndim != 3:
        raise ValueError("X_array and y_array must be 3D arrays with shape (m, T, N).")
    m, T, N = X_array.shape
    if y_array.shape != (m, T, N):
        raise ValueError("y_array must have the same shape as X_array.")
    if group_labels.shape != (N,):
        raise ValueError(f"group_labels must have shape ({N},), got {group_labels.shape}.")

    groups = np.sort(np.unique(group_labels))
    G = len(groups)
    group_to_idx = {g: i for i, g in enumerate(groups)}
    g_idx = np.array([group_to_idx[group_labels[j]] for j in range(N)], dtype=int)

    pi_coup = min(c / G, 0.99)

    if treated.ndim == 2:
        treated_indicator = np.zeros((m, T, N), dtype=bool)
        for ep in range(m):
            for t in range(T):
                idx = int(treated[ep, t])
                if 0 <= idx < N:
                    treated_indicator[ep, t, idx] = True
    elif treated.ndim == 3:
        treated_indicator = treated.astype(bool)
    else:
        raise ValueError("treated must be shape (m,T) or (m,T,N).")

    neighbor_lists = [np.nonzero(A[j] != 0)[0] for j in range(N)]
    neigh_g_idx = [g_idx[neighbor_lists[j]] for j in range(N)]
    degrees = np.array([len(neighbor_lists[j]) for j in range(N)], dtype=float)

    n_base = 4   # intercept, persistence, direct treatment, spillover
    n_feat = n_base + G

    X_per_group = [[] for _ in range(G)]
    y_per_group = [[] for _ in range(G)]

    for j in range(N):
        gi = g_idx[j]
        neigh = neighbor_lists[j]
        ng = neigh_g_idx[j]
        d = len(neigh)
        Nj = degrees[j]
        for ep in range(m):
            for t in range(T):
                row = np.zeros(n_feat)
                row[0] = 1.0
                row[1] = X_array[ep, t, j]
                row[2] = float(treated_indicator[ep, t, j])
                if d > 0:
                    row[3] = treated_indicator[ep, t, neigh].sum()
                    ns = X_array[ep, t, neigh]
                    for gm in range(G):
                        row[n_base + gm] = ns[ng == gm].sum() / (Nj if normalize else 1.0)
                X_per_group[gi].append(row)
                y_per_group[gi].append(int(y_array[ep, t, j]))

    X_per_group = [np.array(x) for x in X_per_group]
    y_per_group = [np.array(y) for y in y_per_group]

    n_total = n_draws * n_chains
    mcmc_draws = {}

    for gi in range(G):
        Xg = X_per_group[gi]
        yg = y_per_group[gi]

        if len(np.unique(yg)) < 2:
            # Fallback: constant prediction encoded as extreme intercept
            fallback = np.zeros(n_feat)
            fallback[0] = 10.0 if yg[0] == 1 else -10.0
            mcmc_draws[gi] = np.tile(fallback, (n_total, 1))
            continue

        if len(yg) > max_rows:
            rng_sub = np.random.default_rng(seed + gi + 100)
            idx_sub = rng_sub.choice(len(yg), size=max_rows, replace=False)
            Xg = Xg[idx_sub]
            yg = yg[idx_sub]

        if verbose:
            print(f"  [MCMC] group {gi} ({len(yg)} rows) ...", end=" ", flush=True)

        with pm.Model():
            baseline = pm.Normal(
                "baseline", mu=0.0, sigma=np.sqrt(sigma_sq), shape=n_base)
            z_coup = pm.NormalMixture(
                "z_coup",
                w=np.array([1.0 - pi_coup, pi_coup]),
                mu=np.zeros(2),
                sigma=np.array([np.sqrt(v0), np.sqrt(v1)]),
                shape=G,
            )
            theta = pm.math.concatenate([baseline, z_coup])
            logit_p = pm.math.dot(Xg, theta)
            _ = pm.Bernoulli("y_obs", logit_p=logit_p, observed=yg.astype(int))
            trace = pm.sample(
                n_draws, tune=n_tune, chains=n_chains,
                random_seed=seed + gi, progressbar=verbose,
                target_accept=0.85,
            )

        _base = trace.posterior["baseline"].values.reshape(-1, n_base)
        _coup = trace.posterior["z_coup"].values.reshape(-1, G)
        mcmc_draws[gi] = np.concatenate([_base, _coup], axis=1)

        if verbose:
            print("done")

    return {
        'method':          'group_mcmc',
        'group_labels':    group_labels,
        'g_idx':           g_idx,
        'groups':          groups,
        'n_groups':        G,
        'normalize':       normalize,
        'neighbor_lists':  neighbor_lists,
        'neigh_g_idx':     neigh_g_idx,
        'degrees':         degrees,
        'mcmc_draws':      mcmc_draws,
        'n_draws':         n_total,
        'hyperparameters': {
            'v0': v0, 'v1': v1, 'c': c, 'sigma_sq': sigma_sq,
            'pi_coup': pi_coup,
            'n_draws': n_draws, 'n_tune': n_tune, 'n_chains': n_chains,
        },
    }


def predict_group_ising_mcmc(
    X_t: np.ndarray,
    action: int,
    result: Dict[str, Any],
    draw_idx: int = 0,
    return_proba: bool = True,
) -> np.ndarray:
    """
    Next-state probabilities for one MCMC posterior draw (delegates to
    predict_group_ising_emvs). draw_idx : 0-indexed draw; action : treated node or -1.
    """
    G = result['n_groups']
    draw_params = {gi: result['mcmc_draws'][gi][draw_idx] for gi in range(G)}
    single_result = {
        'group_params':   draw_params,
        'g_idx':          result['g_idx'],
        'n_groups':       G,
        'normalize':      result.get('normalize', False),
        'neighbor_lists': result['neighbor_lists'],
        'neigh_g_idx':    result['neigh_g_idx'],
        'degrees':        result['degrees'],
    }
    return predict_group_ising_emvs(X_t, action, single_result, A=None,
                                    return_proba=return_proba)
