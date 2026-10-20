"""
Group-pooled covariate Ising fit. Per-node features: [y, treat, own-group covariates 0..K-1,
group_treated]; the fit must find the one effective covariate (EFFECTIVE_IDX) among decoys.
"""
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression


class _DummyModel:
    def __init__(self, constant_class: int, n_features: int):
        self.constant_class = constant_class
        self.coef_ = np.zeros((1, n_features))
        self.intercept_ = np.array([10.0 if constant_class == 1 else -10.0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.constant_class == 1:
            return np.tile([0.0, 1.0], (n, 1))
        return np.tile([1.0, 0.0], (n, 1))


def fit_group_ising_logistic_controllable(
    X_array: np.ndarray,
    y_array: np.ndarray,
    treated: np.ndarray,
    C_own_group_array: np.ndarray,
    group_labels: np.ndarray,
    **logistic_kwargs
) -> Dict[str, Any]:
    """
    X_array, y_array : (m, T, N) node-level binary states / outcomes
    treated : (m, T) treated NODE index (-1 for none)
    C_own_group_array : (m, T, N, n_covariates) node j's OWN GROUP's
        covariate vector at time t (broadcast from (m, T, n_bins, n_cov)
        via group_labels by the caller)
    group_labels : (N,) block membership
    """
    if X_array.ndim != 3 or y_array.ndim != 3 or C_own_group_array.ndim != 4:
        raise ValueError("X_array, y_array must be 3D; C_own_group_array must be 4D.")
    m, T, N = X_array.shape
    n_cov = C_own_group_array.shape[3]

    groups = np.sort(np.unique(group_labels))
    G = len(groups)
    group_to_idx = {g: i for i, g in enumerate(groups)}
    g_idx = np.array([group_to_idx[group_labels[j]] for j in range(N)], dtype=int)

    treated_indicator = np.zeros((m, T, N), dtype=bool)
    treated_group_idx = np.full((m, T), -1, dtype=int)
    for ep in range(m):
        for t in range(T):
            idx = int(treated[ep, t])
            if 0 <= idx < N:
                treated_indicator[ep, t, idx] = True
                treated_group_idx[ep, t] = g_idx[idx]

    n_features = 2 + n_cov + 1  # [y_jt, treat_jt, C_0..C_{n_cov-1}, group_treated_jt]
    lr_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    lr_params.update(logistic_kwargs)

    models = [None] * G
    intercept = np.zeros(G)
    rho = np.zeros(G)
    beta_treat = np.zeros(G)
    gamma_c = np.zeros((G, n_cov))
    gamma_group_treated = np.zeros(G)

    rows_per_group = [[] for _ in range(G)]
    labels_per_group = [[] for _ in range(G)]

    for j in range(N):
        gi = g_idx[j]
        for ep in range(m):
            for t in range(T):
                row = np.zeros(n_features)
                row[0] = X_array[ep, t, j]
                row[1] = float(treated_indicator[ep, t, j])
                row[2:2 + n_cov] = C_own_group_array[ep, t, j]
                row[2 + n_cov] = float(treated_group_idx[ep, t] == gi)
                rows_per_group[gi].append(row)
                labels_per_group[gi].append(int(y_array[ep, t, j]))

    for gi in range(G):
        X_g = np.array(rows_per_group[gi])
        y_g = np.array(labels_per_group[gi])
        unique_classes = np.unique(y_g)
        if len(unique_classes) < 2:
            model = _DummyModel(int(unique_classes[0]), n_features)
        else:
            model = LogisticRegression(**lr_params)
            model.fit(X_g, y_g)
        models[gi] = model

        coefs = model.coef_[0]
        intercept[gi] = model.intercept_[0]
        rho[gi] = coefs[0]
        beta_treat[gi] = coefs[1]
        gamma_c[gi] = coefs[2:2 + n_cov]
        gamma_group_treated[gi] = coefs[2 + n_cov]

    return {
        'method': 'group_logistic_controllable_many',
        'models': models,
        'groups': groups,
        'g_idx': g_idx,
        'n_groups': G,
        'n_covariates': n_cov,
        'extracted': {
            'intercept': intercept, 'rho': rho, 'beta_treat': beta_treat,
            'gamma_c': gamma_c, 'gamma_group_treated': gamma_group_treated,
        },
    }


def predict_group_ising_logistic_controllable(
    X_t: np.ndarray,
    action,
    C_own_group_t: np.ndarray,
    result: Dict[str, Any],
    return_proba: bool = True,
) -> np.ndarray:
    """X_t: (N,) state; C_own_group_t: (N, n_covariates) own-group covariates; action: node or None."""
    models = result['models']
    g_idx = result['g_idx']
    n_cov = result['n_covariates']
    N = len(X_t)

    treated_array = np.zeros(N)
    treated_group = -1
    if action is not None and 0 <= int(action) < N:
        treated_array[int(action)] = 1.0
        treated_group = int(g_idx[int(action)])

    predictions = np.zeros(N)
    for j in range(N):
        gi = g_idx[j]
        features = np.zeros(2 + n_cov + 1)
        features[0] = X_t[j]
        features[1] = treated_array[j]
        features[2:2 + n_cov] = C_own_group_t[j]
        features[2 + n_cov] = float(gi == treated_group)
        model = models[gi]
        if return_proba:
            predictions[j] = model.predict_proba(features.reshape(1, -1))[0, 1]
        else:
            predictions[j] = float(model.predict_proba(features.reshape(1, -1))[0, 1] > 0.5)
    return predictions
