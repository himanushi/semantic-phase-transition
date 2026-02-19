"""Landau fit models and model selection utilities."""

import numpy as np
from scipy.optimize import curve_fit


# --- Fit models ---

def power_law(l_norm: np.ndarray, A: float, beta: float) -> np.ndarray:
    """f(x) = A * x^beta, where x = l/L."""
    return A * np.power(np.clip(l_norm, 1e-10, None), beta)


def tanh_model(l_norm: np.ndarray, A: float, kappa: float, lc: float) -> np.ndarray:
    """f(x) = A * tanh(kappa * (x - lc))."""
    return A * np.tanh(kappa * (l_norm - lc))


def sigmoid_model(l_norm: np.ndarray, A: float, kappa: float, lc: float) -> np.ndarray:
    """f(x) = A / (1 + exp(-kappa * (x - lc)))."""
    return A / (1.0 + np.exp(-kappa * (l_norm - lc)))


def two_stage(l_norm: np.ndarray, A: float, beta: float, B: float, lc: float, w: float) -> np.ndarray:
    """f(x) = A * x^beta + B * sigmoid(w * (x - lc)).

    Gradual power-law growth plus a late sigmoid step for final-layer jump.
    """
    step = 1.0 / (1.0 + np.exp(-w * (l_norm - lc)))
    return A * np.power(np.clip(l_norm, 1e-10, None), beta) + B * step


# --- Fit wrappers ---

FIT_MODELS = {
    "power_law": {
        "func": power_law,
        "p0": [0.5, 1.0],
        "bounds": ([-5, 0.01], [5, 5.0]),
        "n_params": 2,
    },
    "tanh": {
        "func": tanh_model,
        "p0": [0.3, 5.0, 0.5],
        "bounds": ([-5, 0.1, 0.0], [5, 50.0, 1.0]),
        "n_params": 3,
    },
    "sigmoid": {
        "func": sigmoid_model,
        "p0": [0.5, 10.0, 0.5],
        "bounds": ([-5, 0.1, 0.0], [5, 50.0, 1.0]),
        "n_params": 3,
    },
    "two_stage": {
        "func": two_stage,
        "p0": [0.3, 1.0, 0.1, 0.9, 20.0],
        "bounds": ([-5, 0.01, -5, 0.5, 1.0], [5, 5.0, 5, 1.0, 100.0]),
        "n_params": 5,
    },
}


def compute_aic_bic(n: int, k: int, rss: float) -> tuple[float, float]:
    """Compute AIC and BIC from residual sum of squares.

    Args:
        n: Number of data points.
        k: Number of parameters.
        rss: Residual sum of squares.

    Returns:
        Tuple of (AIC, BIC).
    """
    if rss <= 0 or n <= k + 1:
        return float("inf"), float("inf")
    log_likelihood = -n / 2 * np.log(rss / n)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return float(aic), float(bic)


def fit_model(
    l_norm: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> dict | None:
    """Fit a named model to data.

    Args:
        l_norm: Normalized layer positions (l/L), shape (N,).
        y: Target values, shape (N,).
        model_name: Key into FIT_MODELS.

    Returns:
        Dict with params, fitted values, R², AIC, BIC, or None if fit fails.
    """
    spec = FIT_MODELS[model_name]
    func = spec["func"]
    n = len(l_norm)
    k = spec["n_params"]

    # σ の符号でp0を調整
    sign = 1.0 if np.mean(y) >= 0 else -1.0
    p0 = list(spec["p0"])
    p0[0] *= sign

    try:
        popt, pcov = curve_fit(
            func, l_norm, y,
            p0=p0,
            bounds=spec["bounds"],
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    y_fit = func(l_norm, *popt)
    rss = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    aic, bic = compute_aic_bic(n, k, rss)

    param_names = func.__code__.co_varnames[1 : k + 1]
    params = {name: float(val) for name, val in zip(param_names, popt)}

    return {
        "model": model_name,
        "params": params,
        "y_fit": y_fit,
        "r2": r2,
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "n_params": k,
    }


def fit_all_models(
    l_norm: np.ndarray,
    y: np.ndarray,
) -> dict[str, dict]:
    """Fit all models and return results sorted by BIC.

    Args:
        l_norm: Normalized layer positions.
        y: Target values.

    Returns:
        Dict of model_name -> fit result, sorted by BIC (best first).
    """
    results = {}
    for name in FIT_MODELS:
        result = fit_model(l_norm, y, name)
        if result is not None:
            results[name] = result

    # BICでソート
    results = dict(sorted(results.items(), key=lambda x: x[1]["bic"]))
    return results
