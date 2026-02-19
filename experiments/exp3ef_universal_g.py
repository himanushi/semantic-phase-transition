"""Experiment 3E/F/G: Universal permeation function g(l/L).

E. Extract g(l/L) = f(l) / f_max for each word, overlay and average
F. Fit g_mean(l/L) to 5 candidate models (power_law, exponential, sigmoid, erf, log)
G. Compare g(l/L) across GPT-2 small and medium

Reads pre-computed exp3 data from results/data/exp3_{model}.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# g(l/L) candidate models
# -------------------------------------------------------------------

def model_power_law(x: np.ndarray, alpha: float) -> np.ndarray:
    """g(x) = x^alpha"""
    return np.power(np.clip(x, 1e-10, None), alpha)


def model_exponential(x: np.ndarray, tau: float) -> np.ndarray:
    """g(x) = 1 - exp(-x / tau)"""
    return 1.0 - np.exp(-x / np.clip(tau, 1e-10, None))


def model_sigmoid(x: np.ndarray, kappa: float, xc: float) -> np.ndarray:
    """g(x) = 1 / (1 + exp(-kappa * (x - xc)))"""
    return 1.0 / (1.0 + np.exp(-kappa * (x - xc)))


def model_erf(x: np.ndarray, sigma: float) -> np.ndarray:
    """g(x) = erf(x / (sigma * sqrt(2)))

    Solution of diffusion equation: if semantic information diffuses
    through layers, g should follow an error function.
    """
    return erf(x / (np.clip(sigma, 1e-10, None) * np.sqrt(2.0)))


def model_log(x: np.ndarray, tau: float) -> np.ndarray:
    """g(x) = log(1 + x/tau) / log(1 + 1/tau)"""
    tau_c = np.clip(tau, 1e-10, None)
    return np.log(1.0 + x / tau_c) / np.log(1.0 + 1.0 / tau_c)


G_MODELS = {
    "power_law": {
        "func": model_power_law,
        "p0": [1.0],
        "bounds": ([0.01], [10.0]),
        "n_params": 1,
        "label": r"$g(x) = x^{\alpha}$",
    },
    "exponential": {
        "func": model_exponential,
        "p0": [0.3],
        "bounds": ([0.01], [10.0]),
        "n_params": 1,
        "label": r"$g(x) = 1 - e^{-x/\tau}$",
    },
    "sigmoid": {
        "func": model_sigmoid,
        "p0": [10.0, 0.5],
        "bounds": ([0.1, 0.0], [100.0, 1.0]),
        "n_params": 2,
        "label": r"$g(x) = \mathrm{sigmoid}(\kappa(x - x_c))$",
    },
    "erf": {
        "func": model_erf,
        "p0": [0.4],
        "bounds": ([0.01], [10.0]),
        "n_params": 1,
        "label": r"$g(x) = \mathrm{erf}(x / \sigma\sqrt{2})$",
    },
    "log": {
        "func": model_log,
        "p0": [0.3],
        "bounds": ([0.001], [10.0]),
        "n_params": 1,
        "label": r"$g(x) = \log(1+x/\tau) / \log(1+1/\tau)$",
    },
}


def compute_aic_bic(n: int, k: int, rss: float) -> tuple[float, float]:
    """Compute AIC and BIC."""
    if rss <= 0 or n <= k + 1:
        return float("inf"), float("inf")
    log_lik = -n / 2 * np.log(rss / n)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    return float(aic), float(bic)


def fit_g_model(x: np.ndarray, y: np.ndarray, model_name: str) -> dict | None:
    """Fit a g(l/L) model to data."""
    spec = G_MODELS[model_name]
    func = spec["func"]
    n = len(x)
    k = spec["n_params"]

    try:
        popt, pcov = curve_fit(
            func, x, y,
            p0=spec["p0"],
            bounds=spec["bounds"],
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    y_fit = func(x, *popt)
    rss = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    aic, bic = compute_aic_bic(n, k, rss)

    param_names = func.__code__.co_varnames[1 : k + 1]
    params = {name: float(val) for name, val in zip(param_names, popt)}

    return {
        "model": model_name,
        "params": params,
        "y_fit": y_fit.tolist(),
        "r2": float(r2),
        "rss": rss,
        "aic": float(aic),
        "bic": float(bic),
        "n_params": k,
    }


# -------------------------------------------------------------------
# Part E: Extract and overlay g(l/L)
# -------------------------------------------------------------------

def extract_g_from_exp3(exp3_data: dict) -> dict:
    """Extract g_word(l/L) = f(l) / f_max from exp3 linearity data.

    Returns dict with per-word g values and the mean g.
    """
    n_layers = exp3_data["n_layers"]
    words_data = exp3_data["words"]

    g_dict: dict[str, np.ndarray] = {}
    f_dict: dict[str, np.ndarray] = {}
    fmax_dict: dict[str, float] = {}

    for word, wdata in words_data.items():
        slopes = np.array([lin["slope"] for lin in wdata["linearity"]])
        f_max = float(np.max(slopes))

        if f_max < 1e-8:
            print(f"  SKIP {word}: f_max too small ({f_max:.6f})")
            continue

        g = slopes / f_max
        g_dict[word] = g
        f_dict[word] = slopes
        fmax_dict[word] = f_max

    # 語間平均
    if g_dict:
        g_arrays = list(g_dict.values())
        g_mean = np.mean(g_arrays, axis=0)
        g_std = np.std(g_arrays, axis=0)
    else:
        g_mean = np.array([])
        g_std = np.array([])

    n_points = n_layers + 1
    l_norm = np.linspace(0, 1, n_points)

    return {
        "g_dict": g_dict,
        "f_dict": f_dict,
        "fmax_dict": fmax_dict,
        "g_mean": g_mean,
        "g_std": g_std,
        "l_norm": l_norm,
        "n_layers": n_layers,
    }


def compute_residuals(g_dict: dict[str, np.ndarray], g_mean: np.ndarray) -> dict:
    """Compute per-word residuals from g_mean."""
    residuals = {}
    for word, g in g_dict.items():
        resid = g - g_mean
        residuals[word] = {
            "max_abs": float(np.max(np.abs(resid))),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "residuals": resid.tolist(),
        }
    return residuals


# -------------------------------------------------------------------
# Part F: Fit g_mean to candidate models
# -------------------------------------------------------------------

def fit_all_g_models(l_norm: np.ndarray, g_mean: np.ndarray) -> dict:
    """Fit g_mean(l/L) to all candidate models."""
    # L0 を除外（g(0) ≈ 0 は自明でフィット精度を人工的に上げる）
    # ただし L0 も含めた方が物理的に意味があるので、両方試す
    results = {}
    for name in G_MODELS:
        result = fit_g_model(l_norm, g_mean, name)
        if result is not None:
            results[name] = result

    # BIC でソート
    results = dict(sorted(results.items(), key=lambda x: x[1]["bic"]))
    return results


# -------------------------------------------------------------------
# Part G: Cross-model comparison
# -------------------------------------------------------------------

def compare_g_across_models(
    g_data_small: dict,
    g_data_medium: dict,
) -> dict:
    """Compare g(l/L) between GPT-2 small and medium."""
    g_mean_s = g_data_small["g_mean"]
    g_mean_m = g_data_medium["g_mean"]
    l_norm_s = g_data_small["l_norm"]
    l_norm_m = g_data_medium["l_norm"]

    # medium の g_mean を small の l/L グリッドに補間
    from scipy.interpolate import interp1d
    interp_m = interp1d(l_norm_m, g_mean_m, kind="cubic", fill_value="extrapolate")
    g_mean_m_interp = interp_m(l_norm_s)

    # 相関と RMSE
    corr = float(np.corrcoef(g_mean_s, g_mean_m_interp)[0, 1])
    rmse = float(np.sqrt(np.mean((g_mean_s - g_mean_m_interp) ** 2)))
    max_diff = float(np.max(np.abs(g_mean_s - g_mean_m_interp)))

    # 語ごとの cross-model 相関
    word_corrs = {}
    for word in g_data_small["g_dict"]:
        if word in g_data_medium["g_dict"]:
            g_s = g_data_small["g_dict"][word]
            g_m = g_data_medium["g_dict"][word]
            interp_w = interp1d(l_norm_m, g_m, kind="cubic", fill_value="extrapolate")
            g_m_interp = interp_w(l_norm_s)
            word_corrs[word] = float(np.corrcoef(g_s, g_m_interp)[0, 1])

    return {
        "g_mean_correlation": corr,
        "g_mean_rmse": rmse,
        "g_mean_max_diff": max_diff,
        "word_correlations": word_corrs,
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_universal_g(
    g_data: dict,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot g_word(l/L) overlay and g_mean with error band."""
    l_norm = g_data["l_norm"]
    g_dict = g_data["g_dict"]
    g_mean = g_data["g_mean"]
    g_std = g_data["g_std"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: raw g overlay
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(g_dict)))
    for (word, g), color in zip(g_dict.items(), colors):
        ax.plot(l_norm, g, "o-", label=word, markersize=4, color=color)
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("g(l/L) = f(l) / f_max")
    ax.set_title(f"Per-word g(l/L) — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    # Right: g_mean with error band
    ax = axes[1]
    ax.plot(l_norm, g_mean, "k-o", markersize=4, label="g_mean", linewidth=2)
    ax.fill_between(l_norm, g_mean - g_std, g_mean + g_std,
                     alpha=0.2, color="gray", label="±1 std")
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("g_mean(l/L)")
    ax.set_title(f"Universal permeation function — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_g_fit(
    l_norm: np.ndarray,
    g_mean: np.ndarray,
    fit_results: dict,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot g_mean with all candidate model fits."""
    n_models = len(fit_results)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                              squeeze=False)

    x_fine = np.linspace(0, 1, 200)

    for idx, (name, result) in enumerate(fit_results.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        ax.plot(l_norm, g_mean, "ko", markersize=5, label="data", zorder=5)

        spec = G_MODELS[name]
        func = spec["func"]
        params = list(result["params"].values())
        y_fit = func(x_fine, *params)
        ax.plot(x_fine, y_fit, "r-", linewidth=2,
                label=f"fit (R²={result['r2']:.4f})")

        ax.set_xlabel("l/L")
        ax.set_ylabel("g(l/L)")
        ax.set_title(f"{name}\nAIC={result['aic']:.1f}, BIC={result['bic']:.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.1, 1.1)

    # 空きパネルを非表示
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.suptitle(f"g(l/L) model fits — {model_name}", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_cross_model_g(
    g_data_small: dict,
    g_data_medium: dict,
    comparison: dict,
    output_path: Path | None = None,
) -> None:
    """Plot g(l/L) comparison between GPT-2 small and medium."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: g_mean comparison
    ax = axes[0]
    l_s = g_data_small["l_norm"]
    l_m = g_data_medium["l_norm"]
    ax.plot(l_s, g_data_small["g_mean"], "o-", label="GPT-2 small", markersize=4)
    ax.plot(l_m, g_data_medium["g_mean"], "s-", label="GPT-2 medium", markersize=4)
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("g_mean(l/L)")
    ax.set_title(f"Cross-model g_mean (corr={comparison['g_mean_correlation']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: per-word g overlay (small)
    ax = axes[1]
    for word, g in g_data_small["g_dict"].items():
        ax.plot(l_s, g, "o-", label=f"{word} (S)", markersize=3, alpha=0.7)
    for word, g in g_data_medium["g_dict"].items():
        ax.plot(l_m, g, "s--", label=f"{word} (M)", markersize=3, alpha=0.7)
    ax.set_xlabel("l/L")
    ax.set_ylabel("g(l/L)")
    ax.set_title("Per-word g: Small (o) vs Medium (□)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: difference
    ax = axes[2]
    from scipy.interpolate import interp1d
    interp_m = interp1d(l_m, g_data_medium["g_mean"], kind="cubic",
                         fill_value="extrapolate")
    g_m_interp = interp_m(l_s)
    diff = g_data_small["g_mean"] - g_m_interp
    ax.plot(l_s, diff, "o-", color="C3", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("l/L")
    ax.set_ylabel("g_small - g_medium")
    ax.set_title(f"Difference (RMSE={comparison['g_mean_rmse']:.4f})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Universal Permeation Function: Cross-Model Comparison", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_best_fit_overlay(
    g_data_small: dict,
    g_data_medium: dict,
    fits_small: dict,
    fits_medium: dict,
    output_path: Path | None = None,
) -> None:
    """Overlay best-fit curve on both models' g_mean."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    x_fine = np.linspace(0, 1, 200)

    # Data points
    ax.plot(g_data_small["l_norm"], g_data_small["g_mean"],
            "o", color="C0", markersize=6, label="GPT-2 small (data)", zorder=5)
    ax.plot(g_data_medium["l_norm"], g_data_medium["g_mean"],
            "s", color="C1", markersize=6, label="GPT-2 medium (data)", zorder=5)

    # Best fit for each
    best_s_name = next(iter(fits_small))
    best_m_name = next(iter(fits_medium))

    for fits, name, color, ls in [
        (fits_small, best_s_name, "C0", "-"),
        (fits_medium, best_m_name, "C1", "--"),
    ]:
        spec = G_MODELS[name]
        func = spec["func"]
        params = list(fits[name]["params"].values())
        y_fit = func(x_fine, *params)
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in fits[name]["params"].items())
        ax.plot(x_fine, y_fit, color=color, linestyle=ls, linewidth=2,
                label=f"{name} ({param_str})")

    ax.set_xlabel("Normalized layer l/L", fontsize=12)
    ax.set_ylabel("g(l/L)", fontsize=12)
    ax.set_title("Universal Permeation Function g(l/L) — Best Fit", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_single_model(
    exp3_path: Path,
    model_name: str,
    fig_dir: Path,
    data_dir: Path,
) -> tuple[dict, dict]:
    """Run parts E and F for a single model."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    with open(exp3_path) as f:
        exp3_data = json.load(f)

    # --- Part E ---
    print(f"\nPart E: Extracting g(l/L)")
    g_data = extract_g_from_exp3(exp3_data)

    print(f"  Words: {list(g_data['g_dict'].keys())}")
    print(f"  Layers: {g_data['n_layers']}")
    for word, fmax in g_data["fmax_dict"].items():
        print(f"  f_max({word}) = {fmax:.4f}")

    # 残差評価
    residuals = compute_residuals(g_data["g_dict"], g_data["g_mean"])
    print(f"\n  Residuals from g_mean:")
    for word, res in residuals.items():
        print(f"    {word}: RMSE={res['rmse']:.4f}, max|resid|={res['max_abs']:.4f}")

    plot_universal_g(
        g_data, model_name,
        output_path=fig_dir / f"exp3e_universal_g_{model_tag}.png",
    )

    # --- Part F ---
    print(f"\nPart F: Fitting g_mean(l/L) to candidate models")
    fit_results = fit_all_g_models(g_data["l_norm"], g_data["g_mean"])

    print(f"\n  {'Model':<15} {'R²':<10} {'AIC':<10} {'BIC':<10} {'Params'}")
    print(f"  {'-'*65}")
    for name, result in fit_results.items():
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in result["params"].items())
        print(f"  {name:<15} {result['r2']:<10.6f} {result['aic']:<10.2f} {result['bic']:<10.2f} {param_str}")

    best_name = next(iter(fit_results))
    print(f"\n  Best model (by BIC): {best_name}")

    plot_g_fit(
        g_data["l_norm"], g_data["g_mean"], fit_results, model_name,
        output_path=fig_dir / f"exp3f_fit_{model_tag}.png",
    )

    # JSON保存
    output = {
        "model": model_name,
        "n_layers": g_data["n_layers"],
        "words": list(g_data["g_dict"].keys()),
        "fmax": g_data["fmax_dict"],
        "g_per_word": {w: g.tolist() for w, g in g_data["g_dict"].items()},
        "g_mean": g_data["g_mean"].tolist(),
        "g_std": g_data["g_std"].tolist(),
        "l_norm": g_data["l_norm"].tolist(),
        "residuals_from_mean": {w: r for w, r in residuals.items()},
        "fits": {
            name: {k: v for k, v in result.items() if k != "y_fit"}
            for name, result in fit_results.items()
        },
        "best_model": best_name,
    }

    json_path = data_dir / f"exp3ef_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved: {json_path}")

    return g_data, fit_results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3E/F/G: Universal permeation function g(l/L)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="results/data",
        help="Directory containing exp3 JSON files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Base output directory"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    out_data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    # 両モデルのデータファイルを探す
    models = []
    for tag in ["gpt2", "gpt2-medium"]:
        path = data_dir / f"exp3_{tag}.json"
        if path.exists():
            models.append((tag, path))
        else:
            print(f"WARNING: {path} not found, skipping")

    if not models:
        print("ERROR: No exp3 data files found")
        return

    # --- Parts E/F for each model ---
    all_g_data = {}
    all_fits = {}
    for model_tag, path in models:
        model_name = model_tag
        g_data, fits = run_single_model(path, model_name, fig_dir, out_data_dir)
        all_g_data[model_tag] = g_data
        all_fits[model_tag] = fits

    # --- Part G: Cross-model comparison ---
    if len(all_g_data) >= 2 and "gpt2" in all_g_data and "gpt2-medium" in all_g_data:
        print(f"\n{'='*60}")
        print("Part G: Cross-model comparison")
        print(f"{'='*60}")

        comparison = compare_g_across_models(
            all_g_data["gpt2"], all_g_data["gpt2-medium"]
        )

        print(f"\n  g_mean correlation (small vs medium): {comparison['g_mean_correlation']:.4f}")
        print(f"  g_mean RMSE: {comparison['g_mean_rmse']:.4f}")
        print(f"  g_mean max|diff|: {comparison['g_mean_max_diff']:.4f}")

        print(f"\n  Per-word cross-model correlations:")
        for word, corr in comparison["word_correlations"].items():
            print(f"    {word}: {corr:.4f}")

        plot_cross_model_g(
            all_g_data["gpt2"], all_g_data["gpt2-medium"], comparison,
            output_path=fig_dir / "exp3g_cross_model_g.png",
        )

        plot_best_fit_overlay(
            all_g_data["gpt2"], all_g_data["gpt2-medium"],
            all_fits["gpt2"], all_fits["gpt2-medium"],
            output_path=fig_dir / "exp3g_best_fit_overlay.png",
        )

        # 比較結果をJSON追記
        comp_path = out_data_dir / "exp3g_cross_model.json"
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Cross-model data saved: {comp_path}")

    # --- 最終サマリー ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for model_tag in all_g_data:
        g_data = all_g_data[model_tag]
        fits = all_fits[model_tag]
        best = next(iter(fits))
        best_r2 = fits[best]["r2"]
        best_params = fits[best]["params"]

        print(f"\n  {model_tag}:")
        print(f"    Words: {list(g_data['g_dict'].keys())}")
        print(f"    f_max: {g_data['fmax_dict']}")
        print(f"    Best model: {best} (R²={best_r2:.6f})")
        print(f"    Params: {best_params}")

        # erf 結果を特別に表示
        if "erf" in fits:
            erf_r2 = fits["erf"]["r2"]
            erf_sigma = fits["erf"]["params"]["sigma"]
            print(f"    erf model: R²={erf_r2:.6f}, σ={erf_sigma:.4f}")

    if len(all_g_data) >= 2:
        print(f"\n  Cross-model g_mean correlation: {comparison['g_mean_correlation']:.4f}")


if __name__ == "__main__":
    main()
