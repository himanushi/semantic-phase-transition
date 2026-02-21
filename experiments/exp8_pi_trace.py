"""Experiment 8: Trace of π in the universal permeation function g(l/L).

8A. Likelihood ratio test: erf σ_free vs σ_fixed = 1/√π
8B. π-containing model fits: cos(πx), sin(πx/2) etc. with BIC comparison
8C. Inter-layer rotation angle measurement in residual stream

Reads pre-computed exp3ef data from results/data/exp3ef_{model}.json.
Part C requires model loading for residual stream analysis.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SIGMA_PI = 1.0 / np.sqrt(np.pi)  # ≈ 0.5642


# -------------------------------------------------------------------
# Part A: erf σ parameter π test
# -------------------------------------------------------------------

def model_erf_free(x: np.ndarray, sigma: float) -> np.ndarray:
    """g(x) = erf(x / (σ√2)) with free σ."""
    return erf(x / (np.clip(sigma, 1e-10, None) * np.sqrt(2.0)))


def model_erf_fixed(x: np.ndarray) -> np.ndarray:
    """g(x) = erf(x / (σ√2)) with σ = 1/√π."""
    return erf(x / (SIGMA_PI * np.sqrt(2.0)))


def run_part_a(
    data: dict,
    model_name: str,
    fig_dir: Path,
) -> dict:
    """Likelihood ratio test: σ_free vs σ = 1/√π."""
    g_mean = np.array(data["g_mean"])
    l_norm = np.array(data["l_norm"])
    n = len(g_mean)

    # σ=free フィット
    popt, _ = curve_fit(
        model_erf_free, l_norm, g_mean,
        p0=[0.5], bounds=([0.01], [10.0]), maxfev=20000,
    )
    sigma_free = float(popt[0])
    y_free = model_erf_free(l_norm, sigma_free)
    rss_free = float(np.sum((g_mean - y_free) ** 2))

    # σ=1/√π 固定
    y_fixed = model_erf_fixed(l_norm)
    rss_fixed = float(np.sum((g_mean - y_fixed) ** 2))

    # 尤度比検定: χ² = n·ln(RSS_fixed/RSS_free), df=1
    if rss_free > 0 and rss_fixed >= rss_free:
        lr_stat = n * np.log(rss_fixed / rss_free)
    else:
        lr_stat = 0.0
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))

    # R² for both
    ss_tot = float(np.sum((g_mean - np.mean(g_mean)) ** 2))
    r2_free = 1.0 - rss_free / ss_tot if ss_tot > 0 else 0.0
    r2_fixed = 1.0 - rss_fixed / ss_tot if ss_tot > 0 else 0.0

    # σの差
    sigma_diff_pct = abs(sigma_free - SIGMA_PI) / SIGMA_PI * 100

    result = {
        "n_points": n,
        "sigma_free": sigma_free,
        "sigma_pi": float(SIGMA_PI),
        "sigma_diff_pct": sigma_diff_pct,
        "rss_free": rss_free,
        "rss_fixed": rss_fixed,
        "r2_free": r2_free,
        "r2_fixed": r2_fixed,
        "lr_statistic": float(lr_stat),
        "p_value": p_value,
        "reject_H0": p_value < 0.05,
    }

    # --- Plot ---
    model_tag = model_name.replace("/", "_")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_fine = np.linspace(0, 1, 200)

    # Left: data with both fits
    ax = axes[0]
    ax.plot(l_norm, g_mean, "ko", markersize=6, label="g_mean data", zorder=5)
    ax.plot(x_fine, model_erf_free(x_fine, sigma_free), "b-", linewidth=2,
            label=f"erf (σ={sigma_free:.4f}, R²={r2_free:.4f})")
    ax.plot(x_fine, model_erf_fixed(x_fine), "r--", linewidth=2,
            label=f"erf (σ=1/√π={SIGMA_PI:.4f}, R²={r2_fixed:.4f})")
    ax.set_xlabel("l/L")
    ax.set_ylabel("g(l/L)")
    ax.set_title(f"Exp 8A: erf σ test — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)

    # Right: residuals comparison
    ax = axes[1]
    resid_free = g_mean - y_free
    resid_fixed = g_mean - y_fixed
    width = 0.02
    ax.bar(l_norm - width / 2, resid_free, width, color="blue", alpha=0.7,
           label=f"σ_free ({sigma_free:.4f})")
    ax.bar(l_norm + width / 2, resid_fixed, width, color="red", alpha=0.7,
           label=f"σ=1/√π ({SIGMA_PI:.4f})")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("l/L")
    ax.set_ylabel("Residual")
    ax.set_title(f"LR test: χ²={lr_stat:.4f}, p={p_value:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp8a_erf_sigma_test_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return result


# -------------------------------------------------------------------
# Part B: π-containing model fits
# -------------------------------------------------------------------

def model_cos_pi_0p(x: np.ndarray) -> np.ndarray:
    """g(x) = [1 - cos(πx)] / 2  (0 parameters)."""
    return (1.0 - np.cos(np.pi * x)) / 2.0


def model_cos_pi_1p(x: np.ndarray, alpha: float) -> np.ndarray:
    """g(x) = [1 - cos(πx^α)] / 2  (1 parameter)."""
    return (1.0 - np.cos(np.pi * np.power(np.clip(x, 1e-10, None), alpha))) / 2.0


def model_sin_pi_0p(x: np.ndarray) -> np.ndarray:
    """g(x) = sin(πx/2)  (0 parameters)."""
    return np.sin(np.pi * x / 2.0)


def model_sin_pi_1p(x: np.ndarray, alpha: float) -> np.ndarray:
    """g(x) = sin(πx^α / 2)  (1 parameter)."""
    return np.sin(np.pi * np.power(np.clip(x, 1e-10, None), alpha) / 2.0)


PI_MODELS = {
    "cos_pi_0p": {
        "func": model_cos_pi_0p,
        "n_params": 0,
        "label": r"$[1 - \cos(\pi x)] / 2$",
    },
    "cos_pi_1p": {
        "func": model_cos_pi_1p,
        "p0": [1.0],
        "bounds": ([0.01], [10.0]),
        "n_params": 1,
        "label": r"$[1 - \cos(\pi x^{\alpha})] / 2$",
    },
    "sin_pi_0p": {
        "func": model_sin_pi_0p,
        "n_params": 0,
        "label": r"$\sin(\pi x / 2)$",
    },
    "sin_pi_1p": {
        "func": model_sin_pi_1p,
        "p0": [1.0],
        "bounds": ([0.01], [10.0]),
        "n_params": 1,
        "label": r"$\sin(\pi x^{\alpha} / 2)$",
    },
}


def compute_bic(n: int, k: int, rss: float) -> float:
    """Compute BIC = n·ln(RSS/n) + k·ln(n)."""
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def compute_aic(n: int, k: int, rss: float) -> float:
    """Compute AIC = n·ln(RSS/n) + 2k."""
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + 2 * k


def fit_pi_model(x: np.ndarray, y: np.ndarray, model_name: str) -> dict | None:
    """Fit a π-containing model."""
    spec = PI_MODELS[model_name]
    func = spec["func"]
    n = len(x)
    k = spec["n_params"]

    if k == 0:
        # 0パラメータモデル: 直接計算
        y_fit = func(x)
        params = {}
    else:
        try:
            popt, _ = curve_fit(
                func, x, y,
                p0=spec["p0"],
                bounds=spec["bounds"],
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            return None
        y_fit = func(x, *popt)
        param_names = func.__code__.co_varnames[1 : k + 1]
        params = {name: float(val) for name, val in zip(param_names, popt)}

    rss = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    aic = compute_aic(n, k, rss)
    bic = compute_bic(n, k, rss)

    return {
        "model": model_name,
        "params": params,
        "y_fit": y_fit.tolist(),
        "r2": float(r2),
        "rss": rss,
        "aic": float(aic),
        "bic": float(bic),
        "n_params": k,
        "label": spec["label"],
    }


def run_part_b(
    data: dict,
    model_name: str,
    fig_dir: Path,
) -> dict:
    """Fit π-containing models and compare with existing fits."""
    g_mean = np.array(data["g_mean"])
    l_norm = np.array(data["l_norm"])
    n = len(g_mean)

    # π含有モデルをフィット
    pi_results = {}
    for name in PI_MODELS:
        result = fit_pi_model(l_norm, g_mean, name)
        if result is not None:
            pi_results[name] = result

    # 既存モデル (exp3ef) の BIC を再計算（同じ基準で比較するため）
    existing_fits = data.get("fits", {})
    existing_recalc = {}
    for name, fit in existing_fits.items():
        k = fit["n_params"]
        rss = fit["rss"]
        bic = compute_bic(n, k, rss)
        aic = compute_aic(n, k, rss)
        existing_recalc[name] = {
            "model": name,
            "params": fit.get("params", {}),
            "r2": fit["r2"],
            "rss": rss,
            "aic": float(aic),
            "bic": float(bic),
            "n_params": k,
            "label": name,
        }

    # 全モデルを統合してBICソート
    all_models = {}
    all_models.update(existing_recalc)
    all_models.update(pi_results)
    all_models = dict(sorted(all_models.items(), key=lambda x: x[1]["bic"]))

    best_model = next(iter(all_models))
    best_is_pi = best_model in PI_MODELS

    result = {
        "pi_models": {
            name: {k: v for k, v in r.items() if k != "y_fit"}
            for name, r in pi_results.items()
        },
        "all_models_bic_sorted": {
            name: {k: v for k, v in r.items() if k != "y_fit"}
            for name, r in all_models.items()
        },
        "best_model": best_model,
        "best_is_pi_model": best_is_pi,
    }

    # --- Plot ---
    model_tag = model_name.replace("/", "_")

    # (1) π models overlay
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_fine = np.linspace(0, 1, 200)

    ax = axes[0]
    ax.plot(l_norm, g_mean, "ko", markersize=6, label="g_mean data", zorder=5)
    colors = ["C0", "C1", "C2", "C3"]
    for (name, res), color in zip(pi_results.items(), colors):
        spec = PI_MODELS[name]
        func = spec["func"]
        if spec["n_params"] == 0:
            y_plot = func(x_fine)
        else:
            params = list(res["params"].values())
            y_plot = func(x_fine, *params)
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in res["params"].items())
        if param_str:
            param_str = f" ({param_str})"
        ax.plot(x_fine, y_plot, color=color, linewidth=2,
                label=f"{name}{param_str}\nR²={res['r2']:.4f}, BIC={res['bic']:.1f}")
    ax.set_xlabel("l/L")
    ax.set_ylabel("g(l/L)")
    ax.set_title(f"Exp 8B: π-containing models — {model_name}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)

    # (2) BIC comparison bar chart
    ax = axes[1]
    names = list(all_models.keys())
    bics = [all_models[n]["bic"] for n in names]
    is_pi = [n in PI_MODELS for n in names]
    bar_colors = ["C3" if p else "C0" for p in is_pi]
    bars = ax.barh(range(len(names)), bics, color=bar_colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("BIC (lower = better)")
    ax.set_title(f"BIC comparison (red=π model)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = fig_dir / f"exp8b_pi_models_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return result


# -------------------------------------------------------------------
# Part C: Inter-layer rotation angle measurement
# -------------------------------------------------------------------

def load_wikitext2(tokenizer, n_ctx: int, max_tokens: int = 200_000) -> list:
    """Load WikiText-2 test set, chunk into sequences of length n_ctx."""
    import torch
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(dataset["text"])

    tokens = tokenizer.encode(full_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    chunks = []
    for i in range(0, len(tokens) - n_ctx, n_ctx):
        chunk = torch.tensor(tokens[i : i + n_ctx], dtype=torch.long).unsqueeze(0)
        chunks.append(chunk)

    return chunks


def measure_rotation_angles(
    model,
    chunks: list,
    max_samples: int = 2000,
) -> dict:
    """Measure inter-layer rotation angles from residual stream.

    For each token position, compute:
      θ(l) = arccos(cos_sim(resid_post(l), resid_post(l-1)))
    where resid_post(-1) = resid_post of layer 0's input (i.e., embedding).

    Returns dict with per-sample θ arrays and cumulative Θ.
    """
    import torch

    n_layers = model.cfg.n_layers
    device = next(model.parameters()).device

    # θ(l) for each sample: list of arrays, each (n_layers,)
    all_theta = []
    total = 0

    print(f"  Measuring rotation angles ({max_samples} samples)...")
    model.eval()
    with torch.no_grad():
        for ci, chunk in enumerate(chunks):
            if total >= max_samples:
                break
            chunk = chunk.to(device)
            _, cache = model.run_with_cache(chunk)
            seq_len = chunk.shape[1]

            for pos in range(seq_len):
                if total >= max_samples:
                    break

                # Collect resid_post for all layers + embedding
                # resid_pre of layer 0 = embedding output
                vecs = []
                vecs.append(cache["resid_pre", 0][0, pos].cpu().float())
                for l in range(n_layers):
                    vecs.append(cache["resid_post", l][0, pos].cpu().float())

                # Compute angles between consecutive vectors
                thetas = np.zeros(n_layers)
                for l in range(n_layers):
                    v0 = vecs[l].numpy()
                    v1 = vecs[l + 1].numpy()
                    n0 = np.linalg.norm(v0)
                    n1 = np.linalg.norm(v1)
                    if n0 < 1e-10 or n1 < 1e-10:
                        thetas[l] = 0.0
                    else:
                        cos_sim = np.dot(v0, v1) / (n0 * n1)
                        cos_sim = np.clip(cos_sim, -1.0, 1.0)
                        thetas[l] = np.arccos(cos_sim)

                all_theta.append(thetas)
                total += 1

            if (ci + 1) % 10 == 0:
                print(f"    chunk {ci+1}/{len(chunks)}, collected {total} samples")

    all_theta = np.array(all_theta)  # (n_samples, n_layers)
    cumulative = np.sum(all_theta, axis=1)  # (n_samples,)

    print(f"  Collected {total} samples")

    return {
        "all_theta": all_theta,
        "cumulative": cumulative,
        "n_samples": total,
        "n_layers": n_layers,
    }


def measure_rotation_angles_prompts(
    model,
    prompts: dict,
) -> dict:
    """Measure rotation angles for ambiguous word prompts.

    Uses EXPERIMENT_PROMPTS from src/prompts.py.
    Returns per-word, per-condition results.
    """
    import torch

    n_layers = model.cfg.n_layers
    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    results = {}

    model.eval()
    with torch.no_grad():
        for word, conditions in prompts.items():
            word_results = {}
            for cond_name, prompt_text in conditions.items():
                tokens = tokenizer.encode(prompt_text)
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                _, cache = model.run_with_cache(input_ids)

                # Use last token position
                pos = len(tokens) - 1

                vecs = []
                vecs.append(cache["resid_pre", 0][0, pos].cpu().float())
                for l in range(n_layers):
                    vecs.append(cache["resid_post", l][0, pos].cpu().float())

                thetas = np.zeros(n_layers)
                for l in range(n_layers):
                    v0 = vecs[l].numpy()
                    v1 = vecs[l + 1].numpy()
                    n0 = np.linalg.norm(v0)
                    n1 = np.linalg.norm(v1)
                    if n0 < 1e-10 or n1 < 1e-10:
                        thetas[l] = 0.0
                    else:
                        cos_sim = np.dot(v0, v1) / (n0 * n1)
                        cos_sim = np.clip(cos_sim, -1.0, 1.0)
                        thetas[l] = np.arccos(cos_sim)

                cumulative = float(np.sum(thetas))
                word_results[cond_name] = {
                    "thetas": thetas.tolist(),
                    "cumulative": cumulative,
                    "cumulative_over_pi": cumulative / np.pi,
                }

            results[word] = word_results

    return results


def run_part_c(
    model,
    model_name: str,
    fig_dir: Path,
    max_samples: int = 2000,
) -> dict:
    """Run Part C: rotation angle measurement."""
    import torch

    n_layers = model.cfg.n_layers
    model_tag = model_name.replace("/", "_")

    # WikiText-2
    print(f"  Loading WikiText-2...")
    chunks = load_wikitext2(model.tokenizer, model.cfg.n_ctx, max_tokens=200_000)
    print(f"  Loaded {len(chunks)} chunks of {model.cfg.n_ctx} tokens")

    # (1) WikiText-2 rotation angles
    wt_result = measure_rotation_angles(model, chunks, max_samples=max_samples)
    all_theta = wt_result["all_theta"]
    cumulative = wt_result["cumulative"]

    # Stats
    theta_mean = np.mean(all_theta, axis=0)  # (n_layers,)
    theta_std = np.std(all_theta, axis=0)
    cum_mean = float(np.mean(cumulative))
    cum_std = float(np.std(cumulative))
    cum_over_pi = cumulative / np.pi
    cum_over_pi_mean = float(np.mean(cum_over_pi))
    cum_over_pi_std = float(np.std(cum_over_pi))

    # πの整数倍への近さ
    nearest_int = round(cum_over_pi_mean)
    deviation_from_int = abs(cum_over_pi_mean - nearest_int)
    within_01 = float(np.mean(np.abs(cum_over_pi - round(cum_over_pi_mean)) < 0.1))

    # (2) Ambiguous word prompts
    from src.prompts import EXPERIMENT_PROMPTS
    print(f"  Measuring rotation angles for ambiguous word prompts...")
    prompt_result = measure_rotation_angles_prompts(model, EXPERIMENT_PROMPTS)

    # Collect all prompt cumulative/π values
    prompt_cum_over_pi = []
    for word, conds in prompt_result.items():
        for cond, res in conds.items():
            prompt_cum_over_pi.append(res["cumulative_over_pi"])
    prompt_cum_over_pi = np.array(prompt_cum_over_pi)
    prompt_within_01 = float(np.mean(np.abs(prompt_cum_over_pi - np.round(prompt_cum_over_pi)) < 0.1))

    result = {
        "wikitext": {
            "n_samples": wt_result["n_samples"],
            "theta_mean_per_layer": theta_mean.tolist(),
            "theta_std_per_layer": theta_std.tolist(),
            "cumulative_mean": cum_mean,
            "cumulative_std": cum_std,
            "cumulative_over_pi_mean": cum_over_pi_mean,
            "cumulative_over_pi_std": cum_over_pi_std,
            "nearest_integer_multiple": nearest_int,
            "deviation_from_int": deviation_from_int,
            "fraction_within_0.1_of_pi_multiple": within_01,
        },
        "prompts": {
            word: {
                cond: {k: v for k, v in res.items() if k != "thetas_raw"}
                for cond, res in conds.items()
            }
            for word, conds in prompt_result.items()
        },
        "prompts_fraction_within_0.1": prompt_within_01,
        "success_criterion": within_01 >= 0.90,
    }

    # --- Plots ---

    # (1) θ(l) per layer profile
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    layers = np.arange(n_layers)
    ax.errorbar(layers, theta_mean, yerr=theta_std, fmt="o-", capsize=3,
                color="C0", markersize=5)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("θ(l) [radians]")
    ax.set_title(f"Mean rotation angle per layer — {model_name}")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.pi / n_layers, color="red", linestyle="--", alpha=0.5,
               label=f"π/{n_layers} = {np.pi/n_layers:.4f}")
    ax.legend()

    # (2) Θ/π histogram
    ax = axes[0, 1]
    ax.hist(cum_over_pi, bins=50, color="C0", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=cum_over_pi_mean, color="red", linewidth=2,
               label=f"mean={cum_over_pi_mean:.4f}")
    # Mark integer multiples
    for k in range(max(1, int(cum_over_pi_mean) - 1), int(cum_over_pi_mean) + 3):
        ax.axvline(x=k, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Θ / π")
    ax.set_ylabel("Count")
    ax.set_title(f"Cumulative rotation Θ/π (WikiText-2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) Per-word prompt θ profiles
    ax = axes[1, 0]
    cmap = plt.cm.tab10
    word_list = sorted(prompt_result.keys())
    for wi, word in enumerate(word_list):
        for cond in ["neutral", "strong_A", "strong_B"]:
            if cond in prompt_result[word]:
                thetas = prompt_result[word][cond]["thetas"]
                ls = "-" if cond == "neutral" else ("--" if cond == "strong_A" else ":")
                alpha = 1.0 if cond == "neutral" else 0.6
                label = f"{word} ({cond})" if cond == "neutral" else None
                ax.plot(layers, thetas, ls, color=cmap(wi / len(word_list)),
                        alpha=alpha, linewidth=1.5, label=label)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("θ(l) [radians]")
    ax.set_title("Rotation angle for ambiguous word prompts")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (4) Prompt Θ/π comparison
    ax = axes[1, 1]
    positions = []
    labels = []
    values = []
    idx = 0
    for word in word_list:
        for cond in ["neutral", "weak_A", "strong_A", "weak_B", "strong_B"]:
            if cond in prompt_result[word]:
                positions.append(idx)
                labels.append(f"{word[:3]}_{cond[:2]}")
                values.append(prompt_result[word][cond]["cumulative_over_pi"])
                idx += 1
    bar_colors = ["C0" if "ne" in l else ("C1" if "_s" in l else "C2") for l in labels]
    ax.bar(positions, values, color=bar_colors, alpha=0.7)
    ax.axhline(y=cum_over_pi_mean, color="gray", linestyle="--", alpha=0.5,
               label=f"WikiText mean={cum_over_pi_mean:.3f}")
    for k in range(max(1, int(min(values)) - 1), int(max(values)) + 2):
        ax.axhline(y=k, color="green", linestyle=":", alpha=0.3)
    ax.set_xticks(positions[::3])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 3)], rotation=45, fontsize=7)
    ax.set_ylabel("Θ / π")
    ax.set_title("Cumulative rotation for prompts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Exp 8C: Rotation Angle Analysis — {model_name}", fontsize=14)
    plt.tight_layout()
    path = fig_dir / f"exp8c_rotation_angles_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 8: Trace of π in g(l/L)"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--parts", type=str, default="abc",
                        help="Parts to run: a, b, c or combinations")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max samples for Part C rotation measurement")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    out_data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "_")

    all_results: dict = {"model": args.model}

    # --- Part A & B: データのみ（モデルロード不要） ---
    if "a" in args.parts or "b" in args.parts:
        # 両モデルのデータで実行
        for tag in ["gpt2", "gpt2-medium"]:
            exp3ef_path = data_dir / f"exp3ef_{tag}.json"
            if not exp3ef_path.exists():
                print(f"WARNING: {exp3ef_path} not found, skipping")
                continue

            with open(exp3ef_path) as f:
                exp3ef_data = json.load(f)

            if "a" in args.parts:
                print(f"\n{'='*60}")
                print(f"Part A: erf σ parameter π test — {tag}")
                print(f"{'='*60}")

                result_a = run_part_a(exp3ef_data, tag, fig_dir)
                all_results[f"exp8a_{tag}"] = result_a

                # 目立つ表示
                print(f"\n  ╔══════════════════════════════════════════════════╗")
                print(f"  ║  σ_free  = {result_a['sigma_free']:.6f}                          ║")
                print(f"  ║  σ_π     = 1/√π = {result_a['sigma_pi']:.6f}                   ║")
                print(f"  ║  差      = {result_a['sigma_diff_pct']:.2f}%                             ║")
                print(f"  ║  RSS_free  = {result_a['rss_free']:.6f}                        ║")
                print(f"  ║  RSS_fixed = {result_a['rss_fixed']:.6f}                        ║")
                print(f"  ║  χ² = {result_a['lr_statistic']:.4f},  p = {result_a['p_value']:.6f}              ║")
                if result_a["reject_H0"]:
                    print(f"  ║  → H₀(σ=1/√π) 棄却 (p < 0.05)                   ║")
                else:
                    print(f"  ║  → H₀(σ=1/√π) 棄却できず — πと整合的            ║")
                print(f"  ╚══════════════════════════════════════════════════╝")

            if "b" in args.parts:
                print(f"\n{'='*60}")
                print(f"Part B: π-containing model fits — {tag}")
                print(f"{'='*60}")

                result_b = run_part_b(exp3ef_data, tag, fig_dir)
                all_results[f"exp8b_{tag}"] = result_b

                print(f"\n  All models ranked by BIC:")
                print(f"  {'Model':<16} {'k':>3} {'R²':<10} {'BIC':<10} {'π?'}")
                print(f"  {'-'*50}")
                for name, r in result_b["all_models_bic_sorted"].items():
                    is_pi = "★" if name in PI_MODELS else ""
                    print(f"  {name:<16} {r['n_params']:>3} {r['r2']:<10.6f} {r['bic']:<10.2f} {is_pi}")

                print(f"\n  Best model: {result_b['best_model']}"
                      f" (π model: {result_b['best_is_pi_model']})")

    # --- Part C: モデルロード必要 ---
    if "c" in args.parts:
        print(f"\n{'='*60}")
        print(f"Part C: Rotation angle measurement — {args.model}")
        print(f"{'='*60}")

        print(f"  Loading model: {args.model}")
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(args.model, device=args.device)
        print(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

        result_c = run_part_c(model, args.model, fig_dir, max_samples=args.max_samples)
        all_results["exp8c"] = result_c

        wt = result_c["wikitext"]
        print(f"\n  WikiText-2 rotation angles:")
        print(f"    Samples: {wt['n_samples']}")
        print(f"    Cumulative Θ:  {wt['cumulative_mean']:.4f} ± {wt['cumulative_std']:.4f} rad")
        print(f"    Θ/π:           {wt['cumulative_over_pi_mean']:.4f} ± {wt['cumulative_over_pi_std']:.4f}")
        print(f"    Nearest int:   {wt['nearest_integer_multiple']}")
        print(f"    |Θ/π - int|:   {wt['deviation_from_int']:.4f}")
        print(f"    Within ±0.1:   {wt['fraction_within_0.1_of_pi_multiple']*100:.1f}%")

        print(f"\n  Ambiguous word prompts (Θ/π):")
        for word in sorted(result_c["prompts"]):
            conds = result_c["prompts"][word]
            vals = [f"{c[:2]}={conds[c]['cumulative_over_pi']:.3f}" for c in conds]
            print(f"    {word}: {', '.join(vals)}")

        criterion = result_c["success_criterion"]
        print(f"\n  Success criterion (90%+ within π±0.1): {'MET' if criterion else 'NOT MET'}")

    # --- Save ---
    json_path = out_data_dir / f"exp8_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nData saved: {json_path}")

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — exp8 π trace")
    print(f"{'='*60}")

    # Part A summary
    for tag in ["gpt2", "gpt2-medium"]:
        key = f"exp8a_{tag}"
        if key in all_results:
            r = all_results[key]
            status = "CONSISTENT with π" if not r["reject_H0"] else "INCONSISTENT with π"
            print(f"\n  8A ({tag}): σ_free={r['sigma_free']:.4f}, "
                  f"1/√π={r['sigma_pi']:.4f}, p={r['p_value']:.4f} → {status}")

    # Part B summary
    for tag in ["gpt2", "gpt2-medium"]:
        key = f"exp8b_{tag}"
        if key in all_results:
            r = all_results[key]
            print(f"\n  8B ({tag}): Best model = {r['best_model']} "
                  f"(π model: {r['best_is_pi_model']})")

    # Part C summary
    if "exp8c" in all_results:
        r = all_results["exp8c"]
        wt = r["wikitext"]
        print(f"\n  8C: Θ/π = {wt['cumulative_over_pi_mean']:.4f} ± {wt['cumulative_over_pi_std']:.4f}")
        print(f"      Within ±0.1 of integer: {wt['fraction_within_0.1_of_pi_multiple']*100:.1f}%")
        print(f"      Criterion met: {r['success_criterion']}")

    # Overall verdict
    print(f"\n  {'─'*50}")
    verdicts = []
    for tag in ["gpt2", "gpt2-medium"]:
        if f"exp8a_{tag}" in all_results:
            if not all_results[f"exp8a_{tag}"]["reject_H0"]:
                verdicts.append(f"8A({tag})")
        if f"exp8b_{tag}" in all_results:
            if all_results[f"exp8b_{tag}"]["best_is_pi_model"]:
                verdicts.append(f"8B({tag})")
    if "exp8c" in all_results and all_results["exp8c"]["success_criterion"]:
        verdicts.append("8C")

    if verdicts:
        print(f"  π TRACE FOUND in: {', '.join(verdicts)}")
    else:
        print(f"  NO π TRACE FOUND — π does not appear explicitly in g(l/L)")


if __name__ == "__main__":
    main()
