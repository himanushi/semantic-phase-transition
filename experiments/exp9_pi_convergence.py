"""Experiment 9: σ convergence to 1/√π across model sizes.

9A. Measure g(l/L) and erf σ_free for gpt2, gpt2-medium, gpt2-large, gpt2-xl
9B. Measure cumulative rotation angle Θ scaling across model sizes
9C. Verify cos_pi_0p model across all models (uses Part A data)

Designed for Google Colab with A100 GPU.
Each part can be run independently; results are saved to JSON between parts.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# -------------------------------------------------------------------
# JSON helper
# -------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SIGMA_PI = 1.0 / np.sqrt(np.pi)  # ≈ 0.5642

MODELS = [
    ("gpt2", 12, 768),
    ("gpt2-medium", 24, 1024),
    ("gpt2-large", 36, 1280),
    ("gpt2-xl", 48, 1600),
]


def model_tag(model_name: str) -> str:
    return model_name.replace("/", "_")


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def free_model(model, device: str):
    """Delete model and free GPU memory."""
    del model
    import torch
    import gc
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# -------------------------------------------------------------------
# erf model for fitting
# -------------------------------------------------------------------

def model_erf(x: np.ndarray, sigma: float) -> np.ndarray:
    """g(x) = erf(x / (σ√2))."""
    return erf(x / (np.clip(sigma, 1e-10, None) * np.sqrt(2.0)))


# -------------------------------------------------------------------
# Part A: σ convergence test
# -------------------------------------------------------------------

def measure_g_for_model(
    model_name: str,
    device: str,
    words: list[str],
    output_dir: Path,
) -> dict:
    """Measure g(l/L) for a single model from scratch.

    Uses the same functions as exp3_linear_response.py:
    1. compute_gradient_sigmas() for σ profiles
    2. define_h_from_final_layer() for h definition
    3. analyze_linearity_per_layer() for f(l) extraction
    4. g(l/L) = f(l) / f_max (same as exp3ef)

    Falls back to ordinal h if σ_final has no variation (CUDA precision issue).
    """
    import torch
    from transformer_lens import HookedTransformer
    from src.direction import compute_contrastive_direction
    from src.prompts import DIRECTION_PROMPTS, GRADIENT_PROMPTS

    # Import exp3 functions directly to ensure identical logic
    from experiments.exp3_linear_response import (
        compute_gradient_sigmas,
        define_h_from_final_layer,
        analyze_linearity_per_layer,
    )

    print(f"    Loading model: {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float32)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"    Loaded in {fmt_time(time.time()-t0)} — {n_layers} layers, d={d_model}")

    all_slopes = {}  # word -> slopes array (n_layers+1,)
    all_fmax = {}

    for wi, word in enumerate(words):
        if word not in GRADIENT_PROMPTS or word not in DIRECTION_PROMPTS:
            print(f"    [{wi+1}/{len(words)}] SKIP {word}: no prompts defined")
            continue

        dp = DIRECTION_PROMPTS[word]
        print(f"    [{wi+1}/{len(words)}] {word} ({dp['interpretation_A']} vs {dp['interpretation_B']})...")

        # Direction vector (same as exp3)
        try:
            e_diff = compute_contrastive_direction(
                model, dp["prompts_A"], dp["prompts_B"], word, device
            )
        except ValueError as e:
            print(f"      SKIP: {e}")
            continue

        # Gradient sigmas (same function as exp3)
        grad_sigmas = compute_gradient_sigmas(model, word, e_diff, device)
        n_valid_A = sum(1 for s in grad_sigmas["A"] if s is not None)
        n_valid_B = sum(1 for s in grad_sigmas["B"] if s is not None)
        print(f"      Valid prompts: A={n_valid_A}/10, B={n_valid_B}/10")

        if n_valid_A < 3 or n_valid_B < 3:
            print(f"      Too few valid prompts, skipping")
            continue

        # Define h (same function as exp3: h from final-layer σ)
        h_values, sigma_matrix, prompt_indices = define_h_from_final_layer(
            grad_sigmas["A"], grad_sigmas["B"]
        )

        if len(h_values) < 5:
            print(f"      Too few h values ({len(h_values)}), skipping")
            continue

        # Check if h has enough variation
        h_range = float(h_values.max() - h_values.min())
        h_source = "σ_final"

        if h_range < 1e-4:
            # Fallback 1: use σ at middle layer (0.75*L) as h
            # Final layer collapses due to unembedding; middle layers retain context
            h_layer = int(0.75 * n_layers)
            print(f"      σ_final collapsed (range={h_range:.2e}), trying σ at layer {h_layer}...")

            all_h_mid = []
            all_sigma_mid = []
            for s in grad_sigmas["A"]:
                if s is not None:
                    all_h_mid.append(s[h_layer])
                    all_sigma_mid.append(s)
            for s in grad_sigmas["B"]:
                if s is not None:
                    all_h_mid.append(s[h_layer])
                    all_sigma_mid.append(s)

            h_arr_mid = np.array(all_h_mid)
            max_abs = max(abs(h_arr_mid.max()), abs(h_arr_mid.min()))
            if max_abs > 1e-8:
                h_arr_mid = h_arr_mid / max_abs

            h_range = float(h_arr_mid.max() - h_arr_mid.min())
            if h_range > 1e-4:
                h_values = h_arr_mid
                sigma_matrix = np.array(all_sigma_mid)
                order = np.argsort(h_values)
                h_values = h_values[order]
                sigma_matrix = sigma_matrix[order]
                h_source = f"σ_L{h_layer}"
                print(f"      Using σ at layer {h_layer} as h (range={h_range:.3f})")
            else:
                # Fallback 2: ordinal h (prompt ordering)
                print(f"      Middle layer also collapsed, using ordinal h")
                n_A = len(GRADIENT_PROMPTS[word]["A"])
                n_B = len(GRADIENT_PROMPTS[word]["B"])
                h_ordinal = []
                sigma_list = []
                for i, s in enumerate(grad_sigmas["A"]):
                    if s is not None:
                        h_ordinal.append(i / max(n_A - 1, 1))
                        sigma_list.append(s)
                for i, s in enumerate(grad_sigmas["B"]):
                    if s is not None:
                        h_ordinal.append(-i / max(n_B - 1, 1))
                        sigma_list.append(s)
                h_values = np.array(h_ordinal)
                sigma_matrix = np.array(sigma_list)
                order = np.argsort(h_values)
                h_values = h_values[order]
                sigma_matrix = sigma_matrix[order]
                h_range = float(h_values.max() - h_values.min())
                h_source = "ordinal"
                if h_range < 1e-6:
                    print(f"      No h variation at all, skipping")
                    continue

        print(f"      h source: {h_source}, range: [{h_values.min():.3f}, {h_values.max():.3f}], n={len(h_values)}")

        # Linearity analysis (same function as exp3)
        linearity = analyze_linearity_per_layer(h_values, sigma_matrix)
        slopes = np.array([lin["slope"] for lin in linearity])
        r2_values = [lin["r2"] for lin in linearity]

        f_max = float(np.max(slopes))
        if f_max < 1e-8:
            print(f"      f_max too small ({f_max:.6f}), skipping")
            continue

        all_slopes[word] = slopes
        all_fmax[word] = f_max
        r2_mean = np.mean(r2_values)
        print(f"      f_max = {f_max:.4f}, R²_mean = {r2_mean:.4f}")

    # Compute g(l/L) per word and mean
    g_dict = {}
    for word, slopes in all_slopes.items():
        g_dict[word] = slopes / all_fmax[word]

    if not g_dict:
        free_model(model, device)
        return {}

    g_arrays = list(g_dict.values())
    g_mean = np.mean(g_arrays, axis=0)
    g_std = np.std(g_arrays, axis=0)
    l_norm = np.linspace(0, 1, n_layers + 1)

    # erf fit
    try:
        popt, _ = curve_fit(
            model_erf, l_norm, g_mean,
            p0=[0.5], bounds=([0.01], [10.0]), maxfev=20000,
        )
        sigma_free = float(popt[0])
    except (RuntimeError, ValueError):
        sigma_free = float("nan")

    y_fit = model_erf(l_norm, sigma_free)
    rss = float(np.sum((g_mean - y_fit) ** 2))
    ss_tot = float(np.sum((g_mean - np.mean(g_mean)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

    # Per-word σ_free
    sigma_per_word = {}
    for word, g in g_dict.items():
        try:
            popt_w, _ = curve_fit(
                model_erf, l_norm, g,
                p0=[0.5], bounds=([0.01], [10.0]), maxfev=20000,
            )
            sigma_per_word[word] = float(popt_w[0])
        except (RuntimeError, ValueError):
            sigma_per_word[word] = float("nan")

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "words": list(g_dict.keys()),
        "fmax": {w: float(v) for w, v in all_fmax.items()},
        "g_per_word": {w: g.tolist() for w, g in g_dict.items()},
        "g_mean": g_mean.tolist(),
        "g_std": g_std.tolist(),
        "l_norm": l_norm.tolist(),
        "sigma_free": sigma_free,
        "sigma_per_word": sigma_per_word,
        "erf_r2": r2,
        "erf_rss": rss,
    }

    # Save per-model result
    tag = model_tag(model_name)
    json_path = output_dir / "data" / f"exp9_part_a_{tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"    Saved: {json_path}")

    free_model(model, device)
    return result


def run_part_a(
    device: str,
    words: list[str],
    output_dir: Path,
) -> dict:
    """Run Part A: measure σ_free for all 4 model sizes."""
    print("\n" + "=" * 60)
    print("Part A: σ convergence test (erf σ_free across model sizes)")
    print("=" * 60)
    print(f"  Models: {[m[0] for m in MODELS]}")
    print(f"  Words: {words}")
    print(f"  Estimated time: 4 models × ~8min = ~32min total")
    print()

    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mi, (mname, n_layers, d_model) in enumerate(MODELS):
        print(f"\n  [{mi+1}/{len(MODELS)}] {mname} ({n_layers} layers, d={d_model})")
        t_start = time.time()
        tag = model_tag(mname)

        # Check for existing exp3ef data first
        exp3ef_path = data_dir / f"exp3ef_{tag}.json"
        exp9a_path = data_dir / f"exp9_part_a_{tag}.json"

        if exp3ef_path.exists():
            print(f"    Found existing exp3ef data: {exp3ef_path}")
            with open(exp3ef_path) as f:
                exp3ef_data = json.load(f)

            sigma_free = exp3ef_data["fits"]["erf"]["params"]["sigma"]
            erf_r2 = exp3ef_data["fits"]["erf"]["r2"]
            erf_rss = exp3ef_data["fits"]["erf"]["rss"]

            # Per-word σ from exp3ef
            sigma_per_word = {}
            l_norm = np.array(exp3ef_data["l_norm"])
            for word, g_vals in exp3ef_data["g_per_word"].items():
                g = np.array(g_vals)
                try:
                    popt, _ = curve_fit(
                        model_erf, l_norm, g,
                        p0=[0.5], bounds=([0.01], [10.0]), maxfev=20000,
                    )
                    sigma_per_word[word] = float(popt[0])
                except (RuntimeError, ValueError):
                    sigma_per_word[word] = float("nan")

            results[mname] = {
                "model": mname,
                "n_layers": exp3ef_data["n_layers"],
                "d_model": d_model,
                "words": exp3ef_data["words"],
                "fmax": exp3ef_data["fmax"],
                "g_per_word": exp3ef_data["g_per_word"],
                "g_mean": exp3ef_data["g_mean"],
                "g_std": exp3ef_data["g_std"],
                "l_norm": exp3ef_data["l_norm"],
                "sigma_free": sigma_free,
                "sigma_per_word": sigma_per_word,
                "erf_r2": erf_r2,
                "erf_rss": erf_rss,
                "source": "exp3ef",
            }
            elapsed = time.time() - t_start
            print(f"    σ_free = {sigma_free:.4f} (from exp3ef), R² = {erf_r2:.4f}")
            print(f"    Done in {fmt_time(elapsed)}")

        elif exp9a_path.exists():
            print(f"    Found existing exp9a data: {exp9a_path}")
            with open(exp9a_path) as f:
                result = json.load(f)
            results[mname] = result
            elapsed = time.time() - t_start
            print(f"    σ_free = {result['sigma_free']:.4f} (cached), R² = {result['erf_r2']:.4f}")
            print(f"    Done in {fmt_time(elapsed)}")

        else:
            print(f"    No cached data found, measuring from scratch...")
            result = measure_g_for_model(mname, device, words, output_dir)
            if result:
                results[mname] = result
                elapsed = time.time() - t_start
                print(f"    σ_free = {result['sigma_free']:.4f}, R² = {result['erf_r2']:.4f}")
                print(f"    Done in {fmt_time(elapsed)}")
            else:
                print(f"    FAILED — no valid words")

    if len(results) < 2:
        print("\n  ERROR: Need at least 2 models for convergence analysis")
        return {"models": results}

    # --- Convergence analysis ---
    print(f"\n  {'─'*50}")
    print(f"  σ convergence analysis:")
    print(f"  {'Model':<16} {'n_layers':>8} {'σ_free':>8} {'1/√π':>8} {'diff%':>8} {'R²':>8}")
    print(f"  {'-'*58}")

    x_inv = []  # 1/n_layers
    y_sigma = []
    y_sigma_err = []  # std across words

    for mname, _, _ in MODELS:
        if mname not in results:
            continue
        r = results[mname]
        nl = r["n_layers"]
        sf = r["sigma_free"]
        diff_pct = abs(sf - SIGMA_PI) / SIGMA_PI * 100

        # Error from per-word σ
        sw = r.get("sigma_per_word", {})
        sigma_vals = [v for v in sw.values() if not math.isnan(v)]
        sigma_err = float(np.std(sigma_vals)) if len(sigma_vals) >= 2 else 0.0

        x_inv.append(1.0 / nl)
        y_sigma.append(sf)
        y_sigma_err.append(sigma_err)

        print(f"  {mname:<16} {nl:>8} {sf:>8.4f} {SIGMA_PI:>8.4f} {diff_pct:>7.2f}% {r['erf_r2']:>8.4f}")

    x_inv = np.array(x_inv)
    y_sigma = np.array(y_sigma)
    y_sigma_err = np.array(y_sigma_err)

    # Linear regression: σ = a + b * (1/n_layers)
    # Extrapolate to 1/n_layers → 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_inv, y_sigma)
    sigma_extrap = intercept  # value at 1/n_layers = 0

    # 95% CI for intercept
    n_pts = len(x_inv)
    if n_pts > 2:
        x_mean = np.mean(x_inv)
        ss_xx = np.sum((x_inv - x_mean) ** 2)
        s_resid = np.sqrt(np.sum((y_sigma - (slope * x_inv + intercept)) ** 2) / (n_pts - 2))
        se_intercept = s_resid * np.sqrt(1.0 / n_pts + x_mean ** 2 / ss_xx)
        t_crit = stats.t.ppf(0.975, n_pts - 2)
        ci_lower = intercept - t_crit * se_intercept
        ci_upper = intercept + t_crit * se_intercept
    else:
        se_intercept = float("nan")
        ci_lower = float("nan")
        ci_upper = float("nan")

    within_ci = ci_lower <= SIGMA_PI <= ci_upper if not math.isnan(ci_lower) else False

    print(f"\n  Linear extrapolation: σ = {intercept:.4f} + {slope:.4f} × (1/n_layers)")
    print(f"  σ(∞) = {sigma_extrap:.4f} ± {se_intercept:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  1/√π = {SIGMA_PI:.4f}")
    print(f"  Within 95% CI: {'YES' if within_ci else 'NO'}")

    # --- Plot: σ convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: σ vs 1/n_layers
    ax = axes[0]
    ax.errorbar(x_inv, y_sigma, yerr=y_sigma_err, fmt="ko", capsize=5,
                markersize=8, linewidth=2, zorder=5, label="σ_free (±word std)")

    # Extrapolation line
    x_extrap = np.linspace(0, max(x_inv) * 1.1, 100)
    ax.plot(x_extrap, slope * x_extrap + intercept, "b--", linewidth=1.5,
            label=f"Linear fit (R²={r_value**2:.3f})")

    # Extrapolated point
    ax.plot(0, sigma_extrap, "b^", markersize=12, zorder=5,
            label=f"σ(∞) = {sigma_extrap:.4f}")

    # 95% CI band at x=0
    if not math.isnan(ci_lower):
        ax.errorbar(0, sigma_extrap, yerr=[[sigma_extrap - ci_lower], [ci_upper - sigma_extrap]],
                     fmt="none", capsize=8, color="blue", linewidth=2)

    # 1/√π line
    ax.axhline(y=SIGMA_PI, color="red", linestyle="-", linewidth=2, alpha=0.8,
               label=f"1/√π = {SIGMA_PI:.4f}")

    # Model labels
    model_labels = [m[0] for m in MODELS if m[0] in results]
    for xi, yi, label in zip(x_inv, y_sigma, model_labels):
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="gray")

    ax.set_xlabel("1 / n_layers", fontsize=12)
    ax.set_ylabel("erf σ_free", fontsize=12)
    ax.set_title("σ convergence to 1/√π", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.005, max(x_inv) * 1.15)

    # Annotate verdict
    diff_pct = abs(sigma_extrap - SIGMA_PI) / SIGMA_PI * 100
    verdict_text = (
        f"σ(∞) = {sigma_extrap:.4f} ± {se_intercept:.4f}\n"
        f"1/√π = {SIGMA_PI:.4f}\n"
        f"Diff: {diff_pct:.1f}%\n"
        f"Within 95% CI: {'YES' if within_ci else 'NO'}"
    )
    ax.text(0.95, 0.05, verdict_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Right: g(l/L) overlay for all models
    ax = axes[1]
    colors_map = {"gpt2": "C0", "gpt2-medium": "C1", "gpt2-large": "C2", "gpt2-xl": "C3"}
    for mname, _, _ in MODELS:
        if mname not in results:
            continue
        r = results[mname]
        l_norm = np.array(r["l_norm"])
        g_mean = np.array(r["g_mean"])
        color = colors_map.get(mname, "gray")
        ax.plot(l_norm, g_mean, "o-", color=color, markersize=4, linewidth=1.5,
                label=f"{mname} (σ={r['sigma_free']:.3f})")

    # erf with 1/√π
    x_fine = np.linspace(0, 1, 200)
    ax.plot(x_fine, model_erf(x_fine, SIGMA_PI), "r--", linewidth=2, alpha=0.7,
            label=f"erf(σ=1/√π={SIGMA_PI:.4f})")

    ax.set_xlabel("l/L", fontsize=12)
    ax.set_ylabel("g(l/L)", fontsize=12)
    ax.set_title("Universal permeation function — all models", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    path = fig_dir / "exp9a_sigma_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)

    # g overlay only
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for mname, _, _ in MODELS:
        if mname not in results:
            continue
        r = results[mname]
        l_norm = np.array(r["l_norm"])
        g_mean = np.array(r["g_mean"])
        g_std = np.array(r["g_std"])
        color = colors_map.get(mname, "gray")
        ax.plot(l_norm, g_mean, "o-", color=color, markersize=5, linewidth=2,
                label=f"{mname} ({r['n_layers']}L, σ={r['sigma_free']:.4f})")
        ax.fill_between(l_norm, g_mean - g_std, g_mean + g_std,
                         alpha=0.1, color=color)

    ax.plot(x_fine, model_erf(x_fine, SIGMA_PI), "r--", linewidth=2, alpha=0.7,
            label=f"erf(σ=1/√π)")
    ax.set_xlabel("Normalized layer l/L", fontsize=12)
    ax.set_ylabel("g(l/L)", fontsize=12)
    ax.set_title("Universal Permeation Function g(l/L) — 4 Model Sizes", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    path = fig_dir / "exp9a_g_overlay_all.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    convergence = {
        "x_inv_layers": x_inv.tolist(),
        "y_sigma_free": y_sigma.tolist(),
        "y_sigma_err": y_sigma_err.tolist(),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_r2": float(r_value ** 2),
        "sigma_extrapolated": float(sigma_extrap),
        "se_intercept": float(se_intercept),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "sigma_pi": float(SIGMA_PI),
        "within_95_ci": bool(within_ci),
    }

    return {"models": results, "convergence": convergence}


# -------------------------------------------------------------------
# Part B: Θ (cumulative rotation angle) scaling
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


def measure_rotation_for_model(
    model_name: str,
    device: str,
    max_samples: int,
    output_dir: Path,
) -> dict:
    """Measure cumulative rotation angle Θ for a single model."""
    import torch
    from transformer_lens import HookedTransformer

    tag = model_tag(model_name)

    print(f"    Loading model: {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float32)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"    Loaded in {fmt_time(time.time()-t0)} — {n_layers} layers, d={d_model}")

    print(f"    Loading WikiText-2...")
    chunks = load_wikitext2(model.tokenizer, model.cfg.n_ctx, max_tokens=200_000)
    print(f"    {len(chunks)} chunks of {model.cfg.n_ctx} tokens")

    print(f"    Measuring rotation angles ({max_samples} samples)...")
    all_theta = []
    total = 0

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

                all_theta.append(thetas)
                total += 1

            if (ci + 1) % 5 == 0 or total >= max_samples:
                print(f"      chunk {ci+1}/{len(chunks)}, collected {total}/{max_samples} samples")

    all_theta = np.array(all_theta)
    cumulative = np.sum(all_theta, axis=1)

    theta_mean_per_layer = np.mean(all_theta, axis=0).tolist()
    theta_std_per_layer = np.std(all_theta, axis=0).tolist()
    cum_mean = float(np.mean(cumulative))
    cum_std = float(np.std(cumulative))

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_samples": total,
        "theta_mean_per_layer": theta_mean_per_layer,
        "theta_std_per_layer": theta_std_per_layer,
        "cumulative_mean": cum_mean,
        "cumulative_std": cum_std,
        "cumulative_over_pi": cum_mean / np.pi,
        "cumulative_over_pi_sq_half": cum_mean / (np.pi ** 2 / 2),
    }

    json_path = output_dir / "data" / f"exp9_part_b_{tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"    Saved: {json_path}")

    free_model(model, device)
    return result


def run_part_b(
    device: str,
    max_samples: int,
    output_dir: Path,
) -> dict:
    """Run Part B: measure Θ for all 4 model sizes."""
    print("\n" + "=" * 60)
    print("Part B: Cumulative rotation angle Θ scaling")
    print("=" * 60)
    print(f"  Models: {[m[0] for m in MODELS]}")
    print(f"  Max samples per model: {max_samples}")
    print(f"  Estimated time: 4 models × ~5min = ~20min total")
    print()

    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mi, (mname, n_layers, d_model) in enumerate(MODELS):
        print(f"\n  [{mi+1}/{len(MODELS)}] {mname} ({n_layers} layers, d={d_model})")
        t_start = time.time()
        tag = model_tag(mname)

        # Check cache
        cached_path = data_dir / f"exp9_part_b_{tag}.json"
        if cached_path.exists():
            print(f"    Found cached data: {cached_path}")
            with open(cached_path) as f:
                result = json.load(f)
            results[mname] = result
            print(f"    Θ = {result['cumulative_mean']:.4f} ± {result['cumulative_std']:.4f}")
            print(f"    Done in {fmt_time(time.time()-t_start)}")
        else:
            result = measure_rotation_for_model(mname, device, max_samples, output_dir)
            results[mname] = result
            elapsed = time.time() - t_start
            print(f"    Θ = {result['cumulative_mean']:.4f} ± {result['cumulative_std']:.4f}")
            print(f"    Done in {fmt_time(elapsed)}")

    if len(results) < 2:
        print("\n  ERROR: Need at least 2 models for scaling analysis")
        return {"models": results}

    # --- Scaling analysis ---
    print(f"\n  {'─'*50}")
    print(f"  Θ scaling analysis:")
    print(f"  {'Model':<16} {'n_L':>4} {'Θ':>8} {'±':>6} {'Θ/π':>8} {'Θ/(π²/2)':>10} {'Θ/π²':>8}")
    print(f"  {'-'*62}")

    n_layers_arr = []
    theta_arr = []
    theta_err = []

    for mname, nl, _ in MODELS:
        if mname not in results:
            continue
        r = results[mname]
        n_layers_arr.append(r["n_layers"])
        theta_arr.append(r["cumulative_mean"])
        theta_err.append(r["cumulative_std"])

        pi2h = np.pi ** 2 / 2
        print(f"  {mname:<16} {r['n_layers']:>4} {r['cumulative_mean']:>8.3f} "
              f"{r['cumulative_std']:>6.3f} "
              f"{r['cumulative_mean']/np.pi:>8.3f} "
              f"{r['cumulative_mean']/pi2h:>10.3f} "
              f"{r['cumulative_mean']/np.pi**2:>8.3f}")

    n_layers_arr = np.array(n_layers_arr, dtype=float)
    theta_arr = np.array(theta_arr)
    theta_err = np.array(theta_err)

    # Fit candidate scaling models
    scaling_fits = {}

    # Model 1: Θ = constant = π²/2
    pi2h = np.pi ** 2 / 2
    rss_const = float(np.sum((theta_arr - pi2h) ** 2))
    n = len(theta_arr)
    bic_const = n * np.log(rss_const / n) + 0 * np.log(n) if rss_const > 0 else float("inf")
    scaling_fits["constant_pi2_half"] = {
        "formula": "Θ = π²/2",
        "n_params": 0,
        "rss": rss_const,
        "bic": float(bic_const),
        "predicted": [pi2h] * n,
    }

    # Model 2: Θ = a * √n_layers
    try:
        popt, _ = curve_fit(lambda x, a: a * np.sqrt(x), n_layers_arr, theta_arr,
                            p0=[1.0], maxfev=10000)
        y_fit = popt[0] * np.sqrt(n_layers_arr)
        rss = float(np.sum((theta_arr - y_fit) ** 2))
        bic = n * np.log(rss / n) + 1 * np.log(n) if rss > 0 else float("inf")
        scaling_fits["sqrt_n"] = {
            "formula": f"Θ = {popt[0]:.4f} × √n_layers",
            "n_params": 1,
            "a": float(popt[0]),
            "rss": rss,
            "bic": float(bic),
            "predicted": y_fit.tolist(),
        }
    except (RuntimeError, ValueError):
        pass

    # Model 3: Θ = a * log(n_layers)
    try:
        popt, _ = curve_fit(lambda x, a: a * np.log(x), n_layers_arr, theta_arr,
                            p0=[1.0], maxfev=10000)
        y_fit = popt[0] * np.log(n_layers_arr)
        rss = float(np.sum((theta_arr - y_fit) ** 2))
        bic = n * np.log(rss / n) + 1 * np.log(n) if rss > 0 else float("inf")
        scaling_fits["log_n"] = {
            "formula": f"Θ = {popt[0]:.4f} × log(n_layers)",
            "n_params": 1,
            "a": float(popt[0]),
            "rss": rss,
            "bic": float(bic),
            "predicted": y_fit.tolist(),
        }
    except (RuntimeError, ValueError):
        pass

    # Model 4: Θ = a * n_layers + b (linear)
    try:
        popt, _ = curve_fit(lambda x, a, b: a * x + b, n_layers_arr, theta_arr,
                            p0=[0.1, 1.0], maxfev=10000)
        y_fit = popt[0] * n_layers_arr + popt[1]
        rss = float(np.sum((theta_arr - y_fit) ** 2))
        bic = n * np.log(rss / n) + 2 * np.log(n) if rss > 0 else float("inf")
        scaling_fits["linear"] = {
            "formula": f"Θ = {popt[0]:.4f} × n + {popt[1]:.4f}",
            "n_params": 2,
            "a": float(popt[0]),
            "b": float(popt[1]),
            "rss": rss,
            "bic": float(bic),
            "predicted": y_fit.tolist(),
        }
    except (RuntimeError, ValueError):
        pass

    # Sort by BIC
    scaling_fits = dict(sorted(scaling_fits.items(), key=lambda x: x[1]["bic"]))

    print(f"\n  Scaling model comparison (BIC):")
    print(f"  {'Model':<25} {'k':>3} {'BIC':>10} {'Formula'}")
    print(f"  {'-'*65}")
    for name, fit in scaling_fits.items():
        print(f"  {name:<25} {fit['n_params']:>3} {fit['bic']:>10.2f} {fit['formula']}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Θ vs n_layers
    ax = axes[0]
    ax.errorbar(n_layers_arr, theta_arr, yerr=theta_err, fmt="ko", capsize=5,
                markersize=8, linewidth=2, zorder=5, label="Θ (measured)")

    # π²/2 line
    ax.axhline(y=pi2h, color="red", linestyle="-", linewidth=2, alpha=0.8,
               label=f"π²/2 = {pi2h:.4f}")

    # Best fit
    n_fine = np.linspace(min(n_layers_arr) * 0.8, max(n_layers_arr) * 1.2, 100)
    best_name = next(iter(scaling_fits))
    best_fit = scaling_fits[best_name]
    if "sqrt_n" in scaling_fits:
        a = scaling_fits["sqrt_n"]["a"]
        ax.plot(n_fine, a * np.sqrt(n_fine), "b--", linewidth=1.5,
                label=f"√n fit: {scaling_fits['sqrt_n']['formula']}")
    if "log_n" in scaling_fits:
        a = scaling_fits["log_n"]["a"]
        ax.plot(n_fine, a * np.log(n_fine), "g--", linewidth=1.5,
                label=f"log fit: {scaling_fits['log_n']['formula']}")
    if "linear" in scaling_fits:
        a = scaling_fits["linear"]["a"]
        b = scaling_fits["linear"]["b"]
        ax.plot(n_fine, a * n_fine + b, "m--", linewidth=1.5,
                label=f"linear: {scaling_fits['linear']['formula']}")

    # Model labels
    for nl, th, mname in zip(n_layers_arr, theta_arr, [m[0] for m in MODELS if m[0] in results]):
        ax.annotate(mname, (nl, th), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="gray")

    ax.set_xlabel("n_layers", fontsize=12)
    ax.set_ylabel("Θ (cumulative rotation, radians)", fontsize=12)
    ax.set_title("Cumulative rotation angle scaling", fontsize=14)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Right: per-layer θ profiles
    ax = axes[1]
    for mname, nl, _ in MODELS:
        if mname not in results:
            continue
        r = results[mname]
        layers = np.arange(r["n_layers"])
        l_norm = layers / r["n_layers"]
        theta_mean = np.array(r["theta_mean_per_layer"])
        color = {"gpt2": "C0", "gpt2-medium": "C1", "gpt2-large": "C2", "gpt2-xl": "C3"}.get(mname, "gray")
        ax.plot(l_norm, theta_mean, "o-", color=color, markersize=3, linewidth=1.5,
                label=f"{mname} ({r['n_layers']}L)")

    ax.set_xlabel("l / n_layers", fontsize=12)
    ax.set_ylabel("θ(l) mean [radians]", fontsize=12)
    ax.set_title("Per-layer rotation angle profile", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "exp9b_theta_scaling.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)

    return {"models": results, "scaling_fits": scaling_fits}


# -------------------------------------------------------------------
# Part C: cos_pi_0p verification across models
# -------------------------------------------------------------------

def model_cos_pi_0p(x: np.ndarray) -> np.ndarray:
    """g(x) = [1 - cos(πx)] / 2  (0 parameters)."""
    return (1.0 - np.cos(np.pi * x)) / 2.0


def model_sin_pi_1p(x: np.ndarray, alpha: float) -> np.ndarray:
    """g(x) = sin(πx^α / 2)  (1 parameter)."""
    return np.sin(np.pi * np.power(np.clip(x, 1e-10, None), alpha) / 2.0)


def compute_bic(n: int, k: int, rss: float) -> float:
    """BIC = n·ln(RSS/n) + k·ln(n)."""
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def run_part_c(output_dir: Path) -> dict:
    """Run Part C: compare cos_pi_0p, erf, sin_pi_1p across all models."""
    print("\n" + "=" * 60)
    print("Part C: π model verification across model sizes")
    print("=" * 60)
    print(f"  Estimated time: < 1 minute (data only, no model loading)")
    print()

    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mname, n_layers, d_model in MODELS:
        tag = model_tag(mname)

        # Try to load Part A data
        for path_candidate in [
            data_dir / f"exp9_part_a_{tag}.json",
            data_dir / f"exp3ef_{tag}.json",
        ]:
            if path_candidate.exists():
                with open(path_candidate) as f:
                    data = json.load(f)
                break
        else:
            print(f"  SKIP {mname}: no Part A or exp3ef data found")
            continue

        l_norm = np.array(data["l_norm"])
        g_mean = np.array(data["g_mean"])
        n = len(g_mean)

        print(f"  {mname} ({n_layers}L, {n} data points):")

        # --- cos_pi_0p (0 params) ---
        y_cos = model_cos_pi_0p(l_norm)
        rss_cos = float(np.sum((g_mean - y_cos) ** 2))
        ss_tot = float(np.sum((g_mean - np.mean(g_mean)) ** 2))
        r2_cos = 1.0 - rss_cos / ss_tot if ss_tot > 0 else 0.0
        bic_cos = compute_bic(n, 0, rss_cos)

        # --- erf (1 param) ---
        try:
            popt, _ = curve_fit(model_erf, l_norm, g_mean,
                                p0=[0.5], bounds=([0.01], [10.0]), maxfev=20000)
            sigma_erf = float(popt[0])
            y_erf = model_erf(l_norm, sigma_erf)
        except (RuntimeError, ValueError):
            sigma_erf = float("nan")
            y_erf = np.zeros_like(g_mean)
        rss_erf = float(np.sum((g_mean - y_erf) ** 2))
        r2_erf = 1.0 - rss_erf / ss_tot if ss_tot > 0 else 0.0
        bic_erf = compute_bic(n, 1, rss_erf)

        # --- sin_pi_1p (1 param) ---
        try:
            popt, _ = curve_fit(model_sin_pi_1p, l_norm, g_mean,
                                p0=[1.0], bounds=([0.01], [10.0]), maxfev=20000)
            alpha_sin = float(popt[0])
            y_sin = model_sin_pi_1p(l_norm, alpha_sin)
        except (RuntimeError, ValueError):
            alpha_sin = float("nan")
            y_sin = np.zeros_like(g_mean)
        rss_sin = float(np.sum((g_mean - y_sin) ** 2))
        r2_sin = 1.0 - rss_sin / ss_tot if ss_tot > 0 else 0.0
        bic_sin = compute_bic(n, 1, rss_sin)

        delta_bic = bic_cos - bic_erf

        results[mname] = {
            "n_layers": n_layers,
            "n_points": n,
            "cos_pi_0p": {"r2": r2_cos, "bic": float(bic_cos), "rss": rss_cos},
            "erf": {"r2": r2_erf, "bic": float(bic_erf), "rss": rss_erf, "sigma": sigma_erf},
            "sin_pi_1p": {"r2": r2_sin, "bic": float(bic_sin), "rss": rss_sin, "alpha": alpha_sin},
            "delta_bic_cos_minus_erf": float(delta_bic),
        }

        print(f"    cos_pi_0p: R²={r2_cos:.4f}, BIC={bic_cos:.2f}")
        print(f"    erf:       R²={r2_erf:.4f}, BIC={bic_erf:.2f}, σ={sigma_erf:.4f}")
        print(f"    sin_pi_1p: R²={r2_sin:.4f}, BIC={bic_sin:.2f}, α={alpha_sin:.4f}")
        print(f"    ΔBIC(cos-erf) = {delta_bic:+.2f} ({'cos wins' if delta_bic < 0 else 'erf wins'})")

    if not results:
        print("\n  ERROR: No data available for Part C")
        return {"models": results}

    # --- Plot: ΔBIC vs n_points ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: ΔBIC vs data points
    ax = axes[0]
    n_points_list = []
    delta_bic_list = []
    model_names = []
    for mname, _, _ in MODELS:
        if mname not in results:
            continue
        r = results[mname]
        n_points_list.append(r["n_points"])
        delta_bic_list.append(r["delta_bic_cos_minus_erf"])
        model_names.append(mname)

    ax.plot(n_points_list, delta_bic_list, "ko-", markersize=10, linewidth=2, zorder=5)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="ΔBIC = 0 (equal)")
    ax.axhline(y=-2, color="green", linestyle=":", alpha=0.5, label="ΔBIC = -2 (positive evidence)")
    ax.axhline(y=-6, color="green", linestyle="--", alpha=0.5, label="ΔBIC = -6 (strong evidence)")

    for np_val, db, mname in zip(n_points_list, delta_bic_list, model_names):
        ax.annotate(mname, (np_val, db), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)

    ax.set_xlabel("Number of data points (n_layers + 1)", fontsize=12)
    ax.set_ylabel("ΔBIC = BIC(cos_pi_0p) - BIC(erf)", fontsize=12)
    ax.set_title("Does cos(πx) improve with more data?", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Shade regions
    ylim = ax.get_ylim()
    ax.fill_between(ax.get_xlim(), ylim[0], 0, alpha=0.05, color="green")
    ax.fill_between(ax.get_xlim(), 0, ylim[1], alpha=0.05, color="red")
    ax.text(0.02, 0.02, "cos_pi_0p better", transform=ax.transAxes,
            fontsize=9, color="green", alpha=0.7)
    ax.text(0.02, 0.98, "erf better", transform=ax.transAxes,
            fontsize=9, color="red", alpha=0.7, va="top")
    ax.set_ylim(ylim)

    # Right: R² comparison
    ax = axes[1]
    x_pos = np.arange(len(model_names))
    width = 0.25

    r2_cos = [results[m]["cos_pi_0p"]["r2"] for m in model_names]
    r2_erf = [results[m]["erf"]["r2"] for m in model_names]
    r2_sin = [results[m]["sin_pi_1p"]["r2"] for m in model_names]

    bars1 = ax.bar(x_pos - width, r2_cos, width, label="cos_pi_0p (0p)", color="C3", alpha=0.8)
    bars2 = ax.bar(x_pos, r2_erf, width, label="erf (1p)", color="C0", alpha=0.8)
    bars3 = ax.bar(x_pos + width, r2_sin, width, label="sin_pi_1p (1p)", color="C1", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("R² comparison across models", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.8, 1.0)

    plt.tight_layout()
    path = fig_dir / "exp9c_bic_vs_npoints.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)

    return {"models": results}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 9: σ convergence to 1/√π across model sizes"
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda for Colab)")
    parser.add_argument("--parts", type=str, default="abc",
                        help="Parts to run: a, b, c or combinations")
    parser.add_argument("--words", nargs="*", default=["rock", "spring", "bass", "light"],
                        help="Words for Part A measurement")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max WikiText samples per model for Part B")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 9: Does σ converge to 1/√π?")
    print("=" * 60)
    print(f"  Device: {args.device}")
    print(f"  Parts: {args.parts}")
    print(f"  Output: {output_dir}")
    print()

    all_results = {}

    # --- Part A ---
    if "a" in args.parts:
        t0 = time.time()
        result_a = run_part_a(args.device, args.words, output_dir)
        all_results["part_a"] = result_a
        print(f"\n  Part A total time: {fmt_time(time.time()-t0)}")

    # --- Part B ---
    if "b" in args.parts:
        t0 = time.time()
        result_b = run_part_b(args.device, args.max_samples, output_dir)
        all_results["part_b"] = result_b
        print(f"\n  Part B total time: {fmt_time(time.time()-t0)}")

    # --- Part C ---
    if "c" in args.parts:
        t0 = time.time()
        result_c = run_part_c(output_dir)
        all_results["part_c"] = result_c
        print(f"\n  Part C total time: {fmt_time(time.time()-t0)}")

    # --- Save unified results ---
    # Build a serializable summary
    summary = {"experiment": "exp9_pi_convergence"}

    if "part_a" in all_results:
        pa = all_results["part_a"]
        summary["part_a"] = {
            "models": {
                mname: {
                    "n_layers": r["n_layers"],
                    "d_model": r.get("d_model", 0),
                    "sigma_free": r["sigma_free"],
                    "sigma_per_word": r.get("sigma_per_word", {}),
                    "erf_r2": r["erf_r2"],
                }
                for mname, r in pa["models"].items()
            },
        }
        if "convergence" in pa:
            summary["part_a"]["convergence"] = pa["convergence"]

    if "part_b" in all_results:
        pb = all_results["part_b"]
        summary["part_b"] = {
            "models": {
                mname: {
                    "n_layers": r["n_layers"],
                    "cumulative_mean": r["cumulative_mean"],
                    "cumulative_std": r["cumulative_std"],
                    "cumulative_over_pi": r["cumulative_over_pi"],
                }
                for mname, r in pb["models"].items()
            },
        }
        if "scaling_fits" in pb:
            summary["part_b"]["scaling_fits"] = {
                name: {k: v for k, v in fit.items() if k != "predicted"}
                for name, fit in pb["scaling_fits"].items()
            }

    if "part_c" in all_results:
        summary["part_c"] = all_results["part_c"]

    json_path = data_dir / "exp9_convergence.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\nUnified results saved: {json_path}")

    # --- VERDICT ---
    print()
    print("=" * 60)
    print("VERDICT: Does σ converge to 1/√π?")
    print("=" * 60)

    if "part_a" in all_results and "convergence" in all_results["part_a"]:
        conv = all_results["part_a"]["convergence"]
        sigma_vals = conv["y_sigma_free"]
        sigma_ext = conv["sigma_extrapolated"]
        se = conv["se_intercept"]
        ci_lo = conv["ci_95_lower"]
        ci_hi = conv["ci_95_upper"]
        within = conv["within_95_ci"]

        print(f"  σ_free values: {[f'{v:.4f}' for v in sigma_vals]}")
        print(f"  Extrapolated σ(∞) = {sigma_ext:.4f} ± {se:.4f}")
        print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  1/√π = {SIGMA_PI:.4f}")
        diff_pct = abs(sigma_ext - SIGMA_PI) / SIGMA_PI * 100
        print(f"  Difference: {diff_pct:.2f}%")
        print(f"  Within 95% CI: {'YES' if within else 'NO'}")
        print()
        if within:
            print("  >>> π IS INTRINSIC to the permeation function <<<")
        else:
            print("  >>> π agreement was a finite-size coincidence <<<")
    else:
        print("  (Part A not run or insufficient data)")

    if "part_b" in all_results and "scaling_fits" in all_results["part_b"]:
        print()
        print(f"  Θ scaling: best model = {next(iter(all_results['part_b']['scaling_fits']))}")
        for mname in [m[0] for m in MODELS]:
            if mname in all_results["part_b"]["models"]:
                r = all_results["part_b"]["models"][mname]
                print(f"    {mname}: Θ = {r['cumulative_mean']:.3f}, Θ/π = {r['cumulative_over_pi']:.3f}")

    if "part_c" in all_results:
        print()
        print(f"  cos_pi_0p vs erf (ΔBIC):")
        for mname in [m[0] for m in MODELS]:
            if mname in all_results["part_c"]["models"]:
                r = all_results["part_c"]["models"][mname]
                winner = "cos_pi_0p" if r["delta_bic_cos_minus_erf"] < 0 else "erf"
                print(f"    {mname}: ΔBIC = {r['delta_bic_cos_minus_erf']:+.2f} ({winner})")

    print("=" * 60)


if __name__ == "__main__":
    main()
