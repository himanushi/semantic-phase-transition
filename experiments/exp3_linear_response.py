"""Experiment 3: Limits of linear response σ(l, h) = h · f(l).

A. Context gradient: measure σ(l) for 10-step context gradients per direction
B. σ vs h at each layer: linearity check (R² and residual analysis)
C. h*(l): nonlinearity threshold as a function of layer
D. Cross-word comparison of f(l) shapes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer_lens import HookedTransformer

from src.direction import compute_contrastive_direction, find_token_position
from src.order_parameter import compute_order_parameter_contrastive
from src.prompts import DIRECTION_PROMPTS, GRADIENT_PROMPTS


# -------------------------------------------------------------------
# Part A: Compute sigma profiles for gradient prompts
# -------------------------------------------------------------------

def compute_gradient_sigmas(
    model: HookedTransformer,
    word: str,
    e_diff: torch.Tensor,
    device: str,
) -> dict:
    """Compute sigma(l) for all gradient prompts of a word.

    Returns dict with 'A' and 'B' direction sigma arrays.
    Each is a list of (n_layers+1,) arrays, 10 prompts per direction.
    """
    prompts = GRADIENT_PROMPTS[word]
    result = {}

    for direction in ["A", "B"]:
        prompt_list = prompts[direction]
        sigmas = []
        for prompt in prompt_list:
            try:
                sigma = compute_order_parameter_contrastive(
                    model, prompt, word, e_diff
                )
                sigmas.append(sigma)
            except ValueError as e:
                print(f"    [{direction}] SKIP '{prompt[:40]}...': {e}")
                sigmas.append(None)
        result[direction] = sigmas

    return result


# -------------------------------------------------------------------
# Part B: σ vs h at each layer — linearity analysis
# -------------------------------------------------------------------

def define_h_from_final_layer(
    sigmas_A: list[np.ndarray | None],
    sigmas_B: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Define external field h from final-layer sigma values.

    h is defined as: for direction A prompts, h > 0 proportional to sigma_final.
    For direction B prompts, h < 0. Normalized so max |h| = 1.

    Returns:
        h_values: (N,) array of h for all valid prompts
        sigma_matrix: (N, n_layers+1) array of sigma profiles
        prompt_indices: (N,) array of original indices for traceability
    """
    all_h = []
    all_sigma = []
    all_idx = []

    # A 方向の最大σ_final で正規化スケールを決める
    sf_A_vals = [s[-1] for s in sigmas_A if s is not None]
    sf_B_vals = [s[-1] for s in sigmas_B if s is not None]

    if not sf_A_vals or not sf_B_vals:
        return np.array([]), np.array([[]]), np.array([])

    max_abs = max(
        max(abs(v) for v in sf_A_vals),
        max(abs(v) for v in sf_B_vals),
    )

    if max_abs < 1e-8:
        return np.array([]), np.array([[]]), np.array([])

    for i, s in enumerate(sigmas_A):
        if s is not None:
            h = s[-1] / max_abs
            all_h.append(h)
            all_sigma.append(s)
            all_idx.append(("A", i))

    for i, s in enumerate(sigmas_B):
        if s is not None:
            h = s[-1] / max_abs
            all_h.append(h)
            all_sigma.append(s)
            all_idx.append(("B", i))

    # h でソート
    order = np.argsort(all_h)
    h_arr = np.array(all_h)[order]
    sigma_mat = np.array(all_sigma)[order]
    idx_arr = [all_idx[i] for i in order]

    return h_arr, sigma_mat, idx_arr


def analyze_linearity_per_layer(
    h_values: np.ndarray,
    sigma_matrix: np.ndarray,
) -> dict:
    """For each layer, perform linear regression of sigma(l) vs h.

    Returns dict with per-layer R², slope, intercept, and residuals.
    """
    n_prompts, n_layers_plus1 = sigma_matrix.shape
    results = []

    for l in range(n_layers_plus1):
        y = sigma_matrix[:, l]
        slope, intercept, r_value, p_value, std_err = stats.linregress(h_values, y)
        r2 = r_value ** 2
        y_pred = slope * h_values + intercept
        residuals = y - y_pred

        results.append({
            "layer": l,
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "residuals": residuals.tolist(),
            "max_abs_residual": float(np.max(np.abs(residuals))),
        })

    return results


def estimate_hstar(
    h_values: np.ndarray,
    sigma_matrix: np.ndarray,
    threshold: float = 0.05,
) -> list[float | None]:
    """Estimate nonlinearity threshold h* for each layer.

    h* is the smallest |h| where the residual from linear fit exceeds threshold
    times the full sigma range at that layer.

    Returns list of h* per layer (None if always linear).
    """
    n_prompts, n_layers_plus1 = sigma_matrix.shape
    hstar_list = []

    for l in range(n_layers_plus1):
        y = sigma_matrix[:, l]
        slope, intercept, _, _, _ = stats.linregress(h_values, y)
        y_pred = slope * h_values + intercept
        residuals = np.abs(y - y_pred)

        sigma_range = np.max(y) - np.min(y)
        if sigma_range < 1e-8:
            hstar_list.append(None)
            continue

        # 残差が sigma_range * threshold を超える最小の |h|
        rel_residuals = residuals / sigma_range
        nonlinear_mask = rel_residuals > threshold

        if not np.any(nonlinear_mask):
            hstar_list.append(None)  # 全域で線形
        else:
            nonlinear_h = np.abs(h_values[nonlinear_mask])
            hstar_list.append(float(np.min(nonlinear_h)))

    return hstar_list


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_sigma_vs_h(
    h_values: np.ndarray,
    sigma_matrix: np.ndarray,
    linearity: list[dict],
    word: str,
    model_name: str,
    n_layers: int,
    output_path: Path | None = None,
) -> None:
    """Plot sigma(l) vs h for selected layers."""
    n_layers_plus1 = sigma_matrix.shape[1]

    # 6つのレイヤーを選択（等間隔 + 最終層）
    layer_indices = np.linspace(0, n_layers_plus1 - 1, 6, dtype=int)
    layer_indices = sorted(set(layer_indices))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, l in enumerate(layer_indices[:6]):
        ax = axes[ax_idx]
        y = sigma_matrix[:, l]
        lin = linearity[l]

        ax.scatter(h_values, y, c="C0", s=20, zorder=5)

        # 線形フィット
        h_fine = np.linspace(h_values.min(), h_values.max(), 100)
        y_fit = lin["slope"] * h_fine + lin["intercept"]
        ax.plot(h_fine, y_fit, "r--", linewidth=1.5,
                label=f"R²={lin['r2']:.3f}")

        ax.set_xlabel("h (context strength)")
        ax.set_ylabel("σ(l)")
        ax.set_title(f"Layer {l} (l/L={l/n_layers:.2f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle(f'σ vs h — "{word}" ({model_name})', fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_linearity_profile(
    linearity: list[dict],
    hstar: list[float | None],
    word: str,
    model_name: str,
    n_layers: int,
    output_path: Path | None = None,
) -> None:
    """Plot R² and h* as a function of layer."""
    n = len(linearity)
    layers = np.arange(n)
    l_norm = layers / n_layers

    r2_values = [lin["r2"] for lin in linearity]
    slopes = [lin["slope"] for lin in linearity]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # R² per layer
    ax = axes[0]
    ax.plot(l_norm, r2_values, "o-", color="C0", markersize=4)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="R²=0.95")
    ax.axhline(y=0.99, color="green", linestyle="--", alpha=0.5, label="R²=0.99")
    ax.set_ylabel("R² (linearity)")
    ax.set_title(f'Linearity of σ vs h — "{word}" ({model_name})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # slope = f(l)
    ax = axes[1]
    ax.plot(l_norm, slopes, "o-", color="C1", markersize=4)
    ax.set_ylabel("slope f(l) = dσ/dh")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # h* per layer
    ax = axes[2]
    hstar_vals = []
    hstar_layers = []
    for l, hs in enumerate(hstar):
        if hs is not None:
            hstar_vals.append(hs)
            hstar_layers.append(l / n_layers)

    if hstar_vals:
        ax.plot(hstar_layers, hstar_vals, "o-", color="C3", markersize=5)
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("h* (nonlinearity threshold)")
    ax.grid(True, alpha=0.3)
    if not hstar_vals:
        ax.text(0.5, 0.5, "All layers linear (no h* detected)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_fl_comparison(
    all_slopes: dict[str, list[float]],
    model_name: str,
    n_layers: int,
    output_path: Path | None = None,
) -> None:
    """Compare f(l) = dσ/dh across words."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw f(l)
    ax = axes[0]
    for word, slopes in all_slopes.items():
        l_norm = np.linspace(0, 1, len(slopes))
        ax.plot(l_norm, slopes, "o-", label=word, markersize=3)
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("f(l) = dσ/dh")
    ax.set_title(f"Response function f(l) — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normalized f(l) for shape comparison
    ax = axes[1]
    for word, slopes in all_slopes.items():
        s = np.array(slopes)
        smin, smax = s.min(), s.max()
        if smax - smin > 1e-8:
            s_scaled = (s - smin) / (smax - smin)
        else:
            s_scaled = s
        l_norm = np.linspace(0, 1, len(s))
        ax.plot(l_norm, s_scaled, "o-", label=word, markersize=3)
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("Scaled f(l)")
    ax.set_title(f"Normalized response function — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_experiment(
    model_name: str,
    device: str,
    output_dir: Path,
    words: list[str] | None = None,
) -> dict:
    """Run experiment 3: linear response limits."""
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_tag = model_name.replace("/", "_")

    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    n_layers = model.cfg.n_layers
    print(f"  Layers: {n_layers}, Device: {device}")

    if words is None:
        words = list(GRADIENT_PROMPTS.keys())

    all_results: dict[str, dict] = {}
    all_slopes: dict[str, list[float]] = {}

    for word in words:
        if word not in GRADIENT_PROMPTS:
            print(f"  SKIP {word}: no gradient prompts defined")
            continue

        dp = DIRECTION_PROMPTS[word]
        print(f"\n{'='*60}")
        print(f"[{word}] {dp['interpretation_A']} vs {dp['interpretation_B']}")
        print(f"{'='*60}")

        # 方向ベクトル
        try:
            e_diff = compute_contrastive_direction(
                model, dp["prompts_A"], dp["prompts_B"], word, device
            )
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        # --- Part A: Gradient sigma profiles ---
        print(f"\n  Part A: Computing gradient sigma profiles")
        grad_sigmas = compute_gradient_sigmas(model, word, e_diff, device)

        n_valid_A = sum(1 for s in grad_sigmas["A"] if s is not None)
        n_valid_B = sum(1 for s in grad_sigmas["B"] if s is not None)
        print(f"    Valid: A={n_valid_A}/10, B={n_valid_B}/10")

        if n_valid_A < 3 or n_valid_B < 3:
            print(f"  Too few valid prompts, skipping")
            continue

        # --- Part B: Define h and analyze linearity ---
        print(f"\n  Part B: Linearity analysis")
        h_values, sigma_matrix, prompt_indices = define_h_from_final_layer(
            grad_sigmas["A"], grad_sigmas["B"]
        )

        if len(h_values) < 5:
            print(f"  Too few valid h values ({len(h_values)}), skipping")
            continue

        print(f"    h range: [{h_values.min():.3f}, {h_values.max():.3f}]")
        print(f"    N prompts: {len(h_values)}")

        linearity = analyze_linearity_per_layer(h_values, sigma_matrix)

        # Summary
        r2_values = [lin["r2"] for lin in linearity]
        print(f"    R² range: [{min(r2_values):.4f}, {max(r2_values):.4f}]")
        print(f"    R² > 0.95 layers: {sum(1 for r in r2_values if r > 0.95)}/{len(r2_values)}")
        print(f"    R² > 0.99 layers: {sum(1 for r in r2_values if r > 0.99)}/{len(r2_values)}")

        # σ vs h plots
        plot_sigma_vs_h(
            h_values, sigma_matrix, linearity,
            word, model_name, n_layers,
            output_path=fig_dir / f"exp3_sigma_vs_h_{word}_{model_tag}.png",
        )

        # --- Part C: h* estimation ---
        print(f"\n  Part C: Nonlinearity threshold h*")
        hstar = estimate_hstar(h_values, sigma_matrix, threshold=0.05)

        hstar_valid = [(l, hs) for l, hs in enumerate(hstar) if hs is not None]
        if hstar_valid:
            min_hstar_layer, min_hstar_val = min(hstar_valid, key=lambda x: x[1])
            print(f"    Min h* = {min_hstar_val:.3f} at layer {min_hstar_layer} (l/L={min_hstar_layer/n_layers:.2f})")
            print(f"    Layers with h* < 0.5: {sum(1 for _, hs in hstar_valid if hs < 0.5)}")
        else:
            print(f"    No nonlinearity detected (all layers linear)")

        # Linearity profile plots
        plot_linearity_profile(
            linearity, hstar,
            word, model_name, n_layers,
            output_path=fig_dir / f"exp3_linearity_{word}_{model_tag}.png",
        )

        # Collect slopes for cross-word comparison
        slopes = [lin["slope"] for lin in linearity]
        all_slopes[word] = slopes

        # Store results
        all_results[word] = {
            "h_values": h_values.tolist(),
            "sigma_matrix": sigma_matrix.tolist(),
            "prompt_indices": prompt_indices,
            "linearity": linearity,
            "hstar": hstar,
            "prompts_A": GRADIENT_PROMPTS[word]["A"],
            "prompts_B": GRADIENT_PROMPTS[word]["B"],
        }

    # --- Part D: Cross-word comparison ---
    if len(all_slopes) >= 2:
        print(f"\n{'='*60}")
        print("Part D: Cross-word f(l) comparison")
        print(f"{'='*60}")

        plot_fl_comparison(
            all_slopes, model_name, n_layers,
            output_path=fig_dir / f"exp3_fl_comparison_{model_tag}.png",
        )

        # 形状の類似度（相関係数）
        word_list = list(all_slopes.keys())
        print(f"\n  f(l) shape correlation matrix:")
        print(f"  {'':>10}", end="")
        for w in word_list:
            print(f"  {w:>8}", end="")
        print()
        for w1 in word_list:
            print(f"  {w1:>10}", end="")
            for w2 in word_list:
                s1 = np.array(all_slopes[w1])
                s2 = np.array(all_slopes[w2])
                corr = np.corrcoef(s1, s2)[0, 1]
                print(f"  {corr:>8.3f}", end="")
            print()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name} ({n_layers} layers)")
    print(f"{'='*60}")
    print(f"\n{'word':<10} {'R²_mean':<10} {'R²_min':<10} {'min_h*':<10} {'min_h*_layer':<14} {'f_max':<10}")
    print("-" * 64)
    for word, res in all_results.items():
        r2s = [lin["r2"] for lin in res["linearity"]]
        r2_mean = np.mean(r2s)
        r2_min = np.min(r2s)
        slopes = [lin["slope"] for lin in res["linearity"]]
        f_max = max(slopes)

        hstar = res["hstar"]
        hstar_valid = [(l, hs) for l, hs in enumerate(hstar) if hs is not None]
        if hstar_valid:
            min_l, min_hs = min(hstar_valid, key=lambda x: x[1])
            hs_str = f"{min_hs:.3f}"
            hl_str = f"L{min_l} ({min_l/n_layers:.2f})"
        else:
            hs_str = "none"
            hl_str = "—"

        print(f"{word:<10} {r2_mean:<10.4f} {r2_min:<10.4f} {hs_str:<10} {hl_str:<14} {f_max:<10.4f}")

    # JSON保存
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "device": device,
        "words": {},
    }
    for word, res in all_results.items():
        output["words"][word] = {
            "h_values": res["h_values"],
            "linearity": res["linearity"],
            "hstar": res["hstar"],
            "prompts_A": res["prompts_A"],
            "prompts_B": res["prompts_B"],
        }

    json_path = data_dir / f"exp3_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nData saved: {json_path}")

    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Linear response limits")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--words", nargs="*", default=None)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        device=args.device,
        output_dir=Path(args.output_dir),
        words=args.words,
    )


if __name__ == "__main__":
    main()
