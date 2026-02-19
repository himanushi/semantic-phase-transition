"""Experiment 2: Landau fit and universality analysis.

A. Response function extraction: verify σ(l, h) = h · f(l)
B. f(l) Landau fitting with model comparison (AIC/BIC)
C. Logit lens comparison to disentangle unembedding effects
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer_lens import HookedTransformer

from src.direction import compute_contrastive_direction, find_token_position
from src.fitting import fit_all_models, FIT_MODELS
from src.order_parameter import compute_order_parameter_contrastive
from src.prompts import DIRECTION_PROMPTS, EXPERIMENT_PROMPTS


# -------------------------------------------------------------------
# Part A: Response function extraction
# -------------------------------------------------------------------

def extract_response_functions(
    word_sigmas: dict[str, np.ndarray],
    n_layers: int,
) -> dict:
    """Normalize sigma profiles to extract f(l).

    Defines h from sigma_final: h(strong_A)=+1, h(strong_B)=-1,
    others interpolated. Then f(l) = sigma(l) / h.

    Returns dict with h values, f(l) curves, and collapse quality metric.
    """
    sf_A = word_sigmas["strong_A"][-1]
    sf_B = word_sigmas["strong_B"][-1]
    half_range = (sf_A - sf_B) / 2.0
    midpoint = (sf_A + sf_B) / 2.0

    if abs(half_range) < 1e-8:
        return {"valid": False}

    h_values = {}
    f_curves = {}
    for label, sigma in word_sigmas.items():
        h = (sigma[-1] - midpoint) / half_range
        h_values[label] = float(h)
        if abs(h) > 0.05:  # 十分な信号がある条件のみ
            f_curves[label] = sigma / h

    # collapse 品質: f(l) 曲線間の分散（小さいほど良い）
    if len(f_curves) >= 2:
        f_stack = np.stack(list(f_curves.values()))
        variance_per_layer = np.var(f_stack, axis=0)
        mean_variance = float(np.mean(variance_per_layer))
        # f(l) の平均
        f_mean = np.mean(f_stack, axis=0)
    else:
        mean_variance = float("inf")
        f_mean = list(f_curves.values())[0] if f_curves else np.zeros(n_layers + 1)

    return {
        "valid": True,
        "h_values": h_values,
        "f_curves": {k: v.tolist() for k, v in f_curves.items()},
        "f_mean": f_mean,
        "collapse_variance": mean_variance,
    }


def plot_response_function(
    response: dict,
    word: str,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot normalized f(l) curves showing data collapse."""
    if not response["valid"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw f(l) per condition
    ax = axes[0]
    for label, f_list in response["f_curves"].items():
        f = np.array(f_list)
        layers = np.arange(len(f))
        h = response["h_values"][label]
        ax.plot(layers, f, "o-", label=f"{label} (h={h:.2f})", markersize=3)

    f_mean = np.array(response["f_mean"]) if isinstance(response["f_mean"], list) else response["f_mean"]
    layers = np.arange(len(f_mean))
    ax.plot(layers, f_mean, "k--", linewidth=2, label="mean f(l)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("f(l) = σ(l) / h")
    ax.set_title(f'Data Collapse — "{word}" ({model_name})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: residual from mean (collapse quality)
    ax = axes[1]
    for label, f_list in response["f_curves"].items():
        f = np.array(f_list)
        residual = f - f_mean
        ax.plot(layers, residual, "o-", label=label, markersize=3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("f(l) - <f(l)>")
    ax.set_title(f"Residual (var={response['collapse_variance']:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# Part B: Landau fitting
# -------------------------------------------------------------------

def plot_fits(
    l_norm: np.ndarray,
    f_mean: np.ndarray,
    fit_results: dict,
    word: str,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot f(l) data with all fitted models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(l_norm, f_mean, "ko", markersize=5, label="data (mean f(l))", zorder=5)

    colors = {"power_law": "C0", "tanh": "C1", "sigmoid": "C2", "two_stage": "C3"}
    for name, result in fit_results.items():
        # 高解像度のフィット曲線
        l_fine = np.linspace(l_norm.min(), l_norm.max(), 200)
        y_fine = FIT_MODELS[name]["func"](l_fine, *result["params"].values())
        label = f'{name} (R²={result["r2"]:.3f}, BIC={result["bic"]:.1f})'
        ax.plot(l_fine, y_fine, "-", color=colors.get(name, "gray"),
                linewidth=2, label=label)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized layer l/L")
    ax.set_ylabel("f(l)")
    ax.set_title(f'Landau Fits — "{word}" ({model_name})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# Part C: Logit lens
# -------------------------------------------------------------------

def compute_logit_lens(
    model: HookedTransformer,
    prompt: str,
    target_word: str,
) -> dict:
    """Apply unembedding at each layer to track token prediction changes.

    Returns:
        Dict with top tokens and their probs per layer.
    """
    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(prompt)
        pos = find_token_position(tokens, target_word, model)

        n_layers = model.cfg.n_layers
        W_U = model.W_U  # (d_model, d_vocab)
        ln_final = model.ln_final

        results_per_layer = []
        for l in range(n_layers + 1):
            if l == 0:
                resid = cache["resid_pre", 0][0, pos]
            else:
                resid = cache["resid_post", l - 1][0, pos]

            # LayerNorm → unembedding
            normed = ln_final(resid)
            logits = normed @ W_U  # (d_vocab,)
            probs = torch.softmax(logits, dim=-1)

            top5_vals, top5_idx = torch.topk(probs, 5)
            top5_tokens = [model.to_string(idx.unsqueeze(0)) for idx in top5_idx]

            results_per_layer.append({
                "top_tokens": top5_tokens,
                "top_probs": top5_vals.cpu().tolist(),
                "top1": top5_tokens[0],
                "top1_prob": float(top5_vals[0]),
            })

    return {"layers": results_per_layer}


def plot_logit_lens_vs_sigma(
    sigma: np.ndarray,
    logit_lens: dict,
    word: str,
    prompt_label: str,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot sigma(l) alongside logit lens top-1 probability changes."""
    n = len(sigma)
    layers = np.arange(n)
    lens_data = logit_lens["layers"]

    top1_probs = [lens_data[l]["top1_prob"] for l in range(n)]
    top1_tokens = [lens_data[l]["top1"].strip() for l in range(n)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # sigma(l)
    ax1.plot(layers, sigma, "o-", color="C0", markersize=4)
    ax1.set_ylabel("σ(l)")
    ax1.set_title(f'σ(l) vs Logit Lens — "{word}" [{prompt_label}] ({model_name})')
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # dσ/dl
    dsigma = np.diff(sigma)
    ax2.bar(layers[:-1] + 0.5, dsigma, width=0.8, alpha=0.7, color="C1")
    ax2.set_ylabel("Δσ/Δl")
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # logit lens top-1 prob + token labels
    ax3.bar(layers, top1_probs, width=0.8, alpha=0.7, color="C2")
    ax3.set_ylabel("Top-1 prob")
    ax3.set_xlabel("Layer l")
    ax3.grid(True, alpha=0.3)

    # トークンラベルを表示（見やすさのため間引く）
    step = max(1, n // 15)
    for i in range(0, n, step):
        ax3.annotate(
            top1_tokens[i],
            (i, top1_probs[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            rotation=45,
        )

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
    """Run experiment 2."""
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
        words = list(DIRECTION_PROMPTS.keys())

    all_results: dict[str, dict] = {}
    all_f_means: dict[str, np.ndarray] = {}

    for word in words:
        dp = DIRECTION_PROMPTS[word]
        prompts = EXPERIMENT_PROMPTS[word]
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

        # σ(l) を全条件で計算
        word_sigmas: dict[str, np.ndarray] = {}
        for label, prompt_text in prompts.items():
            try:
                sigma = compute_order_parameter_contrastive(
                    model, prompt_text, word, e_diff
                )
                word_sigmas[label] = sigma
            except ValueError as e:
                print(f"  [{label}] SKIP: {e}")

        if len(word_sigmas) < 3:
            print(f"  Too few valid conditions, skipping")
            continue

        # --- Part A: Response function ---
        print(f"\n  Part A: Response function extraction")
        response = extract_response_functions(word_sigmas, n_layers)

        if response["valid"]:
            print(f"    h values: {response['h_values']}")
            print(f"    Collapse variance: {response['collapse_variance']:.6f}")
            print(f"    # curves in collapse: {len(response['f_curves'])}")

            plot_response_function(
                response, word, model_name,
                output_path=fig_dir / f"exp2_collapse_{word}_{model_tag}.png",
            )
        else:
            print(f"    Invalid (no range)")
            continue

        # --- Part B: Landau fits on f_mean ---
        print(f"\n  Part B: Landau fitting")
        f_mean = np.array(response["f_mean"]) if isinstance(response["f_mean"], list) else response["f_mean"]
        l_norm = np.linspace(0, 1, len(f_mean))

        # 最終層を除外したフィット（unembedding効果の分離）
        n_exclude = 1  # 最終1層を除外
        l_norm_trim = l_norm[:-n_exclude]
        f_mean_trim = f_mean[:-n_exclude]

        fit_results = fit_all_models(l_norm_trim, f_mean_trim)

        print(f"    {'Model':<12} {'R²':<8} {'AIC':<10} {'BIC':<10} Params")
        print(f"    {'-'*60}")
        for name, result in fit_results.items():
            params_str = ", ".join(f"{k}={v:.3f}" for k, v in result["params"].items())
            print(f"    {name:<12} {result['r2']:<8.4f} {result['aic']:<10.1f} "
                  f"{result['bic']:<10.1f} {params_str}")

        # フィット用プロット（トリム版）
        plot_fits(
            l_norm_trim, f_mean_trim, fit_results,
            word, model_name,
            output_path=fig_dir / f"exp2_fit_{word}_{model_tag}.png",
        )

        # --- Part C: Logit lens ---
        print(f"\n  Part C: Logit lens")
        # strong_A と strong_B でlogit lensを実行
        logit_lens_results = {}
        for label in ["strong_A", "strong_B"]:
            prompt_text = prompts[label]
            try:
                ll = compute_logit_lens(model, prompt_text, word)
                logit_lens_results[label] = ll

                # top-1 token の変化点を表示
                top1_tokens = [ll["layers"][l]["top1"].strip() for l in range(n_layers + 1)]
                changes = []
                for i in range(1, len(top1_tokens)):
                    if top1_tokens[i] != top1_tokens[i-1]:
                        changes.append(f"L{i}:'{top1_tokens[i]}'")
                print(f"    [{label}] token changes: {', '.join(changes[:8])}")

                plot_logit_lens_vs_sigma(
                    word_sigmas[label], ll,
                    word, label, model_name,
                    output_path=fig_dir / f"exp2_logitlens_{word}_{label}_{model_tag}.png",
                )
            except ValueError as e:
                print(f"    [{label}] Logit lens SKIP: {e}")

        # serialize fit_results
        fit_serializable = {}
        for name, result in fit_results.items():
            fit_serializable[name] = {
                k: v for k, v in result.items() if k != "y_fit"
            }
            fit_serializable[name]["y_fit"] = result["y_fit"].tolist()

        # serialize logit lens
        ll_serializable = {}
        for label, ll in logit_lens_results.items():
            ll_serializable[label] = {
                "layers": [
                    {"top1": ld["top1"], "top1_prob": ld["top1_prob"],
                     "top3": ld["top_tokens"][:3]}
                    for ld in ll["layers"]
                ]
            }

        all_results[word] = {
            "response": {
                "h_values": response["h_values"],
                "collapse_variance": response["collapse_variance"],
                "f_mean": f_mean.tolist(),
            },
            "fits": fit_serializable,
            "logit_lens": ll_serializable,
        }
        all_f_means[word] = f_mean

    # --- Cross-word universality ---
    if len(all_f_means) >= 2:
        print(f"\n{'='*60}")
        print("Cross-word universality of f(l)")
        print(f"{'='*60}")

        fig, ax = plt.subplots(figsize=(12, 6))
        for word, f_mean in all_f_means.items():
            l_norm = np.linspace(0, 1, len(f_mean))
            # f(l) を [0, 1] にスケール
            fmin, fmax = f_mean.min(), f_mean.max()
            if fmax - fmin > 1e-8:
                f_scaled = (f_mean - fmin) / (fmax - fmin)
            else:
                f_scaled = f_mean
            ax.plot(l_norm, f_scaled, "o-", label=word, markersize=3)

        ax.set_xlabel("Normalized layer l/L")
        ax.set_ylabel("Scaled f(l)")
        ax.set_title(f"Cross-word f(l) Universality ({model_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        universality_path = fig_dir / f"exp2_universality_{model_tag}.png"
        plt.savefig(universality_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {universality_path}")
        plt.close(fig)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name} ({n_layers} layers)")
    print(f"{'='*60}")
    print(f"\n{'word':<10} {'collapse_var':<14} {'best_model':<12} {'R²':<8} {'BIC':<10} {'key params'}")
    print("-" * 80)
    for word, res in all_results.items():
        cv = res["response"]["collapse_variance"]
        fits = res["fits"]
        if fits:
            best_name = next(iter(fits))
            best = fits[best_name]
            params_str = ", ".join(f"{k}={v:.3f}" for k, v in best["params"].items())
            print(f"{word:<10} {cv:<14.6f} {best_name:<12} {best['r2']:<8.4f} "
                  f"{best['bic']:<10.1f} {params_str}")

    # JSON保存
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "device": device,
        "n_exclude_final": 1,
        "words": all_results,
    }
    json_path = data_dir / f"exp2_{model_tag}.json"
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
    parser = argparse.ArgumentParser(description="Experiment 2: Landau fit analysis")
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
