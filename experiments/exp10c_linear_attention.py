"""Experiment 10: Phase 2 — Linear Attention (Parts I-L).

10I: Individual linear attention replacement for each head in L1-L11
10J: Best method selection and mixed strategy analysis
10K: Cumulative linearization with best method
10L: Computational cost estimation

Usage:
    python experiments/exp10c_linear_attention.py --model gpt2 --device mps --parts ijkl
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp10_head_ablation import load_wikitext2_validation, compute_perplexity


# -------------------------------------------------------------------
# Linear attention hook utilities
# -------------------------------------------------------------------

METHODS = ["relu", "l1", "identity"]


def make_linear_attention_hooks(
    layer_idx: int,
    head_methods: dict[int, str],
) -> list[tuple[str, callable]]:
    """Create hooks to replace softmax with linear attention for specific heads.

    Uses two hooks per layer:
    1. hook_attn_scores: capture pre-mask scores for target heads
    2. hook_pattern: replace softmax output with linearly-normalized version

    Args:
        layer_idx: Layer index
        head_methods: {head_idx: method} where method is "relu", "l1", or "identity"
    """
    captured_scores: dict[int, torch.Tensor] = {}

    def scores_hook(scores, hook):
        # scores: [batch, n_heads, seq_q, seq_k] — QK^T/√d BEFORE causal mask
        for h in head_methods:
            captured_scores[h] = scores[:, h, :, :].clone()
        return scores

    def pattern_hook(pattern, hook):
        # pattern: [batch, n_heads, seq_q, seq_k] — AFTER softmax
        for h, method in head_methods.items():
            scores = captured_scores[h]  # [batch, seq_q, seq_k]
            # hook_attn_scores already has causal mask applied (-inf for future)
            # Replace -inf with 0 so abs/identity don't produce inf/NaN
            scores_finite = torch.where(
                scores.isfinite(), scores, torch.zeros_like(scores)
            )

            if method == "relu":
                weights = torch.relu(scores_finite)
            elif method == "l1":
                weights = torch.abs(scores_finite)
            elif method == "identity":
                weights = scores_finite
            else:
                raise ValueError(f"Unknown method: {method}")

            # Normalize
            if method == "identity":
                # Identity allows negative weights; normalize by L1 norm
                # to keep total magnitude bounded while preserving signs
                row_sums = weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                # ReLU and L1 produce non-negative weights; sum is always ≥ 0
                row_sums = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            weights = weights / row_sums

            pattern[:, h, :, :] = weights

        return pattern

    return [
        (f"blocks.{layer_idx}.attn.hook_attn_scores", scores_hook),
        (f"blocks.{layer_idx}.attn.hook_pattern", pattern_hook),
    ]


def build_linear_hooks(
    head_method_list: list[tuple[int, int, str]],
) -> list[tuple[str, callable]]:
    """Create linear attention hooks for multiple heads across layers.

    Args:
        head_method_list: [(layer, head, method), ...]
    """
    # Group by layer
    layer_groups: dict[int, dict[int, str]] = {}
    for layer, head, method in head_method_list:
        layer_groups.setdefault(layer, {})[head] = method

    hooks = []
    for layer in sorted(layer_groups):
        hooks.extend(make_linear_attention_hooks(layer, layer_groups[layer]))
    return hooks


# -------------------------------------------------------------------
# 10I: Individual linear attention replacement
# -------------------------------------------------------------------


def run_exp10i(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_baseline: float,
    mean_results: list[dict],
    fig_dir: Path,
    data_dir: Path,
) -> list[dict]:
    """Individual linear attention for each head in L1-L11, all 3 methods."""
    model_tag = model_name.replace("/", "_")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print(f"\n{'='*60}")
    print(f"10I: Individual linear attention — {model_name}")
    print(f"{'='*60}")

    # Verify hooks
    hook_dict = model.hook_dict
    for name in ["blocks.0.attn.hook_attn_scores", "blocks.0.attn.hook_pattern"]:
        if name not in hook_dict:
            print(f"  ERROR: {name} not found in hook_dict")
            return []
    print(f"  Hook verification OK")

    # Cache check
    cache_path = data_dir / f"exp10i_linear_individual_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["results"]

    # Target: L1-L11 heads
    target_heads = [
        (l, h) for l in range(1, n_layers) for h in range(n_heads)
    ]
    n_target = len(target_heads)
    n_total = n_target * len(METHODS)

    results = []
    t0 = time.time()
    idx = 0

    for layer, head in target_heads:
        for method in METHODS:
            hooks = make_linear_attention_hooks(layer, {head: method})
            ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)
            delta_ppl = (ppl - ppl_baseline) / ppl_baseline

            results.append({
                "layer": layer,
                "head": head,
                "method": method,
                "ppl": round(ppl, 4),
                "delta_ppl": round(delta_ppl, 6),
            })

            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_total - idx - 1)
            print(
                f"\r  [{idx+1}/{n_total}] L{layer}H{head} {method:>8}: "
                f"PPL={ppl:.2f} ΔPPL={delta_ppl:+.4f} "
                f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                end="", flush=True,
            )
            idx += 1

    print()

    # Save
    output = {
        "model": model_name,
        "baseline_ppl": round(ppl_baseline, 4),
        "n_heads_tested": n_target,
        "methods": METHODS,
        "results": results,
    }
    with open(cache_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {cache_path}")

    # --- Statistics ---
    for method in METHODS:
        method_results = [r for r in results if r["method"] == method]
        dppl_arr = np.array([r["delta_ppl"] for r in method_results])
        print(f"\n  {method} statistics:")
        print(f"    Mean ΔPPL: {np.mean(dppl_arr):+.4f}")
        print(f"    Median ΔPPL: {np.median(dppl_arr):+.4f}")
        print(f"    Max ΔPPL: {np.max(dppl_arr):+.4f}")
        print(f"    |ΔPPL| < 1%: {np.sum(np.abs(dppl_arr) < 0.01)}/{len(dppl_arr)}")
        print(f"    ΔPPL < 0: {np.sum(dppl_arr < 0)}/{len(dppl_arr)}")

    # --- Plots ---
    # 1. Distribution comparison (3 methods)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, method in zip(axes, METHODS):
        dppl = [r["delta_ppl"] for r in results if r["method"] == method]
        ax.hist(dppl, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("ΔPPL")
        ax.set_ylabel("Count")
        ax.set_title(f"{method} (mean={np.mean(dppl):+.4f})")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Exp 10I: Linear Attention ΔPPL Distribution — {model_name}")
    plt.tight_layout()
    path = fig_dir / f"exp10i_linear_individual_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # 2. Scatter: linear vs mean-ablation
    mean_map = {
        (r["layer"], r["head"]): r["delta_ppl"]
        for r in mean_results if r["layer"] != 0
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, method in zip(axes, METHODS):
        method_results = [r for r in results if r["method"] == method]
        x_mean = [mean_map.get((r["layer"], r["head"]), 0) for r in method_results]
        y_linear = [r["delta_ppl"] for r in method_results]

        ax.scatter(x_mean, y_linear, s=15, alpha=0.6)
        lim = max(
            max(abs(v) for v in x_mean) if x_mean else 0.01,
            max(abs(v) for v in y_linear) if y_linear else 0.01,
        ) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
        ax.plot([-lim, lim], [0, 0], "gray", alpha=0.2)
        ax.plot([0, 0], [-lim, lim], "gray", alpha=0.2)
        ax.set_xlabel("ΔPPL (mean-ablation)")
        ax.set_ylabel(f"ΔPPL ({method})")
        ax.set_title(f"{method}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 対角線より下 = 線形化の方がmean-ablationより良い
        n_better = sum(1 for xm, yl in zip(x_mean, y_linear) if yl < xm)
        ax.text(
            0.05, 0.95, f"Linear < Mean: {n_better}/{len(x_mean)}",
            transform=ax.transAxes, fontsize=9, va="top",
        )

    plt.suptitle(f"Exp 10I: Linear vs Mean-Ablation — {model_name}")
    plt.tight_layout()
    path = fig_dir / f"exp10i_linear_vs_mean_scatter_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# 10J: Best method selection
# -------------------------------------------------------------------


def run_exp10j(
    model_name: str,
    linear_results: list[dict],
    mean_results: list[dict],
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Analyze and select best linear attention method."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10J: Best method selection — {model_name}")
    print(f"{'='*60}")

    # Per-method statistics
    method_stats = {}
    for method in METHODS:
        mr = [r for r in linear_results if r["method"] == method]
        dppl = np.array([r["delta_ppl"] for r in mr])
        method_stats[method] = {
            "mean": round(float(np.mean(dppl)), 6),
            "median": round(float(np.median(dppl)), 6),
            "max": round(float(np.max(dppl)), 6),
            "n_negative": int(np.sum(dppl < 0)),
            "n_small": int(np.sum(np.abs(dppl) < 0.01)),
        }
        print(
            f"\n  {method}: mean={np.mean(dppl):+.4f}, median={np.median(dppl):+.4f}, "
            f"max={np.max(dppl):+.4f}, ΔPPL<0: {np.sum(dppl < 0)}"
        )

    # Best overall method (lowest mean ΔPPL)
    best_method = min(method_stats, key=lambda m: method_stats[m]["mean"])
    print(
        f"\n  Best overall method: {best_method} "
        f"(mean ΔPPL = {method_stats[best_method]['mean']:+.4f})"
    )

    # Per-head best method
    head_best: dict[tuple[int, int], dict] = {}
    for r in linear_results:
        key = (r["layer"], r["head"])
        if key not in head_best or r["delta_ppl"] < head_best[key]["delta_ppl"]:
            head_best[key] = {
                "method": r["method"],
                "delta_ppl": r["delta_ppl"],
            }

    # Mixed strategy stats
    mixed_dppl = np.array([v["delta_ppl"] for v in head_best.values()])
    method_counts: dict[str, int] = {}
    for v in head_best.values():
        method_counts[v["method"]] = method_counts.get(v["method"], 0) + 1

    print(f"\n  Mixed strategy (per-head best):")
    print(f"    Mean ΔPPL: {np.mean(mixed_dppl):+.4f}")
    print(f"    Method distribution: {method_counts}")

    # Comparison with mean-ablation
    mean_map = {
        (r["layer"], r["head"]): r["delta_ppl"]
        for r in mean_results if r["layer"] != 0
    }

    n_better_than_mean = 0
    for (l, h), best_info in head_best.items():
        mean_dppl = mean_map.get((l, h), float("inf"))
        if best_info["delta_ppl"] < mean_dppl:
            n_better_than_mean += 1

    print(f"\n  Linear (best per head) vs mean-ablation:")
    print(f"    Heads where linear < mean: {n_better_than_mean}/{len(head_best)}")

    # Build rankings
    best_method_results = [r for r in linear_results if r["method"] == best_method]
    best_ranking = sorted(best_method_results, key=lambda x: abs(x["delta_ppl"]))

    mixed_ranking = sorted(
        [{"layer": k[0], "head": k[1], **v} for k, v in head_best.items()],
        key=lambda x: abs(x["delta_ppl"]),
    )

    result = {
        "method_stats": method_stats,
        "best_method": best_method,
        "method_counts_mixed": method_counts,
        "mixed_mean_dppl": round(float(np.mean(mixed_dppl)), 6),
        "n_better_than_mean": n_better_than_mean,
        "best_ranking": [
            {"layer": r["layer"], "head": r["head"], "delta_ppl": r["delta_ppl"]}
            for r in best_ranking
        ],
        "mixed_ranking": mixed_ranking,
    }

    # Save
    cache_path = data_dir / f"exp10j_method_comparison_{model_tag}.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {cache_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Method comparison box plot
    ax = axes[0]
    data_for_box = []
    labels_for_box = []
    for method in METHODS:
        dppl = [r["delta_ppl"] for r in linear_results if r["method"] == method]
        data_for_box.append(dppl)
        labels_for_box.append(method)
    data_for_box.append([v["delta_ppl"] for v in head_best.values()])
    labels_for_box.append("mixed")
    # Mean-ablation for comparison
    mean_abl_dppl = [
        mean_map[(r["layer"], r["head"])]
        for r in best_ranking
        if (r["layer"], r["head"]) in mean_map
    ]
    data_for_box.append(mean_abl_dppl)
    labels_for_box.append("mean-abl")

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    colors_box = ["C0", "C1", "C2", "C3", "C4"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("ΔPPL")
    ax.set_title(f"Exp 10J: Method Comparison — {model_name}")
    ax.grid(True, alpha=0.3)

    # Right: Per-head method choice heatmap
    ax = axes[1]
    n_layers = max(r["layer"] for r in linear_results) + 1
    n_heads = max(r["head"] for r in linear_results) + 1
    method_to_int = {m: i for i, m in enumerate(METHODS)}
    grid = np.full((n_layers - 1, n_heads), np.nan)
    for (l, h), info in head_best.items():
        grid[l - 1, h] = method_to_int[info["method"]]

    im = ax.imshow(grid, cmap="Set2", aspect="auto", vmin=-0.5, vmax=2.5)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(n_layers - 1))
    ax.set_yticklabels([f"L{i+1}" for i in range(n_layers - 1)])
    ax.set_xticks(range(n_heads))
    ax.set_title("Per-head best method")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(METHODS)

    plt.tight_layout()
    path = fig_dir / f"exp10j_method_comparison_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return result


# -------------------------------------------------------------------
# 10K: Cumulative linearization
# -------------------------------------------------------------------


def run_exp10k(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_baseline: float,
    method_result: dict,
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Cumulative linearization using best method and mixed strategy."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10K: Cumulative linearization — {model_name}")
    print(f"{'='*60}")

    best_method = method_result["best_method"]
    best_ranking = method_result["best_ranking"]
    mixed_ranking = method_result["mixed_ranking"]

    cache_path = data_dir / f"exp10k_cumulative_linear_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    n_total = len(best_ranking)
    steps = sorted(set(
        list(range(0, n_total + 1, 10))
        + [5, 15, 25, 30, 35, 40, 50, n_total]
    ))
    steps = [s for s in steps if s <= n_total]

    # --- Best single method ---
    print(f"\n  Cumulative linearization ({best_method}):")
    cumul_best = []
    t0 = time.time()

    for i, n in enumerate(steps):
        if n == 0:
            ppl = ppl_baseline
        else:
            heads = best_ranking[:n]
            head_method_list = [
                (r["layer"], r["head"], best_method) for r in heads
            ]
            hooks = build_linear_hooks(head_method_list)
            ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)

        ratio = ppl / ppl_baseline
        cumul_best.append({
            "n_linearized": n,
            "ppl": round(ppl, 4),
            "ratio": round(ratio, 4),
        })

        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(steps)}] {best_method} n={n:>3}: "
            f"PPL={ppl:.2f} ({ratio:.3f}x) [{elapsed:.0f}s]"
        )

    # --- Mixed strategy ---
    print(f"\n  Cumulative linearization (mixed):")
    cumul_mixed = []
    t0 = time.time()

    for i, n in enumerate(steps):
        if n == 0:
            ppl = ppl_baseline
        else:
            heads = mixed_ranking[:n]
            head_method_list = [
                (r["layer"], r["head"], r["method"]) for r in heads
            ]
            hooks = build_linear_hooks(head_method_list)
            ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)

        ratio = ppl / ppl_baseline
        cumul_mixed.append({
            "n_linearized": n,
            "ppl": round(ppl, 4),
            "ratio": round(ratio, 4),
        })

        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(steps)}] mixed n={n:>3}: "
            f"PPL={ppl:.2f} ({ratio:.3f}x) [{elapsed:.0f}s]"
        )

    # Thresholds
    def get_thresholds(cumul):
        thresh = {1.1: 0, 1.5: 0, 2.0: 0}
        for cr in cumul:
            for t in thresh:
                if cr["ratio"] <= t:
                    thresh[t] = max(thresh[t], cr["n_linearized"])
        return thresh

    thresh_best = get_thresholds(cumul_best)
    thresh_mixed = get_thresholds(cumul_mixed)

    print(f"\n  Threshold analysis:")
    print(f"    {'Threshold':>10} {best_method:>12} {'mixed':>12}")
    for t in [1.1, 1.5, 2.0]:
        print(f"    {t:>10.1f}x {thresh_best[t]:>12} {thresh_mixed[t]:>12}")

    result = {
        "best_method": best_method,
        "cumulative_best": cumul_best,
        "cumulative_mixed": cumul_mixed,
        "thresholds_best": {str(k): v for k, v in thresh_best.items()},
        "thresholds_mixed": {str(k): v for k, v in thresh_mixed.items()},
    }

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {cache_path}")

    # --- Plot: Pareto curves ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        [cr["n_linearized"] for cr in cumul_best],
        [cr["ratio"] for cr in cumul_best],
        "o-", color="C0", markersize=5, label=f"Linear ({best_method})",
    )
    ax.plot(
        [cr["n_linearized"] for cr in cumul_mixed],
        [cr["ratio"] for cr in cumul_mixed],
        "s-", color="C3", markersize=5, label="Linear (mixed)",
    )

    # Mean-ablation comparison
    mean_path = data_dir / f"exp10f_cumulative_l0protected_{model_tag}.json"
    if mean_path.exists():
        with open(mean_path) as f:
            mean_data = json.load(f)
        ax.plot(
            [cr["n_ablated"] for cr in mean_data["cumulative_mean"]],
            [cr["ratio"] for cr in mean_data["cumulative_mean"]],
            "^-", color="C2", markersize=5, alpha=0.7,
            label="Mean-ablation (L0 prot)",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for t, c in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        ax.axhline(
            y=t, color=c, linestyle=":", alpha=0.4,
            label=f"{t}x (best={thresh_best[t]}, mixed={thresh_mixed[t]})",
        )

    ax.set_xlabel("Number of L1-L11 heads linearized")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title(f"Exp 10K: Cumulative Linearization — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 3.5)

    plt.tight_layout()
    path = fig_dir / f"exp10k_pareto_linear_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return result


# -------------------------------------------------------------------
# 10L: Cost estimation
# -------------------------------------------------------------------


def run_exp10l(
    model_name: str,
    model_cfg,
    cumul_result: dict,
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Estimate computational savings from linearization."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10L: Computational cost estimation — {model_name}")
    print(f"{'='*60}")

    n_layers = model_cfg.n_layers
    n_heads = model_cfg.n_heads
    d_head = model_cfg.d_head
    total_heads = n_layers * n_heads

    seq_lengths = [128, 512, 1024, 2048]

    results: dict = {"model_config": {
        "n_layers": n_layers, "n_heads": n_heads,
        "d_head": d_head, "total_heads": total_heads,
    }, "seq_lengths": {}}

    for seq_len in seq_lengths:
        n = seq_len

        # FLOPs per head (attention only, one forward pass):
        # QK^T: 2 * n² * d  (matmul)
        # Scale: n²          (div by sqrt(d))
        # Softmax: ~5 * n²   (max, subtract, exp, sum, divide)
        # AV: 2 * n² * d     (matmul)
        flops_qk = 2 * n * n * d_head
        flops_scale = n * n
        flops_softmax = 5 * n * n
        flops_av = 2 * n * n * d_head
        flops_per_head_softmax = flops_qk + flops_scale + flops_softmax + flops_av

        # Linear attention (no exp, just relu/abs + normalize)
        flops_linear_act = n * n     # relu or abs
        flops_linear_norm = 2 * n * n  # sum + divide
        flops_per_head_linear = flops_qk + flops_scale + flops_linear_act + flops_linear_norm + flops_av

        # Kernelized linear attention: O(n * d²) instead of O(n² * d)
        # QΦ(K)^T@V via causal cumsum: ~6 * n * d² (rough estimate)
        flops_per_head_kernel = 6 * n * d_head * d_head

        flops_total_softmax = total_heads * flops_per_head_softmax

        print(f"\n  seq_len={n}:")
        print(f"    FLOPs/head softmax:  {flops_per_head_softmax:>12,}")
        print(f"    FLOPs/head linear:   {flops_per_head_linear:>12,}")
        print(f"    FLOPs/head kernel:   {flops_per_head_kernel:>12,}")
        savings_lin = (1 - flops_per_head_linear / flops_per_head_softmax) * 100
        savings_ker = (1 - flops_per_head_kernel / flops_per_head_softmax) * 100
        print(f"    Savings linear: {savings_lin:.1f}%")
        print(f"    Savings kernel: {savings_ker:.1f}%")

        # Tradeoff per n_linearized
        tradeoffs = []
        cumul_best = cumul_result.get("cumulative_best", [])
        for cr in cumul_best:
            n_lin = cr["n_linearized"]
            n_softmax_heads = total_heads - n_lin

            flops_with_linear = (
                n_softmax_heads * flops_per_head_softmax
                + n_lin * flops_per_head_linear
            )
            flops_with_kernel = (
                n_softmax_heads * flops_per_head_softmax
                + n_lin * flops_per_head_kernel
            )

            tradeoffs.append({
                "n_linearized": n_lin,
                "ppl_ratio": cr["ratio"],
                "savings_linear_pct": round(
                    (1 - flops_with_linear / flops_total_softmax) * 100, 2
                ),
                "savings_kernel_pct": round(
                    (1 - flops_with_kernel / flops_total_softmax) * 100, 2
                ),
            })

        results["seq_lengths"][str(n)] = {
            "flops_per_head_softmax": flops_per_head_softmax,
            "flops_per_head_linear": flops_per_head_linear,
            "flops_per_head_kernel": flops_per_head_kernel,
            "tradeoffs": tradeoffs,
        }

    # Save
    cache_path = data_dir / f"exp10l_cost_estimate_{model_tag}.json"
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {cache_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, savings_key, title in [
        (axes[0], "savings_linear_pct", "Linear (exp() removal only)"),
        (axes[1], "savings_kernel_pct", "Kernel (O(nd²) attention)"),
    ]:
        for seq_len in seq_lengths:
            tradeoffs = results["seq_lengths"][str(seq_len)]["tradeoffs"]
            if not tradeoffs:
                continue
            x = [t[savings_key] for t in tradeoffs]
            y = [t["ppl_ratio"] for t in tradeoffs]
            ax.plot(x, y, "o-", markersize=4, label=f"seq={seq_len}")

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        for t, c in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
            ax.axhline(y=t, color=c, linestyle=":", alpha=0.4)
        ax.set_xlabel("Attention FLOPs savings (%)")
        ax.set_ylabel("PPL / Baseline")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.95, 3.5)

    plt.suptitle(f"Exp 10L: PPL vs Computational Savings — {model_name}")
    plt.tight_layout()
    path = fig_dir / f"exp10l_cost_tradeoff_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Phase 2 — Linear Attention"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--parts", type=str, default="ijkl",
        help="Parts to run: i, j, k, l, or combinations",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens from WikiText-2 validation set",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "_")

    print(f"Loading model: {args.model}")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    print(f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_head: {model.cfg.d_head}")

    # WikiText-2
    print(f"\n  Loading WikiText-2 validation set (max {args.max_tokens} tokens)...")
    chunks = load_wikitext2_validation(
        model.tokenizer, model.cfg.n_ctx, args.max_tokens
    )
    total_tokens = sum(c.numel() for c in chunks)
    print(f"  Loaded {len(chunks)} chunks, {total_tokens} tokens")

    # Baseline PPL
    phase1a_path = data_dir / f"exp10a_{model_tag}.json"
    if phase1a_path.exists():
        with open(phase1a_path) as f:
            ppl_baseline = json.load(f)["baseline_ppl"]
    else:
        print("  Computing baseline PPL...")
        ppl_baseline = compute_perplexity(model, chunks)
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    # Load mean-ablation results
    mean_path = data_dir / f"exp10e_mean_ablation_{model_tag}.json"
    if mean_path.exists():
        with open(mean_path) as f:
            mean_results = json.load(f)["results"]
        print(f"  Loaded mean-ablation results: {mean_path}")
    else:
        print(f"  WARNING: Mean-ablation results not found: {mean_path}")
        mean_results = []

    # --- Part I ---
    linear_results = None
    if "i" in args.parts:
        linear_results = run_exp10i(
            model, args.model, chunks, ppl_baseline,
            mean_results, fig_dir, data_dir,
        )
    else:
        cache_path = data_dir / f"exp10i_linear_individual_{model_tag}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                linear_results = json.load(f)["results"]
            print(f"  Loaded Part I cache: {cache_path}")

    if linear_results is None and any(p in args.parts for p in "jkl"):
        print("  Part I results required for J/K/L. Run with --parts i first.")
        return

    # --- Part J ---
    method_result = None
    if "j" in args.parts:
        method_result = run_exp10j(
            args.model, linear_results, mean_results, fig_dir, data_dir,
        )
    else:
        cache_path = data_dir / f"exp10j_method_comparison_{model_tag}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                method_result = json.load(f)
            print(f"  Loaded Part J cache: {cache_path}")

    if method_result is None and any(p in args.parts for p in "kl"):
        print("  Part J results required for K/L. Run with --parts j first.")
        return

    # --- Part K ---
    cumul_result = None
    if "k" in args.parts:
        cumul_result = run_exp10k(
            model, args.model, chunks, ppl_baseline,
            method_result, fig_dir, data_dir,
        )
    else:
        cache_path = data_dir / f"exp10k_cumulative_linear_{model_tag}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cumul_result = json.load(f)
            print(f"  Loaded Part K cache: {cache_path}")

    if cumul_result is None and "l" in args.parts:
        print("  Part K results required for L. Run with --parts k first.")
        return

    # --- Part L ---
    if "l" in args.parts:
        run_exp10l(args.model, model.cfg, cumul_result, fig_dir, data_dir)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — Phase 2 — {args.model}")
    print(f"{'='*60}")
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    if method_result:
        best = method_result["best_method"]
        print(f"\n  Best method: {best}")
        for m in METHODS:
            stats = method_result["method_stats"].get(m, {})
            if stats:
                print(f"    {m}: mean ΔPPL = {stats['mean']:+.4f}")

    if cumul_result:
        print(f"\n  Cumulative thresholds:")
        for label, key in [("Best", "thresholds_best"), ("Mixed", "thresholds_mixed")]:
            thresh = cumul_result.get(key, {})
            parts = ", ".join(
                f"{t}x={thresh.get(t, '?')}" for t in ["1.1", "1.5", "2.0"]
            )
            print(f"    {label}: {parts}")


if __name__ == "__main__":
    main()
