"""Experiment 10: Phase 1.5 — Mean Ablation and L0-Protected Analysis.

10E: Mean-ablation (individual) — compare with zero-ablation
10F: L0-protected cumulative ablation (mean-ablation)
10G: Three-curve comparison (zero all, zero L0-protected, mean L0-protected)
10H: Harmful heads analysis

Usage:
    python experiments/exp10b_mean_ablation.py --model gpt2 --device mps --parts efgh
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

from exp10_head_ablation import (
    load_wikitext2_validation,
    compute_perplexity,
    make_ablate_hook,
    make_multi_ablate_hook,
)


# -------------------------------------------------------------------
# Mean-ablation utilities
# -------------------------------------------------------------------


def compute_head_means(
    model, chunks: list[torch.Tensor]
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute mean hook_z activation for each head across all tokens.

    Returns dict: (layer, head) -> mean_vector tensor of shape (d_head,).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    device = next(model.parameters()).device

    layer_sums = {
        l: torch.zeros(n_heads, d_head, device=device) for l in range(n_layers)
    }
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            captured: dict[int, torch.Tensor] = {}

            def make_capture_hook(layer_idx: int):
                def hook_fn(value, hook):
                    # value: (batch, seq, n_heads, d_head)
                    captured[layer_idx] = value.sum(dim=(0, 1))  # (n_heads, d_head)
                    return value
                return hook_fn

            hooks = [
                (f"blocks.{l}.attn.hook_z", make_capture_hook(l))
                for l in range(n_layers)
            ]
            _ = model.run_with_hooks(chunk, fwd_hooks=hooks)

            total_tokens += chunk.shape[0] * chunk.shape[1]
            for l in range(n_layers):
                layer_sums[l] += captured[l]

    head_means: dict[tuple[int, int], torch.Tensor] = {}
    for l in range(n_layers):
        means = layer_sums[l] / total_tokens  # (n_heads, d_head)
        for h in range(n_heads):
            head_means[(l, h)] = means[h]  # (d_head,)

    return head_means


def make_mean_ablate_hook(head_idx: int, mean_vec: torch.Tensor):
    """Create a hook that replaces a head's output with its mean vector."""

    def hook_fn(value, hook):
        value[:, :, head_idx, :] = mean_vec
        return value

    return hook_fn


def make_multi_mean_ablate_hook(
    head_mean_pairs: list[tuple[int, torch.Tensor]],
):
    """Create a hook that replaces multiple heads with their mean vectors.

    head_mean_pairs: list of (head_idx, mean_vec)
    """

    def hook_fn(value, hook):
        for h, mean_vec in head_mean_pairs:
            value[:, :, h, :] = mean_vec
        return value

    return hook_fn


# -------------------------------------------------------------------
# Cumulative ablation helpers
# -------------------------------------------------------------------


def build_cumulative_steps(n_total: int) -> list[int]:
    """Build measurement steps for cumulative ablation."""
    steps = sorted(set(
        list(range(0, n_total + 1, 10))
        + [5, 15, 25, 30, 35, 40, 50, n_total]
    ))
    return [s for s in steps if s <= n_total]


def run_cumulative_ablation(
    model,
    chunks: list[torch.Tensor],
    sorted_heads: list[dict],
    ppl_baseline: float,
    head_means: dict[tuple[int, int], torch.Tensor] | None,
    use_mean: bool,
    label: str,
) -> list[dict]:
    """Run cumulative ablation (zero or mean) on a sorted list of heads.

    Args:
        sorted_heads: list of {"layer": l, "head": h, ...} sorted by importance (least first)
        head_means: required if use_mean=True
        use_mean: True for mean-ablation, False for zero-ablation
    """
    n_total = len(sorted_heads)
    steps = build_cumulative_steps(n_total)

    results = []
    t0 = time.time()

    for i, n_ablated in enumerate(steps):
        if n_ablated == 0:
            ppl = ppl_baseline
        else:
            heads_to_ablate = sorted_heads[:n_ablated]
            # レイヤーごとにグループ化
            layer_groups: dict[int, list[dict]] = {}
            for r in heads_to_ablate:
                layer_groups.setdefault(r["layer"], []).append(r)

            hooks = []
            for layer, group in layer_groups.items():
                hook_name = f"blocks.{layer}.attn.hook_z"
                if use_mean:
                    pairs = [
                        (r["head"], head_means[(r["layer"], r["head"])])
                        for r in group
                    ]
                    hook_fn = make_multi_mean_ablate_hook(pairs)
                else:
                    head_list = [r["head"] for r in group]
                    hook_fn = make_multi_ablate_hook(head_list)
                hooks.append((hook_name, hook_fn))

            ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)

        ratio = ppl / ppl_baseline
        results.append({
            "n_ablated": n_ablated,
            "ppl": round(ppl, 4),
            "ratio": round(ratio, 4),
        })

        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(steps)}] {label} ablated={n_ablated:>3}: "
            f"PPL={ppl:.2f} ({ratio:.3f}x) [{elapsed:.0f}s]"
        )

    return results


def compute_thresholds(cumul_results: list[dict], n_total: int) -> dict[float, int]:
    """Find max heads removable at each PPL threshold."""
    thresholds = {1.1: 0, 1.5: 0, 2.0: 0}
    for cr in cumul_results:
        for thresh in thresholds:
            if cr["ratio"] <= thresh:
                thresholds[thresh] = max(thresholds[thresh], cr["n_ablated"])
    return thresholds


# -------------------------------------------------------------------
# 10E: Mean-Ablation (individual)
# -------------------------------------------------------------------


def run_exp10e(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    head_means: dict[tuple[int, int], torch.Tensor],
    ppl_baseline: float,
    zero_results: list[dict],
    fig_dir: Path,
    data_dir: Path,
) -> list[dict]:
    """Individual mean-ablation for each head, with zero-ablation comparison."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10E: Individual mean-ablation — {model_name}")
    print(f"{'='*60}")

    # キャッシュ確認
    cache_path = data_dir / f"exp10e_mean_ablation_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["results"]

    results = []
    t0 = time.time()

    for layer in range(n_layers):
        for head in range(n_heads):
            idx = layer * n_heads + head

            hook_name = f"blocks.{layer}.attn.hook_z"
            mean_vec = head_means[(layer, head)]
            hook_fn = make_mean_ablate_hook(head, mean_vec)
            ppl = compute_perplexity(model, chunks, fwd_hooks=[(hook_name, hook_fn)])
            delta_ppl = (ppl - ppl_baseline) / ppl_baseline

            results.append({
                "layer": layer,
                "head": head,
                "ppl_ablated": round(ppl, 4),
                "delta_ppl": round(delta_ppl, 6),
            })

            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_total - idx - 1)
            print(
                f"\r  [{idx+1}/{n_total}] L{layer}H{head}: "
                f"PPL={ppl:.2f} ΔPPL={delta_ppl:+.4f} "
                f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                end="", flush=True,
            )

    print()

    # 保存
    output = {
        "model": model_name,
        "baseline_ppl": round(ppl_baseline, 4),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "results": results,
    }
    with open(cache_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {cache_path}")

    # --- Zero vs Mean 比較統計 ---
    zero_map = {(r["layer"], r["head"]): r["delta_ppl"] for r in zero_results}
    mean_map = {(r["layer"], r["head"]): r["delta_ppl"] for r in results}

    zero_arr = np.array([zero_map[(r["layer"], r["head"])] for r in results])
    mean_arr = np.array([r["delta_ppl"] for r in results])

    print(f"\n  Zero vs Mean comparison:")
    print(f"    Zero mean ΔPPL: {np.mean(zero_arr):.4f}")
    print(f"    Mean mean ΔPPL: {np.mean(mean_arr):.4f}")
    print(f"    Reduction ratio: {np.mean(mean_arr) / np.mean(zero_arr):.2f}x")

    n_mean_neg = int(np.sum(mean_arr < 0))
    n_mean_small = int(np.sum(np.abs(mean_arr) < 0.10))
    print(f"    Mean-ablation |ΔPPL| < 10%: {n_mean_small}/144")
    print(f"    Mean-ablation ΔPPL < 0: {n_mean_neg}/144")

    # --- 散布図: zero vs mean ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = [r["layer"] for r in results]
    sc = ax.scatter(zero_arr, mean_arr, c=colors, cmap="viridis", s=30, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Layer")
    # 対角線
    lim_max = max(np.max(np.abs(zero_arr)), np.max(np.abs(mean_arr))) * 1.1
    ax.plot([-0.05, lim_max], [-0.05, lim_max], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("ΔPPL (zero-ablation)")
    ax.set_ylabel("ΔPPL (mean-ablation)")
    ax.set_title(f"Exp 10E: Zero vs Mean Ablation — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ズーム版（L0除外）
    ax = axes[1]
    mask_no_l0 = np.array([r["layer"] != 0 for r in results])
    sc = ax.scatter(
        zero_arr[mask_no_l0], mean_arr[mask_no_l0],
        c=np.array(colors)[mask_no_l0], cmap="viridis", s=30, alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, label="Layer")
    lim_zoom = max(
        np.max(np.abs(zero_arr[mask_no_l0])),
        np.max(np.abs(mean_arr[mask_no_l0])),
    ) * 1.1
    ax.plot([-0.01, lim_zoom], [-0.01, lim_zoom], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("ΔPPL (zero-ablation)")
    ax.set_ylabel("ΔPPL (mean-ablation)")
    ax.set_title(f"Exp 10E: Zero vs Mean (L0 excluded) — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp10e_zero_vs_mean_scatter_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# 10F: L0-protected cumulative ablation (mean)
# -------------------------------------------------------------------


def run_exp10f(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    mean_results: list[dict],
    ppl_baseline: float,
    head_means: dict[tuple[int, int], torch.Tensor],
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """L0-protected cumulative ablation using mean-ablation."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10F: L0-protected cumulative ablation (mean) — {model_name}")
    print(f"{'='*60}")

    # L1-L11のヘッドのみ、|ΔPPL|昇順ソート
    l1_l11 = [r for r in mean_results if r["layer"] != 0]
    sorted_heads = sorted(l1_l11, key=lambda x: abs(x["delta_ppl"]))
    n_target = len(sorted_heads)
    print(f"  Target heads (L1-L11): {n_target}")

    # キャッシュ確認
    cache_path = data_dir / f"exp10f_cumulative_l0protected_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        cumul_results = cached["cumulative_mean"]
    else:
        cumul_results = run_cumulative_ablation(
            model, chunks, sorted_heads, ppl_baseline, head_means,
            use_mean=True, label="mean",
        )

        output = {
            "model": model_name,
            "baseline_ppl": round(ppl_baseline, 4),
            "ablation_order": [
                {"layer": r["layer"], "head": r["head"], "delta_ppl": r["delta_ppl"]}
                for r in sorted_heads
            ],
            "cumulative_mean": cumul_results,
        }
        with open(cache_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {cache_path}")

    thresholds = compute_thresholds(cumul_results, n_target)
    print(f"\n  Threshold analysis (L0-protected, mean-ablation):")
    for thresh, n in sorted(thresholds.items()):
        pct = n / n_target * 100
        print(f"    PPL ≤ {thresh}x: ≥{n} heads removable ({pct:.1f}% of L1-L11)")

    # プロット
    n_list = [cr["n_ablated"] for cr in cumul_results]
    ratio_list = [cr["ratio"] for cr in cumul_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_list, ratio_list, "o-", color="C0", markersize=5, label="Mean-ablation (L0 protected)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    for thresh_val, color in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        n_rem = thresholds[thresh_val]
        ax.axhline(
            y=thresh_val, color=color, linestyle=":", alpha=0.5,
            label=f"{thresh_val}x (≥{n_rem} heads)",
        )
    ax.set_xlabel("Number of L1-L11 heads ablated")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title(f"Exp 10F: L0-Protected Cumulative Ablation — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = fig_dir / f"exp10f_pareto_l0protected_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return {
        "cumulative_mean": cumul_results,
        "thresholds_mean": {str(k): v for k, v in thresholds.items()},
    }


# -------------------------------------------------------------------
# 10G: Three-curve comparison
# -------------------------------------------------------------------


def run_exp10g(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    zero_results_a: list[dict],
    mean_results_e: list[dict],
    ppl_baseline: float,
    head_means: dict[tuple[int, int], torch.Tensor],
    cumul_mean_l0prot: list[dict],
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Three-curve comparison: zero-all, zero-L0prot, mean-L0prot."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10G: Three-curve comparison — {model_name}")
    print(f"{'='*60}")

    # Curve 1: Phase 1 Part C (zero-ablation, all heads)
    phase1c_path = data_dir / f"exp10c_{model_tag}.json"
    if phase1c_path.exists():
        with open(phase1c_path) as f:
            phase1c = json.load(f)
        cumul_zero_all = phase1c["cumulative"]
        print(f"  Loaded Phase 1 Part C: {phase1c_path}")
    else:
        print(f"  Phase 1 Part C not found ({phase1c_path}). Run exp10 --parts c first.")
        return {}

    # Curve 2: Zero-ablation, L0 protected
    l1_l11_zero = [r for r in zero_results_a if r["layer"] != 0]
    sorted_zero = sorted(l1_l11_zero, key=lambda x: abs(x["delta_ppl"]))

    cache_path = data_dir / f"exp10g_zero_l0prot_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached zero-L0prot from {cache_path}")
        with open(cache_path) as f:
            cumul_zero_l0prot = json.load(f)["cumulative"]
    else:
        print(f"\n  Running zero-ablation (L0 protected)...")
        cumul_zero_l0prot = run_cumulative_ablation(
            model, chunks, sorted_zero, ppl_baseline, None,
            use_mean=False, label="zero-L0prot",
        )
        with open(cache_path, "w") as f:
            json.dump({"cumulative": cumul_zero_l0prot}, f, indent=2)
        print(f"  Saved: {cache_path}")

    # Curve 3: Mean-ablation, L0 protected (from Part F)
    cumul_mean_l0prot_data = cumul_mean_l0prot

    # 閾値分析
    n_l1l11 = len(l1_l11_zero)
    thresh_zero_all = compute_thresholds(cumul_zero_all, 144)
    thresh_zero_l0p = compute_thresholds(cumul_zero_l0prot, n_l1l11)
    thresh_mean_l0p = compute_thresholds(cumul_mean_l0prot_data, n_l1l11)

    print(f"\n  Threshold comparison (heads removable):")
    print(f"    {'Threshold':>10} {'Zero-All':>10} {'Zero-L0p':>10} {'Mean-L0p':>10}")
    print(f"    {'-'*42}")
    for t in [1.1, 1.5, 2.0]:
        print(
            f"    {t:>10.1f}x {thresh_zero_all[t]:>10} "
            f"{thresh_zero_l0p[t]:>10} {thresh_mean_l0p[t]:>10}"
        )

    # 差分分析: mean vs zero (L0保護条件)
    # 同じn_ablatedでのratio差を計算
    zero_l0p_map = {cr["n_ablated"]: cr["ratio"] for cr in cumul_zero_l0prot}
    mean_l0p_map = {cr["n_ablated"]: cr["ratio"] for cr in cumul_mean_l0prot_data}
    common_steps = sorted(set(zero_l0p_map.keys()) & set(mean_l0p_map.keys()))

    print(f"\n  Mean vs Zero gap (L0-protected):")
    print(f"    {'N ablated':>10} {'Zero ratio':>12} {'Mean ratio':>12} {'Gap':>10}")
    print(f"    {'-'*46}")
    for n in common_steps:
        if n == 0:
            continue
        gap = zero_l0p_map[n] - mean_l0p_map[n]
        print(
            f"    {n:>10} {zero_l0p_map[n]:>12.3f} "
            f"{mean_l0p_map[n]:>12.3f} {gap:>+10.3f}"
        )

    # --- 3本のカーブを重ねてプロット ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: PPL (absolute)
    ax = axes[0]
    ax.plot(
        [cr["n_ablated"] for cr in cumul_zero_all],
        [cr["ppl"] for cr in cumul_zero_all],
        "o-", color="C3", markersize=4, label="(a) Zero, all heads",
    )
    ax.plot(
        [cr["n_ablated"] for cr in cumul_zero_l0prot],
        [cr["ppl"] for cr in cumul_zero_l0prot],
        "s-", color="C0", markersize=4, label="(b) Zero, L0 protected",
    )
    ax.plot(
        [cr["n_ablated"] for cr in cumul_mean_l0prot_data],
        [cr["ppl"] for cr in cumul_mean_l0prot_data],
        "^-", color="C2", markersize=4, label="(c) Mean, L0 protected",
    )
    ax.axhline(y=ppl_baseline, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of heads ablated")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Exp 10G: Three-Curve Comparison — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(200, max(cr["ppl"] for cr in cumul_zero_all) * 1.1))

    # 右: ratio (zoom into 1.0-3.0 range)
    ax = axes[1]
    ax.plot(
        [cr["n_ablated"] for cr in cumul_zero_all],
        [cr["ratio"] for cr in cumul_zero_all],
        "o-", color="C3", markersize=4, label="(a) Zero, all heads",
    )
    ax.plot(
        [cr["n_ablated"] for cr in cumul_zero_l0prot],
        [cr["ratio"] for cr in cumul_zero_l0prot],
        "s-", color="C0", markersize=4, label="(b) Zero, L0 protected",
    )
    ax.plot(
        [cr["n_ablated"] for cr in cumul_mean_l0prot_data],
        [cr["ratio"] for cr in cumul_mean_l0prot_data],
        "^-", color="C2", markersize=4, label="(c) Mean, L0 protected",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for t, c in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        ax.axhline(y=t, color=c, linestyle=":", alpha=0.4)
    ax.set_xlabel("Number of heads ablated")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title(f"Exp 10G: Ratio Comparison (zoomed) — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 3.5)

    plt.tight_layout()
    path = fig_dir / f"exp10g_three_curves_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return {
        "thresholds_zero_all": {str(k): v for k, v in thresh_zero_all.items()},
        "thresholds_zero_l0prot": {str(k): v for k, v in thresh_zero_l0p.items()},
        "thresholds_mean_l0prot": {str(k): v for k, v in thresh_mean_l0p.items()},
    }


# -------------------------------------------------------------------
# 10H: Harmful heads analysis
# -------------------------------------------------------------------


def run_exp10h(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    zero_results: list[dict],
    mean_results: list[dict],
    ppl_baseline: float,
    head_means: dict[tuple[int, int], torch.Tensor],
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Analyze heads whose removal improves PPL."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10H: Harmful heads analysis — {model_name}")
    print(f"{'='*60}")

    # zero-ablationで負のΔPPLを持つヘッド
    harmful_zero = [r for r in zero_results if r["delta_ppl"] < 0]
    harmful_zero.sort(key=lambda x: x["delta_ppl"])

    # mean-ablationで負のΔPPLを持つヘッド
    harmful_mean = [r for r in mean_results if r["delta_ppl"] < 0]
    harmful_mean.sort(key=lambda x: x["delta_ppl"])

    print(f"\n  Harmful heads (zero-ablation ΔPPL < 0): {len(harmful_zero)}")
    for r in harmful_zero:
        print(f"    L{r['layer']}H{r['head']}: ΔPPL={r['delta_ppl']:+.4f}")

    print(f"\n  Harmful heads (mean-ablation ΔPPL < 0): {len(harmful_mean)}")
    for r in harmful_mean:
        print(f"    L{r['layer']}H{r['head']}: ΔPPL={r['delta_ppl']:+.4f}")

    results: dict = {
        "harmful_zero": [
            {"layer": r["layer"], "head": r["head"], "delta_ppl": r["delta_ppl"]}
            for r in harmful_zero
        ],
        "harmful_mean": [
            {"layer": r["layer"], "head": r["head"], "delta_ppl": r["delta_ppl"]}
            for r in harmful_mean
        ],
    }

    if not harmful_zero:
        print("  No harmful heads found (zero-ablation). Skipping joint tests.")
        return results

    # --- Joint ablation of harmful heads ---
    print(f"\n  Joint ablation of {len(harmful_zero)} harmful heads (zero-ablation):")

    # Zero-ablation: 全有害ヘッドを同時にゼロ化
    layer_groups_zero: dict[int, list[int]] = {}
    for r in harmful_zero:
        layer_groups_zero.setdefault(r["layer"], []).append(r["head"])

    hooks_zero = []
    for layer, heads in layer_groups_zero.items():
        hooks_zero.append((
            f"blocks.{layer}.attn.hook_z",
            make_multi_ablate_hook(heads),
        ))
    ppl_joint_zero = compute_perplexity(model, chunks, fwd_hooks=hooks_zero)
    delta_joint_zero = (ppl_joint_zero - ppl_baseline) / ppl_baseline
    print(f"    PPL = {ppl_joint_zero:.4f} (ΔPPL = {delta_joint_zero:+.4f})")

    # Mean-ablation: 全有害ヘッドを同時にmean置換
    layer_groups_mean: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for r in harmful_zero:
        pair = (r["head"], head_means[(r["layer"], r["head"])])
        layer_groups_mean.setdefault(r["layer"], []).append(pair)

    hooks_mean = []
    for layer, pairs in layer_groups_mean.items():
        hooks_mean.append((
            f"blocks.{layer}.attn.hook_z",
            make_multi_mean_ablate_hook(pairs),
        ))
    ppl_joint_mean = compute_perplexity(model, chunks, fwd_hooks=hooks_mean)
    delta_joint_mean = (ppl_joint_mean - ppl_baseline) / ppl_baseline
    print(f"    PPL (mean) = {ppl_joint_mean:.4f} (ΔPPL = {delta_joint_mean:+.4f})")

    results["joint_zero_ppl"] = round(ppl_joint_zero, 4)
    results["joint_zero_delta"] = round(delta_joint_zero, 6)
    results["joint_mean_ppl"] = round(ppl_joint_mean, 4)
    results["joint_mean_delta"] = round(delta_joint_mean, 6)

    # --- 有害ヘッドを先に除去してからの累積ablation ---
    print(f"\n  Cumulative ablation with harmful heads pre-removed...")

    # L1-L11ヘッドから有害ヘッドを除外したリスト
    harmful_keys = {(r["layer"], r["head"]) for r in harmful_zero}
    mean_map = {(r["layer"], r["head"]): r for r in mean_results}

    remaining = [
        r for r in mean_results
        if r["layer"] != 0 and (r["layer"], r["head"]) not in harmful_keys
    ]
    sorted_remaining = sorted(remaining, key=lambda x: abs(x["delta_ppl"]))

    # 有害ヘッドをベースとして常に含めた累積ablation
    # baseline: harmful heads already zero-ablated
    ppl_with_harmful_removed = ppl_joint_zero  # これが新しいbaselineになる

    n_remaining = len(sorted_remaining)
    steps = build_cumulative_steps(n_remaining)

    cumul_harmful_first = []
    t0 = time.time()

    for i, n_add in enumerate(steps):
        if n_add == 0:
            ppl = ppl_with_harmful_removed
        else:
            # 有害ヘッド(zero) + 追加ヘッド(mean)のhooksを合成
            additional = sorted_remaining[:n_add]
            all_hooks = list(hooks_zero)  # 有害ヘッドのzero hooks

            add_layer_groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
            for r in additional:
                pair = (r["head"], head_means[(r["layer"], r["head"])])
                add_layer_groups.setdefault(r["layer"], []).append(pair)

            for layer, pairs in add_layer_groups.items():
                all_hooks.append((
                    f"blocks.{layer}.attn.hook_z",
                    make_multi_mean_ablate_hook(pairs),
                ))

            ppl = compute_perplexity(model, chunks, fwd_hooks=all_hooks)

        # n_totalはharmful + additional
        n_total_ablated = len(harmful_zero) + n_add
        ratio = ppl / ppl_baseline
        cumul_harmful_first.append({
            "n_additional": n_add,
            "n_total_ablated": n_total_ablated,
            "ppl": round(ppl, 4),
            "ratio": round(ratio, 4),
        })

        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(steps)}] +{n_add:>3} (total={n_total_ablated:>3}): "
            f"PPL={ppl:.2f} ({ratio:.3f}x) [{elapsed:.0f}s]"
        )

    results["cumulative_harmful_first"] = cumul_harmful_first

    # 保存
    cache_path = data_dir / f"exp10h_harmful_heads_{model_tag}.json"
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {cache_path}")

    # --- 可視化: 有害ヘッド除去の効果 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: joint ablation bar chart
    ax = axes[0]
    labels_bar = ["Baseline", f"Zero ({len(harmful_zero)}H)", f"Mean ({len(harmful_zero)}H)"]
    values_bar = [ppl_baseline, ppl_joint_zero, ppl_joint_mean]
    colors_bar = ["gray", "C0", "C2"]
    bars = ax.bar(labels_bar, values_bar, color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, values_bar):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", fontsize=10,
        )
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Exp 10H: Joint Harmful Head Removal — {model_name}")
    ax.grid(True, alpha=0.3, axis="y")

    # 右: 累積カーブ比較（harmful-first vs standard mean L0prot）
    ax = axes[1]
    # harmful-first curve (x-axis = total heads ablated)
    ax.plot(
        [cr["n_total_ablated"] for cr in cumul_harmful_first],
        [cr["ratio"] for cr in cumul_harmful_first],
        "^-", color="C4", markersize=4, label="Harmful-first + mean",
    )
    # Load Part F mean-L0prot for comparison
    f_path = data_dir / f"exp10f_cumulative_l0protected_{model_tag}.json"
    if f_path.exists():
        with open(f_path) as f:
            f_data = json.load(f)
        ax.plot(
            [cr["n_ablated"] for cr in f_data["cumulative_mean"]],
            [cr["ratio"] for cr in f_data["cumulative_mean"]],
            "o-", color="C2", markersize=4, label="Standard mean (L0 prot)",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for t, c in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        ax.axhline(y=t, color=c, linestyle=":", alpha=0.4)
    ax.set_xlabel("Total heads ablated")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title(f"Exp 10H: Harmful-First vs Standard — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 3.5)

    plt.tight_layout()
    path = fig_dir / f"exp10h_harmful_heads_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Phase 1.5 — Mean Ablation"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--parts", type=str, default="efgh",
        help="Parts to run: e, f, g, h, or combinations like 'efgh'",
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
    print(f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}")
    print(f"  n_ctx: {model.cfg.n_ctx}, Device: {args.device}")

    # WikiText-2 validation set
    print(f"\n  Loading WikiText-2 validation set (max {args.max_tokens} tokens)...")
    chunks = load_wikitext2_validation(
        model.tokenizer, model.cfg.n_ctx, args.max_tokens
    )
    total_tokens = sum(c.numel() for c in chunks)
    print(f"  Loaded {len(chunks)} chunks, {total_tokens} tokens total")

    # Phase 1 Part A の結果をロード
    phase1a_path = data_dir / f"exp10a_{model_tag}.json"
    if not phase1a_path.exists():
        print(f"  ERROR: Phase 1 Part A results not found: {phase1a_path}")
        print(f"  Run: python experiments/exp10_head_ablation.py --parts a first.")
        return

    with open(phase1a_path) as f:
        phase1a = json.load(f)
    ppl_baseline = phase1a["baseline_ppl"]
    zero_results = phase1a["results"]
    print(f"  Loaded Phase 1 results: baseline PPL = {ppl_baseline:.4f}")

    # head means の計算
    print(f"\n  Computing head mean activations...")
    t0 = time.time()
    head_means = compute_head_means(model, chunks)
    print(f"  Done ({time.time() - t0:.1f}s)")

    # --- 10E ---
    mean_results = None
    if "e" in args.parts:
        mean_results = run_exp10e(
            model, args.model, chunks, head_means, ppl_baseline,
            zero_results, fig_dir, data_dir,
        )
    else:
        cache_path = data_dir / f"exp10e_mean_ablation_{model_tag}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                mean_results = json.load(f)["results"]
            print(f"  Loaded Part E cache: {cache_path}")

    if mean_results is None and any(p in args.parts for p in "fgh"):
        print("  Part E results required for F/G/H. Run with --parts e first.")
        return

    # --- 10F ---
    results_f = None
    if "f" in args.parts:
        results_f = run_exp10f(
            model, args.model, chunks, mean_results, ppl_baseline,
            head_means, fig_dir, data_dir,
        )

    # --- 10G ---
    results_g = None
    if "g" in args.parts:
        # Part Fの結果が必要
        cumul_mean_l0p = None
        if results_f:
            cumul_mean_l0p = results_f["cumulative_mean"]
        else:
            f_path = data_dir / f"exp10f_cumulative_l0protected_{model_tag}.json"
            if f_path.exists():
                with open(f_path) as f:
                    cumul_mean_l0p = json.load(f)["cumulative_mean"]

        if cumul_mean_l0p is None:
            print("  Part F results required for G. Run with --parts f first.")
        else:
            results_g = run_exp10g(
                model, args.model, chunks, zero_results, mean_results,
                ppl_baseline, head_means, cumul_mean_l0p, fig_dir, data_dir,
            )

    # --- 10H ---
    results_h = None
    if "h" in args.parts:
        results_h = run_exp10h(
            model, args.model, chunks, zero_results, mean_results,
            ppl_baseline, head_means, fig_dir, data_dir,
        )

    # --- 統合結果 ---
    output: dict = {
        "model": args.model,
        "baseline_ppl": ppl_baseline,
        "max_tokens": args.max_tokens,
        "phase": "1.5",
    }
    if results_f:
        output["exp10f"] = results_f
    if results_g:
        output["exp10g"] = results_g
    if results_h:
        output["exp10h"] = results_h

    json_path = data_dir / f"exp10_phase1.5_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Combined data saved: {json_path}")

    # --- サマリー ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — Phase 1.5 — {args.model}")
    print(f"{'='*60}")
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    if mean_results:
        mean_dppl = np.array([r["delta_ppl"] for r in mean_results])
        zero_dppl = np.array([r["delta_ppl"] for r in zero_results])
        print(f"\n  10E: Mean vs Zero ablation")
        print(f"    Zero mean ΔPPL: {np.mean(zero_dppl):.4f}")
        print(f"    Mean mean ΔPPL: {np.mean(mean_dppl):.4f}")
        if np.mean(zero_dppl) > 0:
            print(f"    Reduction: {np.mean(mean_dppl)/np.mean(zero_dppl):.2f}x")

    if results_g:
        print(f"\n  10G: Threshold comparison (heads removable)")
        print(f"    {'':>10} {'Zero-All':>10} {'Zero-L0p':>10} {'Mean-L0p':>10}")
        for t in ["1.1", "1.5", "2.0"]:
            za = results_g["thresholds_zero_all"].get(t, "?")
            zl = results_g["thresholds_zero_l0prot"].get(t, "?")
            ml = results_g["thresholds_mean_l0prot"].get(t, "?")
            print(f"    {t:>10}x {za:>10} {zl:>10} {ml:>10}")

    if results_h:
        n_harmful = len(results_h.get("harmful_zero", []))
        if "joint_zero_delta" in results_h:
            print(f"\n  10H: Harmful heads ({n_harmful} heads)")
            print(f"    Joint zero ΔPPL: {results_h['joint_zero_delta']:+.4f}")
            print(f"    Joint mean ΔPPL: {results_h['joint_mean_delta']:+.4f}")


if __name__ == "__main__":
    main()
