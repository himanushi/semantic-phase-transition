"""Experiment 10: Head Ablation — Phase 1 (Parts A-D).

10A: Individual head ablation — measure ΔPPL for each of 144 heads
10B: Importance ranking — histogram, heatmap, power law fit
10C: Cumulative ablation — Pareto curve (heads removed vs PPL)
10D: Layer-wise pattern analysis — correlation with g(l/L)

Usage:
    python experiments/exp10_head_ablation.py --model gpt2 --device mps --parts abcd
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
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# Shared utilities
# -------------------------------------------------------------------


def load_wikitext2_validation(
    tokenizer, n_ctx: int, max_tokens: int = 2048
) -> list[torch.Tensor]:
    """Load WikiText-2 validation set, chunk into sequences of length n_ctx."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    full_text = "\n\n".join(dataset["text"])

    tokens = tokenizer.encode(full_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    chunks = []
    for i in range(0, len(tokens) - n_ctx + 1, n_ctx):
        chunk = torch.tensor(tokens[i : i + n_ctx], dtype=torch.long).unsqueeze(0)
        chunks.append(chunk)

    if not chunks and len(tokens) >= 2:
        # max_tokens < n_ctx の場合、持っている分だけ使う
        chunk = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        chunks.append(chunk)

    return chunks


def compute_perplexity(
    model, chunks: list[torch.Tensor], fwd_hooks: list | None = None
) -> float:
    """Compute perplexity on chunked token sequences, optionally with hooks."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            if fwd_hooks:
                logits = model.run_with_hooks(chunk, fwd_hooks=fwd_hooks)
            else:
                logits = model(chunk)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    return math.exp(total_loss / total_tokens)


def make_ablate_hook(head_idx: int):
    """Create a hook that zeros out a specific head's attention output."""

    def hook_fn(value, hook):
        value[:, :, head_idx, :] = 0
        return value

    return hook_fn


def make_multi_ablate_hook(head_indices: list[int]):
    """Create a hook that zeros out multiple heads' attention output."""

    def hook_fn(value, hook):
        for h in head_indices:
            value[:, :, h, :] = 0
        return value

    return hook_fn


# -------------------------------------------------------------------
# 10A: Individual head ablation
# -------------------------------------------------------------------


def run_exp10a(
    model, model_name: str, chunks: list[torch.Tensor], data_dir: Path
) -> tuple[float, list[dict]]:
    """Ablate each head individually and measure ΔPPL."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10A: Individual head ablation — {model_name}")
    print(f"{'='*60}")
    print(f"  Total heads: {n_total} ({n_layers} layers × {n_heads} heads)")

    # 中間結果キャッシュの確認
    cache_path = data_dir / f"exp10a_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["baseline_ppl"], cached["results"]

    # ベースラインPPL
    print(f"\n  Computing baseline PPL...", end="", flush=True)
    ppl_baseline = compute_perplexity(model, chunks)
    print(f" {ppl_baseline:.4f}")

    # 各ヘッドを個別にablate
    results = []
    t0 = time.time()

    for layer in range(n_layers):
        for head in range(n_heads):
            idx = layer * n_heads + head

            hook_name = f"blocks.{layer}.attn.hook_z"
            hook_fn = make_ablate_hook(head)
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

    # 中間結果を保存
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

    return ppl_baseline, results


# -------------------------------------------------------------------
# 10B: Importance ranking + visualization
# -------------------------------------------------------------------


def run_exp10b(
    ppl_baseline: float,
    results_a: list[dict],
    model_name: str,
    fig_dir: Path,
) -> dict:
    """Rank heads by ΔPPL and create visualizations."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10B: Importance ranking — {model_name}")
    print(f"{'='*60}")

    # ΔPPLで降順ソート（重要なヘッドが先）
    sorted_results = sorted(results_a, key=lambda x: x["delta_ppl"], reverse=True)

    # Top 10 / Bottom 10
    print(f"\n  Baseline PPL: {ppl_baseline:.4f}")
    print(f"\n  Top 10 most important heads (highest ΔPPL):")
    print(f"    {'Rank':>4} {'Layer':>5} {'Head':>4} {'ΔPPL':>10} {'PPL':>10}")
    print(f"    {'-'*35}")
    for i, r in enumerate(sorted_results[:10]):
        print(
            f"    {i+1:>4} L{r['layer']:>4} H{r['head']:>3} "
            f"{r['delta_ppl']:>+10.4f} {r['ppl_ablated']:>10.2f}"
        )

    print(f"\n  Bottom 10 least important heads (lowest ΔPPL):")
    print(f"    {'Rank':>4} {'Layer':>5} {'Head':>4} {'ΔPPL':>10} {'PPL':>10}")
    print(f"    {'-'*35}")
    for i, r in enumerate(sorted_results[-10:]):
        rank = len(sorted_results) - 10 + i + 1
        print(
            f"    {rank:>4} L{r['layer']:>4} H{r['head']:>3} "
            f"{r['delta_ppl']:>+10.4f} {r['ppl_ablated']:>10.2f}"
        )

    # 統計
    delta_ppls = np.array([r["delta_ppl"] for r in results_a])
    n_negligible = int(np.sum(np.abs(delta_ppls) < 0.01))  # < 1%
    n_small = int(np.sum(np.abs(delta_ppls) < 0.10))  # < 10%
    n_negative = int(np.sum(delta_ppls < 0))  # PPLが改善したヘッド

    print(f"\n  Statistics:")
    print(f"    |ΔPPL| < 1% (negligible): {n_negligible}/{len(delta_ppls)} heads")
    print(f"    |ΔPPL| < 10%: {n_small}/{len(delta_ppls)} heads")
    print(f"    ΔPPL < 0 (PPL improved): {n_negative}/{len(delta_ppls)} heads")
    print(f"    Mean ΔPPL: {np.mean(delta_ppls):.4f}")
    print(f"    Median ΔPPL: {np.median(delta_ppls):.4f}")
    print(f"    Max ΔPPL: {np.max(delta_ppls):.4f}")
    print(f"    Min ΔPPL: {np.min(delta_ppls):.4f}")

    # --- 可視化 ---

    # 1. ヒストグラム
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(delta_ppls, bins=30, edgecolor="black", alpha=0.7, color="C0")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="ΔPPL=0")
    ax.axvline(x=0.1, color="orange", linestyle=":", alpha=0.7, label="ΔPPL=0.1 (10%)")
    ax.set_xlabel("ΔPPL (relative)")
    ax.set_ylabel("Count")
    ax.set_title(f"Exp 10B: Head Importance Distribution — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = fig_dir / f"exp10b_histogram_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # 2. ヒートマップ (layer × head)
    n_layers = max(r["layer"] for r in results_a) + 1
    n_heads = max(r["head"] for r in results_a) + 1
    heatmap = np.zeros((n_layers, n_heads))
    for r in results_a:
        heatmap[r["layer"], r["head"]] = r["delta_ppl"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="ΔPPL (relative)")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Exp 10B: Head Importance Heatmap — {model_name}")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    # 値を表示
    vmax = np.max(np.abs(heatmap))
    for l in range(n_layers):
        for h in range(n_heads):
            val = heatmap[l, h]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                h, l, f"{val:.3f}", ha="center", va="center",
                fontsize=6, color=color,
            )
    plt.tight_layout()
    path = fig_dir / f"exp10b_heatmap_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # 3. べき乗則フィット（ランク vs |ΔPPL|）
    sorted_delta = np.sort(np.abs(delta_ppls))[::-1]  # 降順
    ranks = np.arange(1, len(sorted_delta) + 1)

    # Zipf: ΔPPL(rank) = C * rank^(-α)
    mask = sorted_delta > 0
    fit_success = False
    C_fit, alpha_fit = None, None

    if np.sum(mask) > 5:

        def zipf(x, C, alpha):
            return C * np.power(x, -alpha)

        try:
            popt, _ = curve_fit(
                zipf, ranks[mask], sorted_delta[mask],
                p0=[sorted_delta[0], 1.0],
                bounds=([0, 0.01], [np.inf, 5.0]),
            )
            C_fit, alpha_fit = float(popt[0]), float(popt[1])
            fit_success = True
        except RuntimeError:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(ranks, sorted_delta, "o-", markersize=3, color="C0", label="Data")
    if fit_success:
        ax.plot(
            ranks, zipf(ranks, C_fit, alpha_fit), "--", color="red",
            label=f"Zipf: α={alpha_fit:.2f}",
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel("|ΔPPL|")
    ax.set_title(f"Exp 10B: Rank vs |ΔPPL| — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(ranks, sorted_delta + 1e-10, "o", markersize=3, color="C0", label="Data")
    if fit_success:
        ax.loglog(
            ranks, zipf(ranks, C_fit, alpha_fit), "--", color="red",
            label=f"Zipf: C={C_fit:.4f}, α={alpha_fit:.2f}",
        )
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("|ΔPPL| (log)")
    ax.set_title(f"Exp 10B: Log-log rank plot — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp10b_zipf_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    if fit_success:
        print(f"\n  Zipf fit: C={C_fit:.4f}, α={alpha_fit:.2f}")
        print(f"    (α > 1 suggests strong concentration in few heads)")

    return {
        "ranking": [{"rank": i + 1, **r} for i, r in enumerate(sorted_results)],
        "statistics": {
            "n_negligible_1pct": n_negligible,
            "n_small_10pct": n_small,
            "n_negative": n_negative,
            "mean_delta_ppl": float(np.mean(delta_ppls)),
            "median_delta_ppl": float(np.median(delta_ppls)),
            "max_delta_ppl": float(np.max(delta_ppls)),
            "min_delta_ppl": float(np.min(delta_ppls)),
        },
        "zipf_fit": {
            "C": C_fit,
            "alpha": alpha_fit,
        },
    }


# -------------------------------------------------------------------
# 10C: Cumulative ablation
# -------------------------------------------------------------------


def run_exp10c(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    results_a: list[dict],
    ppl_baseline: float,
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Cumulatively ablate heads from least to most important."""
    model_tag = model_name.replace("/", "_")
    n_total = len(results_a)

    print(f"\n{'='*60}")
    print(f"10C: Cumulative ablation — {model_name}")
    print(f"{'='*60}")

    # 中間結果キャッシュの確認
    cache_path = data_dir / f"exp10c_{model_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        cumul_results = cached["cumulative"]
    else:
        # |ΔPPL|の昇順ソート（影響の少ないヘッドが先）
        sorted_by_importance = sorted(results_a, key=lambda x: abs(x["delta_ppl"]))

        # 測定ステップ
        steps = sorted(set(
            list(range(0, n_total + 1, 10)) + [5, 15, 25, 35, 43, 50, n_total]
        ))
        steps = [s for s in steps if s <= n_total]

        cumul_results = []
        t0 = time.time()

        for i, n_ablated in enumerate(steps):
            if n_ablated == 0:
                ppl = ppl_baseline
            else:
                # レイヤーごとにablateするヘッドをグループ化
                heads_to_ablate = sorted_by_importance[:n_ablated]
                layer_heads: dict[int, list[int]] = {}
                for r in heads_to_ablate:
                    layer_heads.setdefault(r["layer"], []).append(r["head"])

                hooks = []
                for layer, head_list in layer_heads.items():
                    hook_name = f"blocks.{layer}.attn.hook_z"
                    hook_fn = make_multi_ablate_hook(head_list)
                    hooks.append((hook_name, hook_fn))

                ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)

            ratio = ppl / ppl_baseline
            cumul_results.append({
                "n_ablated": n_ablated,
                "ppl": round(ppl, 4),
                "ratio": round(ratio, 4),
            })

            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(steps)}] ablated={n_ablated:>3}: "
                f"PPL={ppl:.2f} ({ratio:.3f}x) [{elapsed:.0f}s]"
            )

        # 中間結果を保存
        output = {
            "model": model_name,
            "baseline_ppl": round(ppl_baseline, 4),
            "ablation_order": [
                {"layer": r["layer"], "head": r["head"], "delta_ppl": r["delta_ppl"]}
                for r in sorted_by_importance
            ],
            "cumulative": cumul_results,
        }
        with open(cache_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {cache_path}")

    # 閾値分析: 各PPL倍率以下で削除可能な最大ヘッド数
    thresholds = {1.1: 0, 1.5: 0, 2.0: 0}
    for cr in cumul_results:
        for thresh in thresholds:
            if cr["ratio"] <= thresh:
                thresholds[thresh] = max(thresholds[thresh], cr["n_ablated"])

    print(f"\n  Threshold analysis:")
    for thresh, n in sorted(thresholds.items()):
        pct = n / n_total * 100
        print(f"    PPL ≤ {thresh}x baseline: ≥{n} heads removable ({pct:.1f}%)")

    # プロット
    n_ablated_list = [cr["n_ablated"] for cr in cumul_results]
    ppl_list = [cr["ppl"] for cr in cumul_results]
    ratio_list = [cr["ratio"] for cr in cumul_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(n_ablated_list, ppl_list, "o-", color="C0", markersize=5)
    ax.axhline(
        y=ppl_baseline, color="gray", linestyle="--", alpha=0.7,
        label=f"Baseline={ppl_baseline:.2f}",
    )
    ax.set_xlabel("Number of heads ablated")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Exp 10C: Cumulative Ablation — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(n_ablated_list, ratio_list, "o-", color="C1", markersize=5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    for thresh_val, color in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        n_removable = thresholds[thresh_val]
        ax.axhline(
            y=thresh_val, color=color, linestyle=":", alpha=0.5,
            label=f"{thresh_val}x (≥{n_removable} heads)",
        )
    ax.set_xlabel("Number of heads ablated")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title(f"Exp 10C: Ablation Pareto Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp10c_pareto_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return {
        "cumulative": cumul_results,
        "thresholds": {str(k): v for k, v in thresholds.items()},
    }


# -------------------------------------------------------------------
# 10D: Layer-wise pattern analysis
# -------------------------------------------------------------------


def run_exp10d(
    ppl_baseline: float,
    results_a: list[dict],
    model_name: str,
    n_layers: int,
    fig_dir: Path,
    data_dir: Path,
) -> dict:
    """Analyze layer-wise patterns and correlate with g(l/L)."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"10D: Layer-wise pattern analysis — {model_name}")
    print(f"{'='*60}")

    # レイヤーごとの統計
    layer_mean_dppl = np.zeros(n_layers)
    layer_max_dppl = np.zeros(n_layers)
    layer_heads: list[list[float]] = [[] for _ in range(n_layers)]

    for r in results_a:
        layer_heads[r["layer"]].append(r["delta_ppl"])

    for l in range(n_layers):
        vals = layer_heads[l]
        layer_mean_dppl[l] = np.mean(vals)
        layer_max_dppl[l] = np.max(vals)

    print(f"\n  Layer-wise mean ΔPPL:")
    print(f"    {'Layer':>5} {'Mean ΔPPL':>10} {'Max ΔPPL':>10}")
    print(f"    {'-'*27}")
    for l in range(n_layers):
        print(f"    L{l:>4} {layer_mean_dppl[l]:>+10.4f} {layer_max_dppl[l]:>+10.4f}")

    # --- プロット 1: レイヤーごとの棒グラフ ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    ax = axes[0]
    ax.bar(range(n_layers), layer_mean_dppl, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean ΔPPL")
    ax.set_title(f"Exp 10D: Mean Head Importance by Layer — {model_name}")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{l}" for l in range(n_layers)])
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(range(n_layers), layer_max_dppl, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max ΔPPL")
    ax.set_title(f"Exp 10D: Max Head Importance by Layer — {model_name}")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{l}" for l in range(n_layers)])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = fig_dir / f"exp10d_layer_importance_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # --- プロット 2: g(l/L) との相関 ---
    g_data = None
    exp3ef_path = data_dir / f"exp3ef_{model_tag}.json"
    if exp3ef_path.exists():
        with open(exp3ef_path) as f:
            exp3ef = json.load(f)

        g_mean = np.array(exp3ef["g_mean"])
        l_norm = np.array(exp3ef["l_norm"])

        # g_mean は13値（レイヤー0-12）、モデルは12レイヤー（0-11）
        # dg/d(l/L): 各レイヤー区間の傾き
        dg = np.diff(g_mean)  # 12値
        dl = np.diff(l_norm)
        dg_dl = dg / dl

        from scipy.stats import pearsonr, spearmanr

        corr_pearson, p_pearson = pearsonr(dg_dl, layer_mean_dppl)
        corr_spearman, p_spearman = spearmanr(dg_dl, layer_mean_dppl)

        print(f"\n  Correlation with dg/d(l/L):")
        print(f"    Pearson:  r={corr_pearson:.3f}, p={p_pearson:.4f}")
        print(f"    Spearman: ρ={corr_spearman:.3f}, p={p_spearman:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # オーバーレイプロット
        ax1 = axes[0]
        ax2 = ax1.twinx()
        x = np.arange(n_layers)
        bars1 = ax1.bar(
            x - 0.15, layer_mean_dppl, width=0.3, color="C0", alpha=0.7,
            label="Mean ΔPPL",
        )
        bars2 = ax2.bar(
            x + 0.15, dg_dl, width=0.3, color="C1", alpha=0.7,
            label="dg/d(l/L)",
        )
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean ΔPPL", color="C0")
        ax2.set_ylabel("dg/d(l/L)", color="C1")
        ax1.set_title(f"Exp 10D: ΔPPL vs g(l/L) slope — {model_name}")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"L{l}" for l in range(n_layers)])
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 散布図
        ax = axes[1]
        ax.scatter(dg_dl, layer_mean_dppl, s=80, color="C2", zorder=5)
        for l in range(n_layers):
            ax.annotate(
                f"L{l}", (dg_dl[l], layer_mean_dppl[l]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
            )
        # 線形フィット
        z = np.polyfit(dg_dl, layer_mean_dppl, 1)
        x_fit = np.linspace(min(dg_dl), max(dg_dl), 50)
        ax.plot(
            x_fit, np.polyval(z, x_fit), "--", color="gray", alpha=0.7,
            label=f"r={corr_pearson:.2f}, p={p_pearson:.3f}",
        )
        ax.set_xlabel("dg/d(l/L)")
        ax.set_ylabel("Mean ΔPPL")
        ax.set_title(f"Exp 10D: Scatter — {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = fig_dir / f"exp10d_g_correlation_{model_tag}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

        g_data = {
            "dg_dl": dg_dl.tolist(),
            "corr_pearson": float(corr_pearson),
            "p_pearson": float(p_pearson),
            "corr_spearman": float(corr_spearman),
            "p_spearman": float(p_spearman),
        }
    else:
        print(f"\n  exp3ef data not found ({exp3ef_path}), skipping g(l/L) correlation")

    return {
        "layer_mean_dppl": layer_mean_dppl.tolist(),
        "layer_max_dppl": layer_max_dppl.tolist(),
        "g_correlation": g_data,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Head Ablation — Phase 1"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--parts", type=str, default="abcd",
        help="Parts to run: a, b, c, d, or combinations like 'abcd'",
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

    # WikiText-2 validation set をロード
    print(f"\n  Loading WikiText-2 validation set (max {args.max_tokens} tokens)...")
    chunks = load_wikitext2_validation(
        model.tokenizer, model.cfg.n_ctx, args.max_tokens
    )
    total_tokens = sum(c.numel() for c in chunks)
    print(f"  Loaded {len(chunks)} chunks, {total_tokens} tokens total")

    # --- 10A ---
    ppl_baseline = None
    results_a = None

    if "a" in args.parts:
        ppl_baseline, results_a = run_exp10a(model, args.model, chunks, data_dir)
    else:
        # キャッシュから Part A の結果をロード
        cache_path = data_dir / f"exp10a_{model_tag}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            ppl_baseline = cached["baseline_ppl"]
            results_a = cached["results"]
            print(f"  Loaded Part A cache: {cache_path}")

    if results_a is None:
        print("  Part A results required. Run with --parts a first.")
        return

    # --- 10B ---
    results_b = None
    if "b" in args.parts:
        results_b = run_exp10b(ppl_baseline, results_a, args.model, fig_dir)

    # --- 10C ---
    results_c = None
    if "c" in args.parts:
        results_c = run_exp10c(
            model, args.model, chunks, results_a, ppl_baseline, fig_dir, data_dir
        )

    # --- 10D ---
    results_d = None
    if "d" in args.parts:
        results_d = run_exp10d(
            ppl_baseline, results_a, args.model,
            model.cfg.n_layers, fig_dir, data_dir,
        )

    # --- 統合結果を保存 ---
    output: dict = {
        "model": args.model,
        "baseline_ppl": ppl_baseline,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "max_tokens": args.max_tokens,
    }
    if results_b:
        output["exp10b"] = results_b
    if results_c:
        output["exp10c"] = results_c
    if results_d:
        output["exp10d"] = results_d

    json_path = data_dir / f"exp10_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Combined data saved: {json_path}")

    # --- サマリー ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    if results_b:
        stats = results_b["statistics"]
        print(f"\n  10B: Head importance")
        print(f"    |ΔPPL| < 1%: {stats['n_negligible_1pct']} heads")
        print(f"    |ΔPPL| < 10%: {stats['n_small_10pct']} heads")
        zf = results_b["zipf_fit"]
        if zf["alpha"]:
            print(f"    Zipf α: {zf['alpha']:.2f}")

    if results_c:
        n_total_heads = model.cfg.n_layers * model.cfg.n_heads
        print(f"\n  10C: Cumulative ablation thresholds")
        for thresh, n in sorted(
            results_c["thresholds"].items(), key=lambda x: float(x[0])
        ):
            pct = int(n) / n_total_heads * 100
            print(f"    PPL ≤ {thresh}x: ≥{n} heads removable ({pct:.1f}%)")

    if results_d:
        print(f"\n  10D: Layer patterns")
        if results_d["g_correlation"]:
            g = results_d["g_correlation"]
            print(
                f"    dg/d(l/L) vs mean ΔPPL: "
                f"r={g['corr_pearson']:.3f} (p={g['p_pearson']:.3f})"
            )

    # --- 成功基準チェック ---
    if results_b and results_c:
        stats = results_b["statistics"]
        n_total_heads = model.cfg.n_layers * model.cfg.n_heads

        print(f"\n{'='*60}")
        print(f"SUCCESS CRITERIA CHECK")
        print(f"{'='*60}")

        n_10pct = stats["n_small_10pct"]
        criterion1 = n_10pct >= 43
        print(
            f"  1. ≥30% heads removable at |ΔPPL|<10%: {n_10pct}/{n_total_heads} "
            f"({'PASS' if criterion1 else 'FAIL'} — need ≥43)"
        )

        alpha = results_b["zipf_fit"]["alpha"]
        criterion2 = alpha is not None and alpha > 0.5
        alpha_str = f"{alpha:.2f}" if alpha is not None else "N/A"
        print(
            f"  2. Power law α > 0.5: "
            f"α={alpha_str} "
            f"({'PASS' if criterion2 else 'FAIL'})"
        )

        if results_d and results_d["g_correlation"]:
            corr = results_d["g_correlation"]["corr_pearson"]
            p_val = results_d["g_correlation"]["p_pearson"]
            criterion3 = abs(corr) > 0.5 and p_val < 0.05
            print(
                f"  3. g(l/L) correlation |r|>0.5: "
                f"r={corr:.3f} (p={p_val:.3f}) "
                f"({'PASS' if criterion3 else 'FAIL'})"
            )


if __name__ == "__main__":
    main()
