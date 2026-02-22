"""Experiment 10: Validation + Phase 3 — CoT Recovery (Parts V, P).

Part V: Re-validate Phase 1-2 findings with full WikiText-2 (36K+ tokens)
  V1: Full-token baseline PPL
  V2: Top-10 head individual zero-ablation
  V3: L0H8 mean-ablation re-validation
  V4: ReLU cumulative linearization Pareto curve
  V5: Systematic 2048 vs full-token comparison

Part P: Phase 3 — CoT recovery experiments
  P1: Token generation speed benchmark
  P2: Self-refinement (recursive CoT)
  P3: Equal-FLOPs comparison (best-of-N)
  P4: Longer context window

Usage:
    python experiments/exp10d_validation.py --model gpt2 --device mps --parts v
    python experiments/exp10d_validation.py --model gpt2 --device mps --parts p
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
from exp10b_mean_ablation import (
    compute_head_means,
    make_mean_ablate_hook,
)
from exp10c_linear_attention import (
    build_linear_hooks,
)


# -------------------------------------------------------------------
# V1: Full-token baseline
# -------------------------------------------------------------------


def run_v1(
    model, model_name: str, chunks: list[torch.Tensor], data_dir: Path,
) -> float:
    """Compute baseline PPL on full WikiText-2 validation set."""
    print(f"\n{'='*60}")
    print(f"V1: Full-token baseline PPL — {model_name}")
    print(f"{'='*60}")

    total_tokens = sum(c.numel() for c in chunks)
    print(f"  Chunks: {len(chunks)}, Total tokens: {total_tokens}")

    t0 = time.time()
    ppl = compute_perplexity(model, chunks)
    elapsed = time.time() - t0
    print(f"  Baseline PPL (full): {ppl:.4f}  ({elapsed:.1f}s)")

    return ppl


# -------------------------------------------------------------------
# V2: Top-10 head zero-ablation re-validation
# -------------------------------------------------------------------


def run_v2(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_full: float,
    prev_results: list[dict],
    prev_baseline: float,
) -> list[dict]:
    """Re-validate top-10 most important heads with full tokens."""
    print(f"\n{'='*60}")
    print(f"V2: Top-10 head zero-ablation (full tokens) — {model_name}")
    print(f"{'='*60}")

    # Phase 1のΔPPL降順で上位10ヘッド
    sorted_prev = sorted(prev_results, key=lambda x: x["delta_ppl"], reverse=True)
    top10 = sorted_prev[:10]

    results = []
    t0 = time.time()

    for i, r in enumerate(top10):
        layer, head = r["layer"], r["head"]
        hook_name = f"blocks.{layer}.attn.hook_z"
        hook_fn = make_ablate_hook(head)
        ppl = compute_perplexity(model, chunks, fwd_hooks=[(hook_name, hook_fn)])
        delta_ppl = (ppl - ppl_full) / ppl_full
        prev_dppl = r["delta_ppl"]

        results.append({
            "layer": layer,
            "head": head,
            "ppl_full": round(ppl, 4),
            "delta_ppl_full": round(delta_ppl, 6),
            "ppl_2k": r["ppl_ablated"],
            "delta_ppl_2k": prev_dppl,
            "diff_pct": round((delta_ppl - prev_dppl) / max(abs(prev_dppl), 1e-8) * 100, 1),
        })

        elapsed = time.time() - t0
        print(
            f"  [{i+1}/10] L{layer}H{head}: "
            f"ΔPPL_full={delta_ppl:+.4f} (2K: {prev_dppl:+.4f}) "
            f"[{elapsed:.0f}s]"
        )

    # ランキング比較
    rank_2k = {(r["layer"], r["head"]): i + 1 for i, r in enumerate(top10)}
    sorted_full = sorted(results, key=lambda x: x["delta_ppl_full"], reverse=True)
    rank_full = {(r["layer"], r["head"]): i + 1 for i, r in enumerate(sorted_full)}

    print(f"\n  Ranking comparison (top 10):")
    print(f"    {'Head':>8} {'Rank(2K)':>10} {'Rank(Full)':>12} {'ΔPPL(2K)':>10} {'ΔPPL(Full)':>12}")
    print(f"    {'-'*54}")
    for r in sorted_full:
        key = (r["layer"], r["head"])
        print(
            f"    L{r['layer']}H{r['head']:>2} "
            f"{rank_2k[key]:>10} {rank_full[key]:>12} "
            f"{r['delta_ppl_2k']:>+10.4f} {r['delta_ppl_full']:>+12.4f}"
        )

    return results


# -------------------------------------------------------------------
# V3: L0H8 mean-ablation re-validation
# -------------------------------------------------------------------


def run_v3(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_full: float,
    head_means: dict[tuple[int, int], torch.Tensor],
    prev_zero_results: list[dict],
    prev_mean_results: list[dict],
) -> dict:
    """Re-validate L0H8 zero/mean ablation finding."""
    print(f"\n{'='*60}")
    print(f"V3: L0H8 mean-ablation re-validation — {model_name}")
    print(f"{'='*60}")

    layer, head = 0, 8

    # Zero-ablation
    hook_name = f"blocks.{layer}.attn.hook_z"
    hook_fn = make_ablate_hook(head)
    ppl_zero = compute_perplexity(model, chunks, fwd_hooks=[(hook_name, hook_fn)])
    dppl_zero = (ppl_zero - ppl_full) / ppl_full

    # Mean-ablation
    mean_vec = head_means[(layer, head)]
    hook_fn_mean = make_mean_ablate_hook(head, mean_vec)
    ppl_mean = compute_perplexity(model, chunks, fwd_hooks=[(hook_name, hook_fn_mean)])
    dppl_mean = (ppl_mean - ppl_full) / ppl_full

    reduction = dppl_zero / max(abs(dppl_mean), 1e-8) if dppl_mean != 0 else float("inf")

    # 2Kの値
    prev_zero = next(
        (r for r in prev_zero_results if r["layer"] == 0 and r["head"] == 8), None
    )
    prev_mean = next(
        (r for r in prev_mean_results if r["layer"] == 0 and r["head"] == 8), None
    )

    prev_z_dppl = prev_zero["delta_ppl"] if prev_zero else None
    prev_m_dppl = prev_mean["delta_ppl"] if prev_mean else None
    prev_reduction = (
        prev_z_dppl / max(abs(prev_m_dppl), 1e-8)
        if prev_z_dppl is not None and prev_m_dppl is not None and prev_m_dppl != 0
        else None
    )

    print(f"  L0H8 zero-ablation: ΔPPL = {dppl_zero:+.4f} (2K: {prev_z_dppl:+.4f})")
    print(f"  L0H8 mean-ablation: ΔPPL = {dppl_mean:+.4f} (2K: {prev_m_dppl:+.4f})")
    print(f"  Reduction ratio: {reduction:.1f}x (2K: {prev_reduction:.1f}x)")

    return {
        "zero_ppl_full": round(ppl_zero, 4),
        "zero_dppl_full": round(dppl_zero, 6),
        "mean_ppl_full": round(ppl_mean, 4),
        "mean_dppl_full": round(dppl_mean, 6),
        "reduction_full": round(reduction, 1),
        "zero_dppl_2k": prev_z_dppl,
        "mean_dppl_2k": prev_m_dppl,
        "reduction_2k": round(prev_reduction, 1) if prev_reduction else None,
    }


# -------------------------------------------------------------------
# V4: ReLU cumulative linearization re-validation
# -------------------------------------------------------------------


def run_v4(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_full: float,
    best_ranking: list[dict],
    prev_cumul: list[dict],
) -> list[dict]:
    """Re-validate ReLU cumulative linearization with full tokens."""
    print(f"\n{'='*60}")
    print(f"V4: ReLU cumulative linearization (full tokens) — {model_name}")
    print(f"{'='*60}")

    n_total = len(best_ranking)
    steps = sorted(set(
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, n_total]
    ))
    steps = [s for s in steps if s <= n_total]

    results = []
    t0 = time.time()

    for i, n in enumerate(steps):
        if n == 0:
            ppl = ppl_full
        else:
            heads = best_ranking[:n]
            head_method_list = [(r["layer"], r["head"], "relu") for r in heads]
            hooks = build_linear_hooks(head_method_list)
            ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)

        ratio = ppl / ppl_full
        results.append({
            "n_linearized": n,
            "ppl": round(ppl, 4),
            "ratio": round(ratio, 4),
        })

        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(steps)}] n={n:>3}: "
            f"PPL={ppl:.2f} ({ratio:.3f}x) [{elapsed:.0f}s]"
        )

    # 閾値分析
    thresholds = {1.1: 0, 1.5: 0, 2.0: 0}
    for cr in results:
        for t in thresholds:
            if cr["ratio"] <= t:
                thresholds[t] = max(thresholds[t], cr["n_linearized"])

    # 2K結果との比較
    prev_map = {cr["n_linearized"]: cr for cr in prev_cumul}
    prev_thresh = {1.1: 0, 1.5: 0, 2.0: 0}
    for cr in prev_cumul:
        for t in prev_thresh:
            if cr["ratio"] <= t:
                prev_thresh[t] = max(prev_thresh[t], cr["n_linearized"])

    print(f"\n  Threshold comparison:")
    print(f"    {'Threshold':>10} {'2K tokens':>12} {'Full tokens':>12}")
    for t in [1.1, 1.5, 2.0]:
        print(f"    {t:>10.1f}x {prev_thresh[t]:>12} {thresholds[t]:>12}")

    return results


# -------------------------------------------------------------------
# V5: Systematic 2K vs full comparison
# -------------------------------------------------------------------


def run_v5(
    model_name: str,
    ppl_full: float,
    ppl_2k: float,
    v2_results: list[dict],
    v3_result: dict,
    v4_results: list[dict],
    prev_cumul: list[dict],
    fig_dir: Path,
) -> dict:
    """Create systematic comparison table and visualizations."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"V5: Systematic 2K vs full-token comparison — {model_name}")
    print(f"{'='*60}")

    # --- 比較テーブル ---
    comparisons = []

    # Baseline PPL
    comparisons.append({
        "measurement": "Baseline PPL",
        "value_2k": round(ppl_2k, 4),
        "value_full": round(ppl_full, 4),
        "diff_pct": round((ppl_full - ppl_2k) / ppl_2k * 100, 2),
    })

    # Top-10 zero-ablation ΔPPL
    for r in v2_results:
        comparisons.append({
            "measurement": f"Zero L{r['layer']}H{r['head']} ΔPPL",
            "value_2k": r["delta_ppl_2k"],
            "value_full": r["delta_ppl_full"],
            "diff_pct": r["diff_pct"],
        })

    # L0H8 mean-ablation
    comparisons.append({
        "measurement": "L0H8 zero ΔPPL",
        "value_2k": v3_result["zero_dppl_2k"],
        "value_full": v3_result["zero_dppl_full"],
        "diff_pct": round(
            (v3_result["zero_dppl_full"] - v3_result["zero_dppl_2k"])
            / max(abs(v3_result["zero_dppl_2k"]), 1e-8) * 100, 1
        ),
    })
    comparisons.append({
        "measurement": "L0H8 mean ΔPPL",
        "value_2k": v3_result["mean_dppl_2k"],
        "value_full": v3_result["mean_dppl_full"],
        "diff_pct": round(
            (v3_result["mean_dppl_full"] - v3_result["mean_dppl_2k"])
            / max(abs(v3_result["mean_dppl_2k"]), 1e-8) * 100, 1
        ),
    })
    comparisons.append({
        "measurement": "L0H8 reduction ratio",
        "value_2k": v3_result["reduction_2k"],
        "value_full": v3_result["reduction_full"],
        "diff_pct": round(
            (v3_result["reduction_full"] - v3_result["reduction_2k"])
            / max(abs(v3_result["reduction_2k"]), 1e-8) * 100, 1
        ) if v3_result["reduction_2k"] else None,
    })

    # ReLU cumulative thresholds
    prev_map = {cr["n_linearized"]: cr for cr in prev_cumul}
    full_map = {cr["n_linearized"]: cr for cr in v4_results}

    for n in sorted(set(prev_map.keys()) & set(full_map.keys())):
        if n == 0:
            continue
        r2k = prev_map[n]["ratio"]
        rfull = full_map[n]["ratio"]
        comparisons.append({
            "measurement": f"ReLU cumul n={n} ratio",
            "value_2k": r2k,
            "value_full": rfull,
            "diff_pct": round((rfull - r2k) / max(abs(r2k - 1.0), 1e-8) * 100, 1)
            if abs(r2k - 1.0) > 1e-4 else 0.0,
        })

    # バイアス判定
    diff_pcts = [
        c["diff_pct"] for c in comparisons
        if c["diff_pct"] is not None and abs(c.get("value_2k", 0) or 0) > 1e-4
    ]
    mean_diff = np.mean(diff_pcts) if diff_pcts else 0
    bias_direction = "過大推定" if mean_diff > 5 else ("過小推定" if mean_diff < -5 else "概ね一致")

    print(f"\n  Summary:")
    print(f"    Mean diff%: {mean_diff:+.1f}%")
    print(f"    Bias: {bias_direction}")

    # 主要結論の維持判定
    thresh_full = {1.1: 0, 1.5: 0, 2.0: 0}
    for cr in v4_results:
        for t in thresh_full:
            if cr["ratio"] <= t:
                thresh_full[t] = max(thresh_full[t], cr["n_linearized"])

    thresh_2k = {1.1: 0, 1.5: 0, 2.0: 0}
    for cr in prev_cumul:
        for t in thresh_2k:
            if cr["ratio"] <= t:
                thresh_2k[t] = max(thresh_2k[t], cr["n_linearized"])

    conclusions_maintained = all(
        abs(thresh_full[t] - thresh_2k[t]) <= 10 for t in [1.1, 1.5, 2.0]
    )
    print(f"    Phase 2 conclusions maintained: {'YES' if conclusions_maintained else 'NO'}")

    # --- 可視化 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 累積パレート曲線の比較
    ax = axes[0]
    ax.plot(
        [cr["n_linearized"] for cr in prev_cumul],
        [cr["ratio"] for cr in prev_cumul],
        "o-", color="C0", markersize=5, alpha=0.7, label="2K tokens",
    )
    ax.plot(
        [cr["n_linearized"] for cr in v4_results],
        [cr["ratio"] for cr in v4_results],
        "s-", color="C3", markersize=5, label="Full tokens",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for t, c in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        ax.axhline(y=t, color=c, linestyle=":", alpha=0.4)
    ax.set_xlabel("Heads linearized (ReLU)")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title("ReLU Cumulative: 2K vs Full")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 3.5)

    # 右: Top-10 ΔPPL比較
    ax = axes[1]
    labels = [f"L{r['layer']}H{r['head']}" for r in v2_results]
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, [r["delta_ppl_2k"] for r in v2_results],
                   width, label="2K tokens", color="C0", alpha=0.7)
    bars2 = ax.bar(x + width / 2, [r["delta_ppl_full"] for r in v2_results],
                   width, label="Full tokens", color="C3", alpha=0.7)
    ax.set_xlabel("Head")
    ax.set_ylabel("ΔPPL")
    ax.set_title("Top-10 Zero-Ablation: 2K vs Full")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Exp 10V: Validation — {model_name}")
    plt.tight_layout()
    path = fig_dir / f"exp10v_validation_comparison_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # パレート曲線（全トークン版、単独）
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [cr["n_linearized"] for cr in v4_results],
        [cr["ratio"] for cr in v4_results],
        "o-", color="C0", markersize=6, linewidth=2,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for t, c in [(1.1, "green"), (1.5, "orange"), (2.0, "red")]:
        ax.axhline(
            y=t, color=c, linestyle=":", alpha=0.5,
            label=f"{t}x ({thresh_full[t]} heads)",
        )
    ax.set_xlabel("Heads linearized (ReLU, L0 protected)")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title(f"Exp 10V: Validated Pareto Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 3.5)
    plt.tight_layout()
    path = fig_dir / f"exp10v_pareto_validated_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return {
        "comparisons": comparisons,
        "mean_diff_pct": round(mean_diff, 1),
        "bias_direction": bias_direction,
        "conclusions_maintained": conclusions_maintained,
        "thresholds_full": {str(k): v for k, v in thresh_full.items()},
        "thresholds_2k": {str(k): v for k, v in thresh_2k.items()},
    }


# -------------------------------------------------------------------
# P1: Generation speed benchmark
# -------------------------------------------------------------------


def run_p1(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    best_ranking: list[dict],
    fig_dir: Path,
) -> dict:
    """Benchmark token generation speed under different conditions."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"P1: Generation speed benchmark — {model_name}")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    gen_length = 128
    n_runs = 10
    # 短めのプロンプトを用意（最初のchunkから64トークン）
    prompt_tokens = chunks[0][0, :64].unsqueeze(0).to(device)

    conditions = [
        ("baseline", []),
        ("relu_30", [(r["layer"], r["head"], "relu") for r in best_ranking[:30]]),
        ("relu_70", [(r["layer"], r["head"], "relu") for r in best_ranking[:70]]),
    ]

    results = {}
    for cond_name, head_method_list in conditions:
        hooks = build_linear_hooks(head_method_list) if head_method_list else None

        times = []
        for run in range(n_runs):
            input_ids = prompt_tokens.clone()
            torch.mps.synchronize() if device.type == "mps" else None

            t0 = time.time()
            with torch.no_grad():
                for _ in range(gen_length):
                    if hooks:
                        logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)
                    else:
                        logits = model(input_ids)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)

            if hasattr(torch, "mps") and device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - t0
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        tokens_per_sec = gen_length / mean_time
        latency_ms = mean_time / gen_length * 1000

        results[cond_name] = {
            "mean_time": round(mean_time, 3),
            "std_time": round(std_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "latency_ms": round(latency_ms, 2),
        }

        print(
            f"  {cond_name}: {tokens_per_sec:.1f} tok/s, "
            f"latency={latency_ms:.2f} ms/tok "
            f"(±{std_time:.3f}s)"
        )

    # Theoretical kernel savings
    d_head = model.cfg.d_head
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    total_heads = n_layers * n_heads

    for seq_len in [128, 512, 1024]:
        flops_softmax = 2 * seq_len * seq_len * d_head + 5 * seq_len * seq_len + 2 * seq_len * seq_len * d_head
        flops_kernel = 6 * seq_len * d_head * d_head
        for n_lin in [30, 70]:
            flops_total_orig = total_heads * flops_softmax
            flops_total_lin = (total_heads - n_lin) * flops_softmax + n_lin * flops_kernel
            speedup = flops_total_orig / flops_total_lin
            results[f"theoretical_kernel_speedup_seq{seq_len}_n{n_lin}"] = round(speedup, 3)

    print(f"\n  Theoretical kernel speedup (seq=1024):")
    for n_lin in [30, 70]:
        key = f"theoretical_kernel_speedup_seq1024_n{n_lin}"
        print(f"    {n_lin} heads: {results[key]:.3f}x")

    # --- 可視化 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    cond_names = ["baseline", "relu_30", "relu_70"]
    tps = [results[c]["tokens_per_sec"] for c in cond_names]
    colors = ["C0", "C1", "C3"]
    bars = ax.bar(cond_names, tps, color=colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", fontsize=10,
        )
    ax.set_ylabel("Tokens/sec")
    ax.set_title(f"Exp 10P1: Generation Speed — {model_name}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = fig_dir / f"exp10p1_generation_speed_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# P2: Self-refinement (recursive CoT)
# -------------------------------------------------------------------


def run_p2(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_full: float,
    best_ranking: list[dict],
    fig_dir: Path,
) -> dict:
    """Test self-refinement with recursive generation."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"P2: Self-refinement — {model_name}")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    n_ctx = model.cfg.n_ctx

    # WikiText-2から文の前半を抽出してプロンプトとする
    # 各chunkの前半128トークンをプロンプト、後半をターゲットとして使う
    n_prompts = min(20, len(chunks))  # M1 Macでの時間制約
    prompt_len = 128
    gen_len = 64
    n_iterations = 3

    conditions = [
        ("baseline", []),
        ("relu_30", [(r["layer"], r["head"], "relu") for r in best_ranking[:30]]),
        ("relu_70", [(r["layer"], r["head"], "relu") for r in best_ranking[:70]]),
    ]

    results: dict = {}

    for cond_name, head_method_list in conditions:
        hooks = build_linear_hooks(head_method_list) if head_method_list else None
        iter_ppls = {it: [] for it in range(1, n_iterations + 1)}

        for p_idx in range(n_prompts):
            chunk = chunks[p_idx % len(chunks)]
            prompt = chunk[0, :prompt_len].unsqueeze(0).to(device)

            # 反復生成
            current_input = prompt
            for iteration in range(1, n_iterations + 1):
                with torch.no_grad():
                    generated_tokens = []
                    input_ids = current_input.clone()
                    for _ in range(gen_len):
                        if input_ids.shape[1] >= n_ctx:
                            break
                        if hooks:
                            logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)
                        else:
                            logits = model(input_ids)
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        generated_tokens.append(next_token)

                    if not generated_tokens:
                        break

                    # 生成テキストのPPL（ベースラインモデルで評価）
                    gen_sequence = input_ids[:, :min(input_ids.shape[1], n_ctx)]
                    gen_logits = model(gen_sequence)
                    shift_logits = gen_logits[:, prompt_len - 1:-1, :].contiguous()
                    shift_labels = gen_sequence[:, prompt_len:].contiguous()

                    if shift_labels.numel() > 0:
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction="mean",
                        )
                        gen_ppl = math.exp(loss.item())
                        iter_ppls[iteration].append(gen_ppl)

                    # 次の反復: 生成を入力に追加
                    current_input = input_ids[:, :min(input_ids.shape[1], n_ctx)]

            print(
                f"\r  {cond_name} [{p_idx+1}/{n_prompts}]",
                end="", flush=True,
            )

        print()

        cond_result = {}
        for it in range(1, n_iterations + 1):
            ppls = iter_ppls[it]
            if ppls:
                cond_result[f"iter{it}"] = {
                    "mean_ppl": round(float(np.mean(ppls)), 2),
                    "median_ppl": round(float(np.median(ppls)), 2),
                    "std_ppl": round(float(np.std(ppls)), 2),
                    "n_samples": len(ppls),
                }
                print(
                    f"  {cond_name} iter{it}: mean PPL = {np.mean(ppls):.2f} "
                    f"(median={np.median(ppls):.2f}, n={len(ppls)})"
                )

        results[cond_name] = cond_result

    # GPT-2の限界に関する注記
    results["note"] = (
        "GPT-2 small is not instruction-tuned. Self-refinement relies on "
        "continued generation rather than explicit instruction following. "
        "Results may not reflect true CoT recovery potential."
    )

    # --- 可視化 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, n_iterations + 1)
    width = 0.25
    for i, (cond_name, _) in enumerate(conditions):
        cond_result = results.get(cond_name, {})
        means = [
            cond_result.get(f"iter{it}", {}).get("mean_ppl", float("nan"))
            for it in range(1, n_iterations + 1)
        ]
        ax.bar(x + (i - 1) * width, means, width, label=cond_name, alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean PPL of generated text")
    ax.set_title(f"Exp 10P2: Self-Refinement — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Iter {i}" for i in range(1, n_iterations + 1)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = fig_dir / f"exp10p2_self_refinement_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# P3: Equal-FLOPs comparison (best-of-N)
# -------------------------------------------------------------------


def run_p3(
    model,
    model_name: str,
    chunks: list[torch.Tensor],
    ppl_full: float,
    best_ranking: list[dict],
    fig_dir: Path,
) -> dict:
    """Equal-FLOPs comparison using best-of-N strategy."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"P3: Equal-FLOPs comparison (best-of-N) — {model_name}")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    n_ctx = model.cfg.n_ctx
    d_head = model.cfg.d_head
    n_heads_total = model.cfg.n_layers * model.cfg.n_heads

    prompt_len = 128
    gen_len = 64
    n_prompts = min(20, len(chunks))
    Ns = [1, 2, 3, 5]  # best-of-N candidates

    # FLOPs計算: 1トークンの推論コスト（attention部分のみ、簡略化）
    # full softmax model: per-token FLOPs ≈ n_heads * (4*seq*d + 5*seq)
    # kernel化30: (n_heads-30) * (4*seq*d + 5*seq) + 30 * 6*d²
    # 簡略化: softmax FLOPsに対するkernel化の比率で追加トークン数を計算

    conditions = {
        "baseline": {"n_linear": 0, "hooks": None},
        "relu_30": {
            "n_linear": 30,
            "hooks": build_linear_hooks(
                [(r["layer"], r["head"], "relu") for r in best_ranking[:30]]
            ),
        },
        "relu_70": {
            "n_linear": 70,
            "hooks": build_linear_hooks(
                [(r["layer"], r["head"], "relu") for r in best_ranking[:70]]
            ),
        },
    }

    # Step 1: ベースラインの生成コスト基準
    # Attention FLOPs per token at seq_len ~192 (prompt+gen midpoint)
    avg_seq = prompt_len + gen_len // 2
    flops_softmax = n_heads_total * (4 * avg_seq * d_head + 5 * avg_seq)
    baseline_total_flops = gen_len * flops_softmax

    results: dict = {"flops_analysis": {}}

    for cond_name, cond_info in conditions.items():
        n_lin = cond_info["n_linear"]
        hooks = cond_info["hooks"]

        if n_lin > 0:
            # kernel化ヘッドのFLOPs
            flops_kernel = 6 * d_head * d_head
            flops_per_token = (n_heads_total - n_lin) * (4 * avg_seq * d_head + 5 * avg_seq) + n_lin * flops_kernel
            # 同一FLOPsで生成できる追加トークン数
            extra_tokens = int(baseline_total_flops / flops_per_token) - gen_len
            extra_tokens = max(0, extra_tokens)
            flops_ratio = flops_per_token / flops_softmax
        else:
            flops_per_token = flops_softmax
            extra_tokens = 0
            flops_ratio = 1.0

        results["flops_analysis"][cond_name] = {
            "n_linear": n_lin,
            "flops_per_token_ratio": round(flops_ratio, 4),
            "extra_tokens_at_equal_flops": extra_tokens,
        }

        print(
            f"  {cond_name}: FLOPs ratio={flops_ratio:.4f}, "
            f"extra tokens={extra_tokens}"
        )

    # Step 2: Best-of-N 生成
    print(f"\n  Running best-of-N generation...")

    bon_results: dict = {}
    for cond_name, cond_info in conditions.items():
        hooks = cond_info["hooks"]
        max_N = max(Ns)

        prompt_bon_scores: dict[int, list[float]] = {N: [] for N in Ns}

        for p_idx in range(n_prompts):
            chunk = chunks[p_idx % len(chunks)]
            prompt = chunk[0, :prompt_len].unsqueeze(0).to(device)

            # N個の候補を生成（temperature sampling）
            candidates_ppl = []
            for _ in range(max_N):
                with torch.no_grad():
                    input_ids = prompt.clone()
                    for _ in range(gen_len):
                        if input_ids.shape[1] >= n_ctx:
                            break
                        if hooks:
                            logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)
                        else:
                            logits = model(input_ids)

                        # Temperature sampling for diversity
                        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        input_ids = torch.cat([input_ids, next_token], dim=1)

                    # ベースラインモデルでPPL評価
                    gen_seq = input_ids[:, :min(input_ids.shape[1], n_ctx)]
                    with torch.no_grad():
                        eval_logits = model(gen_seq)
                    shift_logits = eval_logits[:, prompt_len - 1:-1, :].contiguous()
                    shift_labels = gen_seq[:, prompt_len:].contiguous()

                    if shift_labels.numel() > 0:
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction="mean",
                        )
                        candidates_ppl.append(math.exp(loss.item()))
                    else:
                        candidates_ppl.append(float("inf"))

            # Best-of-N: N個からPPL最小を選択
            for N in Ns:
                best_ppl = min(candidates_ppl[:N])
                prompt_bon_scores[N].append(best_ppl)

            print(
                f"\r  {cond_name} [{p_idx+1}/{n_prompts}]",
                end="", flush=True,
            )

        print()

        cond_bon = {}
        for N in Ns:
            scores = prompt_bon_scores[N]
            cond_bon[f"N{N}"] = {
                "mean_ppl": round(float(np.mean(scores)), 2),
                "median_ppl": round(float(np.median(scores)), 2),
            }
            print(
                f"  {cond_name} best-of-{N}: mean PPL = {np.mean(scores):.2f}"
            )
        bon_results[cond_name] = cond_bon

    results["best_of_n"] = bon_results

    # --- 可視化 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(Ns))
    width = 0.25
    for i, cond_name in enumerate(conditions):
        means = [
            bon_results[cond_name].get(f"N{N}", {}).get("mean_ppl", float("nan"))
            for N in Ns
        ]
        ax.bar(x + (i - 1) * width, means, width, label=cond_name, alpha=0.8)

    ax.set_xlabel("Best-of-N")
    ax.set_ylabel("Mean PPL of best generation")
    ax.set_title(f"Exp 10P3: Best-of-N — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = fig_dir / f"exp10p3_equal_flops_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# P4: Longer context window
# -------------------------------------------------------------------


def run_p4(
    model,
    model_name: str,
    tokenizer,
    best_ranking: list[dict],
    fig_dir: Path,
) -> dict:
    """Test PPL across different sequence lengths."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"P4: Context length vs PPL — {model_name}")
    print(f"{'='*60}")

    n_ctx = model.cfg.n_ctx  # 1024 for GPT-2

    # GPT-2のn_ctxは1024なので、それ以上は位置エンコーディングが無い
    # seq_lengths は n_ctx 以下に制限
    seq_lengths = [s for s in [128, 256, 512, 1024] if s <= n_ctx]

    conditions = [
        ("baseline", []),
        ("relu_30", [(r["layer"], r["head"], "relu") for r in best_ranking[:30]]),
        ("relu_70", [(r["layer"], r["head"], "relu") for r in best_ranking[:70]]),
    ]

    results: dict = {}

    for seq_len in seq_lengths:
        print(f"\n  seq_len={seq_len}:")
        # WikiText-2をこのseq_lenでchunk化
        chunks = load_wikitext2_validation(tokenizer, seq_len, max_tokens=50000)
        total_tokens = sum(c.numel() for c in chunks)
        print(f"    Chunks: {len(chunks)}, tokens: {total_tokens}")

        seq_results = {}
        for cond_name, head_method_list in conditions:
            hooks = build_linear_hooks(head_method_list) if head_method_list else None
            ppl = compute_perplexity(model, chunks, fwd_hooks=hooks)
            seq_results[cond_name] = round(ppl, 4)
            print(f"    {cond_name}: PPL = {ppl:.4f}")

        results[str(seq_len)] = seq_results

    # 比率計算
    print(f"\n  PPL ratios vs baseline:")
    for seq_len in seq_lengths:
        seq_key = str(seq_len)
        baseline_ppl = results[seq_key]["baseline"]
        for cond_name in ["relu_30", "relu_70"]:
            ratio = results[seq_key][cond_name] / baseline_ppl
            results.setdefault("ratios", {}).setdefault(seq_key, {})[cond_name] = round(ratio, 4)
            print(f"    seq={seq_len}, {cond_name}: {ratio:.4f}x")

    # --- 可視化 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 絶対PPL
    ax = axes[0]
    for cond_name, _, in conditions:
        ppls = [results[str(s)][cond_name] for s in seq_lengths]
        ax.plot(seq_lengths, ppls, "o-", label=cond_name, markersize=6)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Perplexity")
    ax.set_title("PPL vs Context Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右: 比率
    ax = axes[1]
    for cond_name in ["relu_30", "relu_70"]:
        ratios = [
            results.get("ratios", {}).get(str(s), {}).get(cond_name, 1.0)
            for s in seq_lengths
        ]
        ax.plot(seq_lengths, ratios, "o-", label=cond_name, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for t, c in [(1.1, "green"), (1.5, "orange")]:
        ax.axhline(y=t, color=c, linestyle=":", alpha=0.4)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("PPL / Baseline")
    ax.set_title("PPL Ratio vs Context Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Exp 10P4: Context Length — {model_name}")
    plt.tight_layout()
    path = fig_dir / f"exp10p4_context_length_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Validation + Phase 3 — CoT Recovery"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--parts", type=str, default="vp",
        help="Parts to run: v (validation), p (phase 3), or 'vp' for both",
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

    # --- Load previous results ---
    phase1a_path = data_dir / f"exp10a_{model_tag}.json"
    if not phase1a_path.exists():
        print(f"  ERROR: Phase 1 results not found: {phase1a_path}")
        return
    with open(phase1a_path) as f:
        phase1a = json.load(f)
    ppl_2k = phase1a["baseline_ppl"]
    zero_results = phase1a["results"]
    print(f"  Phase 1 baseline PPL (2K): {ppl_2k:.4f}")

    mean_path = data_dir / f"exp10e_mean_ablation_{model_tag}.json"
    if mean_path.exists():
        with open(mean_path) as f:
            mean_results = json.load(f)["results"]
        print(f"  Loaded mean-ablation results")
    else:
        mean_results = []
        print(f"  WARNING: Mean-ablation results not found")

    method_path = data_dir / f"exp10j_method_comparison_{model_tag}.json"
    if method_path.exists():
        with open(method_path) as f:
            method_result = json.load(f)
        best_ranking = method_result["best_ranking"]
        print(f"  Loaded linearization ranking ({len(best_ranking)} heads)")
    else:
        best_ranking = []
        print(f"  WARNING: Linearization ranking not found")

    cumul_path = data_dir / f"exp10k_cumulative_linear_{model_tag}.json"
    if cumul_path.exists():
        with open(cumul_path) as f:
            cumul_data = json.load(f)
        prev_cumul = cumul_data["cumulative_best"]
        print(f"  Loaded cumulative linearization data")
    else:
        prev_cumul = []
        print(f"  WARNING: Cumulative linearization data not found")

    # ===================================================================
    # Part V: Validation
    # ===================================================================
    if "v" in args.parts:
        print(f"\n{'='*60}")
        print(f"PART V: VALIDATION WITH FULL TOKENS")
        print(f"{'='*60}")

        # Full WikiText-2 (no max_tokens limit)
        print(f"\n  Loading full WikiText-2 validation set...")
        chunks_full = load_wikitext2_validation(
            model.tokenizer, model.cfg.n_ctx, max_tokens=999999
        )
        total_tokens = sum(c.numel() for c in chunks_full)
        print(f"  Loaded {len(chunks_full)} chunks, {total_tokens} tokens")

        # Compute head means on full dataset for V3
        print(f"  Computing head means on full dataset...")
        t0 = time.time()
        head_means = compute_head_means(model, chunks_full)
        print(f"  Done ({time.time() - t0:.1f}s)")

        # V1
        ppl_full = run_v1(model, args.model, chunks_full, data_dir)

        # V2
        v2_results = run_v2(
            model, args.model, chunks_full, ppl_full,
            zero_results, ppl_2k,
        )

        # V3
        v3_result = run_v3(
            model, args.model, chunks_full, ppl_full,
            head_means, zero_results, mean_results,
        )

        # V4
        if best_ranking:
            v4_results = run_v4(
                model, args.model, chunks_full, ppl_full,
                best_ranking, prev_cumul,
            )
        else:
            v4_results = []
            print("  Skipping V4: no linearization ranking available")

        # V5
        v5_result = run_v5(
            args.model, ppl_full, ppl_2k,
            v2_results, v3_result, v4_results, prev_cumul,
            fig_dir,
        )

        # Save validation results
        val_output = {
            "model": args.model,
            "ppl_2k": ppl_2k,
            "ppl_full": round(ppl_full, 4),
            "total_tokens": total_tokens,
            "n_chunks": len(chunks_full),
            "v2_top10": v2_results,
            "v3_l0h8": v3_result,
            "v4_cumulative": v4_results,
            "v5_comparison": v5_result,
        }
        val_path = data_dir / f"exp10v_validation_{model_tag}.json"
        with open(val_path, "w") as f:
            json.dump(val_output, f, indent=2)
        print(f"\n  Saved: {val_path}")

        # Validation summary
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline PPL: 2K={ppl_2k:.4f} → Full={ppl_full:.4f}")
        print(f"  Bias: {v5_result['bias_direction']}")
        print(f"  Conclusions maintained: {'YES' if v5_result['conclusions_maintained'] else 'NO'}")
        print(f"  Thresholds (full):")
        for t in ["1.1", "1.5", "2.0"]:
            n_2k = v5_result["thresholds_2k"].get(t, "?")
            n_full = v5_result["thresholds_full"].get(t, "?")
            print(f"    {t}x: 2K={n_2k} → Full={n_full}")

        # Phase 3 Go/No-Go 判定
        proceed_to_p3 = v5_result["conclusions_maintained"]
    else:
        proceed_to_p3 = True  # Validation skipped, assume OK
        ppl_full = ppl_2k  # fallback
        chunks_full = None

    # ===================================================================
    # Part P: Phase 3
    # ===================================================================
    if "p" in args.parts:
        if not proceed_to_p3:
            print(f"\n  Phase 3 SKIPPED: Validation showed significant discrepancies.")
            print(f"  Re-run Phase 2 with full tokens before proceeding.")
            return

        print(f"\n{'='*60}")
        print(f"PART P: PHASE 3 — CoT RECOVERY")
        print(f"{'='*60}")

        if not best_ranking:
            print("  ERROR: Linearization ranking required for Phase 3.")
            return

        # Load full chunks if not already loaded
        if chunks_full is None:
            print(f"\n  Loading full WikiText-2 validation set...")
            chunks_full = load_wikitext2_validation(
                model.tokenizer, model.cfg.n_ctx, max_tokens=999999
            )
            total_tokens = sum(c.numel() for c in chunks_full)
            print(f"  Loaded {len(chunks_full)} chunks, {total_tokens} tokens")

            ppl_full = compute_perplexity(model, chunks_full)
            print(f"  Baseline PPL (full): {ppl_full:.4f}")

        # P1: Generation speed
        p1_result = run_p1(
            model, args.model, chunks_full, best_ranking, fig_dir,
        )

        # P2: Self-refinement
        p2_result = run_p2(
            model, args.model, chunks_full, ppl_full, best_ranking, fig_dir,
        )

        # P3: Best-of-N
        p3_result = run_p3(
            model, args.model, chunks_full, ppl_full, best_ranking, fig_dir,
        )

        # P4: Context length
        p4_result = run_p4(
            model, args.model, model.tokenizer, best_ranking, fig_dir,
        )

        # Save Phase 3 results
        p_output = {
            "model": args.model,
            "ppl_baseline": round(ppl_full, 4),
            "p1_speed": p1_result,
            "p2_self_refinement": p2_result,
            "p3_best_of_n": p3_result,
            "p4_context_length": p4_result,
        }
        p_path = data_dir / f"exp10p_phase3_{model_tag}.json"
        with open(p_path, "w") as f:
            json.dump(p_output, f, indent=2)
        print(f"\n  Saved: {p_path}")

        # Phase 3 summary
        print(f"\n{'='*60}")
        print(f"PHASE 3 SUMMARY — {args.model}")
        print(f"{'='*60}")

        print(f"\n  P1: Generation speed")
        for cond in ["baseline", "relu_30", "relu_70"]:
            r = p1_result.get(cond, {})
            print(f"    {cond}: {r.get('tokens_per_sec', '?')} tok/s")

        print(f"\n  P2: Self-refinement")
        for cond in ["baseline", "relu_30", "relu_70"]:
            r = p2_result.get(cond, {})
            for it in ["iter1", "iter2", "iter3"]:
                ppl = r.get(it, {}).get("mean_ppl", "?")
                print(f"    {cond} {it}: PPL={ppl}")

        print(f"\n  P3: Best-of-N")
        bon = p3_result.get("best_of_n", {})
        for cond in ["baseline", "relu_30", "relu_70"]:
            for N in [1, 3, 5]:
                ppl = bon.get(cond, {}).get(f"N{N}", {}).get("mean_ppl", "?")
                print(f"    {cond} N={N}: PPL={ppl}")

        print(f"\n  P4: Context length")
        for seq_len in [128, 256, 512, 1024]:
            seq_key = str(seq_len)
            if seq_key in p4_result:
                for cond in ["baseline", "relu_30", "relu_70"]:
                    ppl = p4_result[seq_key].get(cond, "?")
                    print(f"    seq={seq_len} {cond}: PPL={ppl}")


if __name__ == "__main__":
    main()
