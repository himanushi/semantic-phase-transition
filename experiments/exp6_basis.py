"""Experiment 6: Universal Weight Subspace — PCA of W_QK, compression, and clustering.

6A. W_QK の PCA: 累積寄与率
6B. 基底数 K vs perplexity: 圧縮の実用性を直接評価
6C. ヘッドの k-means クラスタリング: 代表ヘッドで置換
6D. GPT-2 medium で同様に実行（6B の結果次第）

Weight injection strategy:
    W_QK_orig = W_Q.T @ W_K  (d_head, d_head)
    W_QK_recon = PCA reconstruction of W_QK_orig
    delta = W_QK_recon - W_QK_orig

    W_Q_new = W_Q_orig + W_K @ inv(W_K.T @ W_K) @ delta.T
    => W_Q_new.T @ W_K = W_QK_recon  (verified)
    => At K=all, delta=0, so W_Q_new = W_Q_orig exactly.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# 6A: W_QK の抽出と PCA
# -------------------------------------------------------------------

def extract_W_QK(model) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract W_QK = W_Q[h].T @ W_K[h] from all heads, flatten to 2D matrix.

    Returns:
        W_flat: (n_heads_total, d_head^2) array
        head_ids: list of (layer, head) tuples
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_ids = []
    W_flat_list = []

    for l in range(n_layers):
        W_Q = model.blocks[l].attn.W_Q.detach().cpu()  # (n_heads, d_model, d_head)
        W_K = model.blocks[l].attn.W_K.detach().cpu()  # (n_heads, d_model, d_head)

        for h in range(n_heads):
            # W_QK = W_Q[h].T @ W_K[h]: (d_head, d_model) @ (d_model, d_head) = (d_head, d_head)
            W_QK = W_Q[h].T @ W_K[h]
            W_flat_list.append(W_QK.numpy().flatten())
            head_ids.append((l, h))

    W_flat = np.array(W_flat_list)  # (n_heads_total, d_head^2)
    return W_flat, head_ids


def run_pca(W_flat: np.ndarray) -> dict:
    """Run PCA on flattened W_QK matrix (with centering)."""
    n_components = min(W_flat.shape[0], W_flat.shape[1])
    pca = PCA(n_components=n_components)
    coeffs = pca.fit_transform(W_flat)  # (n_heads_total, n_components)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    thresholds = {0.5: None, 0.8: None, 0.9: None, 0.95: None}
    for thresh in thresholds:
        idx = np.searchsorted(cumvar, thresh)
        if idx < len(cumvar):
            thresholds[thresh] = int(idx + 1)

    return {
        "pca": pca,
        "coeffs": coeffs,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumvar,
        "thresholds": thresholds,
        "mean": pca.mean_,
    }


def plot_cumvar(
    cumvar: np.ndarray,
    thresholds: dict,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot cumulative explained variance ratio."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    n = len(cumvar)
    ax.plot(range(1, n + 1), cumvar, "b-o", markersize=3)

    for thresh, k in thresholds.items():
        if k is not None:
            ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=k, color="gray", linestyle=":", alpha=0.5)
            ax.annotate(f"{thresh*100:.0f}% -> K={k}", xy=(k, thresh),
                        xytext=(k + n * 0.03, thresh - 0.03), fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Number of principal components K")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title(f"Exp 6A: PCA of W_QK — {model_name}")
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# PCA reconstruction
# -------------------------------------------------------------------

def reconstruct_W_QK(W_flat: np.ndarray, pca_result: dict, K: int) -> np.ndarray:
    """Reconstruct W_QK using top-K principal components.

    Returns (n_heads_total, d_head^2) array.
    """
    coeffs = pca_result["coeffs"]
    pca = pca_result["pca"]

    coeffs_trunc = np.zeros_like(coeffs)
    coeffs_trunc[:, :K] = coeffs[:, :K]

    return coeffs_trunc @ pca.components_ + pca.mean_


def build_W_QK_lookup(
    W_recon_flat: np.ndarray,
    head_ids: list[tuple[int, int]],
    d_head: int,
    device: str,
) -> dict[tuple[int, int], torch.Tensor]:
    """Build lookup dict: (layer, head) -> reconstructed W_QK tensor."""
    lookup = {}
    for idx, (l, h) in enumerate(head_ids):
        W_QK = W_recon_flat[idx].reshape(d_head, d_head)
        lookup[(l, h)] = torch.tensor(W_QK, dtype=torch.float32, device=device)
    return lookup


# -------------------------------------------------------------------
# Weight injection (additive correction — preserves null-space of W_K)
# -------------------------------------------------------------------

def inject_reconstructed_weights(
    model,
    W_QK_lookup: dict[tuple[int, int], torch.Tensor],
) -> dict[tuple[int, int], torch.Tensor]:
    """Inject reconstructed W_QK by modifying W_Q with additive correction.

    For each head:
        delta = W_QK_recon - W_QK_orig
        W_Q_new = W_Q_orig + W_K @ inv(W_K.T @ W_K) @ delta.T

    This ensures W_Q_new.T @ W_K = W_QK_recon, and when delta=0 (K=all),
    W_Q_new = W_Q_orig exactly.

    Returns backup dict: (layer, head) -> original W_Q[h] for restoration.
    """
    backup = {}
    device = next(model.parameters()).device

    # Precompute inv(W_K.T @ W_K) per layer (shared across heads in same layer)
    ktk_inv_cache: dict[int, torch.Tensor] = {}

    for (l, h), W_QK_recon in W_QK_lookup.items():
        W_Q = model.blocks[l].attn.W_Q  # (n_heads, d_model, d_head)
        W_K = model.blocks[l].attn.W_K  # (n_heads, d_model, d_head)

        backup[(l, h)] = W_Q.data[h].clone()

        W_Q_h = W_Q[h].detach()  # (d_model, d_head)
        W_K_h = W_K[h].detach()  # (d_model, d_head)

        # W_QK_orig = W_Q_h.T @ W_K_h
        W_QK_orig = W_Q_h.T @ W_K_h  # (d_head, d_head)

        delta = W_QK_recon.to(device) - W_QK_orig  # (d_head, d_head)

        # inv(W_K.T @ W_K) — per-head (different heads have different W_K)
        KtK = W_K_h.T @ W_K_h  # (d_head, d_head)
        KtK_inv = torch.linalg.inv(KtK)

        # W_Q_new = W_Q_orig + W_K @ inv(W_K.T @ W_K) @ delta.T
        correction = W_K_h @ KtK_inv @ delta.T  # (d_model, d_head)
        W_Q.data[h] = W_Q_h + correction

    return backup


def restore_weights(
    model,
    backup: dict[tuple[int, int], torch.Tensor],
) -> None:
    """Restore original W_Q weights from backup."""
    for (l, h), W_Q_orig in backup.items():
        model.blocks[l].attn.W_Q.data[h] = W_Q_orig


# -------------------------------------------------------------------
# Perplexity computation
# -------------------------------------------------------------------

def load_wikitext2(tokenizer, n_ctx: int, max_tokens: int = 200_000) -> list[torch.Tensor]:
    """Load WikiText-2 test set, chunk into sequences of length n_ctx."""
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


def compute_perplexity(model, chunks: list[torch.Tensor]) -> float:
    """Compute perplexity on chunked token sequences."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            logits = model(chunk)  # (1, seq_len, vocab_size)
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


# -------------------------------------------------------------------
# 6A runner
# -------------------------------------------------------------------

def run_exp6a(model, model_name: str, fig_dir: Path) -> tuple[np.ndarray, list, dict]:
    """Run experiment 6A: PCA of W_QK."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"6A: PCA of W_QK — {model_name}")
    print(f"{'='*60}")

    W_flat, head_ids = extract_W_QK(model)
    print(f"  W_QK matrix shape: {W_flat.shape}")
    print(f"  Total heads: {len(head_ids)}")

    pca_result = run_pca(W_flat)

    print(f"\n  Cumulative variance thresholds:")
    for thresh, k in pca_result["thresholds"].items():
        print(f"    {thresh*100:.0f}% -> K = {k}")

    evr = pca_result["explained_variance_ratio"]
    print(f"\n  Top 10 explained variance ratios:")
    for i in range(min(10, len(evr))):
        print(f"    PC{i+1}: {evr[i]:.4f} (cumul: {pca_result['cumulative_variance'][i]:.4f})")

    plot_cumvar(
        pca_result["cumulative_variance"],
        pca_result["thresholds"],
        model_name,
        output_path=fig_dir / f"exp6a_cumvar_{model_tag}.png",
    )

    return W_flat, head_ids, pca_result


# -------------------------------------------------------------------
# 6B runner
# -------------------------------------------------------------------

def run_exp6b(
    model,
    model_name: str,
    W_flat: np.ndarray,
    head_ids: list[tuple[int, int]],
    pca_result: dict,
    chunks: list[torch.Tensor],
    K_values: list[int],
    fig_dir: Path,
) -> dict:
    """Run experiment 6B: basis count K vs perplexity."""
    model_tag = model_name.replace("/", "_")
    device = str(next(model.parameters()).device)
    d_head = model.cfg.d_head
    n_total = len(head_ids)

    print(f"\n{'='*60}")
    print(f"6B: Basis count vs perplexity — {model_name}")
    print(f"{'='*60}")

    # Baseline
    print(f"\n  Computing baseline perplexity...")
    ppl_baseline = compute_perplexity(model, chunks)
    print(f"  Baseline perplexity: {ppl_baseline:.4f}")

    # K=全数 検証 (additive correction ensures delta=0 => exact recovery)
    K_all = min(n_total, W_flat.shape[1])
    print(f"\n  Verification: K={K_all} (all components)...")
    W_recon_all = reconstruct_W_QK(W_flat, pca_result, K_all)
    lookup_all = build_W_QK_lookup(W_recon_all, head_ids, d_head, device)
    backup = inject_reconstructed_weights(model, lookup_all)
    ppl_all = compute_perplexity(model, chunks)
    restore_weights(model, backup)
    diff = abs(ppl_all - ppl_baseline)
    print(f"  K=all perplexity: {ppl_all:.4f} (diff from baseline: {diff:.6f})")

    if diff > 0.01:
        print(f"  WARNING: K=all differs by {diff:.4f} > 0.01!")
    else:
        print(f"  OK: K=all matches baseline.")

    # 各 K で perplexity 測定
    results: dict = {"baseline": ppl_baseline, "K_all_verification": ppl_all, "K_values": {}}

    for K in K_values:
        if K > K_all:
            continue
        print(f"\n  K={K}...", end="", flush=True)

        W_recon = reconstruct_W_QK(W_flat, pca_result, K)
        lookup = build_W_QK_lookup(W_recon, head_ids, d_head, device)

        backup = inject_reconstructed_weights(model, lookup)
        ppl = compute_perplexity(model, chunks)
        restore_weights(model, backup)

        ratio = ppl / ppl_baseline
        print(f" ppl={ppl:.4f}, ratio={ratio:.4f}")

        results["K_values"][str(K)] = {"perplexity": ppl, "ratio": ratio}

    # Plot
    K_list = sorted(int(k) for k in results["K_values"])
    ppl_list = [results["K_values"][str(k)]["perplexity"] for k in K_list]
    ratio_list = [results["K_values"][str(k)]["ratio"] for k in K_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(K_list, ppl_list, "o-", color="C0", markersize=6)
    ax.axhline(y=ppl_baseline, color="gray", linestyle="--", alpha=0.7,
               label=f"Baseline={ppl_baseline:.2f}")
    ax.set_xlabel("Number of principal components K")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Exp 6B: K vs Perplexity — {model_name}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(K_list, ratio_list, "o-", color="C1", markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline ratio=1.0")
    ax.axhline(y=1.05, color="red", linestyle=":", alpha=0.5, label="1.05x")
    ax.axhline(y=1.10, color="orange", linestyle=":", alpha=0.5, label="1.10x")
    ax.set_xlabel("Number of principal components K")
    ax.set_ylabel("Perplexity / Baseline")
    ax.set_title(f"Exp 6B: Compression ratio — {model_name}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp6b_ppl_vs_K_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)

    return results


# -------------------------------------------------------------------
# 6C runner
# -------------------------------------------------------------------

def run_exp6c(
    model,
    model_name: str,
    W_flat: np.ndarray,
    head_ids: list[tuple[int, int]],
    pca_result: dict,
    chunks: list[torch.Tensor],
    k_values: list[int],
    fig_dir: Path,
    n_pca_dims: int = 20,
) -> dict:
    """Run experiment 6C: head clustering in PCA coefficient space."""
    model_tag = model_name.replace("/", "_")
    device = str(next(model.parameters()).device)
    d_head = model.cfg.d_head

    print(f"\n{'='*60}")
    print(f"6C: Head clustering — {model_name}")
    print(f"{'='*60}")

    coeffs = pca_result["coeffs"][:, :n_pca_dims]
    print(f"  Using top {n_pca_dims} PCA dimensions for clustering")

    ppl_baseline = compute_perplexity(model, chunks)
    print(f"  Baseline perplexity: {ppl_baseline:.4f}")

    results: dict = {"baseline": ppl_baseline, "k_values": {}}

    for k in k_values:
        if k >= len(head_ids):
            print(f"\n  k={k}: skip (>= total heads {len(head_ids)})")
            continue

        print(f"\n  k={k}...", end="", flush=True)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coeffs)

        # 各クラスタの代表ヘッド（重心に最近接）
        representatives = {}
        cluster_sizes = {}
        for c in range(k):
            cluster_indices = np.where(labels == c)[0]
            cluster_sizes[c] = len(cluster_indices)
            centroid = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(coeffs[cluster_indices] - centroid, axis=1)
            representatives[c] = cluster_indices[np.argmin(dists)]

        # 代表ヘッドの W_QK で同クラスタ内の全ヘッドを置換
        W_QK_lookup: dict[tuple[int, int], torch.Tensor] = {}
        for c in range(k):
            rep_idx = representatives[c]
            W_QK_rep = torch.tensor(
                W_flat[rep_idx].reshape(d_head, d_head),
                dtype=torch.float32, device=device,
            )
            for idx in np.where(labels == c)[0]:
                if idx != rep_idx:
                    W_QK_lookup[head_ids[idx]] = W_QK_rep

        backup = inject_reconstructed_weights(model, W_QK_lookup)
        ppl = compute_perplexity(model, chunks)
        restore_weights(model, backup)

        ratio = ppl / ppl_baseline
        print(f" ppl={ppl:.4f}, ratio={ratio:.4f}, replaced={len(W_QK_lookup)}/{len(head_ids)}")

        results["k_values"][str(k)] = {
            "perplexity": ppl,
            "ratio": ratio,
            "n_clusters": k,
            "n_replaced": len(W_QK_lookup),
            "cluster_sizes": {str(c): int(cluster_sizes[c]) for c in range(k)},
            "labels": labels.tolist(),
        }

    # Plot: cluster count vs perplexity
    k_list = sorted(int(k) for k in results["k_values"])
    ppl_list = [results["k_values"][str(k)]["perplexity"] for k in k_list]
    ratio_list = [results["k_values"][str(k)]["ratio"] for k in k_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(k_list, ppl_list, "o-", color="C0", markersize=6)
    ax.axhline(y=ppl_baseline, color="gray", linestyle="--", alpha=0.7,
               label=f"Baseline={ppl_baseline:.2f}")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Exp 6C: Cluster count vs Perplexity — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k_list, ratio_list, "o-", color="C1", markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline ratio=1.0")
    ax.axhline(y=1.05, color="red", linestyle=":", alpha=0.5, label="1.05x")
    ax.axhline(y=1.10, color="orange", linestyle=":", alpha=0.5, label="1.10x")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Perplexity / Baseline")
    ax.set_title(f"Exp 6C: Clustering compression — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp6c_cluster_ppl_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)

    # PCA 2D visualization
    if k_list:
        last_k = k_list[-1]
        labels_viz = np.array(results["k_values"][str(last_k)]["labels"])

        pca_2d = PCA(n_components=2)
        coords_2d = pca_2d.fit_transform(coeffs)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        layers = np.array([lid for lid, _ in head_ids])
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        c=layers, cmap="viridis", s=40, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Layer index")
        ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"Heads colored by layer — {model_name}")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        c=labels_viz, cmap="tab20", s=40, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Cluster ID")
        ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"Heads colored by cluster (k={last_k}) — {model_name}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = fig_dir / f"exp6c_cluster_viz_{model_tag}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 6: Universal Weight Subspace — PCA, compression, clustering"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--parts", type=str, default="ab",
                        help="Parts to run: a, b, c, or combinations like 'ab', 'abc'")
    parser.add_argument("--max-tokens", type=int, default=200_000)
    args = parser.parse_args()

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
    print(f"  n_ctx: {model.cfg.n_ctx}, Device: {args.device}")

    n_total = model.cfg.n_layers * model.cfg.n_heads

    if args.model == "gpt2":
        K_values = [1, 2, 3, 5, 8, 10, 16, 32, 64, 80, 100, 110, 120, 130, 140, 144]
        k_cluster_values = [4, 8, 12, 16, 32, 64]
    elif args.model == "gpt2-medium":
        K_values = [1, 2, 3, 5, 8, 10, 16, 32, 64, 128, 384]
        k_cluster_values = [4, 8, 12, 16, 32, 64]
    else:
        K_values = [1, 2, 3, 5, 8, 10, 16, 32, 64, n_total]
        k_cluster_values = [4, 8, 12, 16, 32, 64]

    # --- 6A ---
    W_flat, head_ids, pca_result = run_exp6a(model, args.model, fig_dir)

    # Load WikiText-2 if needed
    chunks = None
    if "b" in args.parts or "c" in args.parts:
        print(f"\n  Loading WikiText-2...")
        chunks = load_wikitext2(model.tokenizer, model.cfg.n_ctx, max_tokens=args.max_tokens)
        print(f"  Loaded {len(chunks)} chunks of {model.cfg.n_ctx} tokens each")

    # --- 6B ---
    results_6b = None
    if "b" in args.parts and chunks is not None:
        results_6b = run_exp6b(
            model, args.model, W_flat, head_ids, pca_result, chunks, K_values, fig_dir
        )

    # --- 6C ---
    results_6c = None
    if "c" in args.parts and chunks is not None:
        results_6c = run_exp6c(
            model, args.model, W_flat, head_ids, pca_result, chunks, k_cluster_values, fig_dir
        )

    # --- Save ---
    output: dict = {
        "model": args.model,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_head": model.cfg.d_head,
        "n_heads_total": n_total,
        "exp6a": {
            "W_flat_shape": list(W_flat.shape),
            "explained_variance_ratio": pca_result["explained_variance_ratio"].tolist(),
            "cumulative_variance": pca_result["cumulative_variance"].tolist(),
            "thresholds": {str(k): v for k, v in pca_result["thresholds"].items()},
        },
    }

    if results_6b is not None:
        output["exp6b"] = results_6b

    if results_6c is not None:
        results_6c_clean = dict(results_6c)
        results_6c_clean["k_values"] = {
            k: {key: val for key, val in v.items() if key != "labels"}
            for k, v in results_6c["k_values"].items()
        }
        output["exp6c"] = results_6c_clean

    json_path = data_dir / f"exp6_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved: {json_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")

    print(f"\n  6A: PCA of W_QK ({W_flat.shape[0]} heads, {W_flat.shape[1]} dims)")
    for thresh, k in pca_result["thresholds"].items():
        print(f"    {thresh*100:.0f}% variance -> K={k}")

    if results_6b:
        print(f"\n  6B: Basis count vs perplexity (baseline={results_6b['baseline']:.4f})")
        print(f"    {'K':>5} {'PPL':>10} {'Ratio':>8}")
        print(f"    {'-'*25}")
        for K in sorted(int(k) for k in results_6b["K_values"]):
            r = results_6b["K_values"][str(K)]
            marker = " <--" if r["ratio"] <= 1.10 else ""
            print(f"    {K:>5} {r['perplexity']:>10.4f} {r['ratio']:>8.4f}{marker}")

    if results_6c:
        print(f"\n  6C: Head clustering (baseline={results_6c['baseline']:.4f})")
        print(f"    {'k':>5} {'PPL':>10} {'Ratio':>8} {'Replaced':>10}")
        print(f"    {'-'*35}")
        for k in sorted(int(k) for k in results_6c["k_values"]):
            r = results_6c["k_values"][str(k)]
            print(f"    {k:>5} {r['perplexity']:>10.4f} {r['ratio']:>8.4f} {r['n_replaced']:>10}")


if __name__ == "__main__":
    main()
