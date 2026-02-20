"""Experiment 7: Residual Stream Δφ PCA — dynamic write patterns across layers.

7A. Collect Δφ(l) = resid_post(l) - resid_pre(l) from WikiText-2
7B. Per-layer PCA of Δφ(l): cumulative variance analysis
7C. Cross-layer shared basis: can layers share PCA directions?
7D. Low-rank approximation inference: project Δφ onto top-K basis, measure perplexity
7E. Attention vs FFN separation: which sub-component is more compressible?

Motivation: g(l/L) universality (exp3) suggests each layer performs similar write operations.
Unlike exp6 (static W_QK PCA failed), Δφ is dynamic and its PCA large-variance
directions may capture functionally important write patterns.
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# Perplexity (shared with exp6)
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


def compute_perplexity(model, chunks: list[torch.Tensor], hooks=None) -> float:
    """Compute perplexity on chunked token sequences, optionally with forward hooks."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            if hooks:
                logits = model.run_with_hooks(chunk, fwd_hooks=hooks)
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


# -------------------------------------------------------------------
# 7A: Collect Δφ(l) from WikiText-2
# -------------------------------------------------------------------

def collect_delta_phi(
    model,
    chunks: list[torch.Tensor],
    max_samples: int = 1000,
    last_token_only: bool = False,
) -> dict[int, np.ndarray]:
    """Collect Δφ(l) = resid_post(l) - resid_pre(l) for each layer.

    Args:
        model: HookedTransformer
        chunks: tokenized WikiText-2 chunks
        max_samples: max total token positions to collect
        last_token_only: if True, only collect last token position per chunk

    Returns:
        dict: layer_index -> (n_samples, d_model) numpy array
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    device = next(model.parameters()).device

    # Collect into lists, then stack
    delta_lists: dict[int, list] = {l: [] for l in range(n_layers)}
    total_collected = 0

    print(f"  Collecting Δφ from {len(chunks)} chunks (max {max_samples} samples)...")

    model.eval()
    with torch.no_grad():
        for ci, chunk in enumerate(chunks):
            if total_collected >= max_samples:
                break
            chunk = chunk.to(device)
            _, cache = model.run_with_cache(chunk)

            seq_len = chunk.shape[1]

            if last_token_only:
                positions = [seq_len - 1]
            else:
                positions = list(range(seq_len))

            for pos in positions:
                if total_collected >= max_samples:
                    break
                for l in range(n_layers):
                    pre = cache["resid_pre", l][0, pos].cpu().numpy()
                    post = cache["resid_post", l][0, pos].cpu().numpy()
                    delta_lists[l].append(post - pre)
                total_collected += 1

            if (ci + 1) % 20 == 0:
                print(f"    chunk {ci+1}/{len(chunks)}, collected {total_collected} samples")

    delta_phi = {}
    for l in range(n_layers):
        delta_phi[l] = np.array(delta_lists[l], dtype=np.float32)

    print(f"  Collected {total_collected} samples per layer, shape: {delta_phi[0].shape}")
    return delta_phi


def collect_delta_phi_components(
    model,
    chunks: list[torch.Tensor],
    max_samples: int = 1000,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Collect Δφ_attn and Δφ_ffn separately.

    Returns:
        (delta_attn, delta_ffn): each is layer_index -> (n_samples, d_model)
    """
    n_layers = model.cfg.n_layers
    device = next(model.parameters()).device

    attn_lists: dict[int, list] = {l: [] for l in range(n_layers)}
    ffn_lists: dict[int, list] = {l: [] for l in range(n_layers)}
    total_collected = 0

    print(f"  Collecting Δφ_attn and Δφ_ffn from {len(chunks)} chunks...")

    model.eval()
    with torch.no_grad():
        for ci, chunk in enumerate(chunks):
            if total_collected >= max_samples:
                break
            chunk = chunk.to(device)
            _, cache = model.run_with_cache(chunk)

            seq_len = chunk.shape[1]
            for pos in range(seq_len):
                if total_collected >= max_samples:
                    break
                for l in range(n_layers):
                    attn_out = cache["attn_out", l][0, pos].cpu().numpy()
                    mlp_out = cache["mlp_out", l][0, pos].cpu().numpy()
                    attn_lists[l].append(attn_out)
                    ffn_lists[l].append(mlp_out)
                total_collected += 1

            if (ci + 1) % 20 == 0:
                print(f"    chunk {ci+1}/{len(chunks)}, collected {total_collected} samples")

    delta_attn = {l: np.array(attn_lists[l], dtype=np.float32) for l in range(n_layers)}
    delta_ffn = {l: np.array(ffn_lists[l], dtype=np.float32) for l in range(n_layers)}

    print(f"  Collected {total_collected} samples per layer")
    return delta_attn, delta_ffn


# -------------------------------------------------------------------
# 7B: Per-layer PCA
# -------------------------------------------------------------------

def run_per_layer_pca(
    delta_phi: dict[int, np.ndarray],
    max_components: int | None = None,
) -> dict[int, dict]:
    """Run PCA on each layer's Δφ independently.

    Returns:
        dict: layer -> {pca, cumvar, thresholds, n_effective}
    """
    results = {}
    thresholds_targets = [0.5, 0.8, 0.9, 0.95]

    for l in sorted(delta_phi.keys()):
        data = delta_phi[l]
        n_comp = min(data.shape[0], data.shape[1])
        if max_components:
            n_comp = min(n_comp, max_components)

        pca = PCA(n_components=n_comp)
        pca.fit(data)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        thresholds = {}
        for t in thresholds_targets:
            idx = np.searchsorted(cumvar, t)
            thresholds[t] = int(idx + 1) if idx < len(cumvar) else n_comp

        results[l] = {
            "pca": pca,
            "cumvar": cumvar,
            "thresholds": thresholds,
            "n_components": n_comp,
        }

    return results


def plot_cumvar_per_layer(
    pca_results: dict[int, dict],
    model_name: str,
    output_path: Path | None = None,
    title_suffix: str = "",
) -> None:
    """Plot cumulative variance for all layers overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cmap = plt.cm.viridis
    n_layers = len(pca_results)

    # Left: full range
    ax = axes[0]
    for l in sorted(pca_results.keys()):
        cumvar = pca_results[l]["cumvar"]
        color = cmap(l / (n_layers - 1))
        ax.plot(range(1, len(cumvar) + 1), cumvar, color=color, alpha=0.7, linewidth=1)

    for t in [0.5, 0.8, 0.9, 0.95]:
        ax.axhline(y=t, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Number of principal components K")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Exp 7B: Δφ PCA per layer{title_suffix} — {model_name}")
    ax.set_xlim(0, min(200, max(len(r["cumvar"]) for r in pca_results.values())))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers - 1))
    plt.colorbar(sm, ax=ax, label="Layer index")

    # Right: zoomed to K=0..64
    ax = axes[1]
    for l in sorted(pca_results.keys()):
        cumvar = pca_results[l]["cumvar"]
        color = cmap(l / (n_layers - 1))
        ax.plot(range(1, min(65, len(cumvar) + 1)), cumvar[:64], color=color, alpha=0.7, linewidth=1)

    for t in [0.5, 0.8, 0.9]:
        ax.axhline(y=t, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Number of principal components K")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Zoomed (K≤64){title_suffix}")
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# 7C: Cross-layer shared basis
# -------------------------------------------------------------------

def run_shared_pca(
    delta_phi: dict[int, np.ndarray],
    max_components: int | None = None,
) -> dict:
    """Run PCA on all layers' Δφ concatenated."""
    all_data = np.concatenate([delta_phi[l] for l in sorted(delta_phi.keys())], axis=0)
    print(f"  Shared PCA input shape: {all_data.shape}")

    n_comp = min(all_data.shape[0], all_data.shape[1])
    if max_components:
        n_comp = min(n_comp, max_components)

    pca = PCA(n_components=n_comp)
    pca.fit(all_data)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    thresholds = {}
    for t in [0.5, 0.8, 0.9, 0.95]:
        idx = np.searchsorted(cumvar, t)
        thresholds[t] = int(idx + 1) if idx < len(cumvar) else n_comp

    return {
        "pca": pca,
        "cumvar": cumvar,
        "thresholds": thresholds,
        "n_components": n_comp,
    }


def evaluate_shared_basis(
    delta_phi: dict[int, np.ndarray],
    shared_pca: dict,
    K_values: list[int],
) -> dict[int, dict[int, dict]]:
    """Evaluate reconstruction quality using shared basis at various K.

    Returns:
        dict: K -> layer -> {cos_sim, l2_error, relative_l2}
    """
    pca = shared_pca["pca"]
    components = pca.components_  # (n_comp, d_model)
    mean = pca.mean_  # (d_model,)

    results = {}
    for K in K_values:
        U_K = components[:K]  # (K, d_model)
        results[K] = {}

        for l in sorted(delta_phi.keys()):
            data = delta_phi[l]  # (n_samples, d_model)
            centered = data - mean

            # Project and reconstruct
            coeffs = centered @ U_K.T  # (n_samples, K)
            recon = coeffs @ U_K + mean  # (n_samples, d_model)

            # Cosine similarity (per sample, then average)
            norms_orig = np.linalg.norm(data, axis=1, keepdims=True)
            norms_recon = np.linalg.norm(recon, axis=1, keepdims=True)
            # Avoid division by zero
            norms_orig = np.maximum(norms_orig, 1e-10)
            norms_recon = np.maximum(norms_recon, 1e-10)
            cos_sims = np.sum(data * recon, axis=1) / (norms_orig.squeeze() * norms_recon.squeeze())
            cos_sim = float(np.mean(cos_sims))

            # L2 error
            l2_error = float(np.mean(np.linalg.norm(data - recon, axis=1)))
            orig_norm = float(np.mean(np.linalg.norm(data, axis=1)))
            relative_l2 = l2_error / max(orig_norm, 1e-10)

            results[K][l] = {
                "cos_sim": cos_sim,
                "l2_error": l2_error,
                "relative_l2": relative_l2,
            }

    return results


def plot_shared_vs_individual(
    per_layer_results: dict[int, dict],
    shared_result: dict,
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot comparison of individual vs shared PCA cumulative variance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_layers = len(per_layer_results)

    # Left: threshold K needed for 90% variance
    ax = axes[0]
    layers = sorted(per_layer_results.keys())
    k90_individual = [per_layer_results[l]["thresholds"][0.9] for l in layers]
    k90_shared = shared_result["thresholds"][0.9]

    ax.bar(layers, k90_individual, color="C0", alpha=0.7, label="Individual PCA")
    ax.axhline(y=k90_shared, color="C1", linestyle="--", linewidth=2, label=f"Shared PCA (K={k90_shared})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("K for 90% variance")
    ax.set_title(f"Exp 7C: K for 90% variance — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: cumvar comparison at layer 0, mid, last
    ax = axes[1]
    shared_cumvar = shared_result["cumvar"]
    ax.plot(range(1, len(shared_cumvar) + 1), shared_cumvar, "k-", linewidth=2, label="Shared", alpha=0.8)

    for l in [0, n_layers // 2, n_layers - 1]:
        cumvar = per_layer_results[l]["cumvar"]
        ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=1, alpha=0.7, label=f"Layer {l}")

    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of principal components K")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Shared vs Individual PCA — {model_name}")
    ax.set_xlim(0, min(200, len(shared_cumvar)))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# 7D: Low-rank approximation inference
# -------------------------------------------------------------------

def make_lowrank_hooks(
    model,
    basis: dict[int, np.ndarray],
    means: dict[int, np.ndarray],
) -> list:
    """Create forward hooks that project each layer's Δφ onto low-rank basis.

    Args:
        model: HookedTransformer
        basis: layer -> (K, d_model) basis vectors (from PCA)
        means: layer -> (d_model,) mean vector (from PCA)

    Returns:
        list of (hook_name, hook_fn) tuples for model.run_with_hooks
    """
    device = next(model.parameters()).device
    hooks = []

    for l in sorted(basis.keys()):
        U = torch.tensor(basis[l], dtype=torch.float32, device=device)  # (K, d_model)
        mu = torch.tensor(means[l], dtype=torch.float32, device=device)  # (d_model,)

        def hook_fn(value, hook, U=U, mu=mu, layer=l):
            # value shape: (batch, seq_len, d_model) = resid_post
            # resid_pre is available via hook
            # Δφ = value - resid_pre
            # We need resid_pre to reconstruct: resid_post = resid_pre + Δφ_approx
            # Unfortunately transformer-lens resid_post hook doesn't give us resid_pre directly.
            # Instead, use hook on "resid_post" and the model's cached resid_pre.
            # But simpler: hook on resid_post, compute delta from resid_pre hook.
            #
            # Actually, the approach is:
            # Δφ = value - resid_pre  (where resid_pre for this layer)
            # Δφ_approx = U.T @ U @ (Δφ - mu) + mu  (project centered data)
            # new_value = resid_pre + Δφ_approx
            #
            # But we don't have resid_pre in this hook context.
            # Alternative: use a pair of hooks - store resid_pre, then modify resid_post.
            pass

        # Use a class to store state between hooks
        class LayerProjector:
            def __init__(self, U, mu):
                self.U = U
                self.mu = mu
                self.resid_pre = None

            def pre_hook(self, value, hook):
                self.resid_pre = value.clone()
                return value

            def post_hook(self, value, hook):
                if self.resid_pre is None:
                    return value
                delta = value - self.resid_pre
                # Project: centered -> project -> uncenter
                centered = delta - self.mu
                projected = centered @ self.U.T @ self.U  # (batch, seq, d_model)
                delta_approx = projected + self.mu
                return self.resid_pre + delta_approx

        proj = LayerProjector(U, mu)
        hooks.append((f"blocks.{l}.hook_resid_pre", proj.pre_hook))
        hooks.append((f"blocks.{l}.hook_resid_post", proj.post_hook))

    return hooks


def make_lowrank_hooks_shared(
    model,
    U_shared: np.ndarray,
    mu_shared: np.ndarray,
    n_layers: int,
) -> list:
    """Create hooks using a single shared basis for all layers."""
    device = next(model.parameters()).device
    U = torch.tensor(U_shared, dtype=torch.float32, device=device)
    mu = torch.tensor(mu_shared, dtype=torch.float32, device=device)

    hooks = []
    for l in range(n_layers):
        class LayerProjector:
            def __init__(self, U, mu):
                self.U = U
                self.mu = mu
                self.resid_pre = None

            def pre_hook(self, value, hook):
                self.resid_pre = value.clone()
                return value

            def post_hook(self, value, hook):
                if self.resid_pre is None:
                    return value
                delta = value - self.resid_pre
                centered = delta - self.mu
                projected = centered @ self.U.T @ self.U
                delta_approx = projected + self.mu
                return self.resid_pre + delta_approx

        proj = LayerProjector(U, mu)
        hooks.append((f"blocks.{l}.hook_resid_pre", proj.pre_hook))
        hooks.append((f"blocks.{l}.hook_resid_post", proj.post_hook))

    return hooks


# -------------------------------------------------------------------
# 7E: Attention vs FFN cumvar comparison
# -------------------------------------------------------------------

def plot_attn_vs_ffn_cumvar(
    attn_pca: dict[int, dict],
    ffn_pca: dict[int, dict],
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot Attention vs FFN cumulative variance comparison."""
    n_layers = len(attn_pca)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: K for 90% variance per layer
    ax = axes[0]
    layers = sorted(attn_pca.keys())
    k90_attn = [attn_pca[l]["thresholds"][0.9] for l in layers]
    k90_ffn = [ffn_pca[l]["thresholds"][0.9] for l in layers]

    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, k90_attn, w, label="Attention", color="C0", alpha=0.7)
    ax.bar(x + w/2, k90_ffn, w, label="FFN", color="C1", alpha=0.7)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("K for 90% variance")
    ax.set_title(f"Exp 7E: Rank for 90% variance — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: Attention cumvar (all layers)
    ax = axes[1]
    cmap = plt.cm.viridis
    for l in layers:
        cumvar = attn_pca[l]["cumvar"]
        color = cmap(l / (n_layers - 1))
        ax.plot(range(1, min(129, len(cumvar) + 1)), cumvar[:128], color=color, alpha=0.7, linewidth=1)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("K")
    ax.set_ylabel("Cumulative variance")
    ax.set_title("Attention Δφ_attn")
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: FFN cumvar (all layers)
    ax = axes[2]
    for l in layers:
        cumvar = ffn_pca[l]["cumvar"]
        color = cmap(l / (n_layers - 1))
        ax.plot(range(1, min(129, len(cumvar) + 1)), cumvar[:128], color=color, alpha=0.7, linewidth=1)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("K")
    ax.set_ylabel("Cumulative variance")
    ax.set_title("FFN Δφ_ffn")
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# -------------------------------------------------------------------
# Runners
# -------------------------------------------------------------------

def run_exp7a(model, chunks, max_samples: int) -> dict[int, np.ndarray]:
    """Run 7A: collect Δφ."""
    print(f"\n{'='*60}")
    print(f"7A: Collecting Δφ(l) from WikiText-2")
    print(f"{'='*60}")

    delta_phi = collect_delta_phi(model, chunks, max_samples=max_samples)

    # Basic stats
    for l in sorted(delta_phi.keys()):
        d = delta_phi[l]
        print(f"  Layer {l:2d}: mean_norm={np.mean(np.linalg.norm(d, axis=1)):.4f}, "
              f"std_norm={np.std(np.linalg.norm(d, axis=1)):.4f}")

    return delta_phi


def run_exp7b(delta_phi, model_name, fig_dir) -> dict[int, dict]:
    """Run 7B: per-layer PCA."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"7B: Per-layer PCA of Δφ — {model_name}")
    print(f"{'='*60}")

    pca_results = run_per_layer_pca(delta_phi)

    print(f"\n  Thresholds (K needed for X% variance):")
    print(f"  {'Layer':>5} {'50%':>6} {'80%':>6} {'90%':>6} {'95%':>6}")
    print(f"  {'-'*30}")
    for l in sorted(pca_results.keys()):
        t = pca_results[l]["thresholds"]
        print(f"  {l:>5} {t[0.5]:>6} {t[0.8]:>6} {t[0.9]:>6} {t[0.95]:>6}")

    plot_cumvar_per_layer(
        pca_results,
        model_name,
        output_path=fig_dir / f"exp7b_cumvar_per_layer_{model_tag}.png",
    )

    return pca_results


def run_exp7c(delta_phi, per_layer_pca, model_name, fig_dir) -> dict:
    """Run 7C: shared basis."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"7C: Cross-layer shared basis — {model_name}")
    print(f"{'='*60}")

    shared_result = run_shared_pca(delta_phi)

    print(f"\n  Shared PCA thresholds:")
    for t, k in shared_result["thresholds"].items():
        print(f"    {t*100:.0f}% -> K={k}")

    # Evaluate reconstruction
    K_eval = [8, 16, 32, 64, 128, 256]
    shared_eval = evaluate_shared_basis(delta_phi, shared_result, K_eval)

    print(f"\n  Shared basis reconstruction quality (mean cos_sim across layers):")
    print(f"  {'K':>5} {'cos_sim':>10} {'rel_L2':>10}")
    print(f"  {'-'*27}")
    for K in K_eval:
        if K not in shared_eval:
            continue
        cos_sims = [shared_eval[K][l]["cos_sim"] for l in sorted(shared_eval[K].keys())]
        rel_l2s = [shared_eval[K][l]["relative_l2"] for l in sorted(shared_eval[K].keys())]
        print(f"  {K:>5} {np.mean(cos_sims):>10.4f} {np.mean(rel_l2s):>10.4f}")

    plot_shared_vs_individual(
        per_layer_pca,
        shared_result,
        model_name,
        output_path=fig_dir / f"exp7c_shared_vs_individual_{model_tag}.png",
    )

    return {"shared_pca": shared_result, "eval": shared_eval}


def run_exp7d(model, model_name, per_layer_pca, shared_result, chunks, fig_dir) -> dict:
    """Run 7D: low-rank approximation perplexity."""
    model_tag = model_name.replace("/", "_")
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"\n{'='*60}")
    print(f"7D: Low-rank Δφ approximation — {model_name}")
    print(f"{'='*60}")

    # Baseline
    print(f"  Computing baseline perplexity...")
    ppl_baseline = compute_perplexity(model, chunks)
    print(f"  Baseline: {ppl_baseline:.4f}")

    K_values = [8, 16, 32, 64, 128, 256, 512, d_model]

    results = {"baseline": ppl_baseline, "individual": {}, "shared": {}}

    # Individual basis
    print(f"\n  --- Individual basis ---")
    for K in K_values:
        print(f"  K={K}...", end="", flush=True)

        basis = {}
        means = {}
        for l in range(n_layers):
            pca = per_layer_pca[l]["pca"]
            n_avail = min(K, pca.components_.shape[0])
            basis[l] = pca.components_[:n_avail]  # (K, d_model)
            means[l] = pca.mean_

        hooks = make_lowrank_hooks(model, basis, means)
        ppl = compute_perplexity(model, chunks, hooks=hooks)
        ratio = ppl / ppl_baseline
        print(f" ppl={ppl:.4f}, ratio={ratio:.4f}")

        results["individual"][str(K)] = {"perplexity": ppl, "ratio": ratio}

    # Shared basis
    print(f"\n  --- Shared basis ---")
    shared_pca = shared_result["shared_pca"]
    for K in K_values:
        print(f"  K={K}...", end="", flush=True)

        n_avail = min(K, shared_pca["pca"].components_.shape[0])
        U = shared_pca["pca"].components_[:n_avail]
        mu = shared_pca["pca"].mean_

        hooks = make_lowrank_hooks_shared(model, U, mu, n_layers)
        ppl = compute_perplexity(model, chunks, hooks=hooks)
        ratio = ppl / ppl_baseline
        print(f" ppl={ppl:.4f}, ratio={ratio:.4f}")

        results["shared"][str(K)] = {"perplexity": ppl, "ratio": ratio}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    K_ind = sorted(int(k) for k in results["individual"])
    K_sh = sorted(int(k) for k in results["shared"])
    ppl_ind = [results["individual"][str(k)]["perplexity"] for k in K_ind]
    ppl_sh = [results["shared"][str(k)]["perplexity"] for k in K_sh]
    ratio_ind = [results["individual"][str(k)]["ratio"] for k in K_ind]
    ratio_sh = [results["shared"][str(k)]["ratio"] for k in K_sh]

    ax = axes[0]
    ax.plot(K_ind, ppl_ind, "o-", color="C0", label="Individual basis")
    ax.plot(K_sh, ppl_sh, "s-", color="C1", label="Shared basis")
    ax.axhline(y=ppl_baseline, color="gray", linestyle="--", alpha=0.7, label=f"Baseline={ppl_baseline:.2f}")
    ax.set_xlabel("Number of basis vectors K")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Exp 7D: Δφ low-rank approx — {model_name}")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(K_ind, ratio_ind, "o-", color="C0", label="Individual basis")
    ax.plot(K_sh, ratio_sh, "s-", color="C1", label="Shared basis")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(y=1.05, color="red", linestyle=":", alpha=0.5, label="1.05x")
    ax.axhline(y=1.10, color="orange", linestyle=":", alpha=0.5, label="1.10x")
    ax.set_xlabel("Number of basis vectors K")
    ax.set_ylabel("Perplexity / Baseline")
    ax.set_title(f"Compression ratio — {model_name}")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / f"exp7d_ppl_vs_K_{model_tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)

    return results


def run_exp7e(model, chunks, model_name, fig_dir, max_samples: int) -> dict:
    """Run 7E: Attention vs FFN separation."""
    model_tag = model_name.replace("/", "_")

    print(f"\n{'='*60}")
    print(f"7E: Attention vs FFN Δφ — {model_name}")
    print(f"{'='*60}")

    delta_attn, delta_ffn = collect_delta_phi_components(model, chunks, max_samples=max_samples)

    attn_pca = run_per_layer_pca(delta_attn)
    ffn_pca = run_per_layer_pca(delta_ffn)

    print(f"\n  K for 90% variance:")
    print(f"  {'Layer':>5} {'Attn':>6} {'FFN':>6}")
    print(f"  {'-'*20}")
    for l in sorted(attn_pca.keys()):
        k_a = attn_pca[l]["thresholds"][0.9]
        k_f = ffn_pca[l]["thresholds"][0.9]
        print(f"  {l:>5} {k_a:>6} {k_f:>6}")

    plot_attn_vs_ffn_cumvar(
        attn_pca,
        ffn_pca,
        model_name,
        output_path=fig_dir / f"exp7e_attn_vs_ffn_cumvar_{model_tag}.png",
    )

    return {
        "attn_thresholds": {l: attn_pca[l]["thresholds"] for l in attn_pca},
        "ffn_thresholds": {l: ffn_pca[l]["thresholds"] for l in ffn_pca},
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 7: Residual Stream Δφ PCA"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--parts", type=str, default="abcde",
                        help="Parts to run: a, b, c, d, e or combinations")
    parser.add_argument("--max-tokens", type=int, default=200_000)
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max token positions to collect for Δφ")
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
    print(f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}")
    print(f"  d_model: {model.cfg.d_model}, d_head: {model.cfg.d_head}")
    print(f"  n_ctx: {model.cfg.n_ctx}, Device: {args.device}")

    # Load WikiText-2
    print(f"\nLoading WikiText-2...")
    chunks = load_wikitext2(model.tokenizer, model.cfg.n_ctx, max_tokens=args.max_tokens)
    print(f"  Loaded {len(chunks)} chunks of {model.cfg.n_ctx} tokens each")

    all_results: dict = {
        "model": args.model,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "max_samples": args.max_samples,
    }

    # --- 7A ---
    delta_phi = None
    if "a" in args.parts:
        delta_phi = run_exp7a(model, chunks, max_samples=args.max_samples)
        all_results["exp7a"] = {
            "n_samples": delta_phi[0].shape[0],
            "d_model": delta_phi[0].shape[1],
            "mean_norms": {str(l): float(np.mean(np.linalg.norm(delta_phi[l], axis=1)))
                          for l in sorted(delta_phi.keys())},
        }

    # --- 7B ---
    per_layer_pca = None
    if "b" in args.parts and delta_phi is not None:
        per_layer_pca = run_exp7b(delta_phi, args.model, fig_dir)

        # Check: is Δφ low-rank?
        all_layers_90_under_50 = all(
            per_layer_pca[l]["thresholds"][0.9] <= 50
            for l in per_layer_pca
        )
        print(f"\n  All layers 90% in ≤50 components: {all_layers_90_under_50}")

        all_results["exp7b"] = {
            "thresholds": {
                str(l): {str(t): k for t, k in per_layer_pca[l]["thresholds"].items()}
                for l in sorted(per_layer_pca.keys())
            },
            "all_layers_90_under_50": all_layers_90_under_50,
        }

    # --- 7C ---
    shared_result = None
    if "c" in args.parts and delta_phi is not None and per_layer_pca is not None:
        shared_result = run_exp7c(delta_phi, per_layer_pca, args.model, fig_dir)
        all_results["exp7c"] = {
            "shared_thresholds": {str(t): k for t, k in shared_result["shared_pca"]["thresholds"].items()},
            "eval": {
                str(K): {
                    str(l): shared_result["eval"][K][l]
                    for l in sorted(shared_result["eval"][K].keys())
                }
                for K in shared_result["eval"]
            },
        }

    # --- 7D ---
    if "d" in args.parts and per_layer_pca is not None and shared_result is not None:
        results_7d = run_exp7d(model, args.model, per_layer_pca, shared_result, chunks, fig_dir)
        all_results["exp7d"] = results_7d

    # --- 7E ---
    if "e" in args.parts:
        results_7e = run_exp7e(model, chunks, args.model, fig_dir, max_samples=args.max_samples)
        all_results["exp7e"] = {
            "attn_thresholds": {
                str(l): {str(t): k for t, k in v.items()}
                for l, v in results_7e["attn_thresholds"].items()
            },
            "ffn_thresholds": {
                str(l): {str(t): k for t, k in v.items()}
                for l, v in results_7e["ffn_thresholds"].items()
            },
        }

    # --- Save ---
    json_path = data_dir / f"exp7_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nData saved: {json_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")

    if "exp7a" in all_results:
        print(f"\n  7A: Collected {all_results['exp7a']['n_samples']} samples, "
              f"d_model={all_results['exp7a']['d_model']}")

    if "exp7b" in all_results:
        print(f"\n  7B: Per-layer PCA thresholds (K for 90%):")
        for l in sorted(per_layer_pca.keys()):
            k90 = per_layer_pca[l]["thresholds"][0.9]
            print(f"    Layer {l}: K={k90}")
        print(f"  All ≤50? {all_results['exp7b']['all_layers_90_under_50']}")

    if "exp7d" in all_results:
        r = all_results["exp7d"]
        print(f"\n  7D: Perplexity (baseline={r['baseline']:.4f})")
        print(f"    {'K':>5} {'Ind PPL':>10} {'Ind Ratio':>10} {'Sh PPL':>10} {'Sh Ratio':>10}")
        print(f"    {'-'*45}")
        K_all = sorted(set(int(k) for k in r["individual"]) | set(int(k) for k in r["shared"]))
        for K in K_all:
            ind = r["individual"].get(str(K), {})
            sh = r["shared"].get(str(K), {})
            ind_ppl = ind.get("perplexity", float("nan"))
            ind_r = ind.get("ratio", float("nan"))
            sh_ppl = sh.get("perplexity", float("nan"))
            sh_r = sh.get("ratio", float("nan"))
            print(f"    {K:>5} {ind_ppl:>10.4f} {ind_r:>10.4f} {sh_ppl:>10.4f} {sh_r:>10.4f}")

    if "exp7e" in all_results:
        print(f"\n  7E: Attention vs FFN (K for 90%):")
        attn_t = all_results["exp7e"]["attn_thresholds"]
        ffn_t = all_results["exp7e"]["ffn_thresholds"]
        for l in sorted(attn_t.keys(), key=int):
            print(f"    Layer {l}: Attn={attn_t[l]['0.9']}, FFN={ffn_t[l]['0.9']}")


if __name__ == "__main__":
    main()
