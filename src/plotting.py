"""Visualization utilities for phase transition experiments."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_order_parameter(
    sigmas_dict: dict[str, np.ndarray],
    title: str = "Order Parameter σ(l)",
    output_path: str | Path | None = None,
) -> None:
    """Plot order parameter sigma(l) for multiple prompts.

    Args:
        sigmas_dict: Mapping from label to sigma array.
        title: Plot title.
        output_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for label, sigma in sigmas_dict.items():
        layers = np.arange(len(sigma))
        ax.plot(layers, sigma, "o-", label=label, markersize=4)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("Order Parameter σ(l)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_order_parameter_multi_word(
    results: dict[str, dict[str, np.ndarray]],
    output_path: str | Path | None = None,
) -> None:
    """Plot order parameter for multiple ambiguous words in subplots.

    Args:
        results: {word: {prompt_label: sigma_array}}.
        output_path: If provided, save figure to this path.
    """
    n_words = len(results)
    fig, axes = plt.subplots(1, n_words, figsize=(6 * n_words, 5), squeeze=False)

    for idx, (word, sigmas_dict) in enumerate(results.items()):
        ax = axes[0, idx]
        for label, sigma in sigmas_dict.items():
            layers = np.arange(len(sigma))
            ax.plot(layers, sigma, "o-", label=label, markersize=3)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer l")
        ax.set_ylabel("σ(l)")
        ax.set_title(f'"{word}"')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Order Parameter σ(l) by Ambiguous Word", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_derivative(
    sigmas_dict: dict[str, np.ndarray],
    title: str = "dσ/dl — Derivative of Order Parameter",
    output_path: str | Path | None = None,
) -> None:
    """Plot the numerical derivative of sigma to highlight transition layers.

    Args:
        sigmas_dict: Mapping from label to sigma array.
        title: Plot title.
        output_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for label, sigma in sigmas_dict.items():
        dsigma = np.diff(sigma)
        layers = np.arange(len(dsigma)) + 0.5
        ax.plot(layers, dsigma, "o-", label=label, markersize=3)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("dσ/dl")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)
