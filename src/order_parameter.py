"""Order parameter sigma(l) computation."""

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .direction import find_token_position


def compute_order_parameter(
    model: HookedTransformer,
    prompt: str,
    target_word: str,
    e_A: torch.Tensor,
    e_B: torch.Tensor,
) -> np.ndarray:
    """Compute order parameter sigma(l) = cos(phi, e_A) - cos(phi, e_B) for each layer.

    Args:
        model: The HookedTransformer model.
        prompt: Input prompt string.
        target_word: The ambiguous word to track.
        e_A: Normalized direction vector for interpretation A.
        e_B: Normalized direction vector for interpretation B.

    Returns:
        Array of shape (n_layers + 1,) with sigma values.
        Index 0 = embedding layer, index l = after layer l-1.
    """
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(prompt)
        target_pos = find_token_position(tokens, target_word, model)

        sigma = np.zeros(n_layers + 1)

        for l in range(n_layers + 1):
            if l == 0:
                resid = cache["resid_pre", 0][0, target_pos]
            else:
                resid = cache["resid_post", l - 1][0, target_pos]

            resid_norm = resid / resid.norm()

            cos_A = torch.dot(resid_norm, e_A).item()
            cos_B = torch.dot(resid_norm, e_B).item()

            sigma[l] = cos_A - cos_B

    return sigma


def compute_order_parameter_contrastive(
    model: HookedTransformer,
    prompt: str,
    target_word: str,
    e_diff: torch.Tensor,
) -> np.ndarray:
    """Compute order parameter using a contrastive direction: sigma(l) = cos(phi, e_diff).

    Positive sigma means closer to interpretation A, negative to B.

    Args:
        model: The HookedTransformer model.
        prompt: Input prompt string.
        target_word: The ambiguous word to track.
        e_diff: Normalized contrastive direction (mean_A - mean_B).

    Returns:
        Array of shape (n_layers + 1,) with sigma values.
    """
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(prompt)
        target_pos = find_token_position(tokens, target_word, model)

        sigma = np.zeros(n_layers + 1)

        for l in range(n_layers + 1):
            if l == 0:
                resid = cache["resid_pre", 0][0, target_pos]
            else:
                resid = cache["resid_post", l - 1][0, target_pos]

            resid_norm = resid / resid.norm()
            sigma[l] = torch.dot(resid_norm, e_diff).item()

    return sigma
