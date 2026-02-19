"""Direction vector computation for order parameter measurement."""

import torch
from transformer_lens import HookedTransformer


def find_token_position(tokens: torch.Tensor, target_word: str, model: HookedTransformer) -> int:
    """Find the position of target_word in tokenized input.

    Searches backwards through token strings for the last match,
    since the target word typically appears at/near the end of the prompt.

    Args:
        tokens: Tokenized input tensor.
        target_word: The word to locate.
        model: The HookedTransformer model (for tokenizer access).

    Returns:
        Token position index.

    Raises:
        ValueError: If the target word is not found.
    """
    token_strs = model.to_str_tokens(tokens)
    if isinstance(token_strs[0], list):
        token_strs = token_strs[0]

    # 後ろから探索（対象語はプロンプト末尾付近にあることが多い）
    # まず完全一致
    for i in range(len(token_strs) - 1, -1, -1):
        stripped = token_strs[i].strip().lower()
        if stripped == target_word.lower():
            return i

    # 部分一致にフォールバック
    for i in range(len(token_strs) - 1, -1, -1):
        if target_word.lower() in token_strs[i].lower():
            return i

    raise ValueError(
        f"Token '{target_word}' not found in {token_strs}"
    )


def _collect_residuals(
    model: HookedTransformer,
    prompts: list[str],
    target_word: str,
) -> list[torch.Tensor]:
    """Collect final-layer residual stream vectors for target word from multiple prompts."""
    last_layer = model.cfg.n_layers - 1
    vectors = []
    for prompt in prompts:
        with torch.no_grad():
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(prompt)
            pos = find_token_position(tokens, target_word, model)
            vec = cache["resid_post", last_layer][0, pos].clone()
            vectors.append(vec)
    return vectors


def compute_direction_vectors(
    model: HookedTransformer,
    prompts_A: list[str],
    prompts_B: list[str],
    target_word: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute direction vectors by averaging over multiple unambiguous prompts.

    Returns individually normalized mean vectors for each interpretation.

    Args:
        model: The HookedTransformer model.
        prompts_A: List of prompts clearly indicating interpretation A.
        prompts_B: List of prompts clearly indicating interpretation B.
        target_word: The ambiguous word to track.
        device: Computation device.

    Returns:
        Tuple of (e_A, e_B) normalized direction vectors.
    """
    vecs_A = _collect_residuals(model, prompts_A, target_word)
    vecs_B = _collect_residuals(model, prompts_B, target_word)

    mean_A = torch.stack(vecs_A).mean(dim=0)
    mean_B = torch.stack(vecs_B).mean(dim=0)

    e_A = mean_A / mean_A.norm()
    e_B = mean_B / mean_B.norm()

    return e_A, e_B


def compute_contrastive_direction(
    model: HookedTransformer,
    prompts_A: list[str],
    prompts_B: list[str],
    target_word: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute a single contrastive direction vector: mean_A - mean_B, normalized.

    This captures the dimension along which the two meanings maximally differ,
    which is more robust when e_A and e_B have high cosine similarity.

    Args:
        model: The HookedTransformer model.
        prompts_A: List of prompts clearly indicating interpretation A.
        prompts_B: List of prompts clearly indicating interpretation B.
        target_word: The ambiguous word to track.
        device: Computation device.

    Returns:
        Normalized contrastive direction vector (e_A_mean - e_B_mean) / ||...||.
    """
    vecs_A = _collect_residuals(model, prompts_A, target_word)
    vecs_B = _collect_residuals(model, prompts_B, target_word)

    mean_A = torch.stack(vecs_A).mean(dim=0)
    mean_B = torch.stack(vecs_B).mean(dim=0)

    diff = mean_A - mean_B
    return diff / diff.norm()
