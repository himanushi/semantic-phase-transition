"""Direction vector computation for order parameter measurement."""

import torch
from transformer_lens import HookedTransformer


def find_token_position(tokens: torch.Tensor, target_word: str, model: HookedTransformer) -> int:
    """Find the position of target_word in tokenized input.

    Searches through token strings for a match. Handles subword tokenization
    by checking if the target word appears within the decoded token.

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
    # token_strs は list of list の場合がある
    if isinstance(token_strs[0], list):
        token_strs = token_strs[0]

    # まず完全一致を試みる（空白付き）
    for i, t in enumerate(token_strs):
        stripped = t.strip().lower()
        if stripped == target_word.lower():
            return i

    # 部分一致にフォールバック
    for i, t in enumerate(token_strs):
        if target_word.lower() in t.lower():
            return i

    raise ValueError(
        f"Token '{target_word}' not found in {token_strs}"
    )


def compute_direction_vectors(
    model: HookedTransformer,
    prompt_A: str,
    prompt_B: str,
    target_word: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute direction vectors e_A and e_B from unambiguous contexts.

    Uses the final layer residual stream at the target token position
    from two clearly disambiguating prompts.

    Args:
        model: The HookedTransformer model.
        prompt_A: Prompt clearly indicating interpretation A.
        prompt_B: Prompt clearly indicating interpretation B.
        target_word: The ambiguous word to track.
        device: Computation device.

    Returns:
        Tuple of (e_A, e_B) normalized direction vectors.
    """
    with torch.no_grad():
        tokens_A = model.to_tokens(prompt_A)
        _, cache_A = model.run_with_cache(prompt_A)
        pos_A = find_token_position(tokens_A, target_word, model)

        tokens_B = model.to_tokens(prompt_B)
        _, cache_B = model.run_with_cache(prompt_B)
        pos_B = find_token_position(tokens_B, target_word, model)

        # 最終レイヤーの残差ストリームから方向ベクトルを取得
        last_layer = model.cfg.n_layers - 1
        e_A = cache_A["resid_post", last_layer][0, pos_A].clone()
        e_B = cache_B["resid_post", last_layer][0, pos_B].clone()

        # 正規化
        e_A = e_A / e_A.norm()
        e_B = e_B / e_B.norm()

    return e_A, e_B
