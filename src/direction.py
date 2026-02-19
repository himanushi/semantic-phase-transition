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


def compute_direction_vectors(
    model: HookedTransformer,
    prompts_A: list[str],
    prompts_B: list[str],
    target_word: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute direction vectors by averaging over multiple unambiguous prompts.

    For each interpretation, collects the final-layer residual stream at
    the target token position from multiple prompts and averages them.

    Args:
        model: The HookedTransformer model.
        prompts_A: List of prompts clearly indicating interpretation A.
        prompts_B: List of prompts clearly indicating interpretation B.
        target_word: The ambiguous word to track.
        device: Computation device.

    Returns:
        Tuple of (e_A, e_B) normalized direction vectors.
    """
    last_layer = model.cfg.n_layers - 1

    def _collect_vectors(prompts: list[str]) -> torch.Tensor:
        vectors = []
        for prompt in prompts:
            with torch.no_grad():
                tokens = model.to_tokens(prompt)
                _, cache = model.run_with_cache(prompt)
                pos = find_token_position(tokens, target_word, model)
                vec = cache["resid_post", last_layer][0, pos].clone()
                vectors.append(vec)
        # 平均して正規化
        avg = torch.stack(vectors).mean(dim=0)
        return avg / avg.norm()

    e_A = _collect_vectors(prompts_A)
    e_B = _collect_vectors(prompts_B)

    return e_A, e_B
