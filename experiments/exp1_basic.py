"""Experiment 1: Basic phase transition detection.

Measures the order parameter sigma(l) across all layers for ambiguous words
under varying context strengths. Detects whether semantic disambiguation
exhibits phase-transition-like behavior.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer_lens import HookedTransformer

from src.direction import compute_direction_vectors, find_token_position
from src.order_parameter import compute_order_parameter
from src.plotting import (
    plot_derivative,
    plot_order_parameter,
    plot_order_parameter_multi_word,
)
from src.prompts import AMBIGUOUS_WORDS


def run_experiment(
    model_name: str,
    device: str,
    output_dir: Path,
    words: list[str] | None = None,
) -> dict:
    """Run experiment 1 for specified model and words.

    Args:
        model_name: HuggingFace model name (e.g. 'gpt2', 'gpt2-medium').
        device: Torch device string.
        output_dir: Directory to save results.
        words: List of ambiguous words to test. None = all.

    Returns:
        Dictionary of all results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    n_layers = model.cfg.n_layers
    print(f"  Layers: {n_layers}, Device: {device}")

    if words is None:
        words = list(AMBIGUOUS_WORDS.keys())

    all_results: dict[str, dict] = {}
    all_sigmas: dict[str, dict[str, np.ndarray]] = {}

    for word in words:
        info = AMBIGUOUS_WORDS[word]
        print(f"\n{'='*60}")
        print(f"Processing: '{word}' ({info['interpretation_A']} vs {info['interpretation_B']})")
        print(f"{'='*60}")

        # 方向ベクトルの計算
        print(f"  Computing direction vectors...")
        print(f"    Prompt A: \"{info['prompt_A']}\"")
        print(f"    Prompt B: \"{info['prompt_B']}\"")
        e_A, e_B = compute_direction_vectors(
            model, info["prompt_A"], info["prompt_B"], word, device
        )

        # 方向ベクトルの直交性チェック
        cos_AB = torch.dot(e_A, e_B).item()
        print(f"  cos(e_A, e_B) = {cos_AB:.4f}")

        # 各プロンプトで秩序変数を計算
        word_sigmas: dict[str, np.ndarray] = {}
        word_results: dict[str, dict] = {}

        for prompt_label, prompt_text in info["prompts"].items():
            print(f"  [{prompt_label}] \"{prompt_text}\"")

            # トークン位置の確認
            tokens = model.to_tokens(prompt_text)
            token_strs = model.to_str_tokens(tokens)
            if isinstance(token_strs[0], list):
                token_strs = token_strs[0]
            pos = find_token_position(tokens, word, model)
            print(f"    Tokens: {token_strs}")
            print(f"    Target '{word}' at position {pos}")

            sigma = compute_order_parameter(model, prompt_text, word, e_A, e_B)
            word_sigmas[prompt_label] = sigma

            # 基本統計
            dsigma = np.diff(sigma)
            max_jump_layer = int(np.argmax(np.abs(dsigma)))
            max_jump_value = float(dsigma[max_jump_layer])

            result = {
                "prompt": prompt_text,
                "sigma": sigma.tolist(),
                "sigma_final": float(sigma[-1]),
                "sigma_range": float(np.max(sigma) - np.min(sigma)),
                "max_jump_layer": max_jump_layer,
                "max_jump_value": max_jump_value,
                "target_position": pos,
            }
            word_results[prompt_label] = result

            print(f"    σ range: [{sigma.min():.4f}, {sigma.max():.4f}]")
            print(f"    σ final: {sigma[-1]:.4f}")
            print(f"    Max jump: layer {max_jump_layer} → {max_jump_layer+1}, "
                  f"Δσ = {max_jump_value:.4f}")

        all_sigmas[word] = word_sigmas
        all_results[word] = {
            "direction_cos_AB": cos_AB,
            "prompts": word_results,
        }

        # 単語ごとのプロット
        plot_order_parameter(
            word_sigmas,
            title=f'Order Parameter σ(l) — "{word}" ({model_name})',
            output_path=fig_dir / f"sigma_{word}_{model_name.replace('/', '_')}.png",
        )
        plot_derivative(
            word_sigmas,
            title=f'dσ/dl — "{word}" ({model_name})',
            output_path=fig_dir / f"dsigma_{word}_{model_name.replace('/', '_')}.png",
        )

    # 全単語の比較プロット
    plot_order_parameter_multi_word(
        all_sigmas,
        output_path=fig_dir / f"sigma_all_{model_name.replace('/', '_')}.png",
    )

    # サマリー出力
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for word, res in all_results.items():
        print(f"\n  [{word}] cos(e_A, e_B) = {res['direction_cos_AB']:.4f}")
        for label, pr in res["prompts"].items():
            print(f"    {label:15s}: σ_final={pr['sigma_final']:+.4f}, "
                  f"range={pr['sigma_range']:.4f}, "
                  f"max_jump=L{pr['max_jump_layer']}({pr['max_jump_value']:+.4f})")

    # JSON保存
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "device": device,
        "words": all_results,
    }
    json_path = data_dir / f"exp1_{model_name.replace('/', '_')}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nData saved: {json_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Basic phase transition detection")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--words", nargs="*", default=None,
                        help="Ambiguous words to test (default: all)")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        device=args.device,
        output_dir=Path(args.output_dir),
        words=args.words,
    )


if __name__ == "__main__":
    main()
