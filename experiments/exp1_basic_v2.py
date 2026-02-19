"""Experiment 1 v2: Basic phase transition detection (improved).

Improvements over v1:
- All disambiguating context placed BEFORE the target word (causal mask aware)
- Direction vectors: both averaged (e_A, e_B) and contrastive (e_diff = mean_A - mean_B)
- Contrastive order parameter: sigma(l) = cos(phi(l), e_diff), more robust
  when cos(e_A, e_B) is high
- 9 ambiguous words tested
- Multi-model support
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer_lens import HookedTransformer

from src.direction import (
    compute_contrastive_direction,
    compute_direction_vectors,
    find_token_position,
)
from src.order_parameter import compute_order_parameter_contrastive
from src.plotting import plot_derivative, plot_order_parameter, plot_order_parameter_multi_word
from src.prompts import DIRECTION_PROMPTS, EXPERIMENT_PROMPTS


def run_experiment(
    model_name: str,
    device: str,
    output_dir: Path,
    words: list[str] | None = None,
) -> dict:
    """Run experiment 1 v2."""
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_tag = model_name.replace("/", "_")

    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    n_layers = model.cfg.n_layers
    print(f"  Layers: {n_layers}, Device: {device}")

    if words is None:
        words = list(DIRECTION_PROMPTS.keys())

    # --- Phase 1: 方向ベクトル計算 ---
    print(f"\n{'='*60}")
    print("Phase 1: Direction vectors (averaged + contrastive)")
    print(f"{'='*60}")

    direction_info: dict[str, dict] = {}
    valid_words: list[str] = []

    for word in words:
        dp = DIRECTION_PROMPTS[word]
        print(f"\n  [{word}] {dp['interpretation_A']} vs {dp['interpretation_B']}")

        try:
            e_A, e_B = compute_direction_vectors(
                model, dp["prompts_A"], dp["prompts_B"], word, device
            )
            e_diff = compute_contrastive_direction(
                model, dp["prompts_A"], dp["prompts_B"], word, device
            )
        except ValueError as e:
            print(f"    SKIP: {e}")
            continue

        cos_AB = torch.dot(e_A, e_B).item()
        print(f"    cos(e_A, e_B) = {cos_AB:.4f}")

        valid_words.append(word)
        direction_info[word] = {"e_A": e_A, "e_B": e_B, "e_diff": e_diff, "cos_AB": cos_AB}

    # cos(e_A, e_B) でソート（分離度が高い順に表示）
    valid_words.sort(key=lambda w: direction_info[w]["cos_AB"])
    print(f"\n  Words by separation (best first):")
    for w in valid_words:
        print(f"    {w:<10} cos={direction_info[w]['cos_AB']:.4f}")

    # --- Phase 2: 秩序変数の測定（対比的方向ベクトル） ---
    print(f"\n{'='*60}")
    print("Phase 2: Order parameter measurement (contrastive)")
    print(f"{'='*60}")

    all_results: dict[str, dict] = {}
    all_sigmas: dict[str, dict[str, np.ndarray]] = {}

    for word in valid_words:
        info = direction_info[word]
        dp = DIRECTION_PROMPTS[word]
        prompts = EXPERIMENT_PROMPTS[word]
        e_diff = info["e_diff"]

        print(f"\n  --- {word} ({dp['interpretation_A']} vs {dp['interpretation_B']}) ---")

        word_sigmas: dict[str, np.ndarray] = {}
        word_results: dict[str, dict] = {}

        for label, prompt_text in prompts.items():
            tokens = model.to_tokens(prompt_text)
            token_strs = model.to_str_tokens(tokens)
            if isinstance(token_strs[0], list):
                token_strs = token_strs[0]

            try:
                pos = find_token_position(tokens, word, model)
            except ValueError as e:
                print(f"    [{label}] SKIP: {e}")
                continue

            print(f"    [{label}] \"{prompt_text}\"")
            print(f"      '{word}' at pos {pos}/{len(token_strs)-1}, "
                  f"tokens: {token_strs}")

            sigma = compute_order_parameter_contrastive(
                model, prompt_text, word, e_diff
            )
            word_sigmas[label] = sigma

            dsigma = np.diff(sigma)
            max_jump_idx = int(np.argmax(np.abs(dsigma)))

            result = {
                "prompt": prompt_text,
                "sigma": sigma.tolist(),
                "sigma_final": float(sigma[-1]),
                "sigma_range": float(np.ptp(sigma)),
                "max_jump_layer": max_jump_idx,
                "max_jump_value": float(dsigma[max_jump_idx]),
                "target_position": pos,
            }
            word_results[label] = result

            print(f"      σ: [{sigma.min():.4f}, {sigma.max():.4f}], "
                  f"final={sigma[-1]:+.4f}, "
                  f"max_jump=L{max_jump_idx}({dsigma[max_jump_idx]:+.4f})")

        if word_sigmas:
            all_sigmas[word] = word_sigmas
            all_results[word] = {
                "cos_AB": info["cos_AB"],
                "interpretation_A": dp["interpretation_A"],
                "interpretation_B": dp["interpretation_B"],
                "prompts": word_results,
            }

            plot_order_parameter(
                word_sigmas,
                title=f'σ(l) — "{word}" ({model_name}, {n_layers}L)',
                output_path=fig_dir / f"v2_sigma_{word}_{model_tag}.png",
            )
            plot_derivative(
                word_sigmas,
                title=f'dσ/dl — "{word}" ({model_name}, {n_layers}L)',
                output_path=fig_dir / f"v2_dsigma_{word}_{model_tag}.png",
            )

    # 全単語比較プロット
    if all_sigmas:
        plot_order_parameter_multi_word(
            all_sigmas,
            output_path=fig_dir / f"v2_sigma_all_{model_tag}.png",
        )

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name} ({n_layers} layers)")
    print(f"{'='*60}")
    header = (f"{'word':<10} {'cos(eA,eB)':<12} {'σ range':<10} "
              f"{'σ_f(strong_A)':<14} {'σ_f(strong_B)':<14} {'max |Δσ|':<20}")
    print(f"\n{header}")
    print("-" * len(header))

    for word in valid_words:
        if word not in all_results:
            continue
        res = all_results[word]
        pr = res["prompts"]

        sf_A = pr.get("strong_A", {}).get("sigma_final", float("nan"))
        sf_B = pr.get("strong_B", {}).get("sigma_final", float("nan"))

        all_ranges = [p.get("sigma_range", 0) for p in pr.values()]
        sr = max(all_ranges) if all_ranges else 0

        jumps = [(p.get("max_jump_layer", -1), p.get("max_jump_value", 0))
                 for p in pr.values()]
        biggest = max(jumps, key=lambda x: abs(x[1]))

        print(f"{word:<10} {res['cos_AB']:<12.4f} {sr:<10.4f} "
              f"{sf_A:<+14.4f} {sf_B:<+14.4f} L{biggest[0]}({biggest[1]:+.4f})")

    # JSON保存
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "device": device,
        "method": "contrastive",
        "valid_words": valid_words,
        "words": all_results,
    }
    json_path = data_dir / f"exp1_v2_{model_tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nData saved: {json_path}")

    del model
    torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1 v2: Improved phase transition detection"
    )
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
