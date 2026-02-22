# Semantic Phase Transition

LLMの内部における意味処理のダイナミクスを実験的に解明し、重み行列の圧縮可能性を検証したプロジェクト。

## 結論

**当初の目標（重み行列の100x圧縮）は達成できなかった。** しかし、その過程で以下の成果を得た:

1. **意味浸透の普遍法則**: 曖昧語の意味確定過程は `σ(l,h) = h · f_max(word) · g(l/L)` で完全分解でき、浸透関数 g(l/L) は語にもモデルサイズにも依存しない普遍関数（語間相関 > 0.96、モデル間相関 0.98）
2. **PCA分散 ≠ 機能的重要度**: W_QKの分散99.1%を保持してもPPLは2.2x劣化。機能的情報は低分散成分（残り0.6%）に集中
3. **Softmax → ReLUは驚くほど無害**: 144ヘッド中91%がΔPPL < 1%でReLU化可能。ただし浮く計算は微小（8-20%）で、等FLOPsでのCoT回収は困難

---

## 仮説の変遷

```
仮説1: ランダウ相転移 (exp1-2) → 棄却。変化は連続的
仮説2: 意味浸透モデル (exp3) → 確立。σ = h · f_max · g(l/L)
仮説3: 普遍基底による圧縮 (exp6) → 棄却。PCA分散と機能は別物
仮説4: πの痕跡 (exp8-9) → 示唆的だが決定的ではない
仮説5: 線形Attention + CoT回収 (exp10) → 部分的に棄却。ReLU化は無害だが等FLOPs回収は困難
```

---

## 実験結果

詳細は各 `results/exp*_results.md` を参照。

| 実験 | 主題 | 主要結果 |
|------|------|----------|
| exp1 | 秩序変数の測定 | 全9語で扇形分岐 + 二段階構造（漸進分化 + 最終層ジャンプ） |
| exp2 | ランダウフィット | 仮説棄却（β=0.01-2.35）。two_stageモデルが最良 |
| exp3 | 普遍浸透関数 | **σ = h · f_max · g(l/L)** の完全分解。g(l/L)は語・モデル非依存 |
| exp6 | PCA圧縮検証 | K=140/144でようやくPPL 1.07x。低分散成分に機能集中 |
| exp8 | πの痕跡探索 | σ=1/√πを棄却できず(p=0.535)。決定的証拠には至らず |
| exp9 | π収束検証 | CUDA TF32精度問題で中断 |
| exp10 | ヘッド削減+線形化+CoT | 下表参照 |

### exp10 詳細

| Phase | 内容 | 結果 |
|-------|------|------|
| Ph1 | Zero-ablation | L0に機能集中（Zipf α=1.37）。PPL 1.1x以内: 10ヘッド |
| Ph1.5 | Mean-ablation | PPL 1.1x以内: 25ヘッド。L0H8の99.7%が定常バイアス |
| Ph2 | softmax線形化 | **ReLU最良**: PPL 1.1x以内に30ヘッド（2K推定）/ 20ヘッド（250K検証値） |
| Validation | 250Kトークン再検証 | 2K推定は10ヘッド程度楽観的だが定性的結論は維持 |
| Ph3 | CoT回収 | Best-of-N有効（relu_30@N=5 < baseline@N=1）、self-refinement無効 |

---

## セットアップ

```bash
pip install torch transformers transformer-lens numpy matplotlib scipy scikit-learn datasets
```

```bash
# 意味浸透実験 (exp1-3)
python experiments/exp1_basic_v2.py --model gpt2 --device mps
python experiments/exp2_landau.py --model gpt2 --device mps
python experiments/exp3_linear_response.py --model gpt2 --device mps
python experiments/exp3ef_universal_g.py --model gpt2 --device mps

# 普遍基底実験 (exp6)
python experiments/exp6_basis.py --model gpt2 --device mps

# π探索実験 (exp8-9)
python experiments/exp8_pi_trace.py --model gpt2 --device mps --parts abc
python experiments/exp9_pi_convergence.py --device cuda --parts abc

# ヘッド削減実験 (exp10)
python experiments/exp10_head_ablation.py --model gpt2 --device mps --parts abcd
python experiments/exp10b_mean_ablation.py --model gpt2 --device mps --parts efgh
python experiments/exp10c_linear_attention.py --model gpt2 --device mps --parts ijkl
python experiments/exp10d_validation.py --model gpt2 --device mps --parts vp
```

## 参考文献

1. Kaushik et al., "The Universal Weight Subspace Hypothesis" (arXiv:2512.05117, Dec 2025)
2. Sun et al., "Phase Transitions in Large Language Models and the O(N) Model" (arXiv:2501.16241, Jan 2025)
3. "Phase Transitions in the Output Distribution of Large Language Models" (arXiv:2405.17088, May 2024)
4. "Decomposing Behavioral Phase Transitions in LLMs" (arXiv:2508.20015, Aug 2025)
5. Hu et al., "What Affects the Effective Depth of Large Language Models?" (arXiv:2512.14064, Dec 2025)
6. Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models" (Mar 2025)
7. Loshchilov et al., "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" (arXiv:2410.01131, Oct 2024)
