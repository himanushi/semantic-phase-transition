# CLAUDE.md

## プロジェクト概要

LLMの内部で意味がどう処理されるかを実験的に解明し、圧縮への応用を探るリポジトリ。

2つの研究軸がある:

1. **意味浸透の科学 (exp1-3)**: 曖昧語の意味がレイヤーを通じてどう確定するかを秩序変数で追跡。普遍浸透関数 g(l/L) を発見。完了済み。
2. **重み行列の圧縮 (exp6)**: Universal Weight Subspace Hypothesis の追試。W_QKをPCAで分解し、少数の基底でperplexityを維持できるかを検証。

@README.md

## 理論的背景

### 意味浸透モデル (exp1-3 で確立)

- **秩序変数**: `σ(l) = cos(φ(l), ê_diff)`
- **完全分解**: `σ(l, h) = h · f_max(word) · g(l/L)`
  - h: 文脈強度 [-1, 1]
  - f_max(word): 語依存スケーリング定数 (0.30〜0.79)
  - g(l/L): 普遍浸透関数（語間相関>0.96, モデル間相関0.98）
- **g の構造**: l/L < 0.85 でほぼ線形 (α≈0.9) + 最終層ドロップ (unembedding効果)
- **ランダウ相転移仮説は棄却**: β非普遍的 (0.01〜2.35)、一次相転移なし

### Universal Weight Subspace (exp6 で検証予定)

- 先行研究: 16-32個の主成分で重み分散の90%を説明 (Kaushik et al., 2025)
- 検証内容: GPT-2のW_QKをPCA → 少数基底で再構成 → perplexityへの影響を測定

## 技術スタック

- Python 3.10+
- PyTorch
- transformer-lens（Neel Nanda の mechanistic interpretability ライブラリ）
- matplotlib, scipy, numpy, scikit-learn
- 対象モデル: GPT-2 (small/medium)
- デバイス: M1 Mac では `--device mps`（CPU比 2.7x〜4.4x 高速）

## ディレクトリ構成

```
semantic-phase-transition/
├── README.md
├── CLAUDE.md
├── experiments/
│   ├── exp1_basic.py           # 実験1 初期版
│   ├── exp1_basic_v2.py        # 実験1 v2 改良版
│   ├── exp2_landau.py          # 実験2 ランダウフィット + logit lens
│   ├── exp3_linear_response.py # 実験3 線形応答の限界
│   ├── exp3ef_universal_g.py   # 実験3E/F/G 普遍浸透関数
│   └── exp6_basis.py           # 実験6 普遍基底 + 圧縮検証
├── src/
│   ├── order_parameter.py
│   ├── direction.py
│   ├── prompts.py
│   ├── fitting.py
│   └── plotting.py
├── results/
│   ├── figures/
│   └── data/
└── notebooks/
```

## コーディング規約

- 言語: Python。コメントは日本語OK、docstringは英語
- 型ヒント推奨
- 実験スクリプトはargparseで `--model`, `--device`, `--output-dir` を受け付ける
- 結果は `results/data/` にJSON、図は `results/figures/` にpng
- transformer-lensのキャッシュは `cache["resid_post", layer]` の形式で参照

## 実験ステータス

| 実験        | ステータス  | 主要結果                                                |
| ----------- | ----------- | ------------------------------------------------------- |
| exp1 v2     | ✅ 完了     | 全9語で扇形分岐。二段階構造を発見                       |
| exp2        | ✅ 完了     | ランダウ仮説棄却。two_stageモデル最良                   |
| exp3 + 3EFG | ✅ 完了     | σ=h·f_max·g(l/L)の完全分解。g(l/L)の普遍性確認          |
| exp4        | ⏭ スキップ | 臨界レイヤーが無いため感受率測定は不要                  |
| exp5        | ⏭ スキップ | 相転移の境界線がないため相図は不要                      |
| exp6        | 🔜 次に実行 | W_QKのPCA → 基底数 vs perplexity → ヘッドクラスタリング |

## exp6 の実験計画

### 6A. W_QKのPCA

- 全ヘッドから W_QK = W_Q^T @ W_K を抽出、flattenしてPCA
- 累積寄与率 vs 基底数のグラフ作成
- small: 12層×12ヘッド=144個, medium: 24層×16ヘッド=384個

### 6B. 基底数 vs perplexity

- K = [1, 2, 3, 5, 8, 10, 16, 32, 64, 全数] で上位K主成分からW_QKを再構成
- 再構成した重みをモデルに注入してWikiText-2でperplexity測定
- 元モデルとのperplexity比率を記録

### 6C. ヘッドのクラスタリング

- PCA係数空間で全ヘッドをk-meansクラスタリング
- 各クラスタの重心ヘッドで全メンバーを置換してperplexity測定
- k = [4, 8, 12, 16, 32]

### 6D. GPT-2 mediumでも同様に実行

## 重要な注意点

- GPT-2のレイヤー数: small=12 (12ヘッド), medium=24 (16ヘッド)
- W_QKの取得: `model.blocks[l].attn.W_Q[h].T @ model.blocks[l].attn.W_K[h]`
- 各W_QK: d_head × d_head (small: 64×64=4096, medium: 64×64=4096)
- PCA対象行列: (144, 4096) for small, (384, 4096) for medium
- perplexity評価にはdatasetsライブラリでWikiText-2を使用
- 重みの注入は transformer-lens の `model.blocks[l].attn.W_Q` に直接書き込み

## よく使うコマンド

```bash
# 環境構築
pip install torch transformers transformer-lens numpy matplotlib scipy scikit-learn datasets

# 実験6を実行
python experiments/exp6_basis.py --model gpt2 --device mps
python experiments/exp6_basis.py --model gpt2-medium --device mps
```

## 関連研究のキーワード

Universal Weight Subspace Hypothesis, semantic diffusion, linear response, logit lens, residual stream, mechanistic interpretability, layer pruning, weight compression, PCA, attention head clustering
