# CLAUDE.md

## プロジェクト概要

LLMの内部における意味処理のダイナミクスと重み行列の圧縮可能性を検証したプロジェクト。現在、普遍浸透関数 g(l/L) における π の痕跡を探索中。

@README.md

## 実験ステータス

| 実験        | ステータス  | 主要結果                                           |
| ----------- | ----------- | -------------------------------------------------- |
| exp1 v2     | ✅ 完了     | 全9語で扇形分岐。二段階構造を発見                  |
| exp2        | ✅ 完了     | ランダウ仮説棄却。two_stageモデル最良              |
| exp3 + 3EFG | ✅ 完了     | σ=h·f_max·g(l/L)の完全分解。g(l/L)の普遍性確認     |
| exp4        | ⏭ スキップ | 臨界レイヤーが無いため不要                         |
| exp5        | ⏭ スキップ | 相転移の境界線がないため不要                       |
| exp6        | ✅ 完了     | PCA分散≠機能的重要度。K=140/144でようやくPPL 1.07x |
| exp8A       | ✅ 完了     | σ=1/√π を棄却できず（p=0.535, 0.245）             |
| exp8B       | ✅ 完了     | medium で sin(πx^α/2) が BIC 最良                  |
| exp8C       | ✅ 完了     | Θ/π≈1.58、整数倍に非収束だが π/2 に近い            |
| exp9A       | 🔬 予定     | σ_free の4モデルスケーリング（Colab A100）          |
| exp9B       | 🔬 予定     | 累積回転角Θの4モデルスケーリング                   |
| exp9C       | 🔬 予定     | cos_pi_0p の全モデル検証                           |

## 現在の焦点: exp9（π収束検証）

**核心の問い**: σ_free はモデルサイズ→∞ で 1/√π に収束するか？

**exp8の結果**:
1. erf σ が 1/√π と整合的（8A: 両モデルで棄却できず）
2. sin(πx^α/2) が medium で BIC 最良（8B）
3. Θ/π ≈ π/2 の奇妙な一致（8C: 差0.6%）

**exp9の攻め筋**:
1. exp9A: gpt2/medium/large/xl の4点で σ_free を測定し、1/n_layers→0 への外挿
2. exp9B: 累積回転角Θのスケーリング法則を特定
3. exp9C: データ点数が増えるほど cos_pi_0p が有利になるか検証

**実行環境**: Google Colab A100（gpt2-xl まで対応）

## 最終結論（exp1-6）

1. **圧縮目標（100x）は未達成**: PCAベースのW_QK圧縮は原理的に困難
2. **意味浸透の普遍法則を発見**: σ(l,h) = h · f_max · g(l/L)、g の語間相関>0.96
3. **低分散成分が機能を支配**: 分散0.6%の成分でperplexityが2.2x→1.07xに変化

## 技術スタック

- Python 3.10+, PyTorch, transformer-lens, matplotlib, scipy, numpy, scikit-learn, datasets
- 対象モデル: GPT-2 (small/medium/large/xl)
- デバイス: M1 Mac では `--device mps`、Colab では `--device cuda`

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
│   ├── exp6_basis.py           # 実験6 普遍基底 + 圧縮検証
│   ├── exp7_residual_pca.py    # 実験7 残差ストリームΔφ PCA
│   ├── exp8_pi_trace.py        # 実験8 π探索
│   └── exp9_pi_convergence.py  # 実験9 π収束検証（Colab A100用）
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

## 将来の発展可能性

- g(l/L)の普遍性を他アーキテクチャ（Pythia, LLaMA）で検証
- f_max(word)を多数の語で測定し、言語学的性質との相関を分析
- PCA以外の分解手法（ICA, sparse coding）でW_QKの機能的構造を探索
- 低分散成分の「中身」を解釈（どのような注意パターンを符号化しているか）
- **exp9の結果次第**: 超球面拡散モデルの理論的定式化
- **exp9**: σ_free が 1/√π に収束すれば、Pythia/LLaMA 系でも検証
