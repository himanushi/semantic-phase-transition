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
| exp8A       | 🔬 予定     | erf σ≈0.55 vs 1/√π≈0.564 の尤度比検定              |
| exp8B       | 🔬 予定     | cos(πx)系0パラメータモデルのBIC比較                |
| exp8C       | 🔬 予定     | レイヤー間回転角の累積測定                         |

## 現在の焦点: exp8（π探索）

**核心の問い**: exp3efで得た erf σ≈0.55 は 1/√π≈0.5642 と2.5%差。偶然か必然か？

**攻め筋**:

1. exp8A: 既存exp3efデータの再解析（尤度比検定）← まずここから
2. exp8B: π含有0パラメータモデル追加（BIC比較）
3. exp8C: 残差ストリーム回転角の幾何学的測定

**成功基準（事前登録済み）**: README.md参照

## 最終結論（exp1-6）

1. **圧縮目標（100x）は未達成**: PCAベースのW_QK圧縮は原理的に困難
2. **意味浸透の普遍法則を発見**: σ(l,h) = h · f_max · g(l/L)、g の語間相関>0.96
3. **低分散成分が機能を支配**: 分散0.6%の成分でperplexityが2.2x→1.07xに変化

## 技術スタック

- Python 3.10+, PyTorch, transformer-lens, matplotlib, scipy, numpy, scikit-learn, datasets
- 対象モデル: GPT-2 (small/medium)
- デバイス: M1 Mac では `--device mps`

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
│   └── exp8_pi_trace.py        # 実験8 π探索（予定）
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
- **exp8の結果次第**: 超球面拡散モデルの理論的定式化
