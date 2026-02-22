# CLAUDE.md

## プロジェクト概要

LLMの内部における意味処理のダイナミクスと重み行列の圧縮可能性を検証したプロジェクト。現在、ヘッド削減＋線形Attention＋CoT回収による軽量高性能アーキテクチャを探索中。

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
| exp8A       | ✅ 完了     | σ=1/√π を棄却できず（p=0.535, 0.245）              |
| exp8B       | ✅ 完了     | medium で sin(πx^α/2) が BIC 最良                  |
| exp8C       | ✅ 完了     | Θ/π≈1.58、整数倍に非収束だが π/2 に近い            |
| exp9A       | 🔬 中断     | CUDA精度問題で未完了                               |
| exp9B       | 🔬 予定     | 累積回転角Θの4モデルスケーリング                   |
| exp9C       | 🔬 予定     | cos_pi_0p の全モデル検証                           |
| exp10A-D    | ✅ 完了     | Phase 1: L0に機能集中、Zipf α=1.37、累積10ヘッド@1.1x |
| exp10E-H    | ✅ 完了     | Phase 1.5: Mean-ablationで25ヘッド@1.1x（2.5x改善） |
| **exp10 Ph2** | 📋 **NEXT** | Phase 2: softmax線形化（Go判定済）                 |
| exp10 Ph3   | 📋 設計済   | Phase 3: CoT回収検証（Phase 2の結果次第）          |

## 現在の焦点: exp10 Phase 1（ヘッド単位Ablation）

**核心の問い**: 144ヘッド中、何個を削除してもperplexityが維持されるか？機能的重要度の分布はどうなっているか？

**exp6との関係**:

- exp6はPCA基底でW_QKを分解 → 分散0.6%に機能集中を発見
- exp10はヘッド単位で直接ablation → PCAでは見えない機能的構造を測定
- exp6の「ヘッドの非冗長性」結論をヘッド単位で再検証

**最終ゴール（Phase 1-3全体）**:
「Attentionを粗く・高速にし、浮いた計算をCoTに回す」アーキテクチャの実現可能性を検証。

- 小さいモデル × N回再帰推論 ≥ 大きいモデル × 1回推論（同一FLOPs）を目指す

### exp10A: ヘッド個別ablation

- GPT-2 small（12レイヤー × 12ヘッド = 144ヘッド）
- 各ヘッドを1個ずつゼロアウト（attention output を0に置換）
- WikiText-2 validation set でPPL測定（baseline: ~29.79）
- 出力: 144個の ΔPPL 値

### exp10B: 重要度ランキング

- ΔPPL でソートした全144ヘッドのランキング
- ヒストグラム + レイヤー×ヘッドのヒートマップ
- べき乗則フィット（少数ヘッドに機能集中しているか）

### exp10C: 累積ablation

- 重要度の低い順にヘッドを1個ずつ累積削除
- 「削除ヘッド数 vs PPL」のパレート曲線
- PPL 1.1倍、1.5倍、2.0倍の閾値でそれぞれ何個削除可能かを記録

### exp10D: レイヤー内パターン分析

- 重要ヘッドの空間分布（浅い層 vs 深い層）
- レイヤーごとの平均ΔPPL（どのレイヤーが最も機能的に重要か）
- exp3の浸透関数 g(l/L) との対応（g(l/L)の傾きが大きい層のヘッドが重要か？）

### 成功基準（Phase 1）

1. **144ヘッド中30%以上（≥43ヘッド）がPPL劣化1.1倍未満で削除可能**
2. **重要度分布がべき乗則に従う**（少数のヘッドに機能が集中）
3. exp3の g(l/L) と重要度分布に相関がある

## 技術スタック

- Python 3.10+, PyTorch, transformer-lens, matplotlib, scipy, numpy, scikit-learn, datasets
- 対象モデル: GPT-2 (small/medium/large/xl)
- デバイス: M1 Mac では `--device mps`、Colab では `--device cuda`
- **CUDA使用時の注意**: `torch.backends.cuda.matmul.allow_tf32 = False` を必ず設定

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
│   ├── exp9_pi_convergence.py  # 実験9 π収束検証（Colab A100用）
│   └── exp10_head_ablation.py  # 実験10 ヘッド削減 + 線形Attention + CoT回収
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
- **実験が完了したら `results/exp{N}_results.md` に結果レポートを書く**（例: `results/exp10_results.md`）

## 将来の発展可能性

- g(l/L)の普遍性を他アーキテクチャ（Pythia, LLaMA）で検証
- f_max(word)を多数の語で測定し、言語学的性質との相関を分析
- PCA以外の分解手法（ICA, sparse coding）でW_QKの機能的構造を探索
- 低分散成分の「中身」を解釈（どのような注意パターンを符号化しているか）
- **exp9の結果次第**: 超球面拡散モデルの理論的定式化
- **exp10の結果次第**: 線形Attention + CoT再帰アーキテクチャの本格設計
- **長期ビジョン**: 小モデル×高速CoT再帰 ≥ 大モデル×1回推論の等FLOPs検証
