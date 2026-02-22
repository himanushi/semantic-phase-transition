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
| exp10I-L    | ✅ 完了     | Phase 2: ReLU最良、30ヘッド@1.1x、kernel化で19%削減 |
| exp10V      | ✅ 完了     | Validation: 250Kトークンで再検証、閾値-10ヘッド修正 |
| exp10P      | ✅ 完了     | Phase 3: Best-of-N有効、self-refinement無効、仮説5部分棄却 |

## exp10 最終結論

**仮説5「Attentionの精密さを下げ、浮いた計算をCoTに回すことで性能を維持できる」→ 部分的に棄却。**

- ReLU化は驚くほど無害（20ヘッドがPPL 1.1x以内、全トークン検証値）
- しかし浮いた計算は微小（kernel化で8-20%、追加6-16トークン分）
- Self-refinementはGPT-2では退化に向かう
- Best-of-Nは有効だが追加計算が必要（等FLOPsでは困難）

### 残された課題

1. CUDA kernel実装（hook-basedでは速度向上しない）
2. Instruction-tunedモデルでのself-refinement再検証
3. GPT-2 medium/largeでのスケーリング検証
4. ReLU化後のfinetuningによる劣化回収

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
│   ├── exp10_head_ablation.py  # 実験10 Phase 1: ヘッド削減
│   ├── exp10b_mean_ablation.py # 実験10 Phase 1.5: Mean-ablation
│   ├── exp10c_linear_attention.py # 実験10 Phase 2: 線形Attention
│   └── exp10d_validation.py    # 実験10 Validation + Phase 3: CoT回収
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
- **exp10結論**: 等FLOPs CoT回収は困難。CUDA kernel実装 + Instruction-tunedモデルでの再検証が必要
- **長期ビジョン**: 小モデル×高速CoT再帰 ≥ 大モデル×1回推論 → 等FLOPsでは困難と判明、追加計算許容時にのみ有効
