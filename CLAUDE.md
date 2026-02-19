# CLAUDE.md

## プロジェクト概要

LLMが曖昧な文を処理する際、意味の確定が特定のレイヤーで物理学的な「相転移」として起きるかを実験的に検証するリポジトリ。ランダウ相転移理論の枠組みで、秩序変数σ(l)を定義しレイヤーごとに追跡する。

## 理論的背景

- 秩序変数: `σ(l) = cos(φ(l), ê_A) - cos(φ(l), ê_B)` （2つの解釈方向への射影差）
- 仮説: σ(l) が臨界レイヤー l_c で急激にジャンプし、ランダウ理論 `σ ∝ (l - l_c)^β` に従う
- 成功条件: β ≈ 0.5（平均場）、l_c/L がモデルサイズに依存しない普遍性

## 技術スタック

- Python 3.10+
- PyTorch
- transformer-lens（Neel Nanda の mechanistic interpretability ライブラリ）
- matplotlib, scipy, numpy
- 対象モデル: GPT-2 (small/medium/large), Gemma-2-2B

## ディレクトリ構成

```
semantic-phase-transition/
├── README.md              # 実験計画の全体像
├── CLAUDE.md              # このファイル
├── experiments/           # 実験スクリプト
│   ├── exp1_basic.py      # 実験1: 秩序変数の基本測定
│   ├── exp2_landau.py     # 実験2: ランダウフィット
│   ├── exp3_universal.py  # 実験3: モデル横断的な普遍性
│   ├── exp4_suscept.py    # 実験4: 感受率の測定
│   ├── exp5_phase_diag.py # 実験5: 相図の作成
│   └── exp6_basis.py      # 実験6: 普遍基底との接続
├── src/                   # 共通コード
│   ├── order_parameter.py # σ(l) の計算
│   ├── direction.py       # 方向ベクトル ê_A, ê_B の取得
│   ├── prompts.py         # 曖昧プロンプトの定義
│   ├── fitting.py         # ランダウフィット関数
│   └── plotting.py        # 可視化ユーティリティ
├── results/               # 実験結果（図・データ）
│   ├── figures/
│   └── data/
└── notebooks/             # 探索的解析用
```

## コーディング規約

- 言語: Python。コメントは日本語OK、docstringは英語
- 型ヒント推奨
- 実験スクリプトはargparseで `--model`, `--device`, `--output-dir` を受け付ける
- 結果は `results/data/` にJSONまたはnpyで保存、図は `results/figures/` にpng
- transformer-lensのキャッシュは `cache["resid_post", layer]` の形式で参照

## 重要な注意点

- トークン位置の特定が肝。モデルごとにトークナイザの分割が異なるため、`find_token_position` は慎重に実装すること
- GPT-2のレイヤー数: small=12, medium=24, large=36
- 方向ベクトル ê_A, ê_B は**最終レイヤー**の残差ストリームから取得する（そこが最も意味的に分化している）
- GPU メモリ: GPT-2 small は CPU でも動く。large は ~3GB VRAM 必要
- 実験の優先順位: exp1 → exp2 → exp4 → exp5 → exp3 → exp6（まず基本測定、次にフィット）

## よく使うコマンド

```bash
# 環境構築
pip install torch transformers transformer-lens numpy matplotlib scipy

# 実験1を実行
python experiments/exp1_basic.py --model gpt2 --device cpu

# 全実験を順番に実行
python experiments/exp1_basic.py --model gpt2-medium && \
python experiments/exp2_landau.py --model gpt2-medium
```

## 関連研究のキーワード

検索用: Universal Weight Subspace Hypothesis, O(N) model LLM, phase transition training dynamics, effective depth, logit lens, residual stream, mechanistic interpretability, Landau theory, order parameter
