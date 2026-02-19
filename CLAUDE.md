# CLAUDE.md

## プロジェクト概要

LLMが曖昧な文を処理する際、意味の確定が各レイヤーでどのように進行するかを実験的に検証するリポジトリ。当初ランダウ相転移仮説から出発したが、実験1-2の結果を受けて**意味浸透モデル (Semantic Diffusion)** に枠組みを修正した。秩序変数σ(l)を定義し、レイヤーごとに追跡する。

@README.md

## 理論的背景

### 当初の仮説（棄却済み）
- 仮説: σ(l) が臨界レイヤー l_c でランダウ理論 `σ ∝ (l - l_c)^β` に従う
- 結果: βは 0.01〜2.35 と大きく分散し、普遍的臨界指数は存在しない

### 現在の枠組み: 意味浸透モデル
- **秩序変数**: `σ(l) = cos(φ(l), ê_diff)` （対比的方向ベクトルとの内積）
- **線形応答**: `σ(l, h) = h · f(l)` — 文脈強度 h に比例して秩序変数が変化
- **f(l)の性質**: 単調増加関数。意味情報がレイヤーを通じて対象トークンに「浸透」
- **二段階構造**: 中間層での漸進的分化 + 最終層(l/L≈0.92)での unembedding 再構成
- **検証すべき問い**: 線形応答が破れる閾値 h*(l) はどこか？

## 技術スタック

- Python 3.10+
- PyTorch
- transformer-lens（Neel Nanda の mechanistic interpretability ライブラリ）
- matplotlib, scipy, numpy
- 対象モデル: GPT-2 (small/medium)

## ディレクトリ構成

```
semantic-phase-transition/
├── README.md              # 実験計画と結果の全体像
├── CLAUDE.md              # このファイル
├── experiments/           # 実験スクリプト
│   ├── exp1_basic.py      # 実験1: 秩序変数の基本測定（初期版）
│   ├── exp1_basic_v2.py   # 実験1 v2: 改良版（対比的方向ベクトル）
│   ├── exp2_landau.py     # 実験2: ランダウフィットと logit lens
│   ├── exp3_linear_response.py  # 実験3: 線形応答の限界
│   ├── exp5_phase_diag.py # 実験5: 相図の作成
│   └── exp6_basis.py      # 実験6: 普遍基底との接続
├── src/                   # 共通コード
│   ├── order_parameter.py # σ(l) の計算
│   ├── direction.py       # 方向ベクトル ê_diff の取得
│   ├── prompts.py         # プロンプト定義（方向/実験/勾配）
│   ├── fitting.py         # フィット関数
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
- 方向ベクトル ê_diff は**最終レイヤー**の残差ストリームから対比的に取得（mean_A - mean_B を正規化）
- GPU メモリ: GPT-2 small は CPU でも動く。large は ~3GB VRAM 必要
- 実験の優先順位: ~~exp1 → exp2 → exp4~~ → **exp3(線形応答) → exp5(相図) → exp6(普遍基底)**
- **デバイス: M1 Mac では `--device mps` を使用する**（CPU比 2.7x〜4.4x 高速）

## 実験結果のステータス

| 実験 | ステータス | 主要結果 |
|------|-----------|---------|
| exp1 v2 | 完了 | 全9語で扇形分岐パターン確認。二段階構造を発見 |
| exp2 | 完了 | two_stage モデル最良。β非普遍的 → ランダウ仮説棄却 |
| exp3 | 進行中 | 線形応答 σ=h·f(l) の限界を検証 |
| exp5 | 未着手 | 文脈長×レイヤーの2D相図 |
| exp6 | 未着手 | 普遍基底との接続 |

## よく使うコマンド

```bash
# 環境構築
pip install torch transformers transformer-lens numpy matplotlib scipy

# 実験3を実行
python experiments/exp3_linear_response.py --model gpt2 --device mps
python experiments/exp3_linear_response.py --model gpt2-medium --device mps
```

## 関連研究のキーワード

検索用: Universal Weight Subspace Hypothesis, O(N) model LLM, phase transition training dynamics, effective depth, logit lens, residual stream, mechanistic interpretability, Landau theory, order parameter, semantic diffusion, linear response
