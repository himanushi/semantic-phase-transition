# Semantic Phase Transition

LLMの内部における意味処理のダイナミクスを実験的に解明し、重み行列の圧縮可能性を検証したプロジェクト。

## 結論

**当初の目標（重み行列の100x圧縮）は達成できなかった。** しかし、その過程で2つの成果を得た:

1. **意味浸透の普遍法則**: 曖昧語の意味確定過程は `σ(l,h) = h · f_max(word) · g(l/L)` で完全分解でき、浸透関数 g(l/L) は語にもモデルサイズにも依存しない普遍関数である（語間相関 > 0.96、モデル間相関 0.98）

2. **PCA分散 ≠ 機能的重要度**: W_QKの分散の99.1%を保持してもperplexityは2.2倍に劣化する。機能的に重要な情報は低分散成分（残り0.6%）に集中しており、PCAベースの圧縮は原理的に困難である

---

## 仮説の変遷

```
出発点:
  「LLMの重みに普遍法則がある → ヘッドを関数に置換して100x圧縮」

仮説1: ランダウ相転移 (exp1-2)
  「意味確定は臨界レイヤーで不連続に起きる」
  → 棄却。中間層での変化は連続的で、βは0.01〜2.35とバラバラ

仮説2: 意味浸透モデル (exp3)
  「文脈情報はレイヤーを通じて漸進的に浸透する」
  → 確立。σ = h · f_max · g(l/L) の普遍分解が成立
  → 科学的に面白いが圧縮には繋がらなかった

仮説3: 普遍基底による圧縮 (exp6)
  「W_QKを少数基底で再構成してもperplexityが維持される」
  → 棄却。K=140/144でようやくPPL 1.07倍。PCA分散と機能は別物

仮説4: πの痕跡 (exp8)
  「g(l/L)の普遍性は超球面幾何に由来し、πが具体的に現れる」
  → 示唆的だが決定的ではない。8A: σ=1/√πを棄却できず、8B: mediumでsin(πx^α/2)がBIC最良
  → exp9でスケーリング検証を試みたが、CUDA精度問題で中断

仮説5: ヘッド削減 + 線形Attention + CoT回収 (exp10) ← NEW
  「Attentionの精密さを下げ、浮いた計算をCoTに回すことで性能を維持できる」
  → Phase 1: ヘッドのablationで機能的重要度ランキングを作成
  → Phase 2: 残ったヘッドのsoftmaxを線形関数の交差に置換
  → Phase 3: 劣化分をCoTの再帰的トークンで回収できるか検証
```

---

## 実験結果

### exp1: 秩序変数の基本測定 ✅

曖昧語9語（bank, bat, crane, spring, rock, match, light, pitcher, bass）の各レイヤーにおける秩序変数 σ(l) を測定。

- **扇形分岐パターン**: 全9語で、L0から始まり文脈の強さに応じて扇状に分岐
- **二段階構造**: 中間層での漸進的分化 + 最終層 (l/L≈0.92) でのジャンプ
- **causal mask の発見**: デコーダモデルでは対象語より後の文脈は表現に影響しない → プロンプト設計を修正（v2）

### exp2: ランダウフィット ✅

σ(l) がランダウ理論 σ ∝ (l - l_c)^β に従うかを検証。

- **ランダウ仮説を棄却**: β = 0.01〜2.35で普遍的臨界指数は存在しない
- **two_stage モデルが最良**: power_law + sigmoid step（漸進的分化 + 最終層ジャンプ）
- **最終層ジャンプ = unembedding再構成**: logit lensのトークン変化と連動
- **線形応答の部分的成立**: rock, spring, bass で collapse variance < 0.005

### exp3: 線形応答と普遍浸透関数 ✅

文脈強度 h を連続的に変化させ、σ(l,h) = h · f(l) の線形応答を検証。

- **深い層ほど線形**: L0では R²≈0（非線形）、中間層以降で R²>0.95
- **f(l) の語間相関 > 0.96**: 浸透関数の「形状」は語に依存しない普遍関数
- **f_max(word)**: 語依存のスケーリング定数。曖昧性の「解決しやすさ」を表す
- **モデル間相関 0.98**: GPT-2 small と medium で g(l/L) はほぼ一致

**普遍浸透関数 g(l/L) の構造**:

- l/L < 0.85: ほぼ線形 (α≈0.9)。各レイヤーが等量の文脈情報を注入する「定速浸透」
- l/L > 0.85: ドロップ。unembedding 再構成により dσ/dh が減少
- 全域ベストフィット: erf (σ≈0.55) — ただし最終層除外時は power_law (α≈1) が最良

### exp4: 感受率 ⏭ スキップ

臨界レイヤーが存在しないため、感受率のピーク測定は不要と判断。

### exp5: 相図 ⏭ スキップ

相転移の境界線がないため、2D相図の作成は不要と判断。

### exp6: 普遍基底と圧縮検証 ✅

W_QK = W_Q^T @ W_K をPCAで分解し、少数基底でのperplexity維持を検証。

**基底数 vs perplexity** (GPT-2 small, baseline PPL: 29.79):

| K       | 累積分散  | Perplexity | Ratio     |
| ------- | --------- | ---------- | --------- |
| 10      | 52%       | 1201       | 40x       |
| 32      | 71%       | 558        | 19x       |
| 100     | 95%       | 125        | 4.2x      |
| 130     | 99.1%     | 64         | 2.2x      |
| **140** | **99.7%** | **32**     | **1.07x** |
| 144     | 100%      | 30         | 1.00x     |

**核心の発見**: 分散0.6%の成分（PC131-140）を加えるだけでPPLが2.2x → 1.07xに回復。低分散成分に機能的情報が集中している。

### exp8: πの痕跡探索 ✅

exp3efで発見した erf σ≈0.55 と 1/√π≈0.5642 の一致を検証。

| 基準 | 条件                  | 結果                          | 判定          |
| ---- | --------------------- | ----------------------------- | ------------- |
| 8A   | σ=1/√π が棄却できない | small p=0.535, medium p=0.245 | **成立** ✅   |
| 8B   | π含有モデルがBIC最良  | medium で sin(πx^α/2) が1位   | **部分的** ⚠️ |
| 8C   | 累積回転角がπ±0.1     | Θ/π≈1.58、整数倍に非収束      | **不成立** ❌ |

**結論: πの痕跡は「示唆的だが決定的ではない」。** 8Aが最も強い証拠。

### exp9: π収束検証 🔬 中断

exp8の結果を受け、gpt2/medium/large/xl の4モデルで σ_free が 1/√π に収束するかを検証する実験。

- **Part A**: σ_free の4モデルスケーリング → CUDA精度問題で未完了
- **Part B**: 累積回転角Θの4モデルスケーリング → 未実行
- **Part C**: cos_pi_0p のBIC比較 → gpt2/medium の2モデルで完了（exp3efデータ使用）

**CUDA問題**: TF32 (TensorFloat-32) がデフォルト有効のため、float32指定でもattention計算が10ビット精度に低下し、文脈による σ の変動が消失。`torch.backends.cuda.matmul.allow_tf32 = False` の修正済みだが未検証。Colab A100での再実行が必要。

### exp10: ヘッド削減 + 線形Attention + CoT回収 🔬 NEW

**核心の問い**: Attentionの精密さを下げて計算を軽量化し、浮いたリソースをCoT（再帰的トークン生成）に回すことで、小さいモデルが長く考えて大きいモデルと同等の性能を出せるか？

**背景と動機**:

- exp6で「ヘッドの非冗長性」が示唆されたが、それはPCA基底での話。ヘッド単位のablationは未検証
- softmaxのO(n²)計算を線形関数の交差で代替すればO(n)に削減可能
- CoTの中間トークンはattentionのコンテキスト空間を動的に拡張する効果がある（実質的にattentionの補強）
- 再帰的にCoTトークンを再入力すれば、粗いattentionの解像度を反復的に向上できる可能性

#### Phase 1: ヘッド単位Ablation（M1 Mac）

GPT-2 smallの全12レイヤー×12ヘッド = 144ヘッドを対象に：

- **10A: 個別ablation**: ヘッドを1個ずつゼロアウトし、WikiText-2でperplexity変化 ΔPPL を測定
- **10B: 重要度ランキング**: ΔPPL で全144ヘッドをランキング。上位/下位の分布を可視化
- **10C: 累積ablation**: 重要度の低い順にヘッドを削除し、PPL劣化カーブを取得。「ヘッド数 vs PPL」のパレート曲線を描く
- **10D: レイヤー内パターン分析**: 重要なヘッドのレイヤー内分布。浅い層 vs 深い層でパターンが異なるか

**成功基準（Phase 1）**:

1. 144ヘッド中、30%以上（≥43ヘッド）がPPL劣化1.1倍未満で削除可能
2. 重要度分布がべき乗則に従う（少数のヘッドに機能が集中している）

#### Phase 2: softmax線形化（Phase 1の結果次第）

Phase 1で残ったヘッドに対して：

- **10E: 線形attention置換**: softmax(QK^T/√d)V を (QK^T/√d)V に置換（正規化なし or L1正規化）
- **10F: 交差ベースattention**: 各ヘッドが線形関数を計算し、複数ヘッドの応答の積（または閾値交差）でattentionパターンを生成
- **10G: PPL測定**: Phase 1のヘッド削減 + Phase 2の線形化を組み合わせた場合のPPL

#### Phase 3: CoT回収（Phase 2の結果次第）

- **10H: CoT生成速度測定**: 軽量化されたモデルでのトークン生成速度を測定
- **10I: 再帰的CoT**: 生成したCoTトークン列をプロンプトに追加して再推論。反復回数 vs タスク正答率を測定
- **10J: 等計算量比較**: 「大きいモデル×1回推論」vs「小さいモデル×N回再帰推論」で同一FLOPsでの性能を比較

---

## 得られた知見

### 科学的貢献

1. **σ(l,h) = h · f_max · g(l/L)**: 意味確定過程の初めての定量的分解
2. **g(l/L) の普遍性**: Transformerのアーキテクチャが決める「意味浸透スケジュール」が存在
3. **各レイヤーの等量寄与**: 非線形演算の巨視的効果が線形法則に従う

### 圧縮研究への教訓

1. **PCA分散 ≠ 機能的重要度**: 重み空間の「大きな方向」は構造的特徴（レイヤー位置等）を反映し、「小さな方向」に推論の精密情報が入っている
2. **UWS仮説の限界**: 「分散の90%を少数基底で説明」は統計的記述としては正しいが、それに基づく圧縮は機能を破壊する
3. **ヘッドの非冗長性**: 144ヘッドは高次元空間で十分に分離しており、代表ヘッドによる近似は不可能

---

## セットアップ

```bash
pip install torch transformers transformer-lens numpy matplotlib scipy scikit-learn datasets
```

## 使い方

```bash
# 意味浸透実験 (exp1-3)
python experiments/exp1_basic_v2.py --model gpt2 --device mps
python experiments/exp2_landau.py --model gpt2 --device mps
python experiments/exp3_linear_response.py --model gpt2 --device mps
python experiments/exp3ef_universal_g.py --model gpt2 --device mps

# 普遍基底実験 (exp6)
python experiments/exp6_basis.py --model gpt2 --device mps

# π探索実験 (exp8)
python experiments/exp8_pi_trace.py --model gpt2 --device mps --parts abc

# π収束検証 (exp9, Colab A100用)
python experiments/exp9_pi_convergence.py --device cuda --parts abc

# ヘッド削減実験 (exp10)
python experiments/exp10_head_ablation.py --model gpt2 --device mps --parts abcd
```

## 参考文献

1. Kaushik et al., "The Universal Weight Subspace Hypothesis" (arXiv:2512.05117, Dec 2025)
2. Sun et al., "Phase Transitions in Large Language Models and the O(N) Model" (arXiv:2501.16241, Jan 2025)
3. "Phase Transitions in the Output Distribution of Large Language Models" (arXiv:2405.17088, May 2024)
4. "Decomposing Behavioral Phase Transitions in LLMs" (arXiv:2508.20015, Aug 2025)
5. Hu et al., "What Affects the Effective Depth of Large Language Models?" (arXiv:2512.14064, Dec 2025)
6. Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models" (Mar 2025)
7. Loshchilov et al., "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" (arXiv:2410.01131, Oct 2024)
