# 意味の浸透実験：LLMレイヤー内における意味確定のダイナミクス

## 概要

LLMが曖昧な単語を含む文を処理する際、意味がどのレイヤーで・どのように
確定するかを実験的に検証するプロジェクト。

当初はランダウ相転移理論の枠組みで「臨界レイヤーでの不連続な意味確定」
を仮説として出発したが、実験1-2の結果、意味確定は**鋭い相転移ではなく、
文脈情報がレイヤーを通じて漸進的に浸透する過程**であることが明らかに
なった。現在は**意味浸透モデル (Semantic Diffusion)** の枠組みで、
この過程の定量的記述を進めている。

### 仮説の変遷

**当初の仮説（棄却）**:
> Transformerの各レイヤーを通過する過程で、曖昧な単語の意味表現は
> ランダウ型相転移に従い、ある臨界レイヤー l_c で不連続的に確定する。

**現在の仮説（意味浸透モデル）**:
> 文脈情報は各レイヤーで対象トークンの表現に漸進的に浸透し、
> 意味の確定度 σ(l) は文脈強度 h に対して線形応答 σ(l,h) = h·f(l) に
> 従う。f(l) は単語カテゴリに依存する単調増加関数であり、
> 最終層付近で unembedding 再構成による追加的ジャンプを伴う。

### 背景と動機

- **Universal Weight Subspace Hypothesis (2025/12)**: 1100以上のモデルの重み行列に共通の低次元部分空間が存在
- **Phase Transitions in LLM Output (2024/05)**: 学習中の相転移が物理学的手法で検出
- **Effective Depth研究 (2025/12)**: レイヤー中間付近で相転移的な振る舞いが確認
- **O(N)模型としてのLLM (2025/01)**: TransformerをO(N)場の理論として再定式化する試み

**本研究の位置づけ**: 上記は全て「学習時」や「モデルサイズ」の相転移。
本研究は「推論中に、レイヤーを変数として起きる意味確定の過程」を
秩序変数で初めて定量的に追跡し、その数理的構造を明らかにした。

---

## 実験結果のサマリー

### 実験1: 秩序変数の基本測定（完了）

**目的**: 曖昧語の各レイヤーにおける秩序変数 σ(l) を測定し、意味確定のダイナミクスを可視化。

**方法**:
- 秩序変数: σ(l) = cos(φ(l), ê_diff)（対比的方向ベクトルとの内積）
- 対象: 9語 (bank, bat, crane, spring, rock, match, light, pitcher, bass)
- 各語5条件: strong_A, weak_A, neutral, weak_B, strong_B
- モデル: GPT-2 small (12層), GPT-2 medium (24層)

**主要な発見**:
1. **扇形分岐パターン**: 全9語で、L0付近から始まり文脈の強さに応じて扇状に分岐（strong_A > weak_A > neutral > weak_B > strong_B）
2. **二段階構造**: 中間層(L0-L10/L22)での漸進的分化 + 最終層(l/L≈0.92)でのジャンプ
3. **対比的方向ベクトルの有効性**: σ range が v1 比で 4-13倍に改善

### 実験2: ランダウフィットと logit lens（完了）

**目的**: σ(l) がランダウ理論の予測する関数形に従うかを定量的に検証。

**方法**:
- 応答関数 f(l) = σ(l)/h を抽出し、4モデル（power_law, tanh, sigmoid, two_stage）でフィット
- BIC でモデル選択
- Logit lens で unembedding 効果との対応を確認

**主要な発見**:
1. **ランダウ仮説は棄却**: β は 0.01〜2.35 と大きく分散。普遍的臨界指数は存在しない
2. **two_stage モデルが最良**: 大半の語で power_law + sigmoid step が最良フィット → 二段階構造を定量的に裏付け
3. **線形応答の部分的成立**: rock, spring, bass で collapse variance < 0.005（σ=h·f(l) が良好）
4. **最終層ジャンプ = unembedding 再構成**: logit lens の top-1 トークン変化と対応

**最良フィット結果** (R² > 0.99):

| word | model | GPT-2 small R² | GPT-2 medium R² |
|------|-------|---------------|----------------|
| spring | two_stage | 0.995 | 0.993 |
| light | two_stage | 0.966 | 0.996 |

---

## 実験3: 線形応答の限界（進行中）

### 目的

σ(l, h) = h · f(l) の線形応答が成立する範囲と、
非線形領域への逸脱点 h*(l) を特定する。

### 実験設計

#### 3A. 文脈強度の連続変化
4語 (rock, spring, bass, light) について、各解釈方向に10段階の
文脈強度を持つプロンプト系列を作成（計80プロンプト）。

#### 3B. σ(l) vs h プロット
各レイヤーで σ(l) vs h をプロットし、線形回帰の R² と残差から
線形領域と非線形領域の境界を特定。

#### 3C. h*(l) のレイヤー依存性
非線形閾値 h*(l) が最小になるレイヤーが「意味処理の臨界点」。

#### 3D. 語間比較
f(l) 形状の語間差から、具体的/感覚的な語 vs 抽象的な語での
浸透ダイナミクスの違いを検証。

---

## 実験5: 文脈「温度」による相図の作成（計画中）

文脈の長さを連続的に変化させ、σ(context_length, layer) の2D相図を作成。
臨界線の形状から相転移の次数をより正確に判定。

---

## 実験6: 普遍基底との接続（計画中）

Universal Weight Subspace Hypothesis で見つかった普遍基底と、
秩序変数の方向ベクトルの関係を調べる。

---

## 解析と判定基準

### 成功基準（更新版）

| 指標 | 閾値 | exp1-2 結果 | 判定 |
|------|------|------------|------|
| σ(l) の扇形分岐 | 全条件で順序が保持 | 9/9語で確認 | **達成** |
| フィット R² | > 0.9 | 大半の語で達成 | **達成** |
| σ=h·f(l) の collapse | var < 0.01 | 3/9語で達成 | **部分的** |
| β の普遍性 | σ(β)/mean < 0.3 | 大きく分散 | **未達成** |
| l_c/L の普遍性 | モデル間分散 < 0.1 | — | **該当なし** |

### 結果の解釈

```
発見された現象:
  → 意味確定は「相転移」ではなく「浸透」過程
  → 文脈強度に対する線形応答が部分的に成立
  → 最終層ジャンプは unembedding 再構成と不可分

次に検証すべきこと:
  → 線形応答が破れる閾値 h*(l) のレイヤー依存性
  → 文脈長を連続変化させた場合の2D相図
  → モデルアーキテクチャ間での f(l) 形状の比較
```

---

## 参考文献

1. Kaushik et al., "The Universal Weight Subspace Hypothesis" (arXiv:2512.05117, Dec 2025)
2. Sun et al., "Phase Transitions in Large Language Models and the O(N) Model" (arXiv:2501.16241, Jan 2025)
3. "Phase Transitions in the Output Distribution of Large Language Models" (arXiv:2405.17088, May 2024)
4. "Decomposing Behavioral Phase Transitions in LLMs" (arXiv:2508.20015, Aug 2025)
5. Hu et al., "What Affects the Effective Depth of Large Language Models?" (arXiv:2512.14064, Dec 2025)
6. Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models" (Mar 2025)
7. Anthropic, "Tracing Attention Computation Through Feature Interactions" (Jul 2025)
8. Anthropic, "On the Biology of a Large Language Model" (Mar 2025)
9. "The Circuits Research Landscape: Results and Perspectives" - Neuronpedia (Aug 2025)
