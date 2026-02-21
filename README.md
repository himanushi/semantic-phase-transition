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

仮説4: πの痕跡 (exp8) ← NEW
  「g(l/L)の普遍性は超球面幾何に由来し、πが具体的に現れる」
  → 検証中。exp3efでerf最良フィットのσ≈0.55、1/√π≈0.564との一致を検定
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

### exp8: πの痕跡探索 🔬 NEW

**動機**: exp3efで発見した普遍浸透関数 g(l/L) のerfフィットにおいて σ≈0.55。一方 1/√π≈0.5642。この一致は偶然か、超球面幾何の帰結か？

**理論的背景**:

Transformerの埋め込み空間は高次元超球面とみなせる（nGPT, NVIDIA 2024で実証）。球面上のトークン移動・回転・測地線にはπが幾何学的に関与する。もし g(l/L) が球面拡散方程式の解であれば、erf の特性パラメータに π が自然に出現するはず。

**πが現れうる経路**:

1. **erf のσパラメータ**: 球面拡散 → σ = 1/√π (正規化条件)
2. **RoPE の回転**: 位置エンコーディング e^(iπθ) の周期構造
3. **測地線長**: レイヤー間回転の累積角がπの整数倍
4. **cos(πx) フィット**: g(l/L) ∝ [1-cos(π·l/L)]/2 (半周期)

**実験計画**:

#### exp8A: erf σ パラメータの π 検定

exp3ef の既存データを再解析。σ_free (自由パラメータ) vs σ_fixed = 1/√π を尤度比検定。

- H₀: σ = 1/√π（πが本質的）
- H₁: σ ≠ 1/√π（一致は偶然）
- 手法: 尤度比検定 + AIC/BIC比較
- **事前に定めた閾値**: p > 0.05 で H₀ を棄却できない → πとの整合性を支持

#### exp8B: cos(πx) モデルの追加フィット

g(l/L) の候補モデルにπ含有型を追加:

- `g(x) = [1 - cos(π·x)] / 2` (半周期コサイン, 0パラメータ)
- `g(x) = [1 - cos(π·x^α)] / 2` (1パラメータ)
- `g(x) = sin(π·x / 2)` (0パラメータ)
- `g(x) = sin(π·x^α / 2)` (1パラメータ)

既存の power_law, erf, sigmoid 等との BIC 比較。0パラメータモデルがBIC最良なら、πが g(l/L) の本質。

#### exp8C: レイヤー間回転角の測定

残差ストリームの幾何学的軌跡を測定:

- 各レイヤーでの回転角 θ(l) = arccos(cos(resid_post(l), resid_post(l-1)))
- 累積回転角 Θ(L) = Σθ(l) がπの有理数倍に近いか
- 曖昧語 vs 非曖昧語で θ(l) のパターンに差があるか
- WikiText-2 の統計的分析

#### 成功基準（事前登録）

以下の**いずれか1つ**が成立すれば「πの痕跡あり」と判定:

1. erf σ = 1/√π が p > 0.05 で棄却できない（8A）
2. π含有0パラメータモデルが既存モデルよりBIC良好（8B）
3. 累積回転角がπ±0.1以内に集中（8C, 語の90%以上）

**いずれも不成立**なら「πは明示的には現れない」と結論。

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
```

## 参考文献

1. Kaushik et al., "The Universal Weight Subspace Hypothesis" (arXiv:2512.05117, Dec 2025)
2. Sun et al., "Phase Transitions in Large Language Models and the O(N) Model" (arXiv:2501.16241, Jan 2025)
3. "Phase Transitions in the Output Distribution of Large Language Models" (arXiv:2405.17088, May 2024)
4. "Decomposing Behavioral Phase Transitions in LLMs" (arXiv:2508.20015, Aug 2025)
5. Hu et al., "What Affects the Effective Depth of Large Language Models?" (arXiv:2512.14064, Dec 2025)
6. Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models" (Mar 2025)
7. Loshchilov et al., "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" (arXiv:2410.01131, Oct 2024)
