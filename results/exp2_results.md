# 実験2: ランダウフィット解析 — 結果レポート

## 実験条件

- **モデル**: GPT-2 small (12層), GPT-2 medium (24層)
- **デバイス**: MPS (Apple Silicon)
- **対象語**: 9語 (bank, bat, crane, spring, rock, match, light, pitcher, bass)
- **秩序変数**: σ(l) = cos(φ(l), ê_diff)（exp1 v2 と同一）
- **フィットデータ**: 条件平均 f(l) = mean[σ(l)/h] から最終1層を除外
- **フィットモデル**: power_law, tanh, sigmoid, two_stage の4モデル
- **モデル選択基準**: BIC (Bayesian Information Criterion)
- **実行日**: 2026-02-19

## 実験の3パート構成

### Part A: 応答関数 σ(l, h) = h · f(l) の検証
- 外部場 h を σ_final から定義: h(strong_A)=+1, h(strong_B)=-1, 他は線形内挿
- f(l) = σ(l)/h でスケーリング → 全条件が1本のユニバーサルカーブに崩壊するか検証
- **collapse variance** = f(l) 曲線間のレイヤー平均分散（小さいほど良い崩壊）

### Part B: f(l) のランダウフィット
- 4モデルで曲線フィット → AIC/BIC で最良モデルを選択
- 最終1層を除外（unembedding ジャンプの影響を分離）

### Part C: Logit Lens との比較
- 各レイヤーの残差ストリームに unembedding を適用し、top-1 予測トークンを追跡
- σ(l) の変化と語彙予測の変化の対応を確認

---

## Part A: データ崩壊の品質

### Collapse Variance（低いほど σ(l,h) = h·f(l) が成立）

| word | GPT-2 small | GPT-2 medium | 評価 |
|------|------------|-------------|------|
| spring | 0.0021 | 0.0379 | small で極めて良好 |
| bass | 0.0025 | 0.4579 | small のみ良好 |
| rock | 0.0041 | 0.0099 | **両モデルで良好** |
| bank | 0.0143 | 0.0406 | 良好 |
| bat | 0.0619 | 0.0091 | medium で良好 |
| crane | 0.0649 | 0.0713 | 中程度 |
| match | 0.0987 | 0.0644 | 中程度 |
| pitcher | 0.1242 | 0.0500 | 中〜低 |
| light | 0.2918 | 0.0092 | medium で良好、small で不良 |

**考察**: rock は両モデルで collapse variance が低く（0.004, 0.010）、σ(l, h) = h · f(l) の線形応答仮説を最も良く満たす。spring (small), bass (small), light (medium), bat (medium) も良好。ただし完全な崩壊は達成されておらず、非線形効果の存在を示唆する。

---

## Part B: ランダウフィット結果

### GPT-2 small (12層, 最終1層除外)

| word | Best Model | R² | BIC | 主要パラメータ |
|------|-----------|-----|------|---------------|
| bank | tanh | 0.984 | -114.0 | κ=0.10, l_c=0.09 |
| bat | sigmoid | 0.877 | -108.3 | κ=7.75, l_c=0.28 |
| crane | power_law | 0.989 | -124.7 | β=1.02 |
| spring | **two_stage** | **0.995** | -114.0 | β=0.89, l_c=0.86, w=2.85 |
| rock | two_stage | 0.978 | -101.5 | β=0.29, l_c=0.65, w=2.85 |
| match | power_law | 0.976 | -92.0 | β=0.74 |
| light | two_stage | 0.966 | -84.9 | β=2.35, l_c=1.00 |
| pitcher | power_law | 0.570 | -83.1 | β=2.77 |
| bass | two_stage | 0.980 | -112.5 | β=0.04, l_c=0.67, w=2.85 |

### GPT-2 medium (24層, 最終1層除外)

| word | Best Model | R² | BIC | 主要パラメータ |
|------|-----------|-----|------|---------------|
| bank | power_law | 0.916 | -166.3 | β=1.23 |
| bat | power_law | 0.772 | -167.5 | β=1.83 |
| crane | two_stage | 0.863 | -143.3 | β≈0, l_c=0.96, w=28.5 |
| spring | **two_stage** | **0.993** | -215.3 | β=1.28, l_c=0.93, w=29.5 |
| rock | sigmoid | 0.898 | -181.3 | κ=4.86, l_c=0.60 |
| match | two_stage | 0.976 | -183.3 | β=0.88, l_c=1.00, w=31.8 |
| light | **two_stage** | **0.996** | **-230.6** | β=1.74, l_c=1.00, w=3.48 |
| pitcher | two_stage | 0.981 | -205.5 | β=0.82, l_c=0.95, w=46.4 |
| bass | tanh | 0.954 | -125.0 | κ=0.19, l_c=0.43 |

### フィット品質のまとめ

**R² > 0.95 の高品質フィット**:
- GPT-2 small: spring(0.995), crane(0.989), bank(0.984), bass(0.980), rock(0.978), match(0.976), light(0.966)
- GPT-2 medium: light(0.996), spring(0.993), pitcher(0.981), match(0.976), bass(0.954)

**低品質 (R² < 0.9)**:
- GPT-2 small: bat(0.877), pitcher(0.570)
- GPT-2 medium: bat(0.772), crane(0.863), rock(0.898)

---

## 主要な発見

### 1. two_stage モデルの優位性

BIC 基準で最良モデルを集計すると:

| モデル | GPT-2 small | GPT-2 medium | 合計 |
|--------|-----------|-------------|------|
| two_stage | 4語 (spring, rock, light, bass) | 5語 (crane, spring, match, light, pitcher) | 9 |
| power_law | 3語 (crane, match, pitcher) | 2語 (bank, bat) | 5 |
| tanh | 1語 (bank) | 1語 (bass) | 2 |
| sigmoid | 1語 (bat) | 1語 (rock) | 2 |

**two_stage = power_law + sigmoid step** が最も多く選ばれている。これは exp1 v2 で見出された「漸進的分化 + 最終層ジャンプ」の二段階構造を直接反映している。

### 2. 臨界指数 β の分布

two_stage モデルが最良の場合の β 値:

| word | GPT-2 small β | GPT-2 medium β |
|------|--------------|----------------|
| spring | 0.89 | 1.28 |
| rock | 0.29 | — |
| light | 2.35 | 1.74 |
| bass | 0.04 | — |
| match | — | 0.88 |
| pitcher | — | 0.82 |
| crane | — | 0.01 |

- **平均場理論の予測 β = 0.5**: 一部の語（rock/small, spring/small, match/medium, pitcher/medium）で近いが、全体として大きな分散
- β の範囲: 0.01〜2.35 と広い → **単一の普遍的臨界指数は確認されず**
- power_law モデルの β: 0.56〜2.77 とさらに広い

### 3. 臨界層 l_c の位置

two_stage モデルの l_c:

| word | GPT-2 small l_c | GPT-2 medium l_c |
|------|----------------|------------------|
| spring | 0.86 | 0.93 |
| rock | 0.65 | (sigmoid: 0.60) |
| light | 1.00 | 1.00 |
| bass | 0.67 | (tanh: 0.43) |
| match | — | 1.00 |
| pitcher | — | 0.95 |
| crane | — | 0.96 |

- l_c ≈ 0.9-1.0 が多い → 二段階目のステップが最終層近くで起きることを確認
- **l_c/L の普遍性は部分的**: spring, pitcher, crane は 0.86-0.96 の範囲に集中するが、rock, bass はより早い位置

### 4. Logit Lens との対応

代表例（GPT-2 small, rock）:
- **strong_A (stone文脈)**: L0="rock" → L3="formations" → L9="formations" → L11="formations" → L12="samples"
- **strong_B (music文脈)**: L0="rock" → L2="ers" → L6="ers" → L12="songs"

**観察**:
- σ(l) の漸進的増加と、logit lens での予測トークンの文脈依存的変化が対応
- 最終層のジャンプ（L11→L12）で top-1 トークンが大きく変わるケースが多い → unembedding 再構成の証拠
- σ(l) の中間層での変化は、logit lens のトークン変化と必ずしも一致しない → σ は語彙空間ではなく意味空間の変化を捉えている

### 5. Cross-word Universality

GPT-2 small: スケーリング後の f(l) は大半の語で単調増加。ただし bat, pitcher は異なるプロファイルを示す。

GPT-2 medium: より滑らかだが、bass が他と大きく異なるプロファイル。spring, light, match は類似した上昇パターン。

**完全な普遍カーブは確認されず**。形状の多様性はあるが、大半が単調増加で共通の傾向を示す。

---

## 成功基準の評価

| 指標 | 閾値 | 結果 | 判定 |
|------|------|------|------|
| R² | > 0.9 | 大半の語で達成（7/9 small, 5/9 medium） | 部分的に達成 |
| β = 0.5 | ±0.3 | rock(0.29), spring(0.89), match(0.88) — 分散大 | **未達成** |
| β の変動 | < 0.3 | σ(β)/mean(β) >> 0.3 | **未達成** |
| σ(l,h) = h·f(l) | collapse var < 0.01 | rock(0.004), spring(0.002), bass(0.003) で達成 | 部分的 |

---

## 結論

### 1. 相転移の性質
- f(l) は **滑らかな単調増加** が主要パターン（二次相転移 / crossover）
- 最終層ジャンプは unembedding 効果と不可分（logit lens で確認）
- **一次相転移（鋭いジャンプ）は否定的** — 中間層での変化は連続的

### 2. ランダウ理論の適用性
- f(l) ∝ (l/L)^β 型のフィットは R² > 0.9 で形式的には良好
- しかし **β が単語・モデルごとに大きく変動** → 普遍的臨界指数は存在しない
- two_stage モデル（power_law + sigmoid step）が最も記述力が高い → ランダウ二次相転移よりも「漸進的成長 + 最終層ステップ」の現象論的記述が適切

### 3. 線形応答の成立
- σ(l, h) = h · f(l) は一部の語（rock, spring, bass）で良好に成立
- ただし全語で成立するわけではない → 非線形応答の存在

### 4. 次の実験への推奨

**実験4（感受率 χ(l) の測定）への移行を推奨**:
- 感受率 χ(l) = |∂σ/∂h| のレイヤー依存性を直接測定
- 臨界層付近で χ のピークが見えるか検証
- exp2 の f(l) から示唆される l_c ≈ 0.6-0.9 の範囲でピークが期待される

**代替案: 実験5（相図の作成）**:
- 文脈の長さを連続的に変化させ、σ(context_length, layer) の2D相図を作成
- 臨界線の形状から相転移の次数をより正確に判定可能

---

## 生成ファイル

### GPT-2 small
- `results/data/exp2_gpt2.json`
- `results/figures/exp2_fit_{word}_gpt2.png` (9語)
- `results/figures/exp2_collapse_{word}_gpt2.png` (9語)
- `results/figures/exp2_logitlens_{word}_{strong_A/B}_gpt2.png` (18枚)
- `results/figures/exp2_universality_gpt2.png`

### GPT-2 medium
- `results/data/exp2_gpt2-medium.json`
- `results/figures/exp2_fit_{word}_gpt2-medium.png` (9語)
- `results/figures/exp2_collapse_{word}_gpt2-medium.png` (9語)
- `results/figures/exp2_logitlens_{word}_{strong_A/B}_gpt2-medium.png` (18枚)
- `results/figures/exp2_universality_gpt2-medium.png`
