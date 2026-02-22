# Experiment 10: Head Ablation — Phase 1 Results

## Summary

GPT-2 small (12 layers x 12 heads = 144 heads) の全ヘッドに対して個別・累積ablationを実施。
**機能的重要度は L0（第1層）に極度に集中** しており、べき乗則 (α=1.37) に従う。
個別ablationでは 136/144 ヘッドが |ΔPPL| < 10% だが、累積ablationでは効果が急速に複合し、PPL 1.1x 以下で削除可能なのは 10 ヘッドのみ。

## Experimental Setup

- Model: GPT-2 small (12L × 12H, 124M params)
- Dataset: WikiText-2 validation set, 2048 tokens (2 chunks × 1024)
- Baseline PPL: 24.27
- Ablation method: `hook_z` (W_O 射影前の per-head 出力) をゼロに置換
- Device: M1 Mac (MPS)
- Runtime: Part A 62s, Part C 9s

---

## 10A: Individual Head Ablation

各ヘッドを1個ずつゼロアウトし、PPL変化を測定。

**Top 10 most important heads:**

| Rank | Head   | ΔPPL     | Ablated PPL |
|------|--------|----------|-------------|
| 1    | L0H8   | +236.5%  | 81.65       |
| 2    | L0H1   | +73.4%   | 42.08       |
| 3    | L0H11  | +67.8%   | 40.73       |
| 4    | L0H10  | +54.7%   | 37.55       |
| 5    | L0H5   | +24.7%   | 30.26       |
| 6    | L0H9   | +15.9%   | 28.12       |
| 7    | L11H0  | +15.6%   | 28.06       |
| 8    | L0H3   | +13.1%   | 27.45       |
| 9    | L8H2   | +7.2%    | 26.02       |
| 10   | L0H6   | +7.2%    | 26.02       |

**Top 10 のうち 8 個が L0**。L0H8 は単独で PPL を 3.4 倍にする。

**Heads with negative ΔPPL (removal improves PPL):**

| Head | ΔPPL    |
|------|---------|
| L2H5 | -0.38%  |
| L4H1 | -0.34%  |
| L3H1 | -0.26%  |
| L4H8 | -0.22%  |
| L1H1 | -0.17%  |
| L8H5 | -0.05%  |

6 ヘッドは削除すると PPL が僅かに改善。これらは推論に有害な干渉をしている可能性がある。

---

## 10B: Importance Distribution

### Statistics

| Metric | Value |
|--------|-------|
| |ΔPPL| < 1% (negligible) | 78/144 (54.2%) |
| |ΔPPL| < 10% | 136/144 (94.4%) |
| ΔPPL < 0 (improved) | 6/144 (4.2%) |
| Mean ΔPPL | +4.6% |
| Median ΔPPL | +0.9% |
| Max ΔPPL | +236.5% (L0H8) |

### Power Law Fit

ΔPPLのランク分布は Zipf 則 `|ΔPPL|(rank) = C · rank^(-α)` に従う:

- **α = 1.37** (C = 2.35)
- α > 1 は少数ヘッドへの強い機能集中を意味する

### Heatmap

![Heatmap](figures/exp10b_heatmap_gpt2.png)

L0 が圧倒的に「赤い」（重要）。L1-L11 はほぼ均一に「青い」（低重要度）。
特に L0H8 (ΔPPL=2.365) が突出。

---

## 10C: Cumulative Ablation

重要度の低い順にヘッドを累積削除した場合の PPL 変化:

| Heads removed | PPL    | Ratio  |
|---------------|--------|--------|
| 0             | 24.27  | 1.000x |
| 5             | 24.37  | 1.004x |
| 10            | 26.29  | 1.083x |
| 15            | 26.80  | 1.104x |
| 20            | 28.25  | 1.164x |
| 25            | 34.14  | 1.407x |
| 30            | 37.32  | 1.538x |
| 35            | 46.52  | 1.917x |
| 40            | 48.95  | 2.017x |
| 50            | 64.20  | 2.646x |
| 70            | 198.84 | 8.193x |
| 100           | 600.83 | 24.76x |
| 144           | 9064.9 | 373.5x |

### Threshold Analysis

| Threshold | Max heads removable | Percentage |
|-----------|--------------------:|------------|
| PPL ≤ 1.1x | 10 | 6.9% |
| PPL ≤ 1.5x | 25 | 17.4% |
| PPL ≤ 2.0x | 35 | 24.3% |

### Individual vs Cumulative の乖離

個別ablationでは 136/144 (94.4%) が |ΔPPL| < 10% だが、累積では 10 ヘッド削除で既に 1.08x。
**ヘッド間の機能的相互依存**が強く、個別の非冗長性が累積の冗長性を保証しない。

![Pareto](figures/exp10c_pareto_gpt2.png)

---

## 10D: Layer-wise Pattern Analysis

### Layer importance

| Layer | Mean ΔPPL | Max ΔPPL | Note |
|-------|-----------|----------|------|
| **L0**  | **+41.5%** | **+236.5%** | **圧倒的** |
| L1    | +1.4%     | +5.6%    | |
| L2    | +1.3%     | +3.0%    | |
| L3    | +0.6%     | +1.9%    | 最も低い |
| L4    | +0.9%     | +3.9%    | |
| L5    | +1.8%     | +4.7%    | |
| L6    | +1.2%     | +2.7%    | |
| L7    | +1.1%     | +2.5%    | |
| L8    | +1.5%     | +7.2%    | |
| L9    | +1.1%     | +6.5%    | |
| L10   | +1.2%     | +3.1%    | |
| L11   | +1.9%     | +15.6%  | L11H0 が突出 |

**L0 の mean ΔPPL は L1-L11 の平均の 30 倍以上。**

L0 を除外すると、弱い U 字型パターンが見える: L1-L3 (前半) と L11 (最終層) が比較的重要で、L3-L4 (中間) が最も低重要度。

### Correlation with g(l/L)

exp3 の浸透関数 g(l/L) の傾き dg/d(l/L) との相関:

- **Pearson r = 0.24, p = 0.45** (有意でない)
- **Spearman ρ = -0.14, p = 0.66** (有意でない)

L0 が外れ値として支配的であり、意味浸透の傾きとヘッド重要度の間に直接的な対応は見られない。
g(l/L) は「意味情報が各レイヤーで追加される速度」を測るが、ヘッドの機能的重要度はそれとは異なるメカニズムで決まる。

![Layer importance](figures/exp10d_layer_importance_gpt2.png)

---

## Success Criteria Evaluation

| # | Criterion | Result | Verdict |
|---|-----------|--------|---------|
| 1 | ≥30% heads at |ΔPPL| < 10% (individual) | 136/144 = 94.4% | **PASS** |
| 2 | Power law α > 0.5 | α = 1.37 | **PASS** |
| 3 | g(l/L) correlation \|r\| > 0.5 | r = 0.24 | **FAIL** |

**ただし、基準1は個別ablationの結果であり、累積ablationでは 10 ヘッド (6.9%) しか削除できない。**
Phase 1 の「30% 削除可能」の達成は、個別基準では成立するが累積基準では不成立。

---

## Key Insights

### 1. L0 の特殊性

L0 は他のレイヤーとは質的に異なる機能を持つ。推測される役割:
- **位置情報の初期符号化**: Attention pattern が位置的に鋭いピークを持ち、トークン位置の基本構造を確立
- **L0H8**: 単独で PPL を 3.4x にする「最重要ヘッド」。おそらく前方参照 (previous token) パターン

### 2. Individual ≠ Cumulative

exp6 の「PCA 分散 ≠ 機能的重要度」と同じ教訓がヘッドレベルでも成立:
- 個別に見れば大半のヘッドは「不要」に見える
- しかし累積除去すると効果が複合する
- **冗長性の幻想**: 各ヘッドは微小な寄与をしているが、その合計は不可欠

### 3. Phase 2 への示唆

累積ablationで PPL 2.0x 以下に収めるには、最大 35 ヘッド (24.3%) の削除が限界。
これは Phase 2 (softmax 線形化) で以下を意味する:
- 線形化の対象は 35 ヘッド程度に限定すべき
- 残り 109 ヘッドは softmax を維持する必要がある（or ファインチューニングで回収）
- 「粗いAttention + CoT」戦略は、期待したほど大胆なAttention削減はできない可能性

---

## Caveats

- **サンプルサイズ**: 2048 トークン (2 chunks) は PPL 推定のノイズが大きい。より信頼性の高い結果には 50K+ トークンで再実行が望ましい
- **Baseline PPL**: 24.27 (validation 2048 tokens) vs ~29.79 (test 200K tokens, exp6) — サンプルの違いによる差
- **Ablation method**: zero-ablation (出力を 0 に置換) は mean-ablation (平均出力に置換) より保守的な推定を与える可能性がある

---

## Files

- Data: `results/data/exp10a_gpt2.json`, `results/data/exp10c_gpt2.json`, `results/data/exp10_gpt2.json`
- Figures: `results/figures/exp10b_*.png`, `results/figures/exp10c_*.png`, `results/figures/exp10d_*.png`
- Script: `experiments/exp10_head_ablation.py`
