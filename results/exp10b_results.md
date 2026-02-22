# Experiment 10 Phase 1.5: Mean-Ablation Results

## Summary

Phase 1 の zero-ablation は「ヘッドを完全に沈黙させる」上界推定。
Phase 1.5 では mean-ablation（ヘッド出力を平均ベクトルに置換）を用い、**softmax 線形化のより現実的な下限推定**を行った。

**主要結論**: Mean-ablation により削除可能ヘッド数は Phase 1 の **2.5 倍** に改善。
L0 を保護し mean-ablation を使えば、**PPL 1.1x 以内で 25 ヘッド (19%)、1.5x 以内で 50 ヘッド (38%)、2.0x 以内で 60 ヘッド (45%)** を粗視化できる。
Phase 2（線形化）に進む価値は十分にある。

## Experimental Setup

- Model: GPT-2 small (12L × 12H, 124M params)
- Dataset: WikiText-2 validation set, 2048 tokens (2 chunks × 1024)
- Baseline PPL: 24.27
- Mean-ablation method: `hook_z` をデータセット全体の平均ベクトルに置換
- L0 protection: L0 の 12 ヘッドは ablation 対象外（残り 132 ヘッドが対象）
- Device: M1 Mac (MPS)

---

## 10E: Individual Mean-Ablation

各ヘッドの出力を平均ベクトルに置換し、PPL 変化を測定。Zero-ablation との比較:

### Statistics

| Metric | Mean-ablation | Zero-ablation |
|--------|:------------:|:-------------:|
| \|ΔPPL\| < 1% (negligible) | **117/144 (81.3%)** | 78/144 (54.2%) |
| \|ΔPPL\| < 10% | **141/144 (97.9%)** | 136/144 (94.4%) |
| ΔPPL < 0 (improved) | **29/144 (20.1%)** | 6/144 (4.2%) |
| Mean ΔPPL | **+1.50%** | +4.62% |
| Median ΔPPL | **+0.42%** | +0.90% |
| Max ΔPPL | +108.9% (L0H1) | +236.5% (L0H8) |

Mean-ablation は平均 ΔPPL を **0.33x** に低減。29 ヘッド（20%）で PPL が改善する。

### Top 10 most important heads (mean-ablation)

| Rank | Head    | ΔPPL (mean) | ΔPPL (zero) | Ratio |
|------|---------|:-----------:|:-----------:|:-----:|
| 1    | L0H1    | +108.9%     | +73.4%      | 1.48  |
| 2    | L0H5    | +13.8%      | +24.7%      | 0.56  |
| 3    | L0H3    | +13.3%      | +13.1%      | 1.02  |
| 4    | L4H11   | +5.0%       | +3.9%       | 1.28  |
| 5    | L9H11   | +4.3%       | +6.5%       | 0.66  |
| 6    | L0H10   | +3.6%       | +54.7%      | 0.07  |
| 7    | L0H9    | +3.2%       | +15.9%      | 0.20  |
| 8    | L8H10   | +2.6%       | +1.4%       | 1.86  |
| 9    | L8H2    | +2.4%       | +7.2%       | 0.33  |
| 10   | L1H11   | +2.4%       | +5.6%       | 0.43  |

### L0H8 の劇的な変化

**L0H8** は zero-ablation では最重要ヘッド (ΔPPL=+236.5%) だったが、mean-ablation では **ΔPPL=+1.08%** に激減（**219x の低減**）。

**解釈**: L0H8 の機能は主に**定常的バイアス**（平均出力）にあり、トークン間の差分的 attention パターンへの依存は小さい。出力を平均値に固定しても、その主要機能（おそらく位置情報の初期符号化）はほぼ維持される。

### Sign-flipped heads

28 ヘッドが zero→mean で符号反転（zero では有害だが mean では有益、またはその逆）:

| Head | ΔPPL (zero) | ΔPPL (mean) | 解釈 |
|------|:-----------:|:-----------:|------|
| L0H11 | +67.8% | **-0.45%** | 平均出力が本体。差分は不要 |
| L0H6  | +7.2%  | **-0.54%** | 同上 |
| L2H1  | +2.1%  | **-0.09%** | 平均に置換した方が良い |
| L11H3 | +1.0%  | **-0.05%** | 同上 |

L0H11 は特に顕著: zero-ablation では 4 番目に重要だったが、mean-ablation では PPL が改善する。このヘッドの attention 計算は推論に**有害**であり、定常出力の方が良い。

### Layer-wise mean/zero ratio

| Layer | Mean ΔPPL (mean) | Mean ΔPPL (zero) | Ratio |
|-------|:-----------------:|:-----------------:|:-----:|
| **L0**  | +12.0%  | +41.5%  | **0.29** |
| L1    | +0.45%  | +1.40%  | 0.33 |
| L2    | +0.30%  | +1.32%  | 0.23 |
| L3    | +0.29%  | +0.59%  | 0.50 |
| L4    | +0.68%  | +0.88%  | 0.77 |
| L5    | +0.83%  | +1.81%  | 0.46 |
| L6    | +0.69%  | +1.16%  | 0.60 |
| L7    | +0.67%  | +1.13%  | 0.59 |
| L8    | +0.80%  | +1.49%  | 0.54 |
| L9    | +0.56%  | +1.14%  | 0.50 |
| L10   | +0.50%  | +1.17%  | 0.43 |
| **L11** | +0.31% | +1.91%  | **0.16** |

L11 が最も大きく改善 (ratio=0.16)。L11 のヘッドは差分的 attention に強く依存しておらず、出力の大部分は平均活性化で説明される。

![Mean vs Zero scatter](figures/exp10e_zero_vs_mean_scatter_gpt2.png)

---

## 10F: L0-Protected Cumulative Mean-Ablation

L0 の 12 ヘッドを保護し、残り 132 ヘッドを重要度の低い順に累積 mean-ablation:

| Heads removed | PPL    | Ratio  |
|:-------------:|:------:|:------:|
| 0             | 24.27  | 1.000x |
| 5             | 24.30  | 1.001x |
| 10            | 24.57  | 1.013x |
| 15            | 24.92  | 1.027x |
| 20            | 25.59  | 1.054x |
| **25**        | **26.18** | **1.079x** |
| 30            | 27.07  | 1.116x |
| 35            | 27.19  | 1.120x |
| 40            | 28.78  | 1.186x |
| **50**        | **35.37** | **1.457x** |
| **60**        | **47.14** | **1.942x** |
| 70            | 69.92  | 2.881x |
| 80            | 105.0  | 4.327x |
| 100           | 145.7  | 6.005x |
| 132           | 344.0  | 14.18x |

![L0-protected Pareto](figures/exp10f_pareto_l0protected_gpt2.png)

---

## 10G: Three-Curve Comparison

3 つの ablation 戦略の比較:

| Threshold | Zero-All | Zero-L0prot | Mean-L0prot | 改善倍率 |
|:---------:|:--------:|:-----------:|:-----------:|:--------:|
| PPL ≤ 1.1x | 10 | 10 | **25** | **2.5x** |
| PPL ≤ 1.5x | 25 | 25 | **50** | **2.0x** |
| PPL ≤ 2.0x | 35 | 35 | **60** | **1.7x** |

### 分析

1. **Zero-All vs Zero-L0prot が同一**: L0 ヘッドは重要度が高すぎて、累積 ablation の順序（低重要度から）では最後に削除される。L0 を保護してもしなくても同じ閾値に到達。

2. **Mean-L0prot が圧倒的に優位**: PPL 1.1x 以内で 25 ヘッド（10 → 25, 2.5x 改善）。これは mean-ablation が attention の「粗視化」に近い操作であるため。

3. **Gap の拡大**: 1.1x で 2.5x、1.5x で 2.0x、2.0x で 1.7x と、低い閾値ほど改善倍率が大きい。精密な制御が必要な領域で mean-ablation の優位性が際立つ。

![Three curves](figures/exp10g_three_curves_gpt2.png)

---

## 10H: Harmful Heads Analysis

### Zero-ablation で有害な 6 ヘッド

| Head | ΔPPL (zero, individual) | Joint zero ΔPPL |
|------|:-----------------------:|:---------------:|
| L2H5 | -0.38% | — |
| L4H1 | -0.34% | — |
| L3H1 | -0.26% | — |
| L4H8 | -0.22% | — |
| L1H1 | -0.17% | — |
| L8H5 | -0.05% | — |
| **Joint** | **Σ = -1.41%** | **+1.61%** |

**個別効果の合計は -1.41% だが、同時削除では +1.61%**。符号が反転する。
Individual ablation の効果は非線形に相互作用し、加法的に組み合わせることはできない。

### Mean-ablation で有害な 29 ヘッド

Mean-ablation では 29 ヘッド（20.1%）が ΔPPL < 0。これらのヘッドは attention 計算の結果よりも平均出力の方が推論に有益。

トップ 5: L0H6 (-0.54%), L0H11 (-0.45%), L4H8 (-0.44%), L1H5 (-0.35%), L0H2 (-0.35%)

### Harmful-first cumulative strategy

有害ヘッドを先に削除してから通常の累積 ablation を行う戦略:

- 初期 6 ヘッド同時削除後の PPL: 24.66 (1.016x) — 改善ではなく悪化
- 以降の累積 ablation は通常戦略とほぼ同等
- **Harmful-first 戦略は有効ではない**

![Harmful heads](figures/exp10h_harmful_heads_gpt2.png)

---

## Key Insights

### 1. Mean-ablation は softmax 線形化の良い代理指標

Zero-ablation は「ヘッドを殺す」操作であり、削除の上界を推定する。
Mean-ablation は「ヘッドを定数関数に置換する」操作であり、**attention の粗視化**（入力に依存しない固定出力）に対応する。

Softmax の線形化は、この 2 つの間に位置する:
- Zero < **Linear attention** < Mean < Softmax
- Mean-ablation の結果は、線形化で期待される PPL 劣化の**下限**を与える

### 2. L0H8 の本質は定常バイアス

Zero-ablation で最重要だった L0H8 (ΔPPL=+236.5%) が、mean-ablation では +1.08% に低下。
**219x の差**は、このヘッドの機能が attention パターン（入力依存の重み付け）ではなく、出力の定常成分（全トークンに一様に加わるバイアス）にあることを示す。

推測: L0H8 は「previous token head」として知られるタイプで、位置 i のトークンに位置 i-1 の情報を定常的に注入する。この機能は平均出力でほぼ再現される。

### 3. L0H11 の attention は有害

L0H11 は zero-ablation では 4 番目に重要 (+67.8%) だったが、mean-ablation では PPL が改善 (-0.45%)。
このヘッドの attention 計算は推論を**悪化**させており、定常出力に置換した方が良い。

これは L0 の「重要さ」の質的違いを示す:
- **L0H8, L0H10**: 定常バイアスが重要（mean で大幅に低下）
- **L0H1**: 差分的 attention が重要（mean でも +108.9%）
- **L0H11**: attention が有害（mean で改善）

### 4. Individual ≠ Cumulative は健在だが、Gap は縮小

| Metric | Zero-ablation | Mean-ablation |
|--------|:------------:|:-------------:|
| Individual \|ΔPPL\| < 10% | 94.4% | 97.9% |
| Cumulative ≤1.1x | 10 heads | 25 heads |

Mean-ablation でも累積効果は複合するが、Zero に比べて遥かに穏やか。

---

## Phase 2 への判断

### Go/No-Go: **Go（条件付き）**

Phase 1.5 の結果は、Phase 2（softmax 線形化）に進む十分な根拠を提供する:

| 判断基準 | 結果 | 評価 |
|---------|------|------|
| Mean-ablation で ≥20 ヘッド at ≤1.1x | 25 ヘッド (19%) | **PASS** |
| Zero→Mean で改善倍率 ≥1.5x | 2.5x (at 1.1x) | **PASS** |
| L1-L11 のヘッドが主な削除対象 | 全 25 が L1-L11 | **PASS** |

### 推奨戦略

1. **L0 は完全保護**: L0 の 12 ヘッドは softmax を維持。特に L0H1 は mean-ablation でも +108.9% であり、差分的 attention が不可欠

2. **対象は L1-L11 の 132 ヘッド**: ここから 25-50 ヘッドの softmax を線形化

3. **優先順位**: Mean-ablation ΔPPL が小さいヘッドから線形化。特に ΔPPL < 0 の 29 ヘッドは、そもそも attention を粗くした方が良いことが示唆されている

4. **線形化の種類**:
   - L1 正規化 linear attention: `softmax(QK^T/√d)V → normalize(QK^T/√d)V`
   - ReLU attention: `softmax → max(0, QK^T/√d)`
   - Mean-ablation は「入力非依存の定数 attention」に対応するため、線形化はこれより常に良いはず

5. **期待される成果**: Mean-ablation の 25 ヘッド at 1.1x は下限。線形 attention は入力依存性を保持するため、実際にはより多くのヘッドを線形化できる可能性が高い

### リスクと制約

- **サンプルサイズ**: 2048 トークンの PPL 推定にはノイズがある。Phase 2 では 50K+ トークンで再現性を確認すべき
- **累積効果**: 50 ヘッド以上の線形化では PPL が急上昇する可能性。段階的な実験が必要
- **Fine-tuning なし**: 現状は frozen model での線形化。少量の fine-tuning で回収できる PPL 劣化は不明

---

## Files

- Data: `results/data/exp10e_mean_ablation_gpt2.json`, `results/data/exp10f_cumulative_l0protected_gpt2.json`, `results/data/exp10g_zero_l0prot_gpt2.json`, `results/data/exp10h_harmful_heads_gpt2.json`, `results/data/exp10_phase1.5_gpt2.json`
- Figures: `results/figures/exp10e_*.png`, `results/figures/exp10f_*.png`, `results/figures/exp10g_*.png`, `results/figures/exp10h_*.png`
- Script: `experiments/exp10b_mean_ablation.py`
