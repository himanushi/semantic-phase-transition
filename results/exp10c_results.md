# Experiment 10 Phase 2: Linear Attention Results

## Summary

Phase 1.5 の mean-ablation は「ヘッドを定数関数に置換する」操作であり、線形 attention の下限推定だった。
Phase 2 では softmax を3種の線形近似（ReLU, L1, identity）に実際に置換し、PPL劣化を直接測定した。

**主要結論**: **ReLU attention が圧倒的に最良**。個別 ΔPPL は平均 +0.39%（mean-ablation +1.50% の 1/4）で、累積線形化では **PPL 1.1x 以内に 30 ヘッド (23%)、1.5x 以内に 70 ヘッド (53%)、2.0x 以内に 90 ヘッド (68%)** を線形化可能。

Mean-ablation（Phase 1.5）に対する改善: 1.1x で +5 ヘッド (20%↑)、1.5x で +20 ヘッド (40%↑)、2.0x で +30 ヘッド (50%↑)。

## Experimental Setup

- Model: GPT-2 small (12L × 12H, 124M params)
- Dataset: WikiText-2 validation set, 2048 tokens (2 chunks × 1024)
- Baseline PPL: 24.27
- Target: L1-L11 の 132 ヘッド（L0 は完全保護）
- Hook: `hook_attn_scores`（softmax前）でスコアを捕捉 → `hook_pattern`（softmax後）で線形化パターンに置換
- Device: M1 Mac (MPS)
- Runtime: Part I 170s, Part J <1s, Part K 18s, Part L <1s

### 線形化方式

1. **ReLU**: `softmax(QK^T/√d) → ReLU(QK^T/√d) / sum(ReLU(...))`
2. **L1**: `softmax(QK^T/√d) → |QK^T/√d| / sum(|...|)`
3. **Identity**: `softmax(QK^T/√d) → (QK^T/√d) / sum(|...|)` (L1ノルムで正規化、負の重み許容)

### 技術的問題と解決

- **NaN問題**: `hook_attn_scores` は causal mask 適用後（未来位置が -inf）のスコアを返す。`abs(-inf) = inf` により L1/identity で NaN が発生
- **解決**: `-inf` を 0 に置換してから活性化関数を適用 (`torch.where(scores.isfinite(), scores, zeros)`)
- **Identity正規化**: 負のスコアが存在するため、通常の sum 正規化は不安定。L1ノルム (sum of abs) で正規化に変更

---

## 10I: Individual Linear Attention

L1-L11 の 132 ヘッドに対して、各方式で個別に softmax を置換し ΔPPL を測定。

### Statistics

| Metric | ReLU | L1 | Identity |
|--------|:----:|:--:|:--------:|
| Mean ΔPPL | **+0.39%** | +1.85% | +9.74% |
| Median ΔPPL | **+0.23%** | +0.89% | +3.03% |
| Max ΔPPL | +5.74% | +38.9% | +188.7% |
| \|ΔPPL\| < 1% | **120/132 (90.9%)** | 73/132 (55.3%) | 13/132 (9.8%) |
| ΔPPL < 0 | **30/132 (22.7%)** | 11/132 (8.3%) | 3/132 (2.3%) |

**ReLU の圧倒的優位**: mean ΔPPL は L1 の 1/5、identity の 1/25。90% のヘッドが ΔPPL 1% 未満で線形化可能。

### ReLU vs Mean-ablation

| Metric | ReLU | Mean-ablation |
|--------|:----:|:------------:|
| Mean ΔPPL (L1-L11) | **+0.39%** | +0.55% |
| ΔPPL < 0 | **30/132 (22.7%)** | 27/132 (20.5%) |
| Linear < Mean (per-head) | — | **87/132 (65.9%)** |

87/132 ヘッド (66%) で ReLU linearization が mean-ablation より PPL 劣化が小さい。
**「softmax を ReLU に置換しても、定数関数に置換するより良い」** ことが確認された。

### なぜ ReLU が最良か

- **ReLU = softmax の近似**: softmax は正のスコアを保持・増幅し、負のスコアを抑制する。ReLU も同じ（負→0）
- **L1 の問題**: `|score|` は負のスコアの意味を反転する。本来「注目しない」位置に正の重みを割り当ててしまう
- **Identity の問題**: 負の重みは物理的に「情報を減算する」操作であり、softmax の設計意図と根本的に異なる

![Distribution](figures/exp10i_linear_individual_gpt2.png)
![Linear vs Mean scatter](figures/exp10i_linear_vs_mean_scatter_gpt2.png)

---

## 10J: Best Method Selection

### Overall comparison

| Method | Mean ΔPPL | Best for N heads |
|--------|:---------:|:---------------:|
| **ReLU** | **+0.39%** | **104 / 132** |
| L1 | +1.85% | 26 / 132 |
| Identity | +9.74% | 2 / 132 |
| **Mixed** | **+0.26%** | — |

Mixed strategy（各ヘッドでベスト方式を選択）は純 ReLU より僅かに良い (+0.26% vs +0.39%) が、差は微小。

### 方式選択のレイヤーパターン

104/132 ヘッドで ReLU が最良。L1 が最良となる 26 ヘッドは全レイヤーに分散しており、特定のレイヤーパターンは見られない。

![Method comparison](figures/exp10j_method_comparison_gpt2.png)

---

## 10K: Cumulative Linearization

重要度の低い順にヘッドを累積的に線形化。

### Pareto curves

| Heads linearized | ReLU PPL | ReLU ratio | Mixed PPL | Mixed ratio |
|:----------------:|:--------:|:----------:|:---------:|:-----------:|
| 0 | 24.27 | 1.000x | 24.27 | 1.000x |
| 5 | 24.34 | 1.003x | 24.31 | 1.002x |
| 10 | 24.47 | 1.008x | 24.45 | 1.007x |
| 15 | 24.76 | 1.020x | 24.53 | 1.011x |
| 20 | 24.80 | 1.022x | 24.82 | 1.023x |
| 25 | 25.33 | 1.044x | 25.33 | 1.044x |
| **30** | **26.39** | **1.087x** | **26.16** | **1.078x** |
| 35 | 26.78 | 1.104x | 27.68 | 1.141x |
| 40 | 27.71 | 1.142x | 28.87 | 1.190x |
| 50 | 29.45 | 1.214x | 32.88 | 1.355x |
| 60 | 33.14 | 1.366x | 38.99 | 1.607x |
| 70 | 35.94 | 1.481x | 42.58 | 1.755x |
| 80 | 37.23 | 1.534x | 48.55 | 2.001x |
| 90 | 45.64 | 1.881x | 63.50 | 2.616x |
| 100 | 56.57 | 2.331x | 99.22 | 4.088x |
| 132 | 308.00 | 12.69x | 280.84 | 11.57x |

### Threshold analysis

| Threshold | ReLU | Mixed | Mean-ablation | ReLU/Mean改善 |
|:---------:|:----:|:-----:|:------------:|:------------:|
| PPL ≤ 1.1x | **30** | 30 | 25 | **+20%** |
| PPL ≤ 1.5x | **70** | 50 | 50 | **+40%** |
| PPL ≤ 2.0x | **90** | 70 | 60 | **+50%** |

### 重要な発見: 純 ReLU > 混合戦略

Mixed strategy は個別では ReLU より良いヘッドもあるが、累積では ReLU が圧倒的に良い。
- 1.5x: ReLU=70 vs Mixed=50
- 2.0x: ReLU=90 vs Mixed=70

**原因**: L1/identity ヘッドが累積時に非線形に干渉する。ReLU は softmax に最も近い非線形性であり、ヘッド間の相互作用が穏やかに保たれる。

**教訓**: 個別最適化 ≠ 累積最適化。均一な方式の方が相互作用が予測可能で累積効果が安定する。

![Pareto curves](figures/exp10k_pareto_linear_gpt2.png)

---

## 10L: Computational Cost Estimation

### Per-head savings

| seq_len | Softmax FLOPs | Linear FLOPs | Kernel FLOPs | Linear savings | Kernel savings |
|:-------:|:------------:|:----------:|:----------:|:-------------:|:-------------:|
| 128 | 4.3M | 4.3M | 3.1M | 0.8% | **26.7%** |
| 512 | 68.7M | 68.2M | 12.6M | 0.8% | **81.7%** |
| 1024 | 274.7M | 272.6M | 25.2M | 0.8% | **90.8%** |
| 2048 | 1098.9M | 1090.5M | 50.3M | 0.8% | **95.4%** |

**exp() 除去だけでは 0.8% しか節約できない。** 真の削減は kernelization (O(n²d) → O(nd²)) から来る。
seq_len > d_head (= 64) の場合、kernel attention は劇的に高速になる。

### Tradeoff: PPL vs FLOPs savings (seq=1024, ReLU, kernel化想定)

| Heads linearized | PPL ratio | Attention FLOPs savings |
|:----------------:|:---------:|:----------------------:|
| 10 | 1.008x | 6.3% |
| 20 | 1.022x | 12.6% |
| **30** | **1.087x** | **18.9%** |
| 50 | 1.214x | 31.5% |
| **70** | **1.481x** | **44.1%** |
| **90** | **1.881x** | **56.7%** |

seq=2048 では savings がさらに大きくなる（kernel 1ヘッドあたり 95.4% 削減）。

![Cost tradeoff](figures/exp10l_cost_tradeoff_gpt2.png)

---

## Key Insights

### 1. Softmax → ReLU は驚くほど無害

132 ヘッド中 120 ヘッド (91%) が ΔPPL < 1% で ReLU 化可能。これは softmax の「正確な確率分布」が、多くのヘッドでは不要であることを示す。ReLU の「正のスコアのみ通す」という粗い操作で十分な attention パターンが維持される。

### 2. 線形化 > mean-ablation > zero-ablation の序列が確定

| 戦略 | PPL 1.1x で削除/線形化可能 | 性質 |
|------|:------------------------:|------|
| Zero-ablation | 10 heads | ヘッドを完全に殺す |
| Mean-ablation | 25 heads | 定数関数に置換 |
| **ReLU linearization** | **30 heads** | **入力依存の線形関数に置換** |

入力への依存を保持するほど、PPL 劣化は小さくなる。

### 3. 均一方式 > per-head 最適化

個別に最適な方式を選ぶ mixed strategy は、累積では純 ReLU に劣る。これは Phase 1 の「individual ≠ cumulative」と同じ教訓: ヘッド間の相互作用が支配的。

### 4. Kernel 化が真の計算削減

exp() 除去自体は 0.8% の FLOPs 削減にしかならない。しかし ReLU attention は kernel 化 (O(nd²)) が可能:
- seq=1024 で 30 heads 線形化: **attention 19% 削減** (at PPL 1.09x)
- seq=2048 で 30 heads 線形化: **attention 20% 削減** (at PPL 1.09x)

### 5. Phase 3 (CoT) への示唆

PPL 1.1x の劣化を許容して 30 ヘッドを kernel 化した場合:
- attention の 19% を節約（seq=1024）
- この浮いた計算を CoT (追加トークン生成) に回せるか？
- 追加トークンの attention コストは O(n²d) なので、節約した FLOPs で生成できる追加トークン数は限定的
- **より有望なアプローチ**: ReLU attention ヘッドを kernel 化して sequence length を伸ばし、より長いCoTを許容する

---

## Phase 3 への判断

### Go/No-Go: **Go（条件付き）**

| 判断基準 | 結果 | 評価 |
|---------|------|------|
| ReLU で ≥25 heads at ≤1.1x | 30 heads (23%) | **PASS** |
| Mean-ablation より改善 | +20% (at 1.1x) | **PASS** |
| Kernel化で有意なFLOPs削減 | 19% (seq=1024) | **PASS** |

### 推奨戦略

1. **ReLU attention 一択**: L1/identity は不要。混合戦略も不要
2. **30 ヘッドを ReLU 化**: PPL 1.087x で attention FLOPs 19% 削減（seq=1024）
3. **Kernel 化実装**: ReLU attention は `Q·ReLU(K)^T·V = Q·(ReLU(K)^T·V)` に分解可能で O(nd²) に削減
4. **CoT 検証**: 節約した計算で生成できる追加トークン数と、それによるタスク精度改善を測定

### リスク

- **サンプルサイズ**: 2048 トークンの PPL 推定ノイズ
- **Kernel 化の精度**: causal masking 付き kernel attention は実装が複雑で、数値精度の問題が生じる可能性
- **CoT の効果**: 追加トークンが実際にタスク性能を改善するかは未検証

---

## Files

- Data: `results/data/exp10i_linear_individual_gpt2.json`, `results/data/exp10j_method_comparison_gpt2.json`, `results/data/exp10k_cumulative_linear_gpt2.json`, `results/data/exp10l_cost_estimate_gpt2.json`
- Figures: `results/figures/exp10i_*.png`, `results/figures/exp10j_*.png`, `results/figures/exp10k_*.png`, `results/figures/exp10l_*.png`
- Script: `experiments/exp10c_linear_attention.py`
