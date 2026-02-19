# Semantic Phase Transition

LLMの内部における意味処理のダイナミクスを実験的に解明し、重み行列の圧縮可能性を検証するプロジェクト。

## 主要な発見

### 1. 普遍浸透関数 g(l/L) の発見 (exp1-3)

曖昧語（bank, rock, light 等9語）の意味確定過程を秩序変数 σ(l) で追跡した結果、意味の確定は「相転移」ではなく**漸進的な浸透過程**であることが判明。σ は以下のように完全分解できる:

```
σ(l, h) = h · f_max(word) · g(l/L)
```

- **g(l/L)** は語にも モデルサイズにも依存しない普遍関数（語間相関>0.96, モデル間相関0.98）
- g の構造: l/L < 0.85 ではほぼ線形（α≈0.9）、最終層でドロップ（unembedding再構成）
- **f_max(word)** は語依存のスケーリング定数で、曖昧性の「解決しやすさ」を表す

### 2. ランダウ相転移仮説の棄却 (exp2)

当初、意味確定がランダウ型相転移に従うと仮説を立てたが:

- 臨界指数 β は 0.01〜2.35 と大きく分散 → 普遍的臨界指数は存在しない
- 一次相転移（鋭いジャンプ）は観測されず → 中間層での変化は連続的
- 最終層ジャンプは unembedding 再構成であり、意味確定ではない（logit lensで確認）

### 3. 各レイヤーが等量の文脈情報を注入する (exp3)

g(l/L) ≈ (l/L)^0.9 は、Transformer の各レイヤーがほぼ等量の文脈情報を対象トークンに注入していることを意味する。Attention+FFN という非線形演算の巨視的効果が線形法則に従うという、統計力学的に興味深い結果。

---

## 実験一覧

### 完了

| 実験     | 内容                          | 主要結果                                                            |
| -------- | ----------------------------- | ------------------------------------------------------------------- |
| exp1 v2  | 秩序変数の基本測定            | 全9語で扇形分岐パターン。二段階構造を発見                           |
| exp2     | ランダウフィット + logit lens | ランダウ仮説棄却。two_stageモデル最良。R²>0.99達成                  |
| exp3     | 線形応答の限界                | 深い層ほど線形。非線形性は初期層に集中                              |
| exp3 EFG | 普遍浸透関数の決定            | g(l/L)の普遍性確認。erf(全域)、power_law(truncated)がベストフィット |

### スキップ

| 実験        | 理由                                              |
| ----------- | ------------------------------------------------- |
| exp4 感受率 | 臨界レイヤーが存在しないため、χのピーク測定は不要 |
| exp5 相図   | 相転移の境界線がないため、2D相図は不要            |

### 次に実行

| 実験          | 内容                                 | 目的                           |
| ------------- | ------------------------------------ | ------------------------------ |
| exp6 普遍基底 | W_QKのPCA分解 → 基底数 vs perplexity | 重み行列の圧縮可能性を直接検証 |

---

## 仮説の変遷

```
出発点（2025/12 UWS論文に触発）:
  「LLMの重みに普遍法則があるなら、ヘッドを関数に置換して100x圧縮できる」

仮説1: ランダウ相転移（exp1-2で棄却）
  「意味確定は臨界レイヤーで不連続に起きる」
  → 実際は滑らかな浸透過程だった

仮説2: 意味浸透モデル（exp3で確立）
  「σ(l,h) = h · f_max · g(l/L) の普遍分解が成立する」
  → 科学的に面白いが圧縮には直接繋がらない

仮説3: 普遍基底による圧縮（exp6で検証予定）
  「W_QKを少数の基底で再構成してもperplexityが維持される」
  → 当初の直感に最も近い実験
```

---

## exp6: 普遍基底と圧縮の検証

### 背景

Universal Weight Subspace Hypothesis (Kaushik et al., 2025) は、1100以上のモデルで重み行列の分散の90%が16-32個の主成分で説明できることを示した。しかし「重み分散の90%」が「推論品質の維持」に繋がるかは未検証。

### 実験内容

**6A. W_QKのPCA**

- 全ヘッドから W_QK = W_Q^T @ W_K を抽出しPCA
- 累積寄与率のグラフ → 何個の基底で90%に達するか

**6B. 基底数 vs perplexity**

- K = [1, 2, 3, 5, 8, 10, 16, 32, 64, 全数] で再構成
- WikiText-2でperplexity測定
- 圧縮の実用性を直接評価

**6C. ヘッドのクラスタリング**

- PCA係数空間でk-meansクラスタリング
- 重心ヘッドで全メンバーを置換 → perplexity測定

### 期待される結果パターン

```
最良: K=10でperplexity 1.05倍以内 → 圧縮の実用性が証明される
良好: K=16-32で1.1倍以内 → UWS論文と整合、圧縮は限定的
微妙: K=64でも1.2倍超 → 重み分散とperplexityは別の話
最悪: PCAで構造が見えない → GPT-2にはUWSが当てはまらない
```

---

## セットアップ

```bash
pip install torch transformers transformer-lens numpy matplotlib scipy scikit-learn datasets
```

## 使い方

```bash
# 実験1-3（意味浸透）
python experiments/exp1_basic_v2.py --model gpt2 --device mps
python experiments/exp2_landau.py --model gpt2 --device mps
python experiments/exp3_linear_response.py --model gpt2 --device mps
python experiments/exp3ef_universal_g.py --model gpt2 --device mps

# 実験6（普遍基底）
python experiments/exp6_basis.py --model gpt2 --device mps
```

## 参考文献

1. Kaushik et al., "The Universal Weight Subspace Hypothesis" (arXiv:2512.05117, Dec 2025)
2. Sun et al., "Phase Transitions in Large Language Models and the O(N) Model" (arXiv:2501.16241, Jan 2025)
3. "Phase Transitions in the Output Distribution of Large Language Models" (arXiv:2405.17088, May 2024)
4. "Decomposing Behavioral Phase Transitions in LLMs" (arXiv:2508.20015, Aug 2025)
5. Hu et al., "What Affects the Effective Depth of Large Language Models?" (arXiv:2512.14064, Dec 2025)
6. Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models" (Mar 2025)
