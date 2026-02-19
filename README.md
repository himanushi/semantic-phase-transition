# 意味の相転移実験：LLMレイヤー内における意味確定の臨界現象

## 概要

LLMが曖昧な文を処理する際、意味の確定が特定のレイヤーで「相転移」的に
起きるかどうかを検証する実験。物理学のランダウ相転移理論の枠組みを用いて、
意味の確定過程を秩序変数として定量化し、臨界現象の存在を実証する。

### 仮説

> Transformerの各レイヤーを通過する過程で、曖昧な単語の意味表現は
> ランダウ型相転移に従い、ある臨界レイヤー l_c で不連続的に確定する。
> この臨界現象は、意味空間上のポテンシャルの対称性破れとして記述でき、
> その臨界指数はプロンプトの種類に依存しない普遍性を持つ。

### 背景と動機

- **Universal Weight Subspace Hypothesis (2025/12)**: 1100以上のモデルの重み行列に共通の低次元部分空間が存在することが示された
- **Phase Transitions in LLM Output (2024/05)**: 学習中の相転移が物理学的手法で検出された
- **Effective Depth研究 (2025/12)**: レイヤー中間付近で相転移的な振る舞いが確認された
- **O(N)模型としてのLLM (2025/01)**: TransformerをO(N)場の理論として再定式化する試み

**未踏領域**: 上記は全て「学習時」や「モデルサイズ」の相転移。
「1つのプロンプトの推論中に、レイヤーを変数として起きる意味確定の相転移」
を秩序変数で追跡した研究はまだ存在しない。

---

## 実験1: 基本的な相転移の検出

### 目的

曖昧な単語を含む文において、各レイヤーの残差ストリームから
「秩序変数」を定義し、意味確定のダイナミクスを可視化する。

### 使用モデル

```
推奨: GPT-2 (small/medium/large) または Gemma-2-2B
理由: オープンウェイト、中間層へのアクセスが容易、計算コストが低い
```

### 必要なライブラリ

```bash
pip install torch transformers numpy matplotlib scipy
pip install transformer-lens  # Neel Nandaのmech interp用ライブラリ
```

### Step 1: 秩序変数の定義

秩序変数 σ(l) を以下のように定義する:

```
σ(l) = cos(φ(l), ê_A) - cos(φ(l), ê_B)
```

- `φ(l)` : レイヤー l における対象トークン位置の残差ストリームベクトル
- `ê_A`, `ê_B` : 2つの解釈に対応する方向ベクトル
- σ > 0 なら解釈A寄り、σ < 0 なら解釈B寄り、σ ≈ 0 なら未決定

#### 方向ベクトルの取得方法

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-medium")

# 方向ベクトル ê_A, ê_B を明確な文脈から取得
# 例: "bank" の2つの意味
prompt_finance = "I deposited money at the bank"
prompt_river   = "I sat by the river bank"

# 各プロンプトの最終レイヤーにおける "bank" の残差ストリームを取得
_, cache_fin = model.run_with_cache(prompt_finance)
_, cache_riv = model.run_with_cache(prompt_river)

# "bank" トークンの位置を特定 (プロンプトによって異なる)
# 最終レイヤーの残差ストリームを方向ベクトルとして使用
bank_pos_fin = -1  # 最後のトークン（要調整）
bank_pos_riv = -1

e_finance = cache_fin["resid_post", -1][0, bank_pos_fin]  # 最終レイヤー
e_river   = cache_riv["resid_post", -1][0, bank_pos_riv]

# 正規化
e_finance = e_finance / e_finance.norm()
e_river   = e_river / e_river.norm()
```

### Step 2: 曖昧プロンプトの作成

```python
# 段階的に文脈を与えるプロンプトセット
ambiguous_prompts = {
    "bank": {
        # 文脈なし（最大曖昧）
        "neutral": "The bank",
        # 弱い文脈
        "weak_finance": "She went to the bank to",
        "weak_river": "He walked along the bank near",
        # 強い文脈（明確に確定するはず）
        "strong_finance": "She deposited her savings at the bank",
        "strong_river": "The fish swam near the muddy bank of the river",
    },
    "bat": {
        "neutral": "The bat",
        "weak_animal": "The bat flew through",
        "weak_sports": "He swung the bat at",
        "strong_animal": "The bat hung upside down in the dark cave",
        "strong_sports": "The baseball player picked up his wooden bat",
    },
    "crane": {
        "neutral": "The crane",
        "weak_bird": "The crane stood in the shallow",
        "weak_machine": "The crane lifted the heavy",
        "strong_bird": "The white crane flew gracefully over the wetlands",
        "strong_machine": "The construction crane lifted steel beams to the roof",
    },
}
```

### Step 3: 全レイヤーの秩序変数を記録

```python
import numpy as np

def compute_order_parameter(model, prompt, target_token, e_A, e_B):
    """
    各レイヤーの秩序変数 σ(l) を計算する

    Returns:
        sigma: shape (n_layers,) の秩序変数の配列
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(prompt)

    n_layers = model.cfg.n_layers
    sigma = np.zeros(n_layers + 1)  # embedding層を含む

    # target_tokenの位置を特定
    # (注意: トークナイザによって分割が異なるため要確認)
    target_pos = find_token_position(tokens, target_token, model)

    for l in range(n_layers + 1):
        if l == 0:
            resid = cache["resid_pre", 0][0, target_pos]
        else:
            resid = cache["resid_post", l - 1][0, target_pos]

        resid_norm = resid / resid.norm()

        cos_A = torch.dot(resid_norm, e_A).item()
        cos_B = torch.dot(resid_norm, e_B).item()

        sigma[l] = cos_A - cos_B

    return sigma


def find_token_position(tokens, target_word, model):
    """対象トークンの位置を見つける"""
    token_strs = model.to_str_tokens(tokens)
    for i, t in enumerate(token_strs[0]):
        if target_word.lower() in t.lower().strip():
            return i
    raise ValueError(f"Token '{target_word}' not found in {token_strs}")
```

### Step 4: 可視化と解析

```python
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_order_parameter(sigmas_dict, title="Order Parameter σ(l)"):
    """
    複数プロンプトの秩序変数をプロット
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for label, sigma in sigmas_dict.items():
        layers = np.arange(len(sigma))
        ax.plot(layers, sigma, 'o-', label=label, markersize=4)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer l')
    ax.set_ylabel('Order Parameter σ(l)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('order_parameter.png', dpi=150)
    plt.show()
```

### 期待される結果パターン

```
パターンA（相転移あり — 仮説を支持）:
  σ(l) がある臨界レイヤー l_c で急激にジャンプする
  ジャンプ前: σ ≈ 0（未決定）
  ジャンプ後: σ >> 0 または σ << 0（確定）

パターンB（漸次的変化 — 弱い支持）:
  σ(l) が滑らかに単調変化する
  → 二次相転移の可能性、またはcrossover

パターンC（相転移なし — 仮説を棄却）:
  σ(l) が全層でほぼ一定、またはランダムに変動
  → 意味確定はレイヤー横断的に分散している
```

---

## 実験2: ランダウ理論の定量的検証

### 目的

パターンAまたはBが観測された場合、σ(l) がランダウ理論の予測する
関数形に従うかどうかを定量的に検証する。

### ランダウ理論の予測

二次相転移の場合:

```
σ(l) = 0                           (l < l_c)
σ(l) = A · (l - l_c)^β             (l > l_c)
```

平均場理論では臨界指数 β = 1/2（つまり √(l - l_c)）。

一次相転移の場合:

```
σ(l) はl_cで不連続なジャンプ
```

### フィッティング手順

```python
def landau_fit(layers, sigma, transition_type="second_order"):
    """
    ランダウ理論の関数形にフィットする

    transition_type:
      "second_order": σ = A * (l - l_c)^β * Θ(l - l_c)
      "tanh":         σ = A * tanh(κ * (l - l_c))  (crossoverの場合)
    """

    if transition_type == "second_order":
        def model_func(l, A, l_c, beta):
            result = np.zeros_like(l, dtype=float)
            mask = l > l_c
            result[mask] = A * (l[mask] - l_c) ** beta
            return result

        # 初期推定
        # σが最大値の半分を超えるレイヤーを l_c の初期推定とする
        half_max = np.max(np.abs(sigma)) / 2
        l_c_init = layers[np.argmax(np.abs(sigma) > half_max)]

        try:
            popt, pcov = curve_fit(
                model_func, layers.astype(float), sigma,
                p0=[np.max(sigma), l_c_init, 0.5],
                bounds=([0, 0, 0.1], [np.inf, len(layers), 2.0]),
                maxfev=10000
            )
            return {
                "A": popt[0],
                "l_c": popt[1],
                "beta": popt[2],
                "covariance": pcov,
                "type": "second_order"
            }
        except RuntimeError:
            return None

    elif transition_type == "tanh":
        def model_func(l, A, l_c, kappa):
            return A * np.tanh(kappa * (l - l_c))

        l_c_init = len(layers) // 2

        try:
            popt, pcov = curve_fit(
                model_func, layers.astype(float), sigma,
                p0=[np.max(sigma), l_c_init, 0.5]
            )
            return {
                "A": popt[0],
                "l_c": popt[1],
                "kappa": popt[2],
                "covariance": pcov,
                "type": "tanh"
            }
        except RuntimeError:
            return None
```

### 臨界指数βの解析

```python
def analyze_critical_exponent(results_across_prompts):
    """
    複数のプロンプトから得られた臨界指数βの分布を解析

    β ≈ 0.5 → 平均場理論（ランダウ理論）が成立
    β が一定 → 普遍性クラスの存在を示唆
    β がバラバラ → 単純な相転移描像は成立しない
    """
    betas = [r["beta"] for r in results_across_prompts if r is not None]

    print(f"臨界指数 β の統計:")
    print(f"  平均: {np.mean(betas):.3f}")
    print(f"  標準偏差: {np.std(betas):.3f}")
    print(f"  平均場理論の予測 (β=0.5) からのずれ: {abs(np.mean(betas) - 0.5):.3f}")

    # βのヒストグラム
    plt.figure(figsize=(8, 5))
    plt.hist(betas, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Mean field β=0.5')
    plt.xlabel('Critical exponent β')
    plt.ylabel('Count')
    plt.title('Distribution of critical exponents')
    plt.legend()
    plt.savefig('critical_exponents.png', dpi=150)
    plt.show()
```

---

## 実験3: 臨界レイヤーの普遍性

### 目的

臨界レイヤー l_c が以下の条件で普遍的かどうかを検証する:

1. 同じモデル内で異なる曖昧語に対して
2. 異なるモデルサイズ間で（l_c / L の比率として）
3. 異なるモデルアーキテクチャ間で

### 実験設計

```python
# モデルサイズ間の比較
models_to_test = [
    ("gpt2",        12),   # 12 layers
    ("gpt2-medium", 24),   # 24 layers
    ("gpt2-large",  36),   # 36 layers
    # ("gemma-2-2b",  26), # オプション
]

def compare_critical_layers(models_to_test, prompts):
    """
    異なるモデルで臨界レイヤーの相対位置 l_c/L を比較
    """
    results = {}

    for model_name, n_layers in models_to_test:
        model = HookedTransformer.from_pretrained(model_name)

        # 方向ベクトルはモデルごとに再計算
        e_A, e_B = compute_direction_vectors(model, ...)

        model_results = []
        for prompt_name, prompt_text in prompts.items():
            sigma = compute_order_parameter(model, prompt_text, ...)
            fit = landau_fit(np.arange(len(sigma)), sigma)
            if fit:
                model_results.append({
                    "prompt": prompt_name,
                    "l_c": fit["l_c"],
                    "l_c_ratio": fit["l_c"] / n_layers,
                    "beta": fit["beta"],
                })

        results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    return results
```

### 期待される結果

```
もし l_c / L ≈ 0.5 が全モデルで成立する場合:
  → 「意味はモデルの中間層で確定する」という普遍法則
  → 有効深度研究の結果と一致

もし l_c / L がモデルサイズで変化する場合:
  → 意味確定に必要な「計算量」がスケーリングする
  → スケーリング則との接続が可能
```

---

## 実験4: 感受率（susceptibility）の測定

### 目的

物理の相転移では、臨界点の近くで「感受率」（外部摂動に対する応答）が
発散する。LLMでも同様の現象が起きるかを検証する。

### 感受率の定義

```
χ(l) = |∂σ(l) / ∂h|
```

ここで h は「外部場」に相当する小さな摂動。
具体的にはプロンプトに弱いヒントを追加して σ の変化量を測る。

### 実験手順

```python
def compute_susceptibility(model, base_prompt, target_token,
                           hint_words, e_A, e_B):
    """
    プロンプトに弱いヒント語を追加した時の σ の変化量（感受率）を計算

    hint_words: ["money", "fish", "deposit", "water", ...]
    """
    # ベースラインの σ
    sigma_base = compute_order_parameter(
        model, base_prompt, target_token, e_A, e_B
    )

    susceptibilities = {}

    for hint in hint_words:
        # ヒントを追加したプロンプト
        perturbed_prompt = f"{hint}. {base_prompt}"

        sigma_pert = compute_order_parameter(
            model, perturbed_prompt, target_token, e_A, e_B
        )

        # 各レイヤーでの感受率
        chi = np.abs(sigma_pert - sigma_base)
        susceptibilities[hint] = chi

    return susceptibilities


def plot_susceptibility(susceptibilities, title):
    """
    感受率のレイヤー依存性をプロット
    臨界レイヤー付近でピークが見えるはず
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for hint, chi in susceptibilities.items():
        layers = np.arange(len(chi))
        ax.plot(layers, chi, 'o-', label=f'hint: {hint}', markersize=3)

    ax.set_xlabel('Layer l')
    ax.set_ylabel('Susceptibility χ(l)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('susceptibility.png', dpi=150)
    plt.show()
```

### 期待される結果

```
臨界レイヤー l_c の付近で χ(l) にピーク → 相転移の直接的証拠
χ のピーク位置がランダウフィットの l_c と一致 → 理論の整合性を確認
χ のピーク幅が狭い → 鋭い相転移（一次に近い）
χ のピーク幅が広い → 緩やかなcrossover
```

---

## 実験5: 文脈「温度」による制御

### 目的

文脈の量を連続的に変化させることで、相転移の「温度」制御を行い、
相図（phase diagram）を作成する。

### 実験設計

```python
def generate_context_gradient(target_word, interpretation_A, interpretation_B,
                              n_steps=20):
    """
    文脈の「強さ」を段階的に変化させるプロンプト列を生成

    例: bank (finance vs river)
    ステップ0:  "The bank"
    ステップ5:  "She walked to the bank"
    ステップ10: "She walked to the bank with her"
    ステップ15: "She walked to the bank with her paycheck to"
    ステップ20: "She walked to the bank with her paycheck to deposit her savings"
    """
    # 文脈を段階的に追加
    # 実際にはいくつかの戦略がある:

    # 戦略1: 文脈語を1つずつ追加
    finance_context_words = [
        "She", "went", "to", "the", "bank", "with", "her",
        "paycheck", "to", "deposit", "her", "monthly", "savings"
    ]
    river_context_words = [
        "He", "walked", "along", "the", "bank", "of", "the",
        "winding", "river", "watching", "the", "fish", "swim"
    ]

    prompts_A = []
    prompts_B = []
    for i in range(1, len(finance_context_words) + 1):
        prompts_A.append(" ".join(finance_context_words[:i]))
    for i in range(1, len(river_context_words) + 1):
        prompts_B.append(" ".join(river_context_words[:i]))

    return prompts_A, prompts_B


def create_phase_diagram(model, prompts_A, prompts_B,
                         target_token, e_A, e_B):
    """
    横軸: 文脈の長さ（トークン数）= 「逆温度」
    縦軸: レイヤー
    色:   秩序変数 σ の値

    相図として可視化する
    """
    all_prompts = prompts_A + prompts_B
    labels = ['A'] * len(prompts_A) + ['B'] * len(prompts_B)

    # σ(文脈長, レイヤー) の2次元配列を構築
    sigma_map = []

    for prompt in all_prompts:
        sigma = compute_order_parameter(
            model, prompt, target_token, e_A, e_B
        )
        sigma_map.append(sigma)

    sigma_map = np.array(sigma_map)

    # 相図のプロット
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        sigma_map.T,
        aspect='auto',
        cmap='RdBu_r',
        origin='lower',
        vmin=-np.max(np.abs(sigma_map)),
        vmax=np.max(np.abs(sigma_map))
    )
    ax.set_xlabel('Context (prompt index)')
    ax.set_ylabel('Layer l')
    ax.set_title('Phase Diagram: σ(context, layer)')
    plt.colorbar(im, label='Order parameter σ')

    # 臨界線（σ=0の等高線）を重ねる
    ax.contour(sigma_map.T, levels=[0], colors='black', linewidths=2)

    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=150)
    plt.show()
```

---

## 実験6: モデル横断的な普遍基底との接続

### 目的

Universal Weight Subspace Hypothesisで見つかった普遍基底と、
相転移の秩序変数の方向ベクトルの関係を調べる。

### 仮説

もし普遍基底が「意味的力の種類」に対応しているなら、
秩序変数の方向（ê_A - ê_B）は普遍基底の線形結合として
少数の成分で書けるはず。

### 実験手順

```python
def analyze_universal_basis_connection(model, e_A, e_B):
    """
    秩序変数の方向が、Attention重み行列のPCA主成分で
    どの程度説明できるかを解析する
    """
    # W_QK行列を全ヘッド・全レイヤーから抽出
    all_W_QK = []
    for layer in range(model.cfg.n_layers):
        W_Q = model.blocks[layer].attn.W_Q  # (n_heads, d_model, d_head)
        W_K = model.blocks[layer].attn.W_K

        for head in range(model.cfg.n_heads):
            W_QK = W_Q[head].T @ W_K[head]  # (d_head, d_head)
            all_W_QK.append(W_QK.detach().cpu().numpy().flatten())

    all_W_QK = np.array(all_W_QK)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    pca.fit(all_W_QK)

    # 累積寄与率
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"Top  5 components: {cumvar[4]:.3f}")
    print(f"Top 10 components: {cumvar[9]:.3f}")
    print(f"Top 20 components: {cumvar[19]:.3f}")
    print(f"Top 50 components: {cumvar[49]:.3f}")

    # 秩序変数方向を基底で分解
    order_direction = (e_A - e_B).detach().cpu().numpy()
    # この方向がPCA空間でどう表現されるかを調べる
    # (注: 次元が異なるため直接の射影はできない。
    #  ここでは概念的なフレームワークとして記載)

    return pca, cumvar
```

---

## 解析と判定基準

### 成功基準

| 指標                        | 閾値       | 意味                     |
| --------------------------- | ---------- | ------------------------ |
| σ(l) のジャンプ幅           | > 0.1      | 検出可能な相転移が存在   |
| ランダウフィットの R²       | > 0.9      | 理論的予測との良好な一致 |
| β の標準偏差 / 平均         | < 0.3      | 臨界指数の普遍性を示唆   |
| χ のピーク位置と l_c のずれ | < 2 layers | 感受率と秩序変数の整合性 |
| l_c/L のモデル間分散        | < 0.1      | 臨界レイヤーの普遍性     |

### 結果の解釈

```
全指標クリア:
  → 意味確定はランダウ型相転移として記述できる
  → 「意味の物理学」の第一歩
  → 圧縮への直接応用が見えてくる

σのジャンプは見えるがランダウフィットが悪い:
  → 相転移は存在するが、ランダウ理論より複雑な理論が必要
  → それでも臨界レイヤーの存在は圧縮に有用

σのジャンプが見えない:
  → 意味確定は分散的プロセス
  → 相転移の描像は修正が必要
  → ただしレイヤーごとのσ変化パターン自体が有用な情報
```

---

## 発展実験（将来）

### A. 意味ポテンシャルの再構成

相転移が確認された場合、各レイヤーの σ の分布から
ポテンシャル V(σ) を逆問題として再構成する。

```
P(σ, l) ∝ exp(-V(σ, l) / T)

V(σ, l) = -T · log P(σ, l) + const
```

### B. 複数モデルの普遍基底のPCA

Universal Weight Subspace Hypothesisの追試:

```
1. GPT-2, Gemma-2, LLaMA, Mistral, Qwen等の W_QK を収集
2. 全モデルの W_QK を結合してPCA
3. 累積寄与率 vs 基底数のグラフを作成
4. 16-32個で90%以上説明できるか確認
```

### C. 基底の「意味」の解読

PCAで見つかった上位基底を、Attention patternの可視化と
組み合わせて、各基底がどのような「注目の種類」に
対応するかを分類する。

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

---

## メモ: この実験の意義

この実験は「LLMの内部を物理系として扱い、相転移を検出する」最初の試み
の一つとなる。成功した場合、以下の方向に発展する:

1. **圧縮**: FFNの重みを「ポテンシャルの係数」に置換
2. **解釈可能性**: 意味確定のメカニズムの物理的理解
3. **アーキテクチャ設計**: 「必要なレイヤー数」の理論的導出
4. **新たな学問**: 「意味の物理学 (Semantic Physics)」への第一歩
