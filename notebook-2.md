# Part 2：Transformer 架构实现笔记

## 目录

- [1. 张量维度基础](#1-张量维度基础)
- [2. Token vs 向量：Embedding 的作用](#2-token-vs-向量embedding-的作用)
- [3. Batch 与 seq_len：不等长序列怎么处理](#3-batch-与-seq_len不等长序列怎么处理)
- [4. 行向量 vs 列向量：数学和代码的约定差异](#4-行向量-vs-列向量数学和代码的约定差异)
- [5. 参数初始化](#5-参数初始化)
- [6. einops 推荐包](#6-einops-推荐包)
- [7. RMSNorm](#7-rmsnorm)
- [8. Feed-Forward Network（FFN）与 SwiGLU](#8-feed-forward-networkffn与-swiglu)
- [9. 位置编码：绝对位置 vs RoPE](#9-位置编码绝对位置-vs-rope)
- [10. Scaled Dot-Product Attention](#10-scaled-dot-product-attention)
- [11. Multi-Head Self-Attention](#11-multi-head-self-attention)
- [12. TransformerBlock 与 TransformerLM](#12-transformerblock-与-transformerlm)

---

## 1. 张量维度基础

### (batch_size, sequence_length, d_model) 三个维度的含义

| 维度 | 含义 | 典型值 |
|---|---|---|
| `batch_size` | 同时处理多少条样本 | 32 / 64 / 256 |
| `sequence_length` | 每条样本有多少个 token | 256 / 1024 / 8192 |
| `d_model` | 每个 token 用多少维向量表示 | 128 / 768 / 4096 |

### 完整流水线的形状变化

```
                                                  shape
──────────────────────────────────────────────────────────────────────────
原始文本                                          (str)
   │ tokenizer.encode
   ▼
token id 序列                                     (batch_size, seq_len)       ← int 张量
   │ Embedding 查表 (vocab_size, d_model)
   ▼
词向量序列                                        (batch_size, seq_len, d_model)  ← float 张量
   │ TransformerBlock × N
   │   ├ RMSNorm                                  (batch_size, seq_len, d_model)
   │   ├ Multi-Head Attention (内部变 4D)         (batch_size, seq_len, d_model)
   │   ├ + residual                               (batch_size, seq_len, d_model)
   │   ├ RMSNorm                                  (batch_size, seq_len, d_model)
   │   ├ SwiGLU FFN                               (batch_size, seq_len, d_model)
   │   └ + residual                               (batch_size, seq_len, d_model)
   │ 最后一层 RMSNorm
   ▼
hidden states                                     (batch_size, seq_len, d_model)
   │ lm_head (Linear: d_model → vocab_size)
   ▼
logits                                            (batch_size, seq_len, vocab_size)
```

**注意**：3D `(batch_size, seq_len, d_model)` 从 Embedding 后一直保持到最后一层 RMSNorm。只有最后一步 lm_head 把 `d_model` 换成 `vocab_size`。

Attention 内部临时多一个 `num_heads` 维：

```
d_model = num_heads × head_dim

进 attention 前:  (batch, seq, d_model)
内部分头后:       (batch, num_heads, seq, head_dim)   ← 4D
出 attention:     (batch, seq, d_model)               ← 拼回 3D
```

---

## 2. Token vs 向量：Embedding 的作用

"token" 在 LLM 流水线的不同阶段含义不同：

| 阶段 | "token" 是什么 | 形式 |
|---|---|---|
| tokenizer 输出 | 整数 ID | `5234` (int) |
| Embedding 之后 | d_model 维浮点向量 | `[0.1, 0.3, -0.2, ...]` (float) |

**为什么 token id 不能直接当数字算？**

token id 是按 BPE 训练顺序分配的，相邻的 id（如 5234 和 5235）在语义上毫无关系。不能当数字用于矩阵乘，必须转成独立的向量。

**Embedding 的本质：查表**

```python
# weight shape: (vocab_size, d_model)
# 每一行 = 一个 token 对应的 d_model 维向量

def forward(self, token_ids):
    return self.weight[token_ids]   # 用整数 id 当行索引取出对应行
```

```
输入 token_ids = [102, 111, 123]      shape: (3,)  整数
输出            = [weight[102],       shape: (3, d_model)  浮点
                   weight[111],
                   weight[123]]
```

**Embedding 的"语义"从哪里来？**

初始时随机初始化，通过训练（梯度下降）不断更新。训练后语义相近的 token（如 "cat" 和 "dog"）向量会靠近，语义无关的会拉开。

---

## 3. Batch 与 seq_len：不等长序列怎么处理

PyTorch 张量必须是规则形状（每行等长），所以同一个 batch 内的 `seq_len` 必须相同。处理不等长序列有三种方式：

| 方式 | 适用场景 | 说明 |
|---|---|---|
| **Padding** | SFT / RLHF | 短序列补 `<pad>`，配合 attention mask |
| **Packing** | **预训练（Assignment 1 用这种）** | 所有文档拼成长流，按 context_length 切块，无浪费 |
| **Bucket Batching** | SFT / eval | 把长度相近的样本放同一 batch |

**Packing 的流程（Assignment 1 的 get_batch）：**

```
所有 token 拼成 1D 数组：[A, B, C, D, E, F, G, H, ...]
随机采 batch_size 个起始位置：starts = [3, 7, 1, ...]
每个起点取 context_length 长度：
  x = dataset[start : start + context_length]
  y = dataset[start+1 : start+1 + context_length]   ← 错位 1 位作为标签
```

---

## 4. 行向量 vs 列向量：数学和代码的约定差异

| | 数学（论文/教材） | PyTorch 代码 |
|---|---|---|
| 向量方向 | 列向量 | 行向量 |
| Linear 公式 | `y = Wx`（列向量） | `y = xW^T`（行向量） |
| batch 维置于 | 最后（不方便） | **最前**（符合 row-major 内存） |

`W` 的 shape 都是 `(d_out, d_in)`，只是向量方向不同，所以代码里要 `.T`：

```python
# x: (..., d_in)  行向量
return x @ self.weight.T   # (d_in, d_out) → 输出 (..., d_out)
```

用 einsum 可以避免手动转置，直接按维度名字收缩。

---

## 5. 参数初始化

| 模块 | 分布 | 参数 |
|---|---|---|
| Linear weight | 截断正态 | `μ=0, σ²=2/(d_in+d_out)`, 截断 `[-3σ, 3σ]` |
| Embedding weight | 截断正态 | `μ=0, σ²=1`, 截断 `[-3, 3]` |
| RMSNorm weight | 全 1 | — |

**Xavier 初始化直觉**（Linear）：
- 前向：方差稳定要求 `Var(W) ≈ 1/d_in`
- 反向：方差稳定要求 `Var(W) ≈ 1/d_out`
- 折中：`Var(W) = 2/(d_in + d_out)`

**截断正态**：正态分布 + 超出 `[-3σ, 3σ]` 的样本被拒绝重采，彻底排除极端初始值。

```python
std = math.sqrt(2.0 / (d_in + d_out))
nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
```

---

## 6. einops 推荐包

PDF 推荐用 einops 处理维度操作，让代码更可读。

**三大操作：**

```python
from einops import rearrange, einsum, reduce

# rearrange：重排/合并/拆分维度
rearrange(x, "b s (h d) -> b h s d", h=12)   # 拆出多头

# einsum：张量收缩（广义矩阵乘）
attn = einsum(Q, K, "b h q d, b h k d -> b h q k")  # Q @ K^T

# reduce：归约
reduce(x, "a b c -> a b", "mean")   # 沿最后维求均值
```

**einsum 规则**：`->` 左边出现但右边没出现的维度被求和（收缩），两边都出现的维度保留。

---

## 7. RMSNorm

### 为什么需要 Normalization？

每层输出的数值尺度可能大小不一，累积后导致训练不稳定。Norm 把每个 token 向量归一化到合理尺度。

### Pre-norm vs Post-norm

- **Post-norm**（原始 Transformer）：Attention/FFN → Norm
- **Pre-norm**（现代 LLM）：Norm → Attention/FFN → 残差

Pre-norm 把 norm 挪到子层**之前**，训练更稳定：

```
Post-norm: x → sublayer(x) → norm → + x（残差）
Pre-norm:  x → norm(x) → sublayer → + x（残差）   ← 现代做法
```

![pre-norm](file:///data/workspace/llm_course/assignment1-basics/assets/image.png)

### RMSNorm vs LayerNorm

| | LayerNorm | RMSNorm |
|---|---|---|
| 步骤 | 减均值 + 除标准差 + 缩放 + 偏移 | 除 RMS + 缩放 |
| 参数量 | weight + bias（2 × d_model） | 只有 weight（1 × d_model） |
| 速度 | — | 快约 10~15% |
| 现代大模型 | GPT-2 | LLaMA / Mistral / Qwen |

**去掉减均值为什么也能 work？**
- SiLU 等激活函数本身会压负数，向量不容易出现整体偏置
- 残差连接不断加回原始信息，均值偏移不容易累积

### RMSNorm 公式

$$\text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot g_i, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

**计算步骤：**

```
输入 x: (..., d_model)

1. x² 逐元素平方              → (..., d_model)
2. .mean(dim=-1, keepdim=True) → (..., 1)
3. + eps，开方，得到 RMS       → (..., 1)
4. x / rms                    → (..., d_model)   广播
5. * weight                   → (..., d_model)   逐元素缩放
```

**实现：**

```python
def forward(self, x):
    rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
    return x / rms * self.weight
```

---

## 8. Feed-Forward Network（FFN）与 SwiGLU

### FFN 的作用

Transformer 每个 Block 由两部分交替构成：
- **Attention**：负责"看别的 token"，汇聚上下文信息（线性操作）
- **FFN**：负责"处理单个 token"，增加非线性表达能力

MHA 做的都是线性变换，表达能力有限。FFN 通过激活函数引入非线性，大幅提升模型容量。

**FFN 的结构：升维 → 非线性 → 降维**

```
x (d_model) → W1 → (d_ff) → 激活 → (d_ff) → W2 → (d_model)
                    ↑
             d_ff ≈ 8/3 × d_model（取 64 的倍数）
```

**FFN 存储了什么？**（实验发现）每个神经元的 key vector 对应特定语义类别（"体育"、"欧洲地理"等），FFN 相当于一个软查找表/知识库。

### 激活函数演进

| 激活 | 公式 | 问题 |
|---|---|---|
| ReLU | `max(0, x)` | Dying ReLU，梯度消失 |
| SiLU/Swish | `x · σ(x)` | 更平滑，梯度不突然变 0 |
| SwiGLU | `W2(SiLU(W1x) ⊙ W3x)` | 门控机制，效果最好 |

### SwiGLU 公式

$$\text{FFN}(x) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)$$

- `W1, W3`：`(d_model → d_ff)`，升维投影
- `W2`：`(d_ff → d_model)`，降维投影
- `⊙`：逐元素相乘（门控）

**门控直觉**：`SiLU(W1x)` 是"候选内容"，`W3x` 是"开关强度"，两者相乘让模型有选择地传递信息。

实验数据（Shazeer 2020）：SwiGLU 困惑度 17.65，明显优于 ReLU（18.60）和 SiLU（18.22）。

---

## 9. 位置编码：绝对位置 vs RoPE

### 为什么需要位置编码？

Attention 是全局点积，本质上不区分顺序。"狗咬了人" 和 "人咬了狗" 的 token 集合一样，没有位置信息就无法区分。

### 两种方案对比

| | 绝对位置编码（GPT-2） | RoPE（LLaMA） |
|---|---|---|
| 在哪施加 | Embedding 之后（加法） | 每个 Block 的 Q/K 投影之后（旋转） |
| 编码的是 | 绝对位置 i | 相对位置 i-j |
| 可学习参数 | 有（可学版）或无（固定 sin/cos） | **无**（固定公式） |
| 超出训练长度 | 崩溃（可学版）或效果变差 | 效果下降但不崩 |
| 外推性 | 差 | **好** |

### 绝对位置编码

```python
position_emb = nn.Embedding(max_seq_len, d_model)   # 每个位置一个向量
positions = torch.arange(seq_len)
x = token_emb + position_emb(positions)             # 语义 + 位置，一次性加入
```

- Token Embedding 按 token_id 查（词表有多少种词就有多少行）
- Position Embedding 按位置编号查（max_seq_len 行）
- 两者相加，把语义和位置合并

### RoPE（Rotary Position Embedding）

**核心思想**：对 Q/K 施加旋转，使点积 `Q_i · K_j` 只依赖相对位置 `i-j`：

$$Q'_i = R(i) \cdot Q_i, \quad K'_j = R(j) \cdot K_j$$
$$Q'_i \cdot K'_j = Q_i \cdot R(j-i) \cdot K_j \quad \text{（只依赖相对位置）}$$

**旋转方式**：d_k 维向量两两配对，分成 d_k/2 对，每对独立旋转：

$$\theta_{i,k} = \frac{i}{\Theta^{2k/d}}, \quad \Theta = 10000$$

- k 越小（维度对靠前）：分母小，旋转快 → 区分相邻位置
- k 越大（维度对靠后）：分母大，旋转慢 → 区分远距离位置
- 多个频率组合：覆盖所有粒度（类比时钟的时针/分针/秒针）

**为什么 Θ=10000？** 最慢维度旋转一圈需要 62800 个位置，覆盖常见序列长度。LLaMA 3 用 Θ=500000 支持 128k 上下文。

**为什么施加在 Q/K 而非 Embedding？** 位置编码目的是让 Attention score（点积）感知相对位置，只有在点积处施加旋转才能实现这个数学性质。

**实现要点**：
- cos/sin 表在 `__init__` 预计算，用 `register_buffer` 存储（不训练，跟随 `.to(device)` 移动）
- `forward(x, token_positions)` 用 `token_positions` 索引取对应行的 cos/sin
- 旋转公式：

```python
x_even = x[..., 0::2]   # 偶数维
x_odd  = x[..., 1::2]   # 奇数维

x_even_rot = x_even * cos - x_odd * sin
x_odd_rot  = x_even * sin + x_odd * cos

out = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
```

---

## 10. Scaled Dot-Product Attention

### 公式

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 为什么要除 sqrt(d_k)？

`d_k` 越大，点积数值越大，softmax 输出越极端（一个接近 1，其余接近 0），梯度消失。除以 `sqrt(d_k)` 把数值拉回合理范围。

### Causal Mask（因果掩码）

语言模型训练时，位置 i 不能看到未来的 token（否则预测任务变得没意义）：

```
mask = tril(ones(seq, seq)):
[[T, F, F, F],   ← 位置 0 只能看自己
 [T, T, F, F],   ← 位置 1 能看 0 和 1
 [T, T, T, F],
 [T, T, T, T]]

False 位置的 score 设为 -inf，softmax 后变 0
```

### 数值稳定的 Softmax

```
softmax(v)_i = exp(v_i - c) / Σ exp(v_j - c)，c = max(v)
```

减最大值不改变结果，但防止 `exp` 溢出。

### 计算步骤

```python
d_k = Q.shape[-1]
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # 注意用 transpose 而非 .T
if mask is not None:
    scores = scores.masked_fill(~mask, float('-inf'))
return softmax(scores, dim=-1) @ V
```

**注意**：用 `K.transpose(-2, -1)` 而非 `K.T`，后者会反转所有维度，batch 维也会被转置。

---

## 11. Multi-Head Self-Attention

### 为什么要分头？

单头只能学一种"关注模式"。多头让模型并行学多种模式：
- 头 1：专注语法关系（主谓）
- 头 2：专注语义关系（同义词）
- 头 3：专注位置关系（相邻词）

### 数学公式

$$\text{MultiHeadSelfAttention}(x) = W_O \cdot \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$
$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

参数：`W_Q, W_K, W_V` shape `(d_model, d_model)`，`W_O` shape `(d_model, d_model)`

`d_k = d_v = d_model / num_heads`（head_dim）

### Forward 流程

```
x: (batch, seq, d_model)
   │ W_Q/W_K/W_V 投影
   ▼
Q, K, V: (batch, seq, d_model)
   │ reshape + transpose，拆成多头
   ▼
Q, K, V: (batch, num_heads, seq, head_dim)
   │ RoPE(Q), RoPE(K)   ← V 不旋转
   │ causal mask
   │ Scaled Dot-Product Attention
   ▼
out: (batch, num_heads, seq, head_dim)
   │ transpose + reshape，合并多头
   ▼
out: (batch, seq, d_model)
   │ W_O 投影
   ▼
output: (batch, seq, d_model)
```

**关键代码：**

```python
# 拆分多头
Q = Q.view(batch, seq, num_heads, head_dim).transpose(1, 2)   # (batch, heads, seq, head_dim)

# 合并多头
out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
```

---

## 12. TransformerBlock 与 TransformerLM

### TransformerBlock（Pre-norm 结构）

```python
# 第一子层：Attention + 残差
x = x + self.attn(self.norm1(x))

# 第二子层：FFN + 残差
x = x + self.ffn(self.norm2(x))
```

**易错点**：`x = self.norm1(x); x = self.attn(x) + x` 是错的！
这样 `x` 在第一行已经被 norm 覆盖，残差加的不是原始 x，而是 norm 后的 x。
必须先保存原始 x，再加残差。

### TransformerLM（完整模型）

```
token_ids (batch, seq)
   │ token_embeddings: Embedding(vocab_size, d_model)
   ▼
x (batch, seq, d_model)
   │ layers[0]: TransformerBlock
   │ layers[1]: TransformerBlock
   │ ...
   │ layers[N-1]: TransformerBlock
   ▼
x (batch, seq, d_model)
   │ ln_final: RMSNorm
   ▼
x (batch, seq, d_model)
   │ lm_head: Linear(d_model, vocab_size)
   ▼
logits (batch, seq, vocab_size)
```

**注意**：多个 Block 用 `nn.ModuleList` 装，普通 Python list 会导致参数无法被 optimizer 识别。
