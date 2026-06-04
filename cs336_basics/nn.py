import math
import os
from typing import IO, BinaryIO
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, std=std, 
            a=-3.0 * std, b=3.0 * std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (..., in_features)
        # 返回: (..., out_features)
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...)        整数,每个元素在 [0, num_embeddings)
        # 返回:      (..., embedding_dim)   float
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, 
                 device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, 
                 device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        return self.linear2(h * torch.sigmoid(h) * self.gate(x))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 每对维度的频率：shape (d_k/2,)
        # k = 0,1,...,d_k/2-1
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (k / d_k))   # shape (d_k/2,)

        # 每个位置的角度：shape (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_seq_len, d_k/2)

        # 预计算 cos 和 sin，存为 buffer（不训练，随模型移动设备）
        self.register_buffer("cos", torch.cos(angles), persistent=False)  # (max_seq_len, d_k/2)
        self.register_buffer("sin", torch.sin(angles), persistent=False)  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape:               (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)    整数，值在 [0, max_seq_len)

        # 用 token_positions 取对应位置的 cos/sin
        cos = self.cos[token_positions]   # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]   # (..., seq_len, d_k/2)

        # 拆分奇偶维度
        x_even = x[..., 0::2]   # (..., seq_len, d_k/2)
        x_odd  = x[..., 1::2]   # (..., seq_len, d_k/2)

        # 旋转
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_even * sin + x_odd * cos

        # 交织回去：把 even 和 odd 重新合并成 (..., seq_len, d_k)
        out = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (..., seq_len, d_k/2, 2)
        out = out.flatten(-2)                                 # (..., seq_len, d_k)
        return out

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return softmax(scores, dim=-1) @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆分多头：(batch, seq, d_model) → (batch, num_heads, seq, head_dim)
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # causal mask：下三角为 True，防止看到未来 token
        mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()

        # attention: (batch, num_heads, seq, head_dim)
        out = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头：(batch, num_heads, seq, head_dim) → (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        return self.o_proj(out)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆分多头：(batch, seq, d_model) → (batch, num_heads, seq, head_dim)
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # 对 Q 和 K 施加 RoPE（V 不动）
        if token_positions is None:
            token_positions = torch.arange(seq, device=x.device)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # causal mask
        mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()

        # attention
        out = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits:  (..., vocab_size)  原始未归一化的得分
    # targets: (...)              整数，每个值是正确 token 的 id
    c = logits.max(dim=-1, keepdim=True).values          # 减最大值，数值稳定
    shifted = logits - c
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))  # log(Σ exp)
    correct_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # 正确 token 的 logit
    return (-correct_logits + log_sum_exp).mean()

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    # 第 1 步：收集所有有梯度的参数的 grad
    grads = [p.grad for p in parameters if p.grad is not None]
    
    # 第 2 步：算所有梯度拼成一个大向量的 L2 范数
    total_norm = torch.sqrt(sum(g.norm()**2 for g in grads))
    
    # 第 3 步：如果超出，等比缩小
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)   # in-place 修改


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]

                # 初始化 state
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # bias-corrected lr
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # weight decay（直接对参数 in-place）
                p.data.mul_(1 - lr * lam)

                # moment 更新
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # 参数更新
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

        return loss

def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    return min_learning_rate


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s+1 : s+1 + context_length] for s in starts])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)