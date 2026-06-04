# train.py

import argparse
import numpy as np
import torch
from cs336_basics.nn import (
    TransformerLM, AdamW, cross_entropy, softmax,
    gradient_clipping, learning_rate_schedule,
    get_batch, save_checkpoint, load_checkpoint
)
from cs336_basics.tokenizer import Tokenizer

def main(args):
    # 1. 加载数据（用 memmap 内存高效加载）
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data   = np.memmap(args.val_data,   dtype=np.uint16, mode='r')

    # 2. 建模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # 3. 建 optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    # 4. 训练循环
    for step in range(args.max_steps):
        # 学习率调度
        lr = learning_rate_schedule(step, args.lr, args.lr_min,
                                    args.warmup_steps, args.max_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # 取 batch
        x, y = get_batch(train_data, args.batch_size,
                         args.context_length, args.device)

        # forward + loss
        optimizer.zero_grad()
        logits = model(x)                           # (batch, seq, vocab)
        loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

        # backward
        loss.backward()

        # gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # optimizer step
        optimizer.step()

        # 打印 log
        if step % args.log_interval == 0:
            print(f"step {step}: train_loss={loss.item():.4f}, lr={lr:.6f}")

        # 验证集 loss
        if step % args.eval_interval == 0:
            with torch.no_grad():
                xv, yv = get_batch(val_data, args.batch_size,
                                   args.context_length, args.device)
                val_logits = model(xv)
                val_loss = cross_entropy(val_logits.view(-1, args.vocab_size), yv.view(-1))
            print(f"step {step}: val_loss={val_loss.item():.4f}")

        # 保存 checkpoint
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step,
                           f"{args.out_dir}/ckpt_{step}.pt")

def decode(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)          # (1, seq, vocab)
            logits = logits[0, -1, :]    # (vocab,) 只取最后一个位置

            # temperature scaling
            logits = logits / max(temperature, 1e-8)

            # softmax → 概率
            probs = softmax(logits, dim=0)

            # top-p 过滤
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                # 找到累积概率超过 top_p 的位置，之后的全部丢弃
                cutoff = (cumsum - sorted_probs) >= top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs /= sorted_probs.sum()
                # 还原顺序
                probs = torch.zeros_like(probs)
                probs[sorted_idx] = sorted_probs

            # 采样
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                break

            ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

    generated_ids = ids[0].tolist()[len(token_ids):]
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model",    type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads",  type=int, default=8)
    parser.add_argument("--d_ff",       type=int, default=1408)  # 8/3 * 512，取64倍数
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--lr_min",     type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps",  type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_interval",  type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--out_dir",    type=str, default="checkpoints")
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()
    main(args)