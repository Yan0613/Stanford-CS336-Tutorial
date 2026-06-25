# train.py

import argparse
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import wandb
from cs336_basics.nn import (
    TransformerLM, AdamW, cross_entropy, softmax,
    gradient_clipping, learning_rate_schedule,
    get_batch, save_checkpoint, load_checkpoint
)
from cs336_basics.tokenizer import Tokenizer


@dataclass
class TrainConfig:
    train_data: str
    val_data: str
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    lr: float
    lr_min: float
    weight_decay: float
    grad_clip: float
    batch_size: int
    max_steps: int
    warmup_steps: int
    log_interval: int
    eval_interval: int
    eval_batches: int
    save_interval: int
    out_dir: str
    device: str
    wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_entity: str | None


def init_wandb(args: TrainConfig) -> None:
    if not args.wandb:
        return
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=asdict(args),
    )


def loss_to_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def eval_model(
    model: torch.nn.Module,
    val_data: np.memmap,
    args: TrainConfig,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(args.eval_batches):
            xv, yv = get_batch(val_data, args.batch_size, args.context_length, args.device)
            val_logits = model(xv)
            losses.append(
                cross_entropy(val_logits.view(-1, args.vocab_size), yv.view(-1)).item()
            )
    model.train()
    return sum(losses) / len(losses)


def main(args: TrainConfig):
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )

    init_wandb(args)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    try:
        for step in range(args.max_steps):
            lr = learning_rate_schedule(
                step, args.lr, args.lr_min, args.warmup_steps, args.max_steps
            )
            for g in optimizer.param_groups:
                g['lr'] = lr

            x, y = get_batch(train_data, args.batch_size,
                             args.context_length, args.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss = loss.item()
            train_ppl = loss_to_perplexity(train_loss)
            metrics: dict[str, float] = {}

            if step % args.log_interval == 0:
                metrics.update({
                    "train/loss": train_loss,
                    "train/ppl": train_ppl,
                    "lr": lr,
                })
                print(f"step {step}: train_loss={train_loss:.4f}, train_ppl={train_ppl:.1f}, lr={lr:.6f}")

            if step % args.eval_interval == 0:
                val_loss = eval_model(model, val_data, args)
                val_ppl = loss_to_perplexity(val_loss)
                metrics.update({
                    "val/loss": val_loss,
                    "val/ppl": val_ppl,
                })
                print(f"step {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}")

            if args.wandb and metrics:
                wandb.log(metrics, step=step)

            if step % args.save_interval == 0 and step > 0:
                save_checkpoint(model, optimizer, step, f"{args.out_dir}/ckpt_{step}.pt")
    finally:
        if args.wandb:
            wandb.finish()

def decode(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    context_length: int | None = None,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    with torch.no_grad():
        for _ in range(max_tokens):
            model_input = ids
            if context_length is not None and model_input.shape[1] > context_length:
                model_input = model_input[:, -context_length:]
            logits = model(model_input)          # (1, seq, vocab)
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
    parser.add_argument("--eval_batches", type=int, default=10,
                        help="Number of val batches to average for val/loss")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--out_dir",    type=str, default="checkpoints")
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                        help="Log metrics to Weights & Biases (use --no-wandb to disable)")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    ns = parser.parse_args()
    main(TrainConfig(
        train_data=ns.train_data,
        val_data=ns.val_data,
        vocab_size=ns.vocab_size,
        context_length=ns.context_length,
        d_model=ns.d_model,
        num_layers=ns.num_layers,
        num_heads=ns.num_heads,
        d_ff=ns.d_ff,
        rope_theta=ns.rope_theta,
        lr=ns.lr,
        lr_min=ns.lr_min,
        weight_decay=ns.weight_decay,
        grad_clip=ns.grad_clip,
        batch_size=ns.batch_size,
        max_steps=ns.max_steps,
        warmup_steps=ns.warmup_steps,
        log_interval=ns.log_interval,
        eval_interval=ns.eval_interval,
        eval_batches=ns.eval_batches,
        save_interval=ns.save_interval,
        out_dir=ns.out_dir,
        device=ns.device,
        wandb=ns.wandb,
        wandb_project=ns.wandb_project,
        wandb_run_name=ns.wandb_run_name,
        wandb_entity=ns.wandb_entity,
    ))