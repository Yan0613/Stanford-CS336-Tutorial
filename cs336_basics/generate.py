"""Load a trained checkpoint and generate text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.nn import TransformerLM
from cs336_basics.preprocess import load_tokenizer
from cs336_basics.train import decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = load_tokenizer(Path(args.tokenizer_dir))
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint["iteration"]
    print(f"Loaded checkpoint from step {step}")

    output = decode(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        context_length=args.context_length,
    )
    print(args.prompt + output)


if __name__ == "__main__":
    main()
