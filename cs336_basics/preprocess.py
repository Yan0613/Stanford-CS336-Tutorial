"""Train BPE on TinyStories and write tokenized uint16 files for train.py."""

from __future__ import annotations

import argparse
import array
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer, train_bpe


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    tokenizer_dir: Path,
) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    with open(tokenizer_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(tokenizer_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    with open(tokenizer_dir / "special_tokens.pkl", "wb") as f:
        pickle.dump(special_tokens, f)


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    with open(tokenizer_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(tokenizer_dir / "merges.pkl", "rb") as f:
        merges = pickle.load(f)
    with open(tokenizer_dir / "special_tokens.pkl", "rb") as f:
        special_tokens = pickle.load(f)
    return Tokenizer(vocab, merges, special_tokens)


def tokenize_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    """Stream-encode a text file and write tokens as raw uint16 bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = array.array("H")
    token_count = 0
    t0 = time.time()

    with open(input_path, encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        for tid in tokenizer.encode_iterable(f_in):
            if tid >= 65536:
                raise ValueError(f"token id {tid} does not fit in uint16")
            chunk.append(tid)
            token_count += 1
            if len(chunk) >= 1_000_000:
                chunk.tofile(f_out)
                chunk = array.array("H")
        if chunk:
            chunk.tofile(f_out)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"  {input_path.name}: {token_count:,} tokens -> {output_path} "
        f"({size_mb:.1f} MB, {elapsed:.1f}s)"
    )
    return token_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE and tokenize TinyStories")
    parser.add_argument(
        "--train-txt",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Raw training text file",
    )
    parser.add_argument(
        "--valid-txt",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
        help="Raw validation text file",
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Directory to save/load vocab + merges",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Directory for tokenized .bin files",
    )
    parser.add_argument(
        "--skip-train-bpe",
        action="store_true",
        help="Reuse tokenizer from --tokenizer-dir instead of retraining",
    )
    args = parser.parse_args()
    special_tokens = [args.special_token]

    if args.skip_train_bpe:
        print(f"Loading tokenizer from {args.tokenizer_dir} ...")
        tokenizer = load_tokenizer(args.tokenizer_dir)
    else:
        print(f"Training BPE on {args.train_txt} (vocab_size={args.vocab_size}) ...")
        t0 = time.time()
        vocab, merges = train_bpe(args.train_txt, args.vocab_size, special_tokens)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s: {len(vocab)} vocab entries, {len(merges)} merges")
        save_tokenizer(vocab, merges, special_tokens, args.tokenizer_dir)
        print(f"  saved to {args.tokenizer_dir}/")
        tokenizer = Tokenizer(vocab, merges, special_tokens)

    print("Tokenizing ...")
    train_out = args.out_dir / "train.bin"
    valid_out = args.out_dir / "valid.bin"
    train_tokens = tokenize_file(tokenizer, args.train_txt, train_out)
    valid_tokens = tokenize_file(tokenizer, args.valid_txt, valid_out)

    # Sanity check: memmap loads correctly
    train_mmap = np.memmap(train_out, dtype=np.uint16, mode="r")
    valid_mmap = np.memmap(valid_out, dtype=np.uint16, mode="r")
    assert len(train_mmap) == train_tokens
    assert len(valid_mmap) == valid_tokens
    print("Done.")
    print(f"  train: {train_out} ({len(train_mmap):,} tokens)")
    print(f"  valid: {valid_out} ({len(valid_mmap):,} tokens)")


if __name__ == "__main__":
    main()
