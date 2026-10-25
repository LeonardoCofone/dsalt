"""Pre-tokenize PG19 with the Qwen2.5 tokenizer into training shards for the Phase-1
fine-tune of the grafted models.

PG19 books are long enough that one book fills a long window, and each book is one
document. All four grafted operators are fine-tuned on these same shards.

Output, in the format read by ``LongformShardDataset`` (utils/data_utils.py):
  * ``<prefix>_<i>.pt``        : 1-D LongTensor of concatenated token ids;
  * ``<prefix>_<i>.bounds.pt`` : per-document end offsets, so that no window spans
    two books.

Usage::

    python -m utils.prepare_pg19_qwen --qwen Qwen/Qwen2.5-3B         --seq-len 4096 --out ./data_shards_pg19_qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def tokenize_split(ds_iter, tokenizer, max_books: int, eos_id: int, batch: int = 32):
    """Tokenize up to ``max_books`` documents and return (tokens, doc_end_offsets).

    Books are tokenized in batches. Each book ends with EOS; ``bounds`` holds the
    cumulative end offsets.
    """
    all_ids: list[int] = []
    bounds:  list[int] = []
    n = 0
    buf: list[str] = []

    def flush(texts):
        nonlocal n
        if not texts:
            return
        enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            if not ids:
                continue
            all_ids.extend(ids)
            all_ids.append(eos_id)
            bounds.append(len(all_ids))
            n += 1
        print(f"  tokenized {n} books | {len(all_ids):,} tokens")

    for row in ds_iter:
        text = row.get("text") if isinstance(row, dict) else row["text"]
        if text:
            buf.append(text)
        if len(buf) >= batch:
            flush(buf); buf = []
        if max_books and n >= max_books:
            break
    if (not max_books) or n < max_books:
        flush(buf)
    return torch.tensor(all_ids, dtype=torch.long), torch.tensor(bounds, dtype=torch.long)


def save_shard(tokens: torch.Tensor, bounds: torch.Tensor, out_dir: Path, prefix: str, idx: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    tp = out_dir / f"{prefix}_{idx}.pt"
    bp = out_dir / f"{prefix}_{idx}.bounds.pt"
    torch.save(tokens, tp)
    torch.save(bounds, bp)
    print(f"[saved] {tp.name}: {tokens.numel():,} tokens, {bounds.numel():,} docs")


def main():
    ap = argparse.ArgumentParser(description="Pre-tokenize PG19 into DSALT shards (Qwen tokenizer).")
    ap.add_argument("--qwen",        default="Qwen/Qwen2.5-3B")
    ap.add_argument("--seq-len",     type=int, default=4096,
                    help="only used to warn if a book is shorter than one window")
    # A separate directory: the loader globs ``<prefix>_*.pt``, so mixing these shards
    # with others from another tokenizer would give out-of-range token ids.
    ap.add_argument("--out",         default="./data_shards_pg19_qwen")
    ap.add_argument("--train-books", type=int, default=800,
                    help="~80M tokens at ~100k/book; plenty for a 2000-step adaptation FT")
    ap.add_argument("--val-books",   type=int, default=20)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.qwen)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    print(f"[tok] {args.qwen}: vocab={len(tok)}, eos_id={eos}")

    out = Path(args.out)

    # PG19 as parquet (emozilla/pg19), streamed; the first train and validation books
    # are taken.
    print("[pg19] streaming train split ...")
    train_stream = load_dataset("emozilla/pg19", split="train", streaming=True)
    tr_tokens, tr_bounds = tokenize_split(train_stream, tok, args.train_books, eos)
    save_shard(tr_tokens, tr_bounds, out, "train_shard")

    print("[pg19] streaming validation split ...")
    val_stream = load_dataset("emozilla/pg19", split="validation", streaming=True)
    va_tokens, va_bounds = tokenize_split(val_stream, tok, args.val_books, eos)
    save_shard(va_tokens, va_bounds, out, "val_shard")

    win = args.seq_len + 1
    tr_win = int((tr_bounds.diff(prepend=torch.tensor([0])) // win).clamp(min=0).sum())
    print(f"\n[ok] train ~{tr_win:,} windows of {args.seq_len} tokens "
          f"({tr_tokens.numel():,} tokens / {tr_bounds.numel()} books)")
    if tr_win < 100:
        print("[warn] very few training windows; raise --train-books or lower --seq-len.")


if __name__ == "__main__":
    main()
