"""Pre-tokenize a long-context validation set (PG19, optionally arXiv) for the
context-scaling experiment.

Context-scaling windows are confined to a single document, so evaluating at 64k or
128k tokens needs documents at least that long; only documents above
``--min-doc-tokens`` are kept. The text must be tokenized with the tokenizer of the
model under test (Qwen2.5 for the 3B grafts, GPT-2 for the from-scratch models), so
each tokenizer gets its own output directory.

Output, in the format read by LongformShardDataset:
  * ``val_shard_<i>.pt``        : 1-D LongTensor of concatenated token ids;
  * ``val_shard_<i>.bounds.pt`` : per-document end offsets, so that no window spans
    two documents.

Usage, once per tokenizer::

    # 3B grafts (Qwen tokenizer, vocab 151665)
    python -m utils.prepare_longctx_val --tokenizer Qwen/Qwen2.5-3B \
        --sources pg19,arxiv --val-docs 200 --min-doc-tokens 16384 \
        --out ./data_shards_longctx_qwen

    # from-scratch models (GPT-2 tokenizer, vocab 50257)
    python -m utils.prepare_longctx_val --tokenizer gpt2 \
        --sources pg19,arxiv --val-docs 200 --min-doc-tokens 16384 \
        --out ./data_shards_longctx_gpt2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _iter_source(name: str, pg19_split: str = "validation", pg19_skip: int = 0):
    """Yield text rows from a long-document source as a streaming dataset.

    pg19  : full-length English books (emozilla/pg19). The validation split has
            about 50 books; ``--pg19-split train`` gives thousands, and
            ``--pg19-skip`` drops the leading books, which the fine-tune used.
    arxiv : long papers (ccdv/arxiv-summarization). A different domain from the
            PG19 fine-tune, so its perplexity is out of distribution for the grafts.
    """
    from datasets import load_dataset
    if name == "pg19":
        ds = load_dataset("emozilla/pg19", split=pg19_split, streaming=True)
        for i, row in enumerate(ds):
            if i < pg19_skip:
                continue
            yield row["text"]
    elif name == "arxiv":
        ds = load_dataset("ccdv/arxiv-summarization", split="validation", streaming=True)
        for row in ds:
            yield row.get("article") or row.get("text") or ""
    else:
        raise ValueError(f"unknown source {name!r} (use pg19 / arxiv)")


def collect_long_docs(sources, tokenizer, eos_id, val_docs, min_doc_tokens,
                      pg19_split="validation", pg19_skip=0):
    """Tokenize documents from ``sources`` and keep those with at least
    ``min_doc_tokens`` tokens, until ``val_docs`` documents are gathered.

    Each kept document ends with EOS; ``bounds`` holds the cumulative end offsets.
    """
    all_ids: list[int] = []
    bounds: list[int] = []
    kept = 0
    seen = 0
    for src in sources:
        if kept >= val_docs:
            break
        # A failing source keeps the documents already gathered from earlier ones.
        try:
            it = _iter_source(src, pg19_split, pg19_skip) if src == "pg19" else _iter_source(src)
            for text in it:
                seen += 1
                if not text:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(ids) < min_doc_tokens:
                    continue
                all_ids.extend(ids)
                all_ids.append(eos_id)
                bounds.append(len(all_ids))
                kept += 1
                if kept % 10 == 0:
                    print(f"  [{src}] kept {kept}/{val_docs} long docs "
                          f"({len(all_ids):,} tokens, scanned {seen})")
                if kept >= val_docs:
                    break
        except Exception as exc:                      # noqa: BLE001
            print(f"  [{src}] SKIPPED ({type(exc).__name__}: {exc}); "
                  f"continuing with {kept} docs so far")
    if kept == 0:
        raise RuntimeError("no documents collected from any source")
    return (torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(bounds, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser(description="Pre-tokenize a long-context val set (PG19+arXiv) in a chosen tokenizer.")
    ap.add_argument("--tokenizer", required=True,
                    help="HF tokenizer id: 'Qwen/Qwen2.5-3B' (3B) or 'gpt2' (prototypes)")
    ap.add_argument("--sources", default="pg19,arxiv",
                    help="comma list of long-doc sources: pg19,arxiv")
    ap.add_argument("--val-docs", type=int, default=200,
                    help="number of LONG documents to keep")
    ap.add_argument("--min-doc-tokens", type=int, default=16384,
                    help="discard documents shorter than this (must be >= max eval L "
                         "for those docs to host a window at that L)")
    ap.add_argument("--pg19-split", default="validation",
                    help="PG19 split to draw from: 'validation' (~50 books) or 'train' "
                         "(thousands; use with --pg19-skip to stay held-out)")
    ap.add_argument("--pg19-skip", type=int, default=0,
                    help="skip the first K PG19 books (the fine-tune used the leading "
                         "books; skip past them when drawing eval books from 'train')")
    ap.add_argument("--out", required=True, help="output shard dir (keep tokenizers isolated)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos is None:
        eos = 0
    print(f"[tok] {args.tokenizer}: vocab={len(tok)}, eos_id={eos}")

    sources = [s for s in args.sources.split(",") if s]
    toks, bounds = collect_long_docs(sources, tok, eos, args.val_docs, args.min_doc_tokens,
                                     pg19_split=args.pg19_split, pg19_skip=args.pg19_skip)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(toks, out / "val_shard_0.pt")
    torch.save(bounds, out / "val_shard_0.bounds.pt")

    lens = bounds.diff(prepend=torch.tensor([0]))
    print(f"\n[saved] {out}/val_shard_0.pt: {toks.numel():,} tokens, {bounds.numel():,} docs")
    print(f"[docs] longest={int(lens.max()):,}  median={int(lens.median()):,}  "
          f"shortest={int(lens.min()):,}")
    for L in (16384, 32768, 65536, 131072):
        n = int((lens >= L).sum())
        print(f"  docs >= {L:>7,} tokens : {n}")


if __name__ == "__main__":
    main()
