"""Pre-tokenize the training and validation shards of the from-scratch models (GPT-2
tokenizer), in the format read by ``utils/data_utils.py``.

Documents are streamed from three sources and interleaved by document with the
fractions in SOURCES (PG19 0.2, arXiv 0.3, FineWeb-Edu 0.5) until a source runs out;
the shards fill in that order, so the first shards are mostly PG19 books and the
last ones FineWeb-Edu pages. The validation shard uses the validation splits of PG19
and arXiv only. Shard sizes and counts come from CFG (30 x 500M tokens).

Training windows are cut within documents (data_utils.LongformShardDataset), so only
documents longer than ``seq_len + 1`` tokens produce training windows. For the shards
used in the paper about 56%, 15% and 29% of the windows come from PG19, arXiv and
FineWeb-Edu; shard 30 is an exact copy of shard 1.

Missing shards are produced on a rerun, existing ones are skipped.

Usage::

    python -m utils.prepare_train_shards
"""
import os
import time
import warnings
import multiprocessing as mp
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from utils.utils_dsalt import CFG

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "68000"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Token indices sequence length")

TOKENS_PER_SHARD = CFG["tokens_per_shard"]
TARGET_SHARDS    = CFG["target_shards"]
VAL_SHARDS       = CFG["val_shards"]
OUTPUT_DIR       = Path("./data_shards")
NUM_WORKERS      = 24

SOURCES = [
    {"path": "emozilla/pg19",            "config": None,          "field": "text",    "frac": 0.20, "val": True},
    {"path": "ccdv/arxiv-summarization", "config": None,          "field": "article", "frac": 0.30, "val": True},
    {"path": "HuggingFaceFW/fineweb-edu","config": "sample-10BT", "field": "text",    "frac": 0.50, "val": False},
]

_TOK = None
def _get_tok():
    global _TOK
    if _TOK is None:
        _TOK = GPT2TokenizerFast.from_pretrained("gpt2")
    return _TOK


def tokenize_doc(text):
    tok = _get_tok()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ids.append(tok.eos_token_id)
    return ids


def _doc_iter(path, config, field, split):
    ds = load_dataset(path, config, split=split, streaming=True)
    for row in ds:
        text = (row.get(field) or "").strip()
        if text:
            yield text


def _mixed_docs(split):
    srcs  = SOURCES if split == "train" else [s for s in SOURCES if s["val"]]
    gens  = [_doc_iter(s["path"], s["config"], s["field"], split) for s in srcs]
    fracs = [s["frac"] for s in srcs]
    counters = [0.0] * len(gens)
    alive    = [True] * len(gens)
    while any(alive):
        for i, g in enumerate(gens):
            if not alive[i]:
                continue
            counters[i] += fracs[i]
            while counters[i] >= 1.0:
                counters[i] -= 1.0
                try:
                    yield next(g)
                except StopIteration:
                    alive[i] = False
                    break


def _shard_paths(file_prefix, idx):
    spath = OUTPUT_DIR / f"{file_prefix}_{idx:04d}.pt"
    bpath = OUTPUT_DIR / f"{file_prefix}_{idx:04d}.bounds.pt"
    return spath, bpath


def _missing_shards(file_prefix, n_shards):
    missing = []
    for idx in range(1, n_shards + 1):
        spath, bpath = _shard_paths(file_prefix, idx)
        if not (spath.exists() and bpath.exists()):
            missing.append(idx)
    return missing


def _write_shard(file_prefix, idx, token_buffer, doc_bounds, cut):
    spath, bpath = _shard_paths(file_prefix, idx)
    shard_data = token_buffer[:cut]
    in_shard   = [b for b in doc_bounds if b <= cut]
    if not in_shard or in_shard[-1] != cut:
        in_shard.append(cut)
    torch.save(torch.tensor(shard_data, dtype=torch.int32), spath)
    torch.save(torch.tensor(in_shard,  dtype=torch.int64), bpath)
    return spath.name, len(in_shard)


def run_split(file_prefix, split, n_shards, allow_partial):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = _missing_shards(file_prefix, n_shards)
    if not missing:
        print(f"[skip:{file_prefix}] all {n_shards} shards already present")
        return
    print(f"[resume:{file_prefix}] missing {len(missing)}/{n_shards}: {missing}")

    target_set   = set(missing)
    next_target  = missing[0]
    produced     = 0
    pool         = mp.Pool(NUM_WORKERS)
    token_buffer = []
    doc_bounds   = []
    total_tokens = 0
    t_start      = time.time()
    pbar = tqdm(total=len(missing) * TOKENS_PER_SHARD, unit="tok", unit_scale=True, desc=file_prefix)

    imap = pool.imap(tokenize_doc, _mixed_docs(split), chunksize=8)
    try:
        for ids in imap:
            token_buffer.extend(ids)
            doc_bounds.append(len(token_buffer))
            pbar.update(len(ids))

            while len(token_buffer) >= TOKENS_PER_SHARD:
                cut  = TOKENS_PER_SHARD
                name, ndocs = _write_shard(file_prefix, next_target, token_buffer, doc_bounds, cut)
                total_tokens += cut
                produced     += 1

                token_buffer = token_buffer[cut:]
                doc_bounds   = [b - cut for b in doc_bounds if b > cut]

                speed = total_tokens / max(time.time() - t_start, 1)
                tqdm.write(f"[{file_prefix}] wrote {name} ({produced}/{len(missing)}) | "
                           f"{ndocs} docs | {total_tokens/1e9:.2f}B tok | {speed/1e6:.2f}M tok/s")

                target_set.discard(next_target)
                remaining = sorted(target_set)
                if not remaining:
                    break
                next_target = remaining[0]

            if not target_set:
                break
    finally:
        pbar.close()
        pool.terminate()
        pool.join()
        del imap

    # Stream exhausted before every missing shard was filled: write the tail as a
    # partial shard if allowed (validation).
    if target_set and token_buffer:
        if allow_partial:
            cut = len(token_buffer)
            name, ndocs = _write_shard(file_prefix, next_target, token_buffer, doc_bounds, cut)
            total_tokens += cut
            produced     += 1
            tqdm.write(f"[{file_prefix}] flushed partial {name} | {ndocs} docs | "
                       f"{cut/1e9:.3f}B tok")
            target_set.discard(next_target)
        else:
            tqdm.write(f"[{file_prefix}] stream exhausted with {len(token_buffer)} tok "
                       f"in buffer (< {TOKENS_PER_SHARD}); not written (allow_partial=False)")

    still_missing = sorted(target_set)
    print(f"[done:{file_prefix}] produced {produced} shards | {total_tokens/1e9:.2f}B tokens"
          + (f" | STILL MISSING: {still_missing}" if still_missing else " | complete"))


def main():
    print(f"[config] train: {TARGET_SHARDS} x {TOKENS_PER_SHARD/1e6:.0f}M = "
          f"{TARGET_SHARDS*TOKENS_PER_SHARD/1e9:.1f}B tokens | sources: "
          f"{', '.join(s['path'] for s in SOURCES)}")
    run_split("train_shard", "train",      TARGET_SHARDS, allow_partial=False)
    run_split("val_shard",   "validation", VAL_SHARDS,    allow_partial=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
    os._exit(0)
