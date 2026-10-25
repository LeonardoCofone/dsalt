# `utils/`: configuration and data

| File | Contents |
| --- | --- |
| `utils_dsalt.py` | `CFG`, the configuration of the from-scratch runs, and `set_seed`. |
| `data_utils.py` | `LongformShardDataset`, the packing collate function and `make_loaders`. |
| `prepare_train_shards.py` | Builds the training and validation shards of the from-scratch models (PG19, arXiv, FineWeb-Edu; GPT-2 tokenizer). |
| `prepare_longctx_val.py` | Builds the long-context validation shards, once per tokenizer. |
| `prepare_pg19_qwen.py` | Builds the Qwen-tokenized PG19 shards for the fine-tune of the grafts. |

## `utils_dsalt.py`

`CFG` holds every hyperparameter of a from-scratch run: architecture (`d_model`,
`n_layers`, `n_heads`, `seq_len`, and DSALT's `N_min` / `N_max` / `k_lm`), data (shard
counts) and training (seed, batch size, gradient accumulation, steps, learning rate,
`save_dir`). The values in the file are those of the 48-layer runs (seed 6767). All four
from-scratch training scripts read it. `loss_type` selects the loss path (`liger` or
`chunked`).

## `data_utils.py`

`LongformShardDataset` reads pre-tokenized shards, two files per shard:

```
<prefix>_<i>.pt          1-D LongTensor of concatenated token ids
<prefix>_<i>.bounds.pt   per-document end offsets within the shard
```

Windows are cut inside single documents, never across a boundary. Each item is an
`(input_ids, labels)` pair of length `seq_len`, shifted by one token.

`SHARDS_DIR` and `SEQ_LEN` are module-level and read from `CFG` at import time; several
evaluation scripts rebind them at runtime (`d.SHARDS_DIR = ...`).

`dsalt_collate_fn` packs a batch into one flat tensor with `cu_seqlens` and
`max_seqlen`, without padding. `FalconShardDataset` is an alias of
`LongformShardDataset`.

## Data preparation

`prepare_train_shards.py` streams PG19, arXiv and FineWeb-Edu, interleaves them by
document (fractions 0.2, 0.3, 0.5) and writes 30 training shards of 500M tokens. A
source that runs out stops contributing, so the first shards are mostly PG19 and the
last ones FineWeb-Edu. Because windows are cut within documents, only documents longer
than `seq_len + 1` tokens produce training windows: about 56% of the windows come from
PG19, 15% from arXiv and 29% from FineWeb-Edu. Shard 30 is an exact copy of shard 1.

```bash
python -m utils.prepare_train_shards
```

Evaluating at 64k or 128k tokens needs single documents at least that long.
`prepare_longctx_val.py` keeps only documents above `--min-doc-tokens`:

```bash
# Qwen tokenizer (vocab 151665), for the 3B grafts
python -m utils.prepare_longctx_val --tokenizer Qwen/Qwen2.5-3B \
    --sources pg19,arxiv --val-docs 200 --min-doc-tokens 16384 \
    --out ./data_shards_longctx_qwen

# GPT-2 tokenizer (vocab 50257), for the from-scratch models
python -m utils.prepare_longctx_val --tokenizer gpt2 \
    --sources pg19,arxiv --val-docs 200 --min-doc-tokens 16384 \
    --out ./data_shards_longctx_gpt2
```

The fine-tuning shards of the grafts:

```bash
python -m utils.prepare_pg19_qwen --qwen Qwen/Qwen2.5-3B --seq-len 4096 \
    --out ./data_shards_pg19_qwen
```

Perplexity is only meaningful with the model's own tokenizer, so each tokenizer gets
its own directory.
