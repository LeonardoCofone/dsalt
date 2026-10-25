# `train/`: training scripts

Two settings: models trained from scratch (70M/14L, 0.3B/18L, 170M/48L), and a
pretrained Qwen2.5-3B with its attention replaced by each operator and briefly
fine-tuned.

| File | Setting | Contents |
| --- | --- | --- |
| `train_script.py` | from scratch | DSALT: builds a `DSALTConfig` from `CFG` and trains it with `DSALTTrainer`. |
| `train_dense.py` | from scratch | Dense baseline, with its own training loop. |
| `train_swa.py` | from scratch | SWA baseline, same loop as `train_dense.py`. |
| `train_spatra.py` | from scratch | Sparse-T baseline, same loop as `train_dense.py`. |
| `finetune_phase1.py` | graft | Fine-tune of a grafted Qwen2.5 model; `--arch` selects DSALT, dense, SWA or Sparse-T. |

## From scratch

The four scripts take no command-line arguments: every hyperparameter comes from the
`CFG` dict in [`utils/utils_dsalt.py`](../utils/README.md), shared by all four.

```bash
python -m train.train_script    # DSALT
python -m train.train_dense     # dense
python -m train.train_swa       # SWA
python -m train.train_spatra    # Sparse-T
```

The models in the paper were trained on a single A100. The scripts also run under
`torchrun --nproc_per_node=<N>`, which enables DDP.

- **Seeds.** Change `CFG["seed"]` and `CFG["save_dir"]` together, or a second seed
  overwrites the first run's checkpoints (the 48-layer runs use `..._48L_999seed` and
  `..._48L_6767seed`). The SWA and Sparse-T scripts use a fixed seed.
- **Checkpoints.** The baselines derive their directory from `CFG["save_dir"]` by
  replacing the architecture name, and write `<arch>_step<N>.pt`, `<arch>_final.pt` and
  `<arch>_history.json`, with the config inside the `.pt` (`ckpt["cfg"]`). DSALT writes a
  separate `model_config.json` next to its checkpoints.

## Fine-tune of the grafts

```bash
python -m train.finetune_phase1 --arch dsalt --qwen Qwen/Qwen2.5-3B \
    --seq-len 4096 --shards ./data_shards_pg19_qwen --steps 2000 --rope-base 1000000
python -m train.finetune_phase1 --arch dense --qwen Qwen/Qwen2.5-3B \
    --seq-len 4096 --shards ./data_shards_pg19_qwen --steps 2000 --rope-base 1000000
```

Every arm starts from the same backbone and is fine-tuned on the same shards with the
same steps, optimizer and learning rate (1e-5). `--alpha-lr-mult` gives the per-head
landmark balance a larger learning rate, since its gradient is small.

For `--arch dsalt` the window predictor is re-initialized with small random weights and
zero bias, so the window starts mid-range and depends on the token. The near-dense
initialization of `models/load_qwen_into_dsalt.py` saturates the sigmoid and the window
would stay at `n_max` for every token.
