"""Phase-1 fine-tune of a Qwen2.5 model with its attention replaced (DSALT, dense,
SWA or Sparse-T) on PG19 at the training length.

All four arms start from the same Qwen2.5 backbone and get the same shards, steps,
optimizer and learning rate; only the attention operator differs. The dense arm is
fine-tuned too, so every arm receives the same adaptation.

  * weights: ``models.load_qwen_into_dsalt``;
  * data:    ``LongformShardDataset`` over the shards from ``utils.prepare_pg19_qwen``;
  * trainer: ``DSALTTrainer``, which also logs the window, alpha, ρ and rank metrics.

Usage::

    python -m train.finetune_phase1 --arch dsalt --qwen Qwen/Qwen2.5-3B \
        --seq-len 4096 --shards ./data_shards_pg19_qwen --steps 2000 --rope-base 1000000
    python -m train.finetune_phase1 --arch dense --qwen Qwen/Qwen2.5-3B \
        --seq-len 4096 --shards ./data_shards_pg19_qwen --steps 2000 --rope-base 1000000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dsalt.training.gpu_auto import setup_ddp
from dsalt.training          import DSALTTrainer

# Make the top-level packages importable from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _collate(batch):
    """Pack a batch into (input_ids, labels, cu_seqlens, max_seqlen)."""
    input_ids_list = [torch.as_tensor(x[0]) for x in batch]
    labels_list    = [torch.as_tensor(x[1]) for x in batch]
    seqlens        = [t.shape[0] for t in input_ids_list]
    cu_seqlens     = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0, out=cu_seqlens[1:])
    return (torch.cat(input_ids_list), torch.cat(labels_list), cu_seqlens, max(seqlens))


def build_model(arch: str, qwen: str, seq_len: int, n_min: int, n_max: int,
                k_lmk: int, rope_base, window: int = 312, stride: int = 128,
                dtype=torch.bfloat16, device="cuda"):
    """Build the model for ``arch`` and load the Qwen backbone into it.

    ``dsalt``:  DSALT; the window predictor is re-initialized (see below).
    ``dense``:  full causal attention with FlashAttention.
    ``swa``:    fixed sliding window of ``window`` tokens.
    ``spatra``: Sparse Transformer, local window plus strided summary columns.

    SWA and Sparse-T have the same parameter names as the dense model and no extra
    parameters, so they reuse ``copy_qwen_weights_dense``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models.load_qwen_into_dsalt import (
        dsalt_config_from_qwen, copy_qwen_weights, copy_qwen_weights_dense, seed_near_dense,
    )

    tok = AutoTokenizer.from_pretrained(qwen)
    src = AutoModelForCausalLM.from_pretrained(qwen, torch_dtype=dtype)
    cfg = dsalt_config_from_qwen(src.config, n_min, n_max, k_lmk, seq_len, rope_base)

    if arch == "dsalt":
        from dsalt.model.dsalt_lm import DSALTLMHeadModel
        model = DSALTLMHeadModel.from_config(cfg).to(dtype)
        missing = copy_qwen_weights(model, src)
        bad = [m for m in missing if not (m.endswith("win_gate.weight")
                                          or m.endswith("win_gate.bias") or m.endswith("alpha_w"))]
        if bad:
            raise RuntimeError(f"DSALT backbone not fully loaded: {bad}")
        # Window predictor: small random weights and zero bias, so the window starts
        # mid-range and depends on the token. seed_near_dense (zero weights, large
        # bias) saturates the sigmoid at n_max for every token and the window stays
        # frozen there during the fine-tune.
        for layer in model.layers:
            torch.nn.init.normal_(layer.attn.win_gate.weight, std=0.02)
            torch.nn.init.zeros_(layer.attn.win_gate.bias)
    elif arch == "dense":
        from models.model_dense import DenseLM
        model = DenseLM(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, max_seq_len=cfg.max_seq_len, d_ff=cfg.d_ff,
            dropout=0.0, tie_weights=cfg.tie_weights, loss_fn="chunked",
            n_kv_heads=cfg.n_kv_heads, qkv_bias=cfg.qkv_bias, rope_base=cfg.rope_base,
        ).to(dtype)
        missing = copy_qwen_weights_dense(model, src)
        if missing:
            raise RuntimeError(f"dense backbone not fully loaded: {missing}")
    elif arch == "swa":
        from models.model_swa import SWALM
        model = SWALM(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, max_seq_len=cfg.max_seq_len, d_ff=cfg.d_ff,
            dropout=0.0, tie_weights=cfg.tie_weights, loss_fn="chunked",
            n_kv_heads=cfg.n_kv_heads, qkv_bias=cfg.qkv_bias, rope_base=cfg.rope_base,
            window=window,
        ).to(dtype)
        missing = copy_qwen_weights_dense(model, src)
        if missing:
            raise RuntimeError(f"swa backbone not fully loaded: {missing}")
    elif arch == "spatra":
        from models.model_spatra import SparseTLM
        model = SparseTLM(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, max_seq_len=cfg.max_seq_len, d_ff=cfg.d_ff,
            dropout=0.0, tie_weights=cfg.tie_weights, loss_fn="chunked",
            n_kv_heads=cfg.n_kv_heads, qkv_bias=cfg.qkv_bias, rope_base=cfg.rope_base,
            window=window, stride=stride,
        ).to(dtype)
        missing = copy_qwen_weights_dense(model, src)
        if missing:
            raise RuntimeError(f"spatra backbone not fully loaded: {missing}")
    else:
        raise ValueError(f"unknown arch {arch!r} (use 'dsalt', 'dense', 'swa' or 'spatra')")

    del src
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model.to(device), tok, cfg


def main():
    ap = argparse.ArgumentParser(description="Phase-1 short fine-tune on PG19 (DSALT or dense twin).")
    ap.add_argument("--arch", choices=["dsalt", "dense", "swa", "spatra"], required=True)
    ap.add_argument("--qwen", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--shards", default="./data_shards_pg19_qwen",
                    help="dir with the Qwen-tokenized PG19 shards")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--n-min", type=int, default=32,
                    help="smallest adaptive window")
    ap.add_argument("--n-max", type=int, default=512,
                    help="largest adaptive window; must be well below seq_len, otherwise "
                         "nothing pushes the window below full context")
    ap.add_argument("--k-lmk", type=int, default=32)
    # swa / spatra only. 312 is the mean DSALT window on the 3B graft; a stride of
    # 128 gives 4096 / 128 = 32 summary columns, as many as DSALT's landmarks.
    ap.add_argument("--window", type=int, default=312,
                    help="fixed local window for swa / spatra")
    ap.add_argument("--stride", type=int, default=128,
                    help="summary-column stride for spatra")
    ap.add_argument("--rope-base", type=float, default=None, help="Qwen2.5 = 1000000")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1, help="sequences per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5, help="peak learning rate")
    ap.add_argument("--alpha-lr-mult", type=float, default=100.0,
                    help="lr multiplier for alpha_w only; 1.0 keeps the library default")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--metrics-every", type=int, default=100, help="steps between diagnostic metrics")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute activations in the backward pass to save memory")
    args = ap.parse_args()

    # data_utils reads CFG["seq_len"] at import time.
    import utils.utils_dsalt as u
    u.CFG["seq_len"] = args.seq_len
    from utils.data_utils import LongformShardDataset

    rank, local_rank, world_size = setup_ddp()

    os.chdir(Path(args.shards).parent if Path(args.shards).name == "data_shards" else ".")
    import utils.data_utils as d
    d.SHARDS_DIR = Path(args.shards)
    d.SEQ_LEN    = args.seq_len

    train_ds = LongformShardDataset("train_shard")
    val_ds   = LongformShardDataset("val_shard")

    # Shards from another tokenizer would index outside the Qwen embedding.
    max_id = int(train_ds._shards[0].max())
    if max_id >= 151665:
        raise RuntimeError(
            f"shard max token id {max_id} >= Qwen vocab 151665: these shards are NOT "
            f"Qwen-tokenized (wrong --shards dir? mixed with old data?). Re-run "
            f"prepare_pg19_qwen and point --shards at its output."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True, collate_fn=_collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=True, collate_fn=_collate)

    model, tok, cfg = build_model(
        args.arch, args.qwen, args.seq_len, args.n_min, args.n_max, args.k_lmk, args.rope_base,
        window=args.window, stride=args.stride,
    )
    if rank == 0:
        print(f"[phase1:{args.arch}] {model.num_parameters(False)/1e9:.2f}B params, "
              f"{cfg.n_layers}L {cfg.n_heads}H/{cfg.n_kv_heads}KV, seq_len={args.seq_len}, "
              f"steps={args.steps}, lr={args.lr}")

    save_dir = args.save_dir or f"./checkpoints_phase1_{args.arch}"
    # DSALTTrainer saves only the state_dict; the evaluation scripts rebuild the
    # model from this config.
    if rank == 0:
        import json
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        meta = dict(cfg.to_dict())
        meta["arch"] = args.arch
        if args.arch in ("swa", "spatra"):
            meta["window"] = args.window
        if args.arch == "spatra":
            meta["stride"] = args.stride
        (Path(save_dir) / "model_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trainer = DSALTTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        rank=rank, local_rank=local_rank, world_size=world_size,
        lr=args.lr, weight_decay=args.weight_decay, max_grad_norm=args.grad_clip,
        warmup_steps=args.warmup, total_steps=args.steps, grad_accum=args.grad_accum,
        log_every=args.log_every, metrics_every=args.metrics_every, val_every=args.val_every,
        save_every=args.val_every, save_dir=save_dir,
        mixed_precision="auto", compile_model=(not args.no_compile),
    )

    # alpha_w reaches the loss only through the few selected landmarks, so its
    # gradient is much smaller than win_gate's and it barely moves in a short
    # fine-tune. Move it to its own optimizer group with a larger learning rate.
    if args.arch == "dsalt" and args.alpha_lr_mult != 1.0:
        base   = model
        a_ids  = {id(p) for n, p in base.named_parameters() if "alpha_w" in n}
        opt    = trainer.optimizer
        moved  = []
        for grp in opt.param_groups:
            keep = [p for p in grp["params"] if id(p) not in a_ids]
            take = [p for p in grp["params"] if id(p) in a_ids]
            if take:
                grp["params"] = keep
                moved.extend(take)
        if moved:
            opt.add_param_group({
                "params": moved, "weight_decay": 0.0,
                "lr": args.lr * args.alpha_lr_mult, "initial_lr": args.lr * args.alpha_lr_mult,
            })
            if rank == 0:
                print(f"[phase1] alpha_w → dedicated group, lr = {args.lr*args.alpha_lr_mult:g} "
                      f"({args.alpha_lr_mult:g}× base)")

    trainer.train()


if __name__ == "__main__":
    main()
