"""Mechanism metrics (Dobrushin ρ, effective rank, entropy, attention sink, noise
propagation, per layer) for the Qwen2.5-3B Phase-1 checkpoints.

Uses the metric code of ``paper_eval.py`` unchanged and replaces only the loaders:
the grafts have vocab 151665, GQA, q/k/v bias and RoPE base 1e6, and are rebuilt
from the ``model_config.json`` written by ``finetune_phase1``.

Usage::

    python -m results.eval_phase1_3b \
        --dsalt-dir ./checkpoints_phase1_dsalt \
        --dense-dir ./checkpoints_phase1_dense \
        --shards ./data_shards_pg19_qwen --out paper_eval_3b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make the top-level packages importable from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _latest_ckpt(d: Path) -> Path:
    """Checkpoint to evaluate: 'best' if present, else the latest step or final."""
    for name in ("checkpoint_best.pt", "checkpoint_final.pt"):
        if (d / name).exists():
            return d / name
    cks = sorted(d.glob("checkpoint_*.pt"))
    if not cks:
        raise FileNotFoundError(f"no checkpoint_*.pt in {d}")
    return cks[-1]


def load_grafted(ckpt_dir: str, device) -> tuple[torch.nn.Module, str]:
    """Rebuild a Phase-1 model from model_config.json and its state_dict.

    ``model_config.json`` holds every structural field plus an ``arch`` tag selecting
    the class; the checkpoint stores the weights under ``model_state_dict``.
    """
    d   = Path(ckpt_dir)
    meta = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
    arch = meta.pop("arch", "dsalt")
    ckpt = _latest_ckpt(d)
    state = torch.load(ckpt, map_location=device)
    state = state.get("model_state_dict", state.get("model_state", state))

    if arch == "dsalt":
        from dsalt import DSALTConfig, DSALTLMHeadModel
        cfg = DSALTConfig.from_dict(meta)
        cfg.loss_fn = "chunked"
        model = DSALTLMHeadModel.from_config(cfg)
    elif arch in ("dense", "swa", "spatra"):
        # constructor arguments shared by the three baselines
        common = dict(
            vocab_size=meta["vocab_size"], d_model=meta["d_model"], n_layers=meta["n_layers"],
            n_heads=meta["n_heads"], max_seq_len=meta["max_seq_len"], d_ff=meta.get("d_ff"),
            dropout=0.0, tie_weights=meta.get("tie_weights", False), loss_fn="chunked",
            n_kv_heads=meta.get("n_kv_heads"), qkv_bias=meta.get("qkv_bias", False),
            rope_base=meta.get("rope_base", 10000.0),
        )
        if arch == "dense":
            from models.model_dense import DenseLM
            model = DenseLM(**common)
        elif arch == "swa":
            from models.model_swa import SWALM
            model = SWALM(window=meta["window"], **common)
        else:
            from models.model_spatra import SparseTLM
            model = SparseTLM(window=meta["window"], stride=meta["stride"], **common)
    else:
        raise ValueError(f"unknown arch {arch!r} in model_config.json")
    model.load_state_dict(state)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[eval3b] {arch} loaded from {ckpt.name} | params={n:,}")
    return model, arch


def main():
    ap = argparse.ArgumentParser(description="Experiment B (mechanism metrics) on the 3B Phase-1 checkpoints.")
    ap.add_argument("--dsalt-dir",  default="./checkpoints_phase1_dsalt")
    ap.add_argument("--dense-dir",  default="./checkpoints_phase1_dense")
    ap.add_argument("--swa-dir",    default=None,
                    help="SWA Phase-1 checkpoints dir (e.g. ./checkpoints_phase1_swa); "
                         "omit to skip the SWA column")
    ap.add_argument("--spatra-dir", default=None,
                    help="Sparse Transformer Phase-1 checkpoints dir; omit to skip")
    ap.add_argument("--shards",    default="./data_shards_pg19_qwen")
    ap.add_argument("--seq-len",   type=int, default=4096)
    ap.add_argument("--max-batches", type=int, default=100)
    ap.add_argument("--out", default="paper_eval_3b.json")
    args = ap.parse_args()

    # Point the shard loader at the Qwen PG19 shards at the evaluation length.
    import utils.utils_dsalt as u
    u.CFG["seq_len"] = args.seq_len
    import utils.data_utils as d
    d.SHARDS_DIR = Path(args.shards)
    d.SEQ_LEN    = args.seq_len
    from utils.data_utils import LongformShardDataset
    from torch.utils.data import DataLoader
    from train.finetune_phase1 import _collate

    import results.paper_eval as pe
    pe.MAX_BATCHES = args.max_batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = pe._amp_dtype(device)

    val_ds = LongformShardDataset("val_shard")
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=True, drop_last=True, collate_fn=_collate)

    # DSALT first (reference column), then dense and the optional baselines; a directory
    # set to None is skipped.
    model_dirs = [("dsalt", args.dsalt_dir), ("dense", args.dense_dir)]
    if args.swa_dir:
        model_dirs.append(("swa", args.swa_dir))
    if args.spatra_dir:
        model_dirs.append(("spatra", args.spatra_dir))

    results = {}
    for tag, ckpt_dir in model_dirs:
        model, kind = load_grafted(ckpt_dir, device)
        acc = pe.WelfordSet()
        torch.manual_seed(42)
        pe.eval_model(model, kind, loader, device, amp, args.max_batches, acc)
        results[tag] = acc.summary()
        if "_val_ppl_token_weighted" in acc.store:
            results[tag]["val_ppl"] = {"mean": acc.store["_val_ppl_token_weighted"][0], "std": 0.0, "n": 1}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[eval3b] wrote {args.out}\n")

    tags = [t for t, _ in model_dirs]
    hdr  = {"dsalt": "DSALT", "dense": "DENSE", "swa": "SWA", "spatra": "SparseT"}

    # Scalars and per-layer Dobrushin ρ, as in paper_eval.main.
    keys = ["val_ppl", "hidden_eff_rank_final", "attn_eff_rank", "attn_entropy",
            "head_spec_std", "attn_sink", "noise_norm", "noise_rel"]
    print(f"{'metric':24s}" + "".join(f"{hdr[t]:>16s}" for t in tags))
    print("-" * (24 + 16 * len(tags)))
    for key in keys:
        row = "".join(f"{results[t].get(key, {}).get('mean', float('nan')):16.4f}" for t in tags)
        print(f"{key:24s}{row}")

    n_layers = sum(1 for k in results["dsalt"] if k.startswith("dobrushin_L"))
    print("\n=== Dobrushin rho per layer (1 = the layer does not contract) ===")
    print(f"{'layer':8s}" + "".join(f"{hdr[t]:>12s}" for t in tags))
    for li in range(n_layers):
        row = "".join(
            f"{results[t].get(f'dobrushin_L{li}', {}).get('mean', float('nan')):12.4f}" for t in tags
        )
        print(f"L{li:<7d}{row}")


if __name__ == "__main__":
    main()
