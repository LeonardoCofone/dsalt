from __future__ import annotations

import math
import os
import sys
import time
import json
from pathlib import Path

from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Make the top-level packages importable from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_utils import make_loaders, dsalt_collate_fn  # noqa: E402
from utils.utils_dsalt import CFG, set_seed  # noqa: E402

from models.model_swa import SWALM, WINDOW_FROM_DSALT  # noqa: E402

VOCAB_SIZE = 50257


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main(rank):
    return rank == 0


def unwrap(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


@torch.no_grad()
def compute_metrics(model, ids, cu_seqlens, max_seqlen=None, heavy: bool = True) -> dict:
    """Diagnostics of the SWA baseline, computed as in train/train_dense.py.
    ``_last_P`` is [H, T, T] for the first sequence (see model_swa.py).
    """
    m = unwrap(model)
    m.eval()
    device = ids.device

    # Every metric is defined on the first sequence of the packed batch; running
    # the probe on the whole batch does not fit in memory at 48 layers.
    T0 = int((cu_seqlens[1] - cu_seqlens[0]).item())
    ids = ids[:T0].contiguous()
    cu_seqlens = torch.tensor([0, T0], device=device, dtype=torch.int32)
    total_len = T0

    x = m._to_autocast_dtype(m.embed_tokens(ids))
    rope_cs, n_seqs, seq_len, uniform = m._packed_meta(cu_seqlens, total_len, m.layers[0].attn)

    layer_hiddens = [x.clone()]
    for layer in m.layers:
        x = layer(x, cu_seqlens, total_len, rope_cs, n_seqs, seq_len, uniform)
        layer_hiddens.append(x.clone())

    out = {k: float("nan") for k in
           ("sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm",
            "token_dist", "head_spec_std", "attn_sink")}
    out.update({k: [] for k in
                ("sigma2_per_layer", "entropy_per_layer", "noise_per_layer",
                 "eff_rank_per_layer", "eff_rank_per_layer_attn",
                 "res_per_layer", "token_dist_per_layer")})

    h_final = layer_hiddens[-1]
    h_mean = h_final.mean(dim=0, keepdim=True)
    res_norm = ((h_final - h_mean).norm() / (layer_hiddens[0].norm() + 1e-9)).item()
    out["res_norm"] = res_norm

    if heavy:
        # One pass over the layers, freeing each dense [H, T, T] map after use.
        n_layers = len(m.layers)
        last_idx = n_layers - 1
        for li, layer in enumerate(m.layers):
            lp = layer.attn._last_P
            if lp is None:
                out["sigma2_per_layer"].append(float("nan"))
                out["eff_rank_per_layer_attn"].append(float("nan"))
                out["entropy_per_layer"].append(float("nan"))
                continue
            Pf = lp.float()
            sv_l = torch.linalg.svdvals(Pf.mean(dim=0).detach())
            out["sigma2_per_layer"].append(sv_l[1].item() if sv_l.shape[0] > 1 else 0.0)
            sv_ln = sv_l / (sv_l.sum() + 1e-9)
            out["eff_rank_per_layer_attn"].append(torch.exp(-(sv_ln * (sv_ln + 1e-9).log()).sum()).item())
            Pc = Pf.clamp(min=1e-9)
            out["entropy_per_layer"].append(-(Pc * Pc.log()).sum(dim=-1).mean().item())
            if li == last_idx:
                out["sigma2"] = out["sigma2_per_layer"][-1]
                out["eff_rank"] = out["eff_rank_per_layer_attn"][-1]
                out["attn_entropy"] = out["entropy_per_layer"][-1]
                eh = -(Pc * Pc.log()).sum(dim=-1).mean(dim=-1)
                out["head_spec_std"] = eh.std(correction=0).item() if eh.numel() > 0 else float("nan")
                out["attn_sink"] = Pf[:, :, 0].mean().item()
            del lp, Pf, Pc, sv_l, sv_ln
            layer.attn._last_P = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for h_l in layer_hiddens[1:]:
            sv_h = torch.linalg.svdvals(h_l.float().detach())
            sv_hn = sv_h / (sv_h.sum() + 1e-9)
            out["eff_rank_per_layer"].append(torch.exp(-(sv_hn * (sv_hn + 1e-9).log()).sum()).item())

        h0_norm = layer_hiddens[0].norm().item()
        for h_l in layer_hiddens[1:]:
            xm = h_l.mean(dim=0, keepdim=True)
            out["res_per_layer"].append(((h_l - xm).norm() / (h0_norm + 1e-9)).item())

        min_dist = 64
        n_pairs = min(64, total_len - min_dist)
        if n_pairs > 0:
            pairs_i = torch.randint(min_dist, total_len, (n_pairs,), device=device)
            pairs_j = pairs_i - min_dist
            out["token_dist"] = (layer_hiddens[-1][pairs_i] - layer_hiddens[-1][pairs_j]).norm(dim=-1).mean().item()
            for h_l in layer_hiddens[1:]:
                out["token_dist_per_layer"].append((h_l[pairs_i] - h_l[pairs_j]).norm(dim=-1).mean().item())

    seq0_len = int((cu_seqlens[1] - cu_seqlens[0]).item())
    run_noise = heavy and seq0_len > 128 and math.isfinite(res_norm) and res_norm < 200.0
    if run_noise:
        inject_pos = int(seq0_len // 4)
        ids_pert = ids.clone()
        ids_pert[inject_pos] = torch.randint(0, m.vocab_size, (1,), device=device).item()
        xp = m._to_autocast_dtype(m.embed_tokens(ids_pert))
        hp = [xp.clone()]
        # The noise pass needs only the hidden states, not the attention maps.
        for layer in m.layers:
            layer.attn._collect_last_P = False
        for layer in m.layers:
            xp = layer(xp, cu_seqlens, total_len, rope_cs, n_seqs, seq_len, uniform)
            hp.append(xp.clone())
        for layer in m.layers:
            layer.attn._collect_last_P = True
        far_start = inject_pos + 64
        far_end = int(cu_seqlens[1].item())
        if far_end > far_start:
            out["noise_norm"] = (hp[-1][far_start:far_end] - layer_hiddens[-1][far_start:far_end]).norm(dim=-1).mean().item()
            for ho, hpp in zip(layer_hiddens[1:], hp[1:]):
                out["noise_per_layer"].append((hpp[far_start:far_end] - ho[far_start:far_end]).norm(dim=-1).mean().item())

    m.train()
    return out


def _resolve_amp_dtype(device):
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _fs(v):
    return f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else "nan"


@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype, max_batches):
    model.eval()
    # In eval mode every layer would rebuild its dense attention map, which
    # validation does not need and which stays in memory afterwards.
    m = unwrap(model)
    for layer in m.layers:
        layer.attn._collect_last_P = False
    total_loss = torch.zeros((), device=device)
    total_tok = torch.zeros((), device=device)
    for i, (ids, labels, cu, _) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        cu = cu.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = model(ids, cu_seqlens=cu, labels=labels)["loss"]
        n = (labels.view(-1) != -100).sum()
        total_loss += loss.float() * n
        total_tok += n
    for layer in m.layers:
        layer.attn._collect_last_P = True
        layer.attn._last_P = None
    model.train()
    avg = (total_loss / total_tok.clamp(min=1)).item()
    return avg, math.exp(min(avg, 20.0))


def train_fn():
    rank, local_rank, world_size = setup_ddp()
    set_seed(42 + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, cfg = make_loaders(limit=None)

    # Default window: the mean adaptive window of the DSALT config. Override with
    # SWA_WINDOW or a "window" key in CFG.
    window = int(os.environ.get("SWA_WINDOW", cfg.get("window", WINDOW_FROM_DSALT)))

    if world_size > 1:
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_loader.dataset, batch_size=cfg["batch_size"], sampler=train_sampler,
                                  num_workers=2, pin_memory=True, drop_last=True, collate_fn=dsalt_collate_fn)
        val_loader = DataLoader(val_loader.dataset, batch_size=cfg["batch_size"], sampler=val_sampler,
                                num_workers=2, pin_memory=True, drop_last=True, collate_fn=dsalt_collate_fn)
    else:
        train_sampler = None

    save_dir = Path(str(cfg.get("save_dir", "./checkpoints_dsalt")).replace("dsalt", "SWA"))
    if is_main(rank):
        save_dir.mkdir(parents=True, exist_ok=True)

    model = SWALM(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg["seq_len"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        tie_weights=cfg.get("tie_weights", True),
        loss_fn=cfg["loss_type"],
        window=window,
    ).to(device)

    if is_main(rank):
        print(f">>> SWA baseline | params = {model.num_parameters():,} | "
              f"d_model={cfg['d_model']} layers={cfg['n_layers']} heads={cfg['n_heads']} "
              f"seq={cfg['seq_len']} | window={window} | loss={cfg['loss_type']}")

    amp_dtype = _resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, gradient_as_bucket_view=True)

    if cfg.get("compile", True) and hasattr(torch, "compile"):
        try:
            if world_size > 1 and hasattr(torch, "_dynamo"):
                torch._dynamo.config.optimize_ddp = False
            # seq_len and the flex_attention BlockMask are static.
            model = torch.compile(model, fullgraph=False, dynamic=False)
        except Exception as e:
            print(f"[swa] torch.compile failed ({type(e).__name__}); eager.")

    decay, no_decay = [], []
    for n, p in unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.dim() < 2 or "norm" in n) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
        fused=device.type == "cuda",
    )

    warmup = cfg["warmup_steps"]
    total = cfg["max_steps"]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_every = cfg["log_every"]
    metrics_every = cfg.get("metrics_every") or log_every * 20
    val_every = cfg["val_every"]
    grad_accum = cfg["grad_accum"]
    max_val_batches = cfg.get("max_val_batches")
    tokens_per_step = cfg["batch_size"] * grad_accum * cfg["seq_len"] * world_size

    history = {k: [] for k in (
        "train_loss", "train_ppl", "it_s", "tok_s", "gpu_mem_gb", "gpu_peak_gb",
        "sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm", "token_dist",
        "head_spec_std", "attn_sink", "sigma2_per_layer", "entropy_per_layer",
        "noise_per_layer", "eff_rank_per_layer", "eff_rank_per_layer_attn",
        "res_per_layer", "token_dist_per_layer",
        "val_ppl", "val_steps", "metrics_steps")}

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    loader_iter = iter(train_loader)
    last_batch = None
    t0 = time.time()

    while global_step < total:
        if train_sampler is not None:
            train_sampler.set_epoch(global_step)
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                ids, labels, cu, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                ids, labels, cu, _ = next(loader_iter)
            ids = ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            cu = cu.to(device, non_blocking=True)
            last_batch = (ids, cu)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(ids, cu_seqlens=cu, labels=labels)["loss"] / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if is_main(rank):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] step {global_step}")

        if is_main(rank) and global_step % log_every == 0:
            torch.cuda.synchronize(device)
            dt = (time.time() - t0) / log_every
            it_s = 1.0 / dt if dt > 0 else 0.0
            tok_s = tokens_per_step * it_s
            mem_gb = torch.cuda.memory_allocated(device) / 1e9
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            torch.cuda.reset_peak_memory_stats(device)
            lr_now = scheduler.get_last_lr()[0]
            ppl = math.exp(min(accum_loss, 20.0))

            history["train_loss"].append(accum_loss)
            history["train_ppl"].append(ppl)
            history["it_s"].append(it_s)
            history["tok_s"].append(tok_s)
            history["gpu_mem_gb"].append(mem_gb)
            history["gpu_peak_gb"].append(peak_gb)

            heavy = (global_step % metrics_every == 0)
            if not heavy:
                print(f"  step {global_step:6d}  loss {accum_loss:.4f}  ppl {ppl:9.2f}  "
                      f"lr {lr_now:.2e}  {it_s:.2f} it/s  {tok_s/1e3:.1f}k tok/s  peak {peak_gb:.1f} GB")
            else:
                ids_m, cu_m = last_batch
                mm = compute_metrics(model, ids_m, cu_m, heavy=True)
                ts = datetime.now().strftime("%H:%M:%S")
                print("─" * 72)
                print(f"  {ts}   step {global_step:<6d}  {it_s:.2f} it/s   {tok_s/1e3:.1f}k tok/s   "
                      f"cur {mem_gb:.2f}·peak {peak_gb:.2f} GB / {total_gb:.0f} tot")
                print("─" * 72)
                print(f"        loss   {accum_loss:12.4f}           ppl   {ppl:14.4f}")
                print(f"          σ²   {_fs(mm['sigma2']):>12}          rank   {_fs(mm['eff_rank']):>14}")
                print(f"    residual   {_fs(mm['res_norm']):>12}     entropy H   {_fs(mm['attn_entropy']):>14}")
                print(f"       noise   {_fs(mm['noise_norm']):>12}          sink   {_fs(mm['attn_sink']):>14}")
                print(f"    head_std   {_fs(mm['head_spec_std']):>12}            lr   {lr_now:14.2e}")
                print(f"  token_dist   {_fs(mm['token_dist']):>12}     attention   {f'swa (w={window})':>14}")
                print("─" * 72)
                for k in ("sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm",
                          "token_dist", "head_spec_std", "attn_sink", "sigma2_per_layer",
                          "entropy_per_layer", "noise_per_layer", "eff_rank_per_layer",
                          "eff_rank_per_layer_attn",
                          "res_per_layer", "token_dist_per_layer"):
                    history[k].append(mm[k])
                history["metrics_steps"].append(global_step)
            t0 = time.time()

        if global_step % val_every == 0:
            avg, ppl = evaluate(model, val_loader, device, amp_dtype, max_val_batches)
            if is_main(rank):
                history["val_ppl"].append(ppl)
                history["val_steps"].append(global_step)
                print(f"  *** val loss {avg:.4f} | val ppl {ppl:.2f} @ step {global_step} ***")
                ckpt = save_dir / f"swa_step{global_step}.pt"
                torch.save({"step": global_step, "model_state": unwrap(model).state_dict(),
                            "cfg": dict(cfg), "window": window, "history": history}, ckpt)
                (save_dir / "swa_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                print(f"  checkpoint: {ckpt}")
            t0 = time.time()

    if is_main(rank):
        torch.save({"step": global_step, "model_state": unwrap(model).state_dict(),
                    "cfg": dict(cfg), "window": window, "history": history}, save_dir / "swa_final.pt")
        (save_dir / "swa_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(">>> SWA baseline training done.")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train_fn()
