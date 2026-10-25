import math
from pathlib import Path
import contextlib

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datetime import datetime

from .gpu_auto import (
    get_device,
    get_gpu_memory_stats,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    barrier,
)
from .logging_config import get_logger, StepTimer


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP and torch.compile wrappers, in either nesting order."""
    seen = 0
    while seen < 4:
        if isinstance(model, DDP):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            break
        seen += 1
    return model


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    heavy: bool = True,
    ) -> dict:
    """Model diagnostics: rank, entropy, sink, noise, window and alpha.

    Two tiers, selected with ``heavy``:
      * ``heavy=False`` (every ``log_every``): per-head ``alpha``, the spread of the
        adaptive window and the kernel scan cost, from one forward. No SVD.
      * ``heavy=True`` (every ``metrics_every``): also effective ranks (GPU SVD),
        attention entropy, attention sink, residual norms, token distance and the
        noise-propagation probe (a second, perturbed forward).

    The heavy tier needs the per-layer dense attention ``_last_P``, which the attention
    module stores only in eval mode.
    """
    m = _unwrap_model(model)
    m.eval()

    device = ids.device
    total_len = ids.shape[0]

    x = m.embed_tokens(ids)
    layer_hiddens = [x.clone()]

    for i, layer in enumerate(m.layers):
        out = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = out[0] if isinstance(out, tuple) else out
        layer_hiddens.append(x.clone())

    last_attn = m.layers[-1].attn
    sigma2 = float("nan")
    eff_rank = float("nan")
    sigma2_per_layer = []
    eff_rank_per_layer_attn = []
    eff_rank_per_layer = []
    res_per_layer = []
    attn_entropy = float("nan")
    entropy_per_layer = []
    token_dist = float("nan")
    token_dist_per_layer = []

    # The residual norm is cheap and gates the noise probe below.
    h_final = layer_hiddens[-1]
    h_mean = h_final.mean(dim=0, keepdim=True)
    res_norm = ((h_final - h_mean).norm() / (layer_hiddens[0].norm() + 1e-9)).item()

    if heavy:
        if hasattr(last_attn, "_last_P") and last_attn._last_P is not None:
            # SVD on the GPU: on the CPU it dominated the metrics pass.
            P = last_attn._last_P.float()
            P_avg = P.mean(dim=0).detach()
            sv = torch.linalg.svdvals(P_avg)
            sigma2 = sv[1].item() if sv.shape[0] > 1 else 0.0
            sv_norm = sv / (sv.sum() + 1e-9)
            eff_rank = torch.exp(-(sv_norm * (sv_norm + 1e-9).log()).sum()).item()
        else:
            print("[trainer] compute_metrics: _last_P unavailable (training mode / packed), skipping rank/entropy stats")

        for li, layer in enumerate(m.layers):
            attn = layer.attn
            if hasattr(attn, "_last_P") and attn._last_P is not None:
                sv_l = torch.linalg.svdvals(attn._last_P.float().mean(dim=0).detach())
                sigma2_per_layer.append(sv_l[1].item() if sv_l.shape[0] > 1 else 0.0)
                sv_ln = sv_l / (sv_l.sum() + 1e-9)
                eff_rank_per_layer_attn.append(torch.exp(-(sv_ln * (sv_ln + 1e-9).log()).sum()).item())
            else:
                sigma2_per_layer.append(float("nan"))
                eff_rank_per_layer_attn.append(float("nan"))

        for li, h_l in enumerate(layer_hiddens[1:]):
            sv_h = torch.linalg.svdvals(h_l.float().detach())
            sv_hn = sv_h / (sv_h.sum() + 1e-9)
            er = torch.exp(-(sv_hn * (sv_hn + 1e-9).log()).sum()).item()
            eff_rank_per_layer.append(er)

        h0_norm = layer_hiddens[0].norm().item()
        for li, h_l in enumerate(layer_hiddens[1:]):
            xm = h_l.mean(dim=0, keepdim=True)
            rn = ((h_l - xm).norm() / (h0_norm + 1e-9)).item()
            res_per_layer.append(rn)

        if hasattr(last_attn, "_last_P") and last_attn._last_P is not None:
            P_safe = last_attn._last_P.float().clamp(min=1e-9)
            attn_entropy = -(P_safe * P_safe.log()).sum(dim=-1).mean().item()

        for li, layer in enumerate(m.layers):
            attn = layer.attn
            if hasattr(attn, "_last_P") and attn._last_P is not None:
                P_l = attn._last_P.float().clamp(min=1e-9)
                H = -(P_l * P_l.log()).sum(dim=-1).mean().item()
                entropy_per_layer.append(H)
            else:
                entropy_per_layer.append(float("nan"))

        min_dist = 64
        n_pairs = min(64, total_len - min_dist)
        if n_pairs > 0:
            pairs_i = torch.randint(min_dist, total_len, (n_pairs,), device=device)
            pairs_j = pairs_i - min_dist
            hi = layer_hiddens[-1][pairs_i]
            hj = layer_hiddens[-1][pairs_j]
            token_dist = (hi - hj).norm(dim=-1).mean().item()
            for li, h_l in enumerate(layer_hiddens[1:]):
                hi_l = h_l[pairs_i]
                hj_l = h_l[pairs_j]
                td = (hi_l - hj_l).norm(dim=-1).mean().item()
                token_dist_per_layer.append(td)

    noise_norm = float("nan")
    noise_per_layer = []
    seq0_len = (cu_seqlens[1] - cu_seqlens[0]).item()

    # The noise probe skips degenerate hidden states (NaN/inf or a blown-up residual).
    # ``res_norm`` grows with depth (about 320 on Qwen2.5-3B), so the cap scales with
    # the number of layers.
    res_cap  = 60.0 * (len(m.layers) ** 0.5)
    run_noise = heavy and seq0_len > 128 and math.isfinite(res_norm) and res_norm < res_cap

    if run_noise:
        inject_pos = int(seq0_len // 4)
        ids_pert = ids.clone()
        ids_pert[inject_pos] = torch.randint(0, m.vocab_size, (1,), device=device).item()

        x_pert = m.embed_tokens(ids_pert)
        layer_hiddens_pert = [x_pert.clone()]
        for i, layer in enumerate(m.layers):
            out_pert = layer(x_pert, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            x_pert = out_pert[0] if isinstance(out_pert, tuple) else out_pert
            layer_hiddens_pert.append(x_pert.clone())

        far_start = inject_pos + 64
        far_end = int(cu_seqlens[1].item())

        if far_end > far_start:
            noise_norm = (
                layer_hiddens_pert[-1][far_start:far_end] - layer_hiddens[-1][far_start:far_end]
            ).norm(dim=-1).mean().item()

            for li, (hl_orig, hl_pert) in enumerate(zip(layer_hiddens[1:], layer_hiddens_pert[1:])):
                nl = (hl_pert[far_start:far_end] - hl_orig[far_start:far_end]).norm(dim=-1).mean().item()
                noise_per_layer.append(nl)

    head_spec_std = float("nan")
    attn_sink = float("nan")
    if heavy and hasattr(last_attn, "_last_P") and last_attn._last_P is not None:
        P = last_attn._last_P.float().clamp(min=1e-9)
        entropy_heads = -(P * P.log()).sum(dim=-1).mean(dim=-1)
        head_spec_std = entropy_heads.std(correction=0).item() if entropy_heads.numel() > 0 else float("nan")
        attn_sink = last_attn._last_P.float()[:, :, 0].mean().item()

    # Per-head balance alpha = σ(alpha_w), min/mean/max over heads and layers.
    alpha_per_head = []
    alpha_vals = []
    for li, layer in enumerate(m.layers):
        attn = layer.attn
        if hasattr(attn, "alpha_w"):
            av = torch.sigmoid(attn.alpha_w).detach().cpu().tolist()
            alpha_per_head.append(av)
            alpha_vals.extend(av)
    if alpha_vals:
        at = torch.tensor(alpha_vals)
        alpha_min, alpha_mean, alpha_max = at.min().item(), at.mean().item(), at.max().item()
    else:
        alpha_min = alpha_mean = alpha_max = float("nan")

    # Adaptive window w(i) = n_min + σ(f(x_i))·(n_max-n_min), recomputed from the input
    # of each block (floor, as in eval).
    win_min = win_mean = win_max = float("nan")
    win_per_layer = []
    win_vals_all = []
    for li, layer in enumerate(m.layers):
        attn = layer.attn
        if hasattr(attn, "_window_sizes"):
            w = attn._window_sizes(layer_hiddens[li], floor=True)  # [T]
            wmin, wmean, wmax = w.min().item(), w.mean().item(), w.max().item()
            win_per_layer.append([wmin, wmean, wmax])
            win_vals_all.append(w)
    if win_vals_all:
        w_all   = torch.cat(win_vals_all)
        win_min, win_mean, win_max = w_all.min().item(), w_all.mean().item(), w_all.max().item()

    # Kernel scan cost. The Triton kernels scan key blocks back to ``m_start -
    # w_max_block + 1``, where ``w_max_block`` is the largest window in each block of
    # queries, so one wide query widens the scan of its whole block. This metric
    # averages the per-block maximum; ``scan_ratio`` divides it by the global mean
    # window.
    scan_block_max = float("nan")
    scan_ratio     = float("nan")
    if win_vals_all:
        _BM = 32  # BLOCK_M tiling of the kernel
        block_maxes = []
        for w in win_vals_all:
            T = w.numel()
            pad = (-T) % _BM
            wp = w if pad == 0 else torch.cat([w, w.new_zeros(pad)])
            block_maxes.append(wp.view(-1, _BM).max(dim=1).values)
        bm_all = torch.cat(block_maxes).float()
        scan_block_max = bm_all.mean().item()
        if win_mean and win_mean > 0:
            scan_ratio = scan_block_max / win_mean

    oow_mass_per_layer = []
    if heavy:
        for li, layer in enumerate(m.layers):
            attn = layer.attn
            if hasattr(attn, "_last_P") and attn._last_P is not None and hasattr(attn, "n_min"):
                P_l = attn._last_P.float()
                T_l = P_l.shape[-1]
                positions = torch.arange(T_l, device=P_l.device).float()
                dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
                in_window = (dist < attn.n_min).unsqueeze(0).unsqueeze(0)
                oow = P_l.masked_fill(in_window, 0.0).sum(dim=-1).mean().item()
                oow_mass_per_layer.append(oow)
            else:
                oow_mass_per_layer.append(float("nan"))

    m.train()

    return {
        "sigma2": sigma2,
        "eff_rank": eff_rank,
        "res_norm": res_norm,
        "attn_entropy": attn_entropy,
        "noise_norm": noise_norm,
        "token_dist": token_dist,
        "head_spec_std": head_spec_std,
        "attn_sink": attn_sink,
        "sigma2_per_layer": sigma2_per_layer,
        "entropy_per_layer": entropy_per_layer,
        "noise_per_layer": noise_per_layer,
        "eff_rank_per_layer": eff_rank_per_layer,
        "res_per_layer": res_per_layer,
        "token_dist_per_layer": token_dist_per_layer,
        "alpha_per_head": alpha_per_head,
        "alpha_min": alpha_min,
        "alpha_mean": alpha_mean,
        "alpha_max": alpha_max,
        "win_min": win_min,
        "win_mean": win_mean,
        "win_max": win_max,
        "win_per_layer": win_per_layer,
        "scan_block_max": scan_block_max,
        "scan_ratio": scan_ratio,
        "oow_mass_per_layer": oow_mass_per_layer,
    }


class DSALTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 0.5,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        grad_accum: int = 1,
        log_every: int = 100,
        metrics_every: int | None = None,
        val_every: int = 500,
        max_val_batches: int | None = None,
        save_every: int = 1000,
        save_dir: str = "./checkpoints_dsalt",
        mixed_precision: str = "auto",
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        ddp_backend: str = "nccl",
        seed: int = 42,
    ):
        self.rank       = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main    = is_main_process(rank)

        self.lr                     = lr
        self.weight_decay           = weight_decay
        self.max_grad_norm          = max_grad_norm
        self.warmup_steps           = warmup_steps
        self.total_steps            = total_steps
        self.grad_accum             = grad_accum
        self.log_every              = log_every
        # Heavy diagnostics (SVD rank, entropy, sink, noise probe) cost an extra forward
        # and per-layer SVDs, so by default they run every 20 logs; the cheap metrics
        # print every ``log_every``.
        self.metrics_every          = metrics_every if metrics_every is not None else log_every * 20
        self.val_every              = val_every
        self.max_val_batches        = max_val_batches
        self.save_every             = save_every
        self.save_dir               = Path(save_dir)
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_model          = compile_model
        self.ddp_backend            = ddp_backend
        self.seed                   = seed

        torch.manual_seed(seed + rank)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("dsalt.trainer", log_dir=str(self.save_dir))
        self.device = get_device(local_rank)

        # TF32 for fp32 GEMMs on Ampere and newer (no effect on sm_75); cudnn.benchmark
        # for the fixed shapes.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True

        self._amp_dtype = self._resolve_amp_dtype(mixed_precision)
        self._use_amp   = self._amp_dtype is not None
        self._scaler    = (
            torch.amp.GradScaler("cuda")
            if self._use_amp and self._amp_dtype == torch.float16
            else None
        )

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        model = model.to(self.device)

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        # Compile after the DDP wrap. The Triton kernels are opaque to dynamo, so
        # compile fuses the code around them; a compile failure falls back to eager.
        #
        # Under DDP, Dynamo's DDPOptimizer is disabled: it splits the graph at the
        # allreduce buckets and breaks the backward link across the custom autograd
        # Function (DSALTTrainFunction), so ``loss.backward()`` fails.
        if compile_model and hasattr(torch, "compile"):
            try:
                if world_size > 1 and hasattr(torch, "_dynamo"):
                    torch._dynamo.config.optimize_ddp = False
                # dynamic=True: variable-length packing changes the token count almost
                # every step, which with static shapes triggers a recompile per step.
                model = torch.compile(model, fullgraph=False, dynamic=True)
            except Exception as e:
                print(f"[trainer] torch.compile failed ({type(e).__name__}); running eager.")

        self.model = model

        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.optimizer)

        self.global_step       = 0
        self.best_val_ppl      = float("inf")
        self._timer            = StepTimer(window=50, device=self.device)
        self._tokens_per_batch = 0
        self._step_tokens      = 0
        self._accum_loss_sum   = 0.0
        self._accum_loss_steps = 0

        self.history = {k: [] for k in [
            "train_loss", "train_ppl", "sigma2", "eff_rank", "res_norm", "attn_entropy",
            "noise_norm", "token_dist", "head_spec_std", "attn_sink",
            "sigma2_per_layer", "entropy_per_layer", "noise_per_layer",
            "eff_rank_per_layer", "res_per_layer", "token_dist_per_layer",
            "alpha_per_head", "alpha_min", "alpha_mean", "alpha_max",
            "win_min", "win_mean", "win_max", "win_per_layer",
            "scan_block_max", "scan_ratio",
            "oow_mass_per_layer",
            "val_ppl", "val_steps", "gpu_mem_gb", "it_s", "tok_s",
            # Cheap history is appended every log_every and heavy metrics every
            # metrics_every; metrics_steps aligns the heavy entries with the timeline.
            "metrics_steps",
        ]}

    def _resolve_amp_dtype(self, mixed_precision: str) -> torch.dtype | None:
        if self.device.type != "cuda":
            return None
        if mixed_precision == "auto":
            # bf16 only with native support (sm_80+). torch.cuda.is_bf16_supported() is
            # also True on sm_75, where bf16 is emulated, so the compute capability is
            # checked directly.
            major, _ = torch.cuda.get_device_capability(self.device)
            mixed_precision = "bf16" if major >= 8 else "fp16"
        if mixed_precision == "bf16":
            return torch.bfloat16
        if mixed_precision == "fp16":
            return torch.float16
        return None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        base = _unwrap_model(self.model)
        decay, nodecay, dsalt_params = [], [], []

        # win_gate and alpha_w are trained through the soft window edge and the landmark
        # weight. Their gradient is weaker than that of the projections, so they get
        # their own group with lr x2 and no weight decay.
        for name, p in base.named_parameters():
            if not p.requires_grad:
                continue
            if "alpha_w" in name or "win_gate" in name:
                dsalt_params.append(p)
            elif p.ndim < 2 or any(k in name for k in ("norm", "bias", "embed")):
                nodecay.append(p)
            else:
                decay.append(p)

        n_decay   = sum(p.numel() for p in decay)
        n_nodecay = sum(p.numel() for p in nodecay)
        n_dsalt   = sum(p.numel() for p in dsalt_params)

        opt = torch.optim.AdamW(
            [
                {"params": decay,        "weight_decay": self.weight_decay, "lr": self.lr},
                {"params": nodecay,      "weight_decay": 0.0,               "lr": self.lr},
                {"params": dsalt_params, "weight_decay": 0.0,               "lr": self.lr * 2.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.device.type == "cuda",
        )
        return opt

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _extract_batch(self, batch):
        ids, labels, cu_seqlens, max_seqlen = batch
        ids        = ids.to(self.device, non_blocking=True)
        labels     = labels.to(self.device, non_blocking=True)
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        max_seqlen = int(max_seqlen)
        return ids, labels, cu_seqlens, max_seqlen

    def _forward_step(self, batch) -> torch.Tensor:
        ids, labels, cu_seqlens, max_seqlen = self._extract_batch(batch)
        self._tokens_per_batch = ids.numel()
        self._step_tokens     += ids.numel()
        self._last_ids         = ids
        self._last_cu_seqlens  = cu_seqlens
        self._last_max_seqlen  = max_seqlen

        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            out  = self.model(
                ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                labels=labels,
                gradient_checkpointing=self.gradient_checkpointing,
            )
            loss = out["loss"]

        return loss

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0

        # Cap the number of validation batches; None evaluates the whole set.
        for vi, batch in enumerate(self.val_loader):
            if self.max_val_batches is not None and vi >= self.max_val_batches:
                break
            ids, labels, cu_seqlens, max_seqlen = self._extract_batch(batch)
            valid_tokens  = (labels != -100).sum().item()

            out           = self.model(
                ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                labels=labels,
                gradient_checkpointing=False,
            )
            batch_loss = out["loss"].item()
            total_loss   += batch_loss * valid_tokens
            total_tokens += valid_tokens

        self.model.train()

        if self.world_size > 1:
            t = torch.tensor([total_loss, float(total_tokens)], device=self.device)
            torch.distributed.all_reduce(t)
            total_loss, total_tokens = t[0].item(), t[1].item()

        avg_loss = total_loss / max(total_tokens, 1)
        val_ppl  = math.exp(min(avg_loss, 20.0))
        return val_ppl

    def _save_checkpoint(self, tag: str) -> None:
        if not self.is_main:
            return
        ckpt = {
            "step":                 self.global_step,
            "model_state_dict":     _unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl":         self.best_val_ppl,
            "history":              self.history,
        }
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        self.logger.info(f"checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        _unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step  = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        self.history      = ckpt.get("history", self.history)
        if self.is_main:
            self.logger.info(f"checkpoint resumed from step {self.global_step}")

    def _log_step(self, accum_loss: float) -> None:
        if not self.is_main:
            return

        stats  = self._timer.stop()
        it_s   = stats.get("it_s", 0.0)
        # Global tokens per second: tokens per step × it/s × world_size (each DDP rank
        # counts only its own tokens).
        steps_since_log = max(1, self.log_every)
        tokens_per_step = self._step_tokens / steps_since_log
        tok_s           = tokens_per_step * it_s * self.world_size
        self._step_tokens = 0
        mem_gb = 0.0
        peak_gb = 0.0
        if self.device.type == "cuda":
            stats   = get_gpu_memory_stats(self.device)
            mem_gb  = stats.get("allocated_gb", 0.0)
            peak_gb = stats.get("peak_gb", 0.0)
            torch.cuda.reset_peak_memory_stats(self.device)
        lr_now    = self.scheduler.get_last_lr()[0]
        train_ppl = math.exp(min(accum_loss, 20.0))

        def _fs(v) -> str:
            return f"{v:.6f}" if math.isfinite(v) else "nan"

        extra = {"it_s": it_s, "tok_s": tok_s, "mem_gb": mem_gb,
                 "peak_gb": peak_gb, "total_gb": stats.get("total_gb", 0.0)}

        self.history["gpu_peak_gb"] = self.history.get("gpu_peak_gb", [])
        self.history["gpu_peak_gb"].append(peak_gb)
        self.history["train_loss"].append(accum_loss)
        self.history["train_ppl"].append(train_ppl)
        self.history["it_s"].append(it_s)
        self.history["tok_s"].append(tok_s)
        self.history["gpu_mem_gb"].append(mem_gb)

        heavy = (self.global_step % self.metrics_every == 0)

        # cheap line, every log_every
        if not heavy:
            self.logger.info(
                f"kind=cheap | step={self.global_step} | "
                f"loss={accum_loss:.4f} | ppl={train_ppl:.4f} | lr={lr_now:.2e}",
                extra=extra,
            )
            return

        # heavy diagnostics, every metrics_every
        metrics = compute_metrics(
            _unwrap_model(self.model),
            self._last_ids,
            self._last_cu_seqlens,
            self._last_max_seqlen,
            heavy=True,
        )
        self.logger.info(
            f"kind=metrics | step={self.global_step} | "
            f"loss={accum_loss:.4f} | ppl={train_ppl:.4f} | lr={lr_now:.2e} | "
            f"σ²={_fs(metrics['sigma2'])} | "
            f"rank={_fs(metrics['eff_rank'])} | "
            f"res={_fs(metrics['res_norm'])} | "
            f"H={_fs(metrics['attn_entropy'])} | "
            f"noise={_fs(metrics['noise_norm'])} | "
            f"sink={_fs(metrics['attn_sink'])} | "
            f"head_std={_fs(metrics['head_spec_std'])} | "
            f"token_dist={_fs(metrics['token_dist'])} | "
            f"win={metrics['win_min']:.0f}/{metrics['win_mean']:.1f}/{metrics['win_max']:.0f} | "
            f"scan={metrics['scan_block_max']:.1f} x{metrics['scan_ratio']:.2f} | "
            f"alpha={metrics['alpha_min']:.3f}/{metrics['alpha_mean']:.3f}/{metrics['alpha_max']:.3f}",
            extra=extra,
        )
        for k in ["sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm",
                  "token_dist", "head_spec_std", "attn_sink",
                  "sigma2_per_layer", "entropy_per_layer", "noise_per_layer",
                  "eff_rank_per_layer", "res_per_layer", "token_dist_per_layer",
                  "alpha_per_head", "alpha_min", "alpha_mean", "alpha_max",
                  "win_min", "win_mean", "win_max", "win_per_layer",
                  "scan_block_max", "scan_ratio",
                  "oow_mass_per_layer"]:
            self.history[k].append(metrics[k])
        self.history["metrics_steps"].append(self.global_step)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        data_iter = iter(self.train_loader)

        if self.is_main:
            n_params = sum(p.numel() for p in _unwrap_model(self.model).parameters() if p.requires_grad)
            mode = f"DDP×{self.world_size} (backend={self.ddp_backend})" if self.world_size > 1 else "1×GPU"
            self.logger.info(
                f"training start | mode={mode} | steps={self.total_steps} | "
                f"grad_accum={self.grad_accum} | device={self.device} | "
                f"amp={self._amp_dtype} | gc={self.gradient_checkpointing} | "
                f"params={n_params:,}"
            )

        ddp_active = self.world_size > 1 and isinstance(self.model, DDP)

        while self.global_step < self.total_steps:
            accum_loss = torch.zeros((), device=self.device)

            for accum_i in range(self.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if isinstance(self.train_loader.sampler, DistributedSampler):
                        self.train_loader.sampler.set_epoch(self.global_step)
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                is_last  = accum_i == self.grad_accum - 1
                sync_ctx = self.model.no_sync() if (ddp_active and not is_last) else contextlib.nullcontext()

                with sync_ctx:
                    loss = self._forward_step(batch) / self.grad_accum
                    if self._scaler is not None:
                        self._scaler.scale(loss).backward()
                    else:
                        loss.backward()
                accum_loss += loss.detach()

            accum_loss = accum_loss.item()

            grad_norm = None
            if self._scaler is not None:
                if self.max_grad_norm > 0:
                    self._scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    grads = [p.grad.detach().norm() ** 2 for p in self.model.parameters() if p.grad is not None]
                    if grads:
                        grad_norm = torch.stack(grads).sum() ** 0.5
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            gn_val = grad_norm.item() if grad_norm is not None else 0.0
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                self._log_step(accum_loss)

            if self.global_step % self.val_every == 0:
                barrier(self.rank, self.world_size)
                val_ppl = self._validate()
                if self.is_main:
                    self.history["val_ppl"].append(val_ppl)
                    self.history["val_steps"].append(self.global_step)
                    is_best = val_ppl < self.best_val_ppl
                    self.logger.info(
                        f"step={self.global_step} | val_ppl={val_ppl:.4f}" + (" ← best" if is_best else "")
                    )
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best")

            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            if self.global_step >= self.total_steps:
                break

            self._timer.start()

        self._save_checkpoint("final")
        if self.is_main:
            self.logger.info(f"done | best_val_ppl={self.best_val_ppl:.4f}")
        cleanup_ddp()