"""Context-scaling experiment: perplexity, peak memory and latency against context
length, for DSALT and the baselines.

For each context length L, forward only, on long real text:
  - perplexity, token-weighted over all tokens of each window, with the per-window
    mean, std, min and max;
  - peak GPU memory (GB);
  - median forward latency (ms);
  - whether the model runs out of memory (then that model stops scaling).

Evaluation choices:
  * Windows are anchored to the tails of the same documents for every L, so a
    change across L is not a change of text.
  * Per-window std, min and max are reported next to the token-weighted mean.
  * --rope-mode {ntk,linear,none} selects the RoPE extension; every model gets the
    same scheme, relative to its own rope_base.

Two checkpoint formats, selected with --mode:
  * "3b"    : Qwen2.5-3B grafts, rebuilt from model_config.json via
              eval_phase1_3b.load_grafted (vocab 151665).
  * "proto" : from-scratch models. DSALT has model_config.json next to the
              checkpoint; the baselines store their cfg inside the .pt (ckpt["cfg"]).

Run (3B, four operators)::

    DSALT_COMPILE_FLEX=1 python -m results.context_scaling --mode 3b \
        --dsalt-dir ./checkpoints_phase1_dsalt --dense-dir ./checkpoints_phase1_dense \
        --swa-dir ./checkpoints_phase1_swa --spatra-dir ./checkpoints_phase1_spatra \
        --shards ./data_shards_pg19_qwen --lengths 2048,4096,8192,16384,32768 \
        --rope-mode ntk --out context_scaling_3b.json

DSALT_COMPILE_FLEX=1 is required at long context when SWA or Sparse-T are
included: eager flex_attention materializes the full [L, L] scores.

Run (from-scratch, 48 layers, four operators)::

    python -m results.context_scaling --mode proto \
        --dsalt-dir checkpoints_dsalt_48L --dense-ckpt checkpoints_dense_48L/dense_step5000.pt \
        --swa-ckpt checkpoints_swa_48L/swa_step5000.pt \
        --spatra-ckpt checkpoints_spatra_48L/spatra_step5000.pt \
        --shards ./data_shards_proto --lengths 2048,4096,8192,16384 \
        --rope-mode ntk --vocab 50257 --out context_scaling_proto48.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

# Make the top-level packages importable from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dsalt.kernels.window_utils import build_rope_cache

# Baselines with the dense interface (kind -> module, class), loaded by load_iso() from
# the cfg stored in their .pt.
_ISO_ARCH = {
    "dense":  ("models.model_dense",  "DenseLM"),
    "swa":    ("models.model_swa",    "SWALM"),
    "spatra": ("models.model_spatra", "SparseTLM"),
}

# Defaults, overridable on the CLI. L_TRAIN is the training length, the reference for
# RoPE scaling and for the slowdown table.
L_TRAIN = 4096


# device / dtype helpers
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype(device):
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _abs(path):
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


# RoPE extension (same scheme for every model)
def _rope_scale_for(L, rope_mode):
    """Effective length scale (L / L_train), >= 1, or 1 for 'none'."""
    if rope_mode == "none":
        return 1.0
    return max(1.0, L / L_TRAIN)


def _rope_params_for(L, base0, head_dim, rope_mode):
    """Map target length L to (base, pos_scale) for build_rope_cache.

    - "ntk":    raw positions, stretched base
                base' = base0 * s**(head_dim/(head_dim-2)), s = L/L_train.
    - "linear": raw base, positions / s (position interpolation).
    - "none":   raw base, raw positions (extrapolation).
    """
    s = _rope_scale_for(L, rope_mode)
    if rope_mode == "ntk":
        return base0 * (s ** (head_dim / (head_dim - 2))), 1.0
    if rope_mode == "linear":
        return base0, s
    return base0, 1.0


def _patch_rope(model, L, device, rope_mode):
    """Rebuild every layer's RoPE cache for length L with the chosen scheme.

    The base is derived from each layer's own rope_base (1e6 for Qwen2.5, 10000 for the
    from-scratch models), and every model gets the same (base, pos_scale).
    """
    head_dim = model.layers[0].attn.head_dim
    base0 = float(getattr(model.layers[0].attn, "rope_base", 10000.0))
    base, pos_scale = _rope_params_for(L, base0, head_dim, rope_mode)
    cos, sin = build_rope_cache(L, head_dim, device, base=base, scale=pos_scale)
    for layer in model.layers:
        layer.attn.rope_cos = cos
        layer.attn.rope_sin = sin
    return _rope_scale_for(L, rope_mode)


def _disable_last_P(model):
    """Skip the dense _last_P reconstruction in every layer.

    At inference each attention module rebuilds a dense [H, T, T] matrix for the rank
    and Dobrushin metrics, which costs O(T^2) memory; perplexity and memory do not
    need it.
    """
    for layer in model.layers:
        layer.attn._collect_last_P = False


# loaders
def load_dsalt(ckpt_dir, device, mode, vocab):
    """DSALT loader; both modes rebuild the model from model_config.json."""
    if mode == "3b":
        from results.eval_phase1_3b import load_grafted
        model, arch = load_grafted(str(_abs(ckpt_dir)), device)
        assert arch == "dsalt", f"{ckpt_dir} is arch={arch}, expected dsalt"
    else:
        from dsalt import DSALTConfig, DSALTLMHeadModel
        d = _abs(ckpt_dir)
        cfg = DSALTConfig.load(d / "model_config.json")
        cfg.loss_fn = "chunked"
        model = DSALTLMHeadModel.from_config(cfg).to(device)
        ckpt_path = next((d / n for n in ("checkpoint_best.pt", "checkpoint_final.pt")
                          if (d / n).exists()), None) or sorted(d.glob("checkpoint_*.pt"))[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state)
    model.eval()
    _disable_last_P(model)
    print(f"[ctx] dsalt ready ({mode}) | layers={len(model.layers)}")
    return model


def load_dense(ckpt_dir, device, mode, vocab, naive=False, dense_ckpt=None):
    """Dense loader.

    - mode "3b":    graft directory with model_config.json (load_grafted).
    - mode "proto": cfg stored inside the .pt at --dense-ckpt (ckpt["cfg"]).
    ``naive`` materializes the [T, T] attention matrix; otherwise FlashAttention.
    """
    if mode == "3b":
        from results.eval_phase1_3b import load_grafted
        model, arch = load_grafted(str(_abs(ckpt_dir)), device)
        assert arch == "dense", f"{ckpt_dir} is arch={arch}, expected dense"
    else:
        model = load_iso("dense", dense_ckpt, device, vocab)
    model.eval()
    _disable_last_P(model)
    if naive:
        for layer in model.layers:
            layer.attn._force_naive_attn = True
    print(f"[ctx] {'dense-naive' if naive else 'dense-flash'} ready ({mode}) | "
          f"layers={len(model.layers)}")
    return model


def load_iso(kind, ckpt_path, device, vocab):
    """Proto-mode loader for the baselines (dense / swa / spatra).

    The cfg is stored inside the .pt (ckpt["cfg"]), including the ``window`` of SWA and
    Sparse-T and the ``stride`` of Sparse-T, so the training-time geometry is
    reproduced. The model uses the dense forward path.
    """
    import importlib
    mod_name, cls_name = _ISO_ARCH[kind]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    ckpt_path = _abs(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    kwargs = dict(
        vocab_size=vocab, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], max_seq_len=cfg["seq_len"], d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0), tie_weights=cfg.get("tie_weights", True),
        loss_fn="chunked",
    )
    if kind in ("swa", "spatra") and ckpt.get("window") is not None:
        kwargs["window"] = int(ckpt["window"])
    if kind == "spatra" and ckpt.get("stride") is not None:
        kwargs["stride"] = int(ckpt["stride"])
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_iso_baseline(kind, ckpt_path, device, vocab):
    """load_iso, then eval mode and _disable_last_P."""
    model = load_iso(kind, ckpt_path, device, vocab)
    model.eval()
    _disable_last_P(model)
    print(f"[ctx] {kind} ready (proto) | layers={len(model.layers)}")
    return model


def load_grafted_baseline(kind, ckpt_dir, device):
    """[3b] SWA / Sparse-T graft directories, loaded with eval_phase1_3b.load_grafted.
    """
    from results.eval_phase1_3b import load_grafted
    model, arch = load_grafted(str(_abs(ckpt_dir)), device)
    assert arch == kind, f"{ckpt_dir} is arch={arch}, expected {kind}"
    model.eval()
    _disable_last_P(model)
    print(f"[ctx] {kind} ready (3b) | layers={len(model.layers)}")
    return model


# data: one token stream and a set of evaluation windows shared across lengths
def _load_long_token_stream(device, shards_dir, want):
    """Concatenated 1-D token tensor and per-document end offsets from the val shards.
    """
    import utils.data_utils as d
    d.SHARDS_DIR = Path(shards_dir)
    from utils.data_utils import LongformShardDataset
    val_ds = LongformShardDataset("val_shard")

    # Shift each shard's bounds by the running offset so that document spans refer to
    # the concatenated stream.
    parts, all_bounds, offset = [], [], 0
    for sp in sorted(Path(shards_dir).glob("val_shard_*.pt")):
        if sp.name.endswith(".bounds.pt"):
            continue
        toks = torch.load(sp, map_location="cpu").view(-1).long()
        bp = sp.with_suffix(".bounds.pt")
        bounds = (torch.load(bp, map_location="cpu").long()
                  if bp.exists() else torch.tensor([toks.numel()], dtype=torch.long))
        parts.append(toks)
        all_bounds.append(bounds + offset)
        offset += toks.numel()
    stream = torch.cat(parts).to(device)
    bounds = torch.cat(all_bounds)
    print(f"[ctx] token stream = {stream.numel():,} tokens | {bounds.numel():,} docs "
          f"(want {want:,})")
    return stream, bounds


def _doc_spans(bounds):
    """[(start, end), ...] for every document from cumulative end offsets."""
    ends = bounds.tolist()
    spans, prev = [], 0
    for e in ends:
        if e > prev:
            spans.append((prev, e))
        prev = e
    return spans


def _window_starts(bounds, L, n_windows):
    """Start offsets of up to n_windows windows of length L, each inside one document.

    The longest documents that admit an L-window are taken first, with the window
    anchored to the end of the document, round-robin until n_windows windows are found.
    Fewer windows are returned if too few documents are long enough.
    """
    spans = [(s, e) for (s, e) in _doc_spans(bounds) if e - s >= L]
    spans.sort(key=lambda se: se[1] - se[0], reverse=True)
    if not spans:
        return []
    starts = []
    for s, e in spans:
        starts.append(e - L)
        if len(starts) >= n_windows:
            break
    return starts


@torch.no_grad()
def _ppl_at_length(model, kind, stream, bounds, L, device, amp, rope_mode, n_windows):
    """Token-weighted perplexity over the shared windows, with the per-window spread."""
    _patch_rope(model, L, device, rope_mode)
    starts = _window_starts(bounds, L, n_windows)
    if not starts:
        return {"ppl": float("nan"), "n_windows": 0, "ppl_std": float("nan"),
                "ppl_min": float("nan"), "ppl_max": float("nan")}
    tot_nll = 0.0
    tot_tok = 0
    per_win = []
    for s in starts:
        seg = stream[s:s + L].contiguous()
        labels = seg.clone()
        labels[:-1] = seg[1:]
        labels[-1] = -100
        cu = torch.tensor([0, L], device=device, dtype=torch.int32)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            if kind == "dsalt":
                out = model(seg, cu_seqlens=cu, max_seqlen=L, labels=labels)
            else:
                out = model(seg, cu_seqlens=cu, labels=labels)
        nll = out["loss"].item()
        ntok = int((labels != -100).sum().item())
        tot_nll += nll * ntok
        tot_tok += ntok
        per_win.append(math.exp(min(nll, 20.0)))
    mean_ppl = math.exp(min(tot_nll / max(1, tot_tok), 20.0))
    pw = torch.tensor(per_win)
    return {"ppl": mean_ppl, "n_windows": len(starts),
            "ppl_std": pw.std(unbiased=False).item() if len(pw) > 1 else 0.0,
            "ppl_min": pw.min().item(), "ppl_max": pw.max().item(),
            "window_starts": starts}


@torch.no_grad()
def _latency_at_length(model, kind, stream, L, device, amp, rope_mode,
                       n_warmup=2, n_iter=5):
    """Median forward latency (ms) at length L."""
    _patch_rope(model, L, device, rope_mode)
    seg = stream[:L].contiguous()
    cu = torch.tensor([0, L], device=device, dtype=torch.int32)

    def _one():
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            if kind == "dsalt":
                model(seg, cu_seqlens=cu, max_seqlen=L)
            else:
                model(seg, cu_seqlens=cu)

    for _ in range(n_warmup):
        _one()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _one()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _run_model(model, label, call_kind, stream, bounds, device, amp, lengths, rope_mode, n_windows):
    results = {}
    oom_at = None
    for L in lengths:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            stat = _ppl_at_length(model, call_kind, stream, bounds, L, device, amp, rope_mode, n_windows)
            lat = _latency_at_length(model, call_kind, stream, L, device, amp, rope_mode)
            peak = (torch.cuda.max_memory_allocated(device) / 1e9
                    if device.type == "cuda" else float("nan"))
            results[L] = {**stat, "peak_gb": peak, "latency_ms": lat, "oom": False,
                          "rope_scale": _rope_scale_for(L, rope_mode)}
            print(f"[{label}] L={L:>7d} | ppl {stat['ppl']:9.3f} "
                  f"(±{stat['ppl_std']:6.3f}, {stat['n_windows']}w) | "
                  f"peak {peak:6.2f} GB | {lat:8.1f} ms | "
                  f"rope {_rope_scale_for(L, rope_mode):.2f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            oom_at = L
            results[L] = {"ppl": float("nan"), "peak_gb": float("nan"),
                          "latency_ms": float("nan"), "oom": True,
                          "rope_scale": _rope_scale_for(L, rope_mode)}
            print(f"[{label}] L={L:>7d} | *** OOM *** (stop scaling here)", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    return results, oom_at


def _free(model, device):
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# main
def main():
    global L_TRAIN
    ap = argparse.ArgumentParser(description="Context-scaling (paper Exp. A), 3B or prototype.")
    ap.add_argument("--mode", choices=["3b", "proto"], default="3b",
                    help="checkpoint format: 3b (grafted dirs) or proto (dense cfg in .pt)")
    ap.add_argument("--dsalt-dir", required=True, help="DSALT checkpoint dir (has model_config.json)")
    ap.add_argument("--dense-dir", default=None, help="[3b] dense grafted dir")
    ap.add_argument("--dense-ckpt", default=None, help="[proto] dense .pt with cfg inside")
    ap.add_argument("--swa-ckpt", default=None,
                    help="[proto] optional SWA baseline .pt with cfg inside")
    ap.add_argument("--spatra-ckpt", default=None,
                    help="[proto] optional Sparse-Transformer baseline .pt with cfg inside")
    ap.add_argument("--swa-dir", default=None,
                    help="[3b] optional SWA grafted dir (model_config.json + checkpoint)")
    ap.add_argument("--spatra-dir", default=None,
                    help="[3b] optional Sparse-Transformer grafted dir")
    ap.add_argument("--shards", default="./data_shards_pg19_qwen")
    ap.add_argument("--lengths", default="2048,4096,8192,16384,32768",
                    help="comma-separated context lengths")
    ap.add_argument("--rope-mode", choices=["ntk", "linear", "none"], default="ntk")
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--l-train", type=int, default=L_TRAIN)
    ap.add_argument("--vocab", type=int, default=50257, help="[proto] dense vocab size")
    ap.add_argument("--out", default="context_scaling.json")
    args = ap.parse_args()

    L_TRAIN = args.l_train
    lengths = [int(x) for x in args.lengths.split(",") if x]
    if args.mode == "3b" and not args.dense_dir:
        ap.error("--dense-dir is required for --mode 3b")
    if args.mode == "proto" and not args.dense_ckpt:
        ap.error("--dense-ckpt is required for --mode proto")

    device = _device()
    amp = _amp_dtype(device)
    torch.manual_seed(42)

    want = max(lengths) * args.n_windows
    stream, bounds = _load_long_token_stream(device, args.shards, want)
    # Windows lie inside single documents, so what matters is how many documents are at
    # least L tokens long.
    spans = _doc_spans(bounds)
    for L in lengths:
        n_long = sum(1 for s, e in spans if e - s >= L)
        if n_long == 0:
            raise RuntimeError(
                f"no validation document is >= {L:,} tokens; cannot eval at L={L}. "
                f"Add longer documents (arXiv papers / long PG19 books) or drop {L}.")
        if n_long < args.n_windows:
            print(f"[ctx] WARNING: only {n_long} docs >= {L:,} tokens "
                  f"(< {args.n_windows} windows); L={L} ppl will use {n_long} windows")

    out = {"config": {"mode": args.mode, "l_train": L_TRAIN, "rope_mode": args.rope_mode,
                      "lengths": lengths, "n_windows": args.n_windows,
                      "stream_tokens": int(stream.numel())}}

    m = load_dsalt(args.dsalt_dir, device, args.mode, args.vocab)
    dsalt_res, dsalt_oom = _run_model(m, "dsalt", "dsalt", stream, bounds, device, amp,
                                      lengths, args.rope_mode, args.n_windows)
    _free(m, device)

    m = load_dense(args.dense_dir, device, args.mode, args.vocab, naive=False,
                   dense_ckpt=args.dense_ckpt)
    flash_res, flash_oom = _run_model(m, "dense-flash", "dense", stream, bounds, device, amp,
                                      lengths, args.rope_mode, args.n_windows)
    _free(m, device)

    m = load_dense(args.dense_dir, device, args.mode, args.vocab, naive=True,
                   dense_ckpt=args.dense_ckpt)
    naive_res, naive_oom = _run_model(m, "dense-naive", "dense", stream, bounds, device, amp,
                                      lengths, args.rope_mode, args.n_windows)
    _free(m, device)

    out["dsalt"] = {"by_length": dsalt_res, "oom_at": dsalt_oom}
    out["dense_flash"] = {"by_length": flash_res, "oom_at": flash_oom}
    out["dense_naive"] = {"by_length": naive_res, "oom_at": naive_oom}

    # Optional SWA and Sparse-T baselines, which use the dense forward path.
    #   proto mode: .pt with cfg inside (--swa-ckpt / --spatra-ckpt)
    #   3b mode:    directories with model_config.json (--swa-dir / --spatra-dir)
    baseline_res = {}
    for kind in ("swa", "spatra"):
        if args.mode == "3b":
            src = args.swa_dir if kind == "swa" else args.spatra_dir
            if not src:
                continue
            m = load_grafted_baseline(kind, src, device)
        else:
            src = args.swa_ckpt if kind == "swa" else args.spatra_ckpt
            if not src:
                continue
            m = load_iso_baseline(kind, src, device, args.vocab)
        res, oom = _run_model(m, kind, "dense", stream, bounds, device, amp,
                              lengths, args.rope_mode, args.n_windows)
        _free(m, device)
        out[kind] = {"by_length": res, "oom_at": oom}
        baseline_res[kind] = (res, oom)
    # Non-finite floats (OOM rows) become null so that the file is valid JSON.
    def _clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    Path(args.out).write_text(json.dumps(_clean(out), indent=2, allow_nan=False),
                              encoding="utf-8")
    print(f"\nsaved -> {args.out}\n")

    # perplexity with its per-window spread, then peak memory for DSALT, dense with
    # FlashAttention and dense with the materialized matrix
    print(f"{'L':>8s} {'DSALT ppl':>16s} {'DENSE ppl':>16s} | "
          f"{'DSALT GB':>9s} {'D-flash GB':>11s} {'D-naive GB':>11s}")
    print("-" * 82)
    for L in lengths:
        d = dsalt_res.get(L, {})
        f = flash_res.get(L, {})
        nv = naive_res.get(L, {})
        def _pp(x):
            if x.get("oom"):
                return "OOM"
            return f"{x.get('ppl', float('nan')):8.3f}±{x.get('ppl_std', float('nan')):5.2f}"
        ng = "OOM" if nv.get("oom") else f"{nv.get('peak_gb', float('nan')):9.2f}"
        print(f"{L:>8d} {_pp(d):>16s} {_pp(f):>16s} | "
              f"{d.get('peak_gb', float('nan')):9.2f} {f.get('peak_gb', float('nan')):11.2f} {ng:>11s}")

    print(f"\nOOM  ->  DSALT: {dsalt_oom}   dense-flash: {flash_oom}   dense-naive: {naive_oom}")

    for kind, (res, oom) in baseline_res.items():
        print(f"\n=== {kind.upper()} (sparse baseline) ===")
        print(f"{'L':>8s} {kind.upper() + ' ppl':>16s} {kind.upper() + ' GB':>10s}")
        print("-" * 36)
        for L in lengths:
            x = res.get(L, {})
            if x.get("oom"):
                print(f"{L:>8d} {'OOM':>16s} {'OOM':>10s}")
            else:
                print(f"{L:>8d} "
                      f"{x.get('ppl', float('nan')):8.3f}±{x.get('ppl_std', float('nan')):5.2f} "
                      f"{x.get('peak_gb', float('nan')):10.2f}")
        print(f"OOM -> {kind}: {oom}")

    # latency, and the FlashAttention / DSALT ratio
    print(f"\n{'L':>8s} {'DSALT ms':>10s} {'D-flash ms':>11s} {'D-naive ms':>11s} {'flash/DSALT':>12s}")
    print("-" * 56)
    for L in lengths:
        d = dsalt_res.get(L, {}).get("latency_ms", float("nan"))
        f = flash_res.get(L, {}).get("latency_ms", float("nan"))
        nv = naive_res.get(L, {})
        ntag = "OOM" if nv.get("oom") else f"{nv.get('latency_ms', float('nan')):9.1f}"
        ratio = f / d if (d and math.isfinite(d) and math.isfinite(f)) else float("nan")
        print(f"{L:>8d} {d:10.1f} {f:11.1f} {ntag:>11s} {ratio:12.2f}")

    # slowdown relative to L_train: about 4x per doubling for O(L^2), 2x for O(L)
    def _base(res):
        b = res.get(L_TRAIN, {}).get("latency_ms", float("nan"))
        return b if (b and math.isfinite(b)) else float("nan")
    b_d, b_f, b_n = _base(dsalt_res), _base(flash_res), _base(naive_res)
    print(f"\n  slowdown vs L={L_TRAIN} (xN):")
    print(f"{'L':>8s} {'DSALT':>9s} {'D-flash':>9s} {'D-naive':>9s}")
    print("-" * 38)
    for L in lengths:
        def _nx(res, base):
            v = res.get(L, {})
            if v.get("oom"):
                return "OOM"
            ms = v.get("latency_ms", float("nan"))
            return f"{ms / base:7.2f}x" if math.isfinite(ms) and math.isfinite(base) else "  nan"
        print(f"{L:>8d} {_nx(dsalt_res, b_d):>9s} {_nx(flash_res, b_f):>9s} {_nx(naive_res, b_n):>9s}")


if __name__ == "__main__":
    main()
