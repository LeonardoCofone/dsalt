"""Regenerate the figures of the paper from the result JSONs in this directory.

The JSON-driven figures need only matplotlib: no GPU, no checkpoints. Figures are
written to ``images/`` as PDF and PNG.

  * The 170M/48L DSALT and dense models have three seeds (1, 999, 6767): lines are
    the seed mean and the shaded band is the min/max over seeds. Every other model,
    and SWA and Sparse-T, are single runs.
  * Each method keeps one colour, marker and line style across all figures.

The attention-pattern figure needs a forward pass through the trained checkpoints.
By default it is cropped from the committed render
``images/figure2_attention_patterns.png``; ``--attention`` renders it again (GPU,
checkpoints and a shard directory required).

Run::

    python -m results.make_figures
    python -m results.make_figures --attention --mode proto \\
        --dsalt-dir checkpoints_dsalt_48L \\
        --dense-ckpt checkpoints_dense_48L/dense_best.pt \\
        --swa-ckpt checkpoints_swa_48L/swa_best.pt \\
        --spatra-ckpt checkpoints_spatra_48L/spatra_best.pt \\
        --lm-shards ./data_shards_proto --layer 24 --seq-len 1024 --crop 256
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter

DATA = Path(__file__).resolve().parent
OUT = DATA / "images"

# tag -> (context_scaling json, paper_eval json, panel title)
MODELS = {
    "70M": ("context_scaling_proto70.json", "paper_eval_results_70M.json", "70M, 14 layers"),
    "0_3B": ("context_scaling_proto18.json", "paper_eval_results_0_3B.json", "0.3B, 18 layers"),
    "48L": ("context_scaling_proto48.json", "paper_eval_results_48L_full.json", "170M, 48 layers (3 seeds)"),
    "3B": ("context_scaling_3b.json", "paper_eval_results_3b.json", "Qwen2.5-3B graft, 36 layers"),
}
ORDER = ["70M", "0_3B", "48L", "3B"]
SHORT = {"70M": "70M, 14L", "0_3B": "0.3B, 18L", "48L": "170M, 48L (3 seeds)", "3B": "Qwen2.5-3B graft, 36L"}

# extra seeds of the 48-layer pair (the files in MODELS are seed 1)
SEEDS_CS = ["context_scaling_proto48_seed999.json", "context_scaling_proto48_seed6767.json"]
SEEDS_PE = ["paper_eval_results_48L_999seed.json", "paper_eval_results_48L_6767seed.json"]
SEEDED = ("dsalt", "dense", "dense_flash")
# block-causal evaluation of the 48-layer DSALT models, seeds 1, 999, 6767 in file order
CAUSAL = "causal_landmark_check.json"

# method -> (colour, line style, marker, legend name)
STYLE = {
    "dsalt": ("#2a78d6", "-", "s", "DSALT"),
    "dense": ("#eb6834", "-", "o", "Dense"),
    "dense_flash": ("#eb6834", "-", "o", "Dense"),
    "swa": ("#1baf7a", "--", "^", "SWA"),
    "spatra": ("#eda100", "-.", "v", "Sparse-T"),
    "dense_naive": ("#6250d6", ":", "D", "Dense, materialized"),
}
INK = "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#b5b4ae",
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "lines.markersize": 3,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _by_len(block):
    raw = (block or {}).get("by_length", block or {})
    out = {}
    for k, v in raw.items():
        if str(k).isdigit():
            out[int(k)] = v
    return dict(sorted(out.items()))


def _series(block, field):
    """(lengths, values) with OOM and missing points dropped."""
    Ls, ys = [], []
    for L, v in _by_len(block).items():
        y = v.get(field)
        if not v.get("oom") and isinstance(y, (int, float)) and math.isfinite(y):
            Ls.append(L)
            ys.append(y)
    return np.array(Ls, float), np.array(ys, float)


def _per_layer(block, prefix):
    items = []
    for k, v in (block or {}).items():
        if k.startswith(prefix) and k[len(prefix):].isdigit():
            items.append((int(k[len(prefix):]), v.get("mean") if isinstance(v, dict) else v))
    items.sort()
    if not items:
        return np.array([]), np.array([])
    li, val = zip(*items)
    return np.array(li), np.array(val, float)


def _stack(curves):
    """Mean, min and max over seeds; the curves must share the same x."""
    xs = curves[0][0]
    for x, _ in curves[1:]:
        assert len(x) == len(xs) and (x == xs).all(), "seed curves are not aligned"
    arr = np.vstack([y for _, y in curves])
    return xs, arr.mean(0), arr.min(0), arr.max(0)


def _causal_curve():
    """Block-causal perplexity of the 48-layer DSALT models against length, over seeds."""
    curves = []
    for run in _load(CAUSAL)["seeds"].values():
        pts = sorted((int(L), v["block_causal"]) for L, v in run["sweep"].items())
        curves.append((np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)))
    return _stack(curves) + (len(curves),)


def _cs_curve(tag, method, field):
    curves = [_series(_load(MODELS[tag][0]).get(method), field)]
    if tag == "48L" and method in SEEDED:
        curves += [_series(_load(f).get(method), field) for f in SEEDS_CS]
    return _stack(curves) + (len(curves),)


def _pe_curve(tag, method, prefix):
    curves = [_per_layer(_load(MODELS[tag][1]).get(method), prefix)]
    if not len(curves[0][0]):
        return None
    if tag == "48L" and method in SEEDED:
        curves += [_per_layer(_load(f).get(method), prefix) for f in SEEDS_PE]
    return _stack(curves) + (len(curves),)


def _draw(ax, method, x, y, lo, hi, n, ms=4, lw=1.8):
    col, ls, mk, name = STYLE[method]
    if n > 1:
        ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
    ax.plot(x, y, ls, color=col, lw=lw, marker=mk, ms=ms, label=name,
            markeredgecolor="white", markeredgewidth=0.6)


def _len_axis(ax, Ls, ltrain=None):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"{int(L / 1024)}k" for L in Ls], fontsize=6.5)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if ltrain:
        ax.axvline(ltrain, color="#9a998f", lw=0.9, ls="--")
    ax.set_xlabel("Context length (tokens)")


def _log_y(ax):
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_dobrushin():
    """Per-layer Dobrushin coefficient, one panel per model."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.0))
    axes = axes.flat
    for ax, tag in zip(axes, ORDER):
        for m in ("dense", "swa", "spatra", "dsalt"):
            c = _pe_curve(tag, m, "dobrushin_L")
            if c is not None:
                _draw(ax, m, *c, ms=3)
        ax.axhline(1.0, color="#9a998f", lw=0.8, ls="--", zorder=1)
        ax.set_ylim(0.5, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(MODELS[tag][2])
        ax.set_xlabel("Layer")
    axes[0].set_ylabel(r"Dobrushin coefficient $\rho$")
    axes[2].set_ylabel(r"Dobrushin coefficient $\rho$")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, "dobrushin")


def fig_ppl():
    """Perplexity against context length, one panel per model, in one row."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    axes = axes.flat
    for ax, tag in zip(axes, ORDER):
        data = _load(MODELS[tag][0])
        ltrain = data.get("config", {}).get("l_train", 4096)
        Ls = set()
        for m in ("dense_flash", "swa", "spatra", "dsalt"):
            if m not in data:
                continue
            if tag == "48L" and m == "dsalt":
                x, y, lo, hi, n = _causal_curve()
            else:
                x, y, lo, hi, n = _cs_curve(tag, m, "ppl")
            _draw(ax, m, x, y, lo, hi, n)
            Ls |= set(x.tolist())
        _len_axis(ax, sorted(Ls), ltrain)
        _log_y(ax)
        ax.set_title(SHORT[tag])
        ax.set_xlabel("Context length")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    axes[0].set_ylabel("Perplexity (log scale)")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    _save(fig, "ppl_context")


def fig_counterfactual():
    """Delta = ppl(distant context replaced) - ppl(true), and ppl(true), against context length."""
    # On the graft only DSALT and SWA are drawn: the dense and Sparse-T graft runs of
    # this test disagree with the context sweep on the same models (dense 96.8 against
    # 8.55 at 16k) for reasons not yet traced, so the paper does not use them.
    scratch = _load("counterfactual_proto48.json")["models"]
    graft = _load("bench_3b_4way.json")["counterfactual"]["models"]
    seeds = [s["counterfactual"] for s in _load(CAUSAL)["seeds"].values()]

    def single(block, key):
        pts = sorted((int(L), v[key]) for L, v in block.items()
                     if isinstance(v, dict) and isinstance(v.get("delta"), float))
        x, y = (np.array(a, float) for a in zip(*pts))
        return x, y, y, y, 1

    def dsalt_seeds(key):
        Ls = sorted(int(L) for L in seeds[0])
        Y = np.array([[s[str(L)]["block_causal"][key] for L in Ls] for s in seeds])
        return np.array(Ls, float), Y.mean(0), Y.min(0), Y.max(0), len(seeds)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panels = [(axes[0], scratch, ("dense", "spatra", "swa", "dsalt"), "delta", "From scratch, 48 layers"),
              (axes[1], scratch, ("dense", "spatra", "swa", "dsalt"), "ppl_true", "From scratch, 48 layers"),
              (axes[2], graft, ("swa", "dsalt"), "delta", "Qwen2.5-3B graft")]
    for ax, data, kinds, key, title in panels:
        for m in kinds:
            curve = dsalt_seeds(key) if (m == "dsalt" and data is scratch) else single(data[m], key)
            _draw(ax, m, *curve)
        _len_axis(ax, [16384, 32768, 65536, 131072])
        ax.set_title(title)
    ax = axes[0]
    ax.axhline(0.0, color="#9a998f", lw=0.9, ls="--", zorder=1)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    ax.set_ylim(-0.06, 40)
    ax.set_yticks([0, 0.1, 1, 10])
    ax.set_yticklabels(["0", "0.1", "1", "10"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r"$\Delta$ = ppl$_{\mathrm{replaced}}$ $-$ ppl$_{\mathrm{true}}$")
    ax = axes[1]
    _log_y(ax)
    ax.set_ylim(10, 140)
    ax.set_yticks([16, 32, 64, 128])
    ax.set_ylabel("Perplexity, true context")
    ax = axes[2]
    ax.axhline(0.0, color="#9a998f", lw=0.9, ls="--", zorder=1)
    ax.set_ylim(-0.02, 0.1)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.set_ylabel(r"$\Delta$ (linear scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "counterfactual")


def fig_latency_memory():
    """Forward latency, and peak memory on the 0.3B model (where the materialized baseline runs)."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.0))
    axes = list(axes.flat)
    for ax, tag in zip(axes, ("70M", "48L", "3B")):
        Ls = set()
        for m in ("dense_flash", "dsalt"):
            x, y, lo, hi, n = _cs_curve(tag, m, "latency_ms")
            _draw(ax, m, x, y, lo, hi, n)
            Ls |= set(x.tolist())
        _len_axis(ax, sorted(Ls))
        _log_y(ax)
        ax.set_title(MODELS[tag][2])
    axes[0].set_ylabel("Forward latency (ms, log scale)")
    axes[2].set_ylabel("Forward latency (ms, log scale)")
    ax = axes[3]
    data = _load(MODELS["0_3B"][0])
    for m in ("dense_flash", "dense_naive", "dsalt"):
        x, y = _series(data.get(m), "peak_gb")
        _draw(ax, m, x, y, y, y, 1)
    oom = (data.get("dense_naive") or {}).get("oom_at")
    if oom:
        ax.annotate("out of memory", (oom, ax.get_ylim()[1] * 0.9), ha="center", fontsize=6.5, color=INK)
    _len_axis(ax, sorted(_by_len(data.get("dsalt"))))
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title("Peak memory, 0.3B")
    h, l = ax.get_legend_handles_labels()
    l = ["Dense (FlashAttention)" if x == "Dense" else x for x in l]
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, "latency_memory")


def _grid(rows, name):
    """Per-layer DSALT and dense, one row per metric and one column per model."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 1.9 * len(rows) + 0.3), squeeze=False)
    for r, (prefix, ylabel) in enumerate(rows):
        for c, tag in enumerate(ORDER):
            ax = axes[r][c]
            for m in ("dense", "dsalt"):
                cv = _pe_curve(tag, m, prefix)
                if cv is not None:
                    _draw(ax, m, *cv, ms=3, lw=1.5)
            if r == 0:
                ax.set_title(SHORT[tag])
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if r == len(rows) - 1:
                ax.set_xlabel("Layer")
    h, l = axes[0][2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    _save(fig, name)


def fig_attention_patterns_crop():
    """Crop the title off the committed render of the measured attention patterns."""
    from PIL import Image
    img = Image.open(OUT / "figure2_attention_patterns.png")
    img.crop((0, 48, img.width, img.height)).save(OUT / "attention_patterns.png")
    print("saved attention_patterns")


def _attention_matrix(model, kind, ids, device, amp, layer_idx):
    """One forward; the head-averaged attention matrix of ``layer_idx`` as a [T, T] array.

    Each attention module stores the matrix of the first sequence in ``_last_P`` in
    eval mode."""
    import torch

    for layer in model.layers:
        layer.attn._collect_last_P = True
        layer.attn._last_P = None
    ids = ids.reshape(-1)
    T = ids.shape[0]
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            model(ids, cu_seqlens=cu, max_seqlen=T)
    P = model.layers[layer_idx].attn._last_P
    if P is None:
        raise RuntimeError(f"{kind}: _last_P is None at layer {layer_idx}")
    P = P.float().cpu()
    while P.dim() > 2:
        P = P.mean(dim=0)
    return P.numpy()


def fig_attention_patterns(args):
    """Render the measured attention patterns: one panel per model, the last ``crop``
    query rows of the same real sequence, shared log colour scale."""
    import torch
    from matplotlib.colors import LogNorm

    import results.bench as bench

    device = bench._device()
    amp = bench._amp_dtype(device)
    stream, bounds = bench._load_stream(args.lm_shards, device)
    start = bench._doc_starts(bounds, args.seq_len, 1)[0]
    ids = stream[start:start + args.seq_len].to(device)

    panels = [
        ("Dense", "dense", args.dense_dir or args.dense_ckpt),
        ("SWA", "swa", args.swa_dir or args.swa_ckpt),
        ("Sparse-T", "spatra", args.spatra_dir or args.spatra_ckpt),
        ("DSALT", "dsalt", args.dsalt_dir),
    ]
    panels = [p for p in panels if p[2] is not None]
    if not panels:
        raise SystemExit("--attention needs at least one checkpoint (e.g. --dsalt-dir)")

    crops = []
    for name, kind, src in panels:
        model = bench.load_model(src, kind, device, mode=args.mode, vocab=args.vocab)
        model.eval()
        P = _attention_matrix(model, kind, ids, device, amp, args.layer)
        crops.append((name, P[-min(args.crop, P.shape[0]):, :]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    allpos = np.concatenate([c[c > 0].ravel() for _, c in crops])
    norm = LogNorm(vmin=max(allpos.min(), 1e-6), vmax=allpos.max())
    n = len(crops)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for col, (ax, (name, Pc)) in enumerate(zip(axes, crops)):
        im = ax.imshow(Pc, aspect="auto", cmap="magma", norm=norm)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("key position")
        if col == 0:
            ax.set_ylabel(f"query position (last {args.crop})")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="attention weight")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "attention_patterns.png", dpi=160)
    fig.savefig(OUT / "attention_patterns.pdf")
    plt.close(fig)
    print("saved attention_patterns")


def _parse_args():
    ap = argparse.ArgumentParser(description="Regenerate the paper figures from the result JSONs.")
    ap.add_argument("--attention", action="store_true",
                    help="render the attention patterns again (GPU, checkpoints and --lm-shards)")
    ap.add_argument("--mode", choices=["3b", "proto"], default="proto")
    ap.add_argument("--dsalt-dir", type=str, default=None)
    ap.add_argument("--dense-dir", type=str, default=None)
    ap.add_argument("--dense-ckpt", type=str, default=None)
    ap.add_argument("--swa-dir", type=str, default=None)
    ap.add_argument("--swa-ckpt", type=str, default=None)
    ap.add_argument("--spatra-dir", type=str, default=None)
    ap.add_argument("--spatra-ckpt", type=str, default=None)
    ap.add_argument("--lm-shards", type=str, default=None)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--crop", type=int, default=256)
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.attention:
        fig_attention_patterns(args)
    else:
        fig_attention_patterns_crop()
    fig_dobrushin()
    fig_ppl()
    fig_counterfactual()
    fig_latency_memory()
    _grid([("attn_eff_rank_L", "Attention effective rank"),
           ("hidden_eff_rank_L", "Hidden-state effective rank")], "per_layer_rank")
    _grid([("entropy_L", "Attention entropy"),
           ("sink_L", "Attention sink score")], "per_layer_attention")


if __name__ == "__main__":
    main()
