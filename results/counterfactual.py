"""Counterfactual-context test: does a model use distant context?

For each within-document window of length L (the tail-anchored windows of
bench.run_lm_utilization), the perplexity of the final ``last_k`` tokens is
measured twice:
  (a) true:     the real document context;
  (b) replaced: the last ``keep_local`` tokens are kept and everything before them
      is replaced by text from a different document (the same donor for every
      model).
delta = ppl(replaced) - ppl(true). A model that reads the distant context should
get worse when it is replaced (delta > 0).

Expected readings:
  * SWA: beyond its receptive radius (window x n_layers) the replaced tokens cannot
    reach the scored tokens, so delta is 0 by construction.
  * DSALT: delta > 0 shows that predictions depend on the distant context.
  * Dense: sees everything, but its long-context degradation enters delta too.

Usage (3B graft)::

    DSALT_COMPILE_FLEX=1 python -m results.counterfactual \
        --dsalt-dir ckpt_dsalt --dense-dir ckpt_dense \
        --swa-dir ckpt_swa --spatra-dir ckpt_spatra \
        --lengths 16384,32768,65536,131072 \
        --merge-into bench_3b_4way.json --out bench_3b_4way.json

Usage (from-scratch, 48 layers)::

    python -m results.counterfactual --mode proto --dsalt-dir ck48_dsalt \
        --dense-ckpt dense.pt --swa-ckpt swa.pt --spatra-ckpt spatra.pt \
        --lm-shards ./data_shards_proto --vocab 50257 \
        --lengths 16384,32768,65536,131072 --out counterfactual_proto48.json

With --merge-into, results go under the ``counterfactual`` key and the other
sections of the file are kept (the previous file is backed up as .bak).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

import results.bench as bench


def _doc_spans(bounds):
    """[(start, end)] document spans from the cumulative bounds tensor."""
    ends = bounds.tolist()
    spans, prev = [], 0
    for e in ends:
        if e > prev:
            spans.append((prev, e))
        prev = e
    return spans


def pick_donor(spans, stream_len, win_start, win_end, need, rng):
    """Start offset of a ``need``-token donor segment that does not overlap the window,
    preferably beginning at a document start.
    """
    docs = [(s, e) for (s, e) in spans
            if e - s >= need and (s + need <= win_start or s >= win_end)]
    if docs:
        s, _ = docs[rng.randrange(len(docs))]
        return s
    ranges = []
    if win_start >= need:
        ranges.append((0, win_start - need))
    if stream_len - win_end >= need:
        ranges.append((win_end, stream_len - need))
    if not ranges:
        return None
    lo, hi = ranges[rng.randrange(len(ranges))]
    return rng.randint(lo, hi)


@torch.no_grad()
def _tail_ppl(model, kind, seg, last_k, device, amp):
    """Perplexity of the final last_k tokens of one segment (as in
    bench.run_lm_utilization, single window).
    """
    L = seg.numel()
    labels = seg.clone()
    labels[:-1] = seg[1:]
    labels[-1] = -100
    labels[: max(0, L - last_k)] = -100
    cu = torch.tensor([0, L], device=device, dtype=torch.int32)
    with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
        out = model(seg, cu_seqlens=cu, max_seqlen=L, labels=labels) if kind == "dsalt" \
            else model(seg, cu_seqlens=cu, labels=labels)
    ntok = int((labels != -100).sum().item())
    return out["loss"].item() * ntok, ntok


@torch.no_grad()
def run_counterfactual(model, kind, stream, bounds, L, keep_local, last_k,
                       n_windows, device, amp, seed=1234):
    """{ppl_true, ppl_corrupt, delta, n_windows} at one length L.

    The donor for window i at length L depends only on (seed, L, i), so every model
    scores the same token sequences.
    """
    bench._patch_rope(model, L, device, "ntk")
    starts = bench._doc_starts(bounds, L, n_windows)
    if not starts:
        return {"ppl_true": float("nan"), "ppl_corrupt": float("nan"),
                "delta": float("nan"), "n_windows": 0}
    spans = _doc_spans(bounds)
    corrupt_len = L - keep_local
    nll_t = nll_c = tok_t = tok_c = 0
    for i, s in enumerate(starts):
        rng = random.Random(f"{seed}:{L}:{i}")
        seg = stream[s:s + L].contiguous()
        d = pick_donor(spans, stream.numel(), s, s + L, corrupt_len, rng)
        if d is None:
            continue
        seg_c = seg.clone()
        seg_c[:corrupt_len] = stream[d:d + corrupt_len]
        a, na = _tail_ppl(model, kind, seg, last_k, device, amp)
        b, nb = _tail_ppl(model, kind, seg_c, last_k, device, amp)
        nll_t += a; tok_t += na
        nll_c += b; tok_c += nb
    if not tok_t:
        return {"ppl_true": float("nan"), "ppl_corrupt": float("nan"),
                "delta": float("nan"), "n_windows": 0}
    pt = math.exp(min(nll_t / tok_t, 20.0))
    pc = math.exp(min(nll_c / tok_c, 20.0))
    return {"ppl_true": pt, "ppl_corrupt": pc, "delta": pc - pt,
            "n_windows": len(starts)}


def main():
    ap = argparse.ArgumentParser(description="Counterfactual-context ablation (4-way).")
    ap.add_argument("--mode", choices=["3b", "proto"], default="3b")
    ap.add_argument("--dsalt-dir", required=True)
    ap.add_argument("--dense-dir", default=None, help="[3b] dense grafted dir")
    ap.add_argument("--swa-dir", default=None, help="[3b] SWA grafted dir (needs DSALT_COMPILE_FLEX=1 at long L)")
    ap.add_argument("--spatra-dir", default=None, help="[3b] SparseT grafted dir")
    ap.add_argument("--dense-ckpt", default=None, help="[proto] dense .pt")
    ap.add_argument("--swa-ckpt", default=None, help="[proto] SWA .pt")
    ap.add_argument("--spatra-ckpt", default=None, help="[proto] SparseT .pt")
    ap.add_argument("--vocab", type=int, default=50257, help="[proto] baseline vocab")
    ap.add_argument("--lm-shards", default="./data_shards_longctx_qwen")
    ap.add_argument("--lengths", default="16384,32768,65536,131072",
                    help="context lengths; each must exceed --keep-local")
    ap.add_argument("--keep-local", type=int, default=4096,
                    help="tail tokens kept intact (= the training window, so the "
                         "local regime is identical across models); everything "
                         "before is swapped")
    ap.add_argument("--last-k", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--merge-into", default=None,
                    help="existing bench JSON: results grafted under 'counterfactual', "
                         "everything else preserved (previous file backed up as .bak)")
    ap.add_argument("--out", default="counterfactual.json")
    args = ap.parse_args()

    import os
    if os.environ.get("DSALT_COMPILE_FLEX") != "1":
        torch._dynamo.config.disable = True
    device = bench._device()
    amp = bench._amp_dtype(device)

    lengths = [int(x) for x in args.lengths.split(",") if x]
    bad = [L for L in lengths if L <= args.keep_local]
    if bad:
        raise SystemExit(f"[cfx] lengths {bad} <= keep_local {args.keep_local}: "
                         f"nothing to ablate there, drop them")

    if args.mode == "proto":
        dirs = [("dsalt", args.dsalt_dir)]
        if args.dense_ckpt:
            dirs.append(("dense", args.dense_ckpt))
        if args.swa_ckpt:
            dirs.append(("swa", args.swa_ckpt))
        if args.spatra_ckpt:
            dirs.append(("spatra", args.spatra_ckpt))
    else:
        if not args.dense_dir:
            ap.error("--dense-dir is required in 3b mode")
        dirs = [("dsalt", args.dsalt_dir), ("dense", args.dense_dir)]
        if args.swa_dir:
            dirs.append(("swa", args.swa_dir))
        if args.spatra_dir:
            dirs.append(("spatra", args.spatra_dir))

    stream, bounds = bench._load_stream(args.lm_shards, device)
    print(f"[cfx] stream {stream.numel():,} tokens | {bounds.numel():,} docs")
    if args.mode == "proto" and int(stream.max()) >= args.vocab:
        raise SystemExit(f"[cfx] shard token id {int(stream.max())} >= vocab {args.vocab}: "
                         f"wrong shards for this vocabulary")
    n_fwd = 2 * args.n_windows * len(lengths) * len(dirs)
    print(f"[cfx] budget: {n_fwd:,} forwards total ({len(dirs)} models, "
          f"{len(lengths)} lengths, {args.n_windows} windows x 2 passes)")
    # Expected reading for SWA, printed before the numbers.
    if args.mode == "3b" and args.swa_dir:
        meta = json.loads((Path(args.swa_dir) / "model_config.json").read_text(encoding="utf-8"))
        w, nl = meta["window"], meta["n_layers"]
        print(f"[cfx] SWA window={w} x n_layers={nl} -> receptive radius {w * nl:,} tokens: "
              f"beyond it delta ~ 0 is STRUCTURAL, not a finding")
    print("[cfx] PREDICTION: the measurand is DSALT's delta > 0 (landmarks carry usable "
          "distant signal); a local-bubble model cannot notice the swap (delta ~ 0)\n")

    results = {"keep_local": args.keep_local, "last_k": args.last_k,
               "n_windows": args.n_windows, "seed": args.seed,
               "lengths": lengths, "models": {}}
    for kind, ckpt in dirs:
        print(f"[cfx] ===== {kind} =====")
        model = bench.load_model(ckpt, kind, device, mode=args.mode, vocab=args.vocab)
        results["models"][kind] = {}
        for L in lengths:
            r = bench._safe(run_counterfactual, model, kind, stream, bounds, L,
                            args.keep_local, args.last_k, args.n_windows, device, amp,
                            seed=args.seed, device=device, label=f"cfx L={L}")
            results["models"][kind][str(L)] = r
            if r.get("oom"):
                continue
            print(f"  L={L:>7d} | true {r['ppl_true']:7.2f}  ablated {r['ppl_corrupt']:7.2f}  "
                  f"delta {r['delta']:+7.2f}  ({r['n_windows']} win)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = results
    out = Path(args.out)
    if args.merge_into:
        base = json.loads(Path(args.merge_into).read_text(encoding="utf-8"))
        base["counterfactual"] = results
        payload = base
        if out.exists():
            bak = out.with_suffix(out.suffix + ".bak")
            bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[cfx] backed up previous results -> {bak}")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out}\n")

    tags = [k for k, _ in dirs]
    hdr = {"dsalt": "DSALT", "dense": "DENSE", "swa": "SWA", "spatra": "SparseT"}
    print("=== Counterfactual delta (ppl ablated - true; >0 = uses distant context) ===")
    print(f"{'L':>8s}" + "".join(f"{hdr[t]:>10s}" for t in tags))
    for L in lengths:
        row = f"{L:>8d}"
        for t in tags:
            c = results["models"].get(t, {}).get(str(L), {})
            d = c.get("delta")
            row += f"{d:>+10.2f}" if isinstance(d, float) and math.isfinite(d) else f"{'--':>10s}"
        print(row)


if __name__ == "__main__":
    main()
