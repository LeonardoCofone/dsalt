"""Block-causal evaluation of the DSALT landmark selection.

At inference DSALT selects its landmarks once per sequence and head: the hybrid score
is standardized over the whole sequence and the top-k is taken over all positions
(dsalt/kernels/selectors.py). A query never attends to a later token, but which earlier
tokens are available as landmarks depends on later tokens, through the standardization
statistics and the competition for the k slots. This script evaluates the trained
48-layer DSALT models again with a selection that uses only earlier tokens.

Three modes for DSALT (dense, SWA and Sparse-T are not affected):
  * "triton": the original path (whole-sequence selection, Triton kernel);
  * "global": a PyTorch implementation of the same whole-sequence selection, used only
              to check that it reproduces "triton";
  * "block":  block-causal selection. Queries in block [b0, b0+B) use only landmarks
              chosen from positions < b0, with the standardization statistics of that
              prefix; the window is unchanged. Every prediction then depends only on
              earlier tokens.

The protocols are those of the paper, with the original functions:
  * context sweep: results.context_scaling._ppl_at_length (8 tail windows per length,
    perplexity over all tokens, NTK RoPE);
  * counterfactual: results.counterfactual.run_counterfactual (keep_local 4096,
    last_k 512, same donors).

Decision rule, fixed on 22/10/2026 before the run: if for every seed and length the
block-causal perplexity is within 2% of the whole-sequence perplexity of the same run,
the paper keeps the whole-sequence numbers; otherwise it reports the block-causal ones.
Result (causal_landmark_check.json): the largest difference is 2.31% (seed 6767,
131,072 tokens), so the paper reports the block-causal numbers for the 48-layer models.

Usage (one A100; absolute paths, since relative ones are resolved from results/)::

    python -m results.causal_landmark_check         --dsalt-dirs /path/ck48_dsalt_seed1,/path/ck48_dsalt_seed999,/path/ck48_dsalt_seed6767         --shards /path/data_shards_longctx_gpt2 --out causal_landmark_check.json

Sanity check only (a few minutes)::

    python -m results.causal_landmark_check --dsalt-dirs /path/ck48_dsalt_seed1         --shards /path/data_shards_longctx_gpt2 --sanity-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dsalt.modules import dsalt_attention as DA
from dsalt.kernels.window_utils import apply_rotary_emb

THRESHOLD = 0.02                       # fixed before the run, see the module docstring
SWEEP_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
CFX_LENGTHS = [16384, 32768, 65536, 131072]

_ORIG_PACKED = DA.DSALTAttention._packed
_CFG = {"mode": "triton", "block": 512}


# --------------------------------------------------------------------------- #
# landmark selection
# --------------------------------------------------------------------------- #
def _token_signals(attn, x):
    """Per-token norms, computed exactly as hybrid_scores_per_head does.

    Both are functions of a single token's hidden state, so they are causal; only
    their standardization (mean / std over a set of tokens) can leak the future."""
    dh = attn.head_dim
    W_V = attn.v_proj.weight.detach()
    T = x.shape[0]
    x_norm = x.norm(dim=-1).float()                                     # [T]
    n_kv = W_V.shape[0] // dh
    xwv = (x @ W_V.T).view(T, n_kv, dh).norm(dim=-1).float()            # [T, n_kv]
    return x_norm, xwv


def _prefix_stats(v):
    """Running sums for the unbiased mean / std of every prefix of v (float64)."""
    v = v.double()
    return torch.cumsum(v, 0), torch.cumsum(v * v, 0)


def _std_at(S1, S2, n):
    """(mean, std) over the first n entries, torch.std() convention (unbiased)."""
    mean = S1[n - 1] / n
    var = (S2[n - 1] - n * mean * mean) / max(n - 1, 1)
    return mean, var.clamp(min=0).sqrt().clamp(min=1e-6)


def _select(attn, x_norm, xwv, covered, alpha, n, sx, sv):
    """Top-k landmark positions per head among positions [0, n), using statistics
    of that prefix only. Returns [H, k] long, -1 where no valid candidate."""
    H, k = attn.n_heads, attn.k_lmk
    out = torch.full((H, k), -1, dtype=torch.long, device=x_norm.device)
    if n < 2:
        return out
    mx, sdx = _std_at(*sx, n)
    mv, sdv = _std_at(*sv, n)
    z_x = ((x_norm[:n].double() - mx) / sdx).float()                    # [n]
    z_v = ((xwv[:n].double() - mv) / sdv).float()                       # [n, n_kv]
    if z_v.shape[1] != H:
        z_v = z_v.repeat_interleave(H // z_v.shape[1], dim=1)
    scores = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(1)             # [n, H]
    scores = scores.masked_fill(covered[:n].unsqueeze(1), float("-inf"))
    k_eff = min(k, n)
    val, idx = torch.topk(scores.T, k_eff, dim=1, sorted=False)         # [H, k_eff]
    out[:, :k_eff] = torch.where(torch.isfinite(val), idx, torch.full_like(idx, -1))
    return out


# --------------------------------------------------------------------------- #
# attention with an explicit landmark set per query block
# --------------------------------------------------------------------------- #
def _block_attention(q, k, v, w, lmk, b0, b1, scale):
    """Exact kernel semantics for queries [b0, b1): window keys j in [i-w(i)+1, i],
    plus landmark keys at positions p with p <= i and p < i-w(i)+1."""
    H = q.shape[1]
    wb = w[b0:b1].long()
    lo = max(0, b0 - int(wb.max()) + 1)
    qb = q[b0:b1].float().transpose(0, 1)                               # [H, B, D]
    kw = k[lo:b1].float().transpose(0, 1)                               # [H, Kw, D]
    vw = v[lo:b1].float().transpose(0, 1)
    i = torch.arange(b0, b1, device=q.device).unsqueeze(1)              # [B, 1]
    j = torch.arange(lo, b1, device=q.device).unsqueeze(0)              # [1, Kw]
    win_lo = i - wb.unsqueeze(1) + 1                                    # [B, 1]
    in_win = (j >= win_lo) & (j <= i)                                   # [B, Kw]
    s_w = torch.matmul(qb, kw.transpose(-1, -2)) * scale
    s_w = s_w.masked_fill(~in_win.unsqueeze(0), float("-inf"))

    hidx = torch.arange(H, device=q.device).unsqueeze(1)
    safe = lmk.clamp(min=0)                                             # [H, k]
    lk = k[safe, hidx].float()                                          # [H, k, D]
    lv = v[safe, hidx].float()
    s_l = torch.einsum("hbd,hkd->hbk", qb, lk) * scale                  # [H, B, k]
    pos = lmk.unsqueeze(1)                                              # [H, 1, k]
    ok = (lmk >= 0).unsqueeze(1) & (pos <= i.unsqueeze(0)) & (pos < win_lo.unsqueeze(0))
    s_l = s_l.masked_fill(~ok, float("-inf"))

    p = torch.softmax(torch.cat([s_w, s_l], dim=-1), dim=-1)            # the query itself is always in-window
    Kw = kw.shape[1]
    out = torch.matmul(p[..., :Kw], vw) + torch.matmul(p[..., Kw:], lv)
    return out.transpose(0, 1)                                          # [B, H, D]


def _packed_checked(self, x, cu_seqlens, total_len, device, rope_cs=None, cu_list=None):
    if self.training or _CFG["mode"] == "triton":
        return _ORIG_PACKED(self, x, cu_seqlens, total_len, device, rope_cs, cu_list)
    if cu_seqlens.numel() != 2:
        raise RuntimeError("causal_landmark_check supports one sequence per forward")

    # q / k / v exactly as DSALTAttention._packed
    q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
    kk = self.k_proj(x).view(total_len, self.n_kv_heads, self.head_dim)
    vv = self.v_proj(x).view(total_len, self.n_kv_heads, self.head_dim)
    if rope_cs is None:
        pos = torch.arange(total_len, device=device)
        cos, sin = self.rope_cos[pos], self.rope_sin[pos]
    else:
        cos, sin = rope_cs
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q, kk = apply_rotary_emb(q, kk, cos.unsqueeze(1), sin.unsqueeze(1))
    kk = self._repeat_kv(kk)
    vv = self._repeat_kv(vv)
    aux = self._aux_zero(x.device, x.dtype)

    T = total_len
    w = self._window_sizes(x, floor=True)
    alpha = self._alpha().detach().float()
    xd = x.detach()
    x_norm, xwv = _token_signals(self, xd)
    covered = torch.arange(T, device=device) < w.long()                 # same exclusion as selectors.py
    sx, sv = _prefix_stats(x_norm), _prefix_stats(xwv)
    scale = 1.0 / math.sqrt(self.head_dim)

    B = _CFG["block"]
    out = torch.empty(T, self.n_heads, self.head_dim, device=device, dtype=torch.float32)
    lmk_global = _select(self, x_norm, xwv, covered, alpha, T, sx, sv) if _CFG["mode"] == "global" else None
    for b0 in range(0, T, B):
        b1 = min(T, b0 + B)
        lmk = lmk_global if lmk_global is not None else _select(self, x_norm, xwv, covered, alpha, b0, sx, sv)
        out[b0:b1] = _block_attention(q, kk, vv, w, lmk, b0, b1, scale)
    out = out.to(q.dtype)
    self._last_P = None
    return self._apply_out_proj(out.view(T, self.d_model)), aux


DA.DSALTAttention._packed = _packed_checked


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _with_mode(mode, fn, *a, **kw):
    _CFG["mode"] = mode
    try:
        return fn(*a, **kw)
    finally:
        _CFG["mode"] = "triton"


def _rel(c, t):
    return abs(c - t) / t if (t and math.isfinite(t) and math.isfinite(c)) else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Causal-landmark check for DSALT (eval only).")
    ap.add_argument("--dsalt-dirs", required=True, help="comma-separated DSALT checkpoint dirs (one per seed)")
    ap.add_argument("--shards", default="./data_shards_proto")
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--sweep-lengths", default=",".join(map(str, SWEEP_LENGTHS)))
    ap.add_argument("--cfx-lengths", default=",".join(map(str, CFX_LENGTHS)))
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--out", default="causal_landmark_check.json")
    args = ap.parse_args()

    torch._dynamo.config.disable = True
    import results.context_scaling as cs
    import results.counterfactual as cf
    import results.bench as bench

    _CFG["block"] = args.block
    device = cs._device()
    amp = cs._amp_dtype(device)
    sweep_L = [int(s) for s in args.sweep_lengths.split(",") if s]
    cfx_L = [int(s) for s in args.cfx_lengths.split(",") if s]
    dirs = [d for d in args.dsalt_dirs.split(",") if d]

    print(f"[causal] PRE-REGISTERED RULE: block-causal ppl within {THRESHOLD:.0%} of the "
          f"whole-sequence ppl at every length and seed -> 'selection does not change the "
          f"results'; otherwise the block-causal numbers become the main numbers.\n")

    stream, bounds = cs._load_long_token_stream(device, args.shards, max(sweep_L) * args.n_windows)
    cf_stream, cf_bounds = bench._load_stream(args.shards, device)
    out = {"threshold": THRESHOLD, "block": args.block, "n_windows": args.n_windows,
           "rule": "block-causal ppl within threshold of whole-sequence ppl at every length and seed",
           "seeds": {}}

    for d in dirs:
        model = cs.load_dsalt(d, device, "proto", args.vocab)
        res = {"sanity": {}, "sweep": {}, "counterfactual": {}}

        # 1) sanity: the PyTorch re-implementation must reproduce the Triton kernel
        for L in (4096, 16384):
            t = _with_mode("triton", cs._ppl_at_length, model, "dsalt", stream, bounds, L,
                           device, amp, "ntk", 2)["ppl"]
            g = _with_mode("global", cs._ppl_at_length, model, "dsalt", stream, bounds, L,
                           device, amp, "ntk", 2)["ppl"]
            res["sanity"][str(L)] = {"triton": t, "global_torch": g, "rel_diff": _rel(g, t)}
            print(f"[causal] {d} sanity L={L}: triton {t:.4f} | torch-global {g:.4f} | "
                  f"rel diff {_rel(g, t):.2e}", flush=True)
        worst = max(v["rel_diff"] for v in res["sanity"].values())
        res["sanity_ok"] = bool(worst < 2e-3)
        if not res["sanity_ok"]:
            print(f"[causal] SANITY FAILED (rel diff {worst:.2e} >= 2e-3): the re-implementation "
                  f"does not match the kernel, causal numbers would be meaningless. Stopping.")
            out["seeds"][d] = res
            break
        if args.sanity_only:
            out["seeds"][d] = res
            continue

        # 2) context sweep, same protocol as the paper
        for L in sweep_L:
            t0 = time.time()
            t = _with_mode("triton", bench._safe, cs._ppl_at_length, model, "dsalt", stream, bounds,
                           L, device, amp, "ntk", args.n_windows, device=device, label=f"triton L={L}")
            c = _with_mode("block", bench._safe, cs._ppl_at_length, model, "dsalt", stream, bounds,
                           L, device, amp, "ntk", args.n_windows, device=device, label=f"block L={L}")
            res["sweep"][str(L)] = {"whole_sequence": t.get("ppl"), "block_causal": c.get("ppl"),
                                    "rel_diff": _rel(c.get("ppl"), t.get("ppl"))}
            print(f"[causal] {d} sweep L={L:>6d}: whole {t.get('ppl'):9.4f} | causal "
                  f"{c.get('ppl'):9.4f} | rel {res['sweep'][str(L)]['rel_diff']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # 3) counterfactual, same protocol as the paper
        for L in cfx_L:
            t = _with_mode("triton", cf.run_counterfactual, model, "dsalt", cf_stream, cf_bounds, L,
                           4096, 512, args.n_windows, device, amp)
            c = _with_mode("block", cf.run_counterfactual, model, "dsalt", cf_stream, cf_bounds, L,
                           4096, 512, args.n_windows, device, amp)
            res["counterfactual"][str(L)] = {"whole_sequence": t, "block_causal": c}
            print(f"[causal] {d} cfx L={L:>6d}: whole true {t['ppl_true']:.3f} delta {t['delta']:+.3f} | "
                  f"causal true {c['ppl_true']:.3f} delta {c['delta']:+.3f}", flush=True)

        out["seeds"][d] = res
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # verdict
    rels = [v["rel_diff"] for r in out["seeds"].values() for v in r.get("sweep", {}).values()]
    rels += [_rel(v["block_causal"]["ppl_true"], v["whole_sequence"]["ppl_true"])
             for r in out["seeds"].values() for v in r.get("counterfactual", {}).values()]
    rels = [x for x in rels if math.isfinite(x)]
    if rels:
        out["max_rel_diff"] = max(rels)
        out["verdict"] = ("within threshold: the non-causal selection does not change the results"
                          if max(rels) <= THRESHOLD else
                          "outside threshold: report the block-causal numbers as the main numbers")
        print(f"\n[causal] max relative difference {max(rels):.4f} -> {out['verdict']}")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[causal] saved -> {args.out}")


if __name__ == "__main__":
    main()
