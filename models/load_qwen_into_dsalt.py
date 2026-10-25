"""Load a pretrained Qwen2.5 backbone into DSALT and into the baselines.

The window predictor (``win_gate``) and the per-head landmark balance (``alpha_w``)
do not exist in a dense model, so the grafted models need a short fine-tune
(``train.finetune_phase1``). This module:

  1. builds a :class:`DSALTLMHeadModel` with the architecture of the backbone
     (depth, width, heads, GQA kv-heads, SwiGLU FFN, RMSNorm, RoPE base, q/k/v
     bias, untied lm_head);
  2. copies the Qwen weights into it by name; k/v stay at ``n_kv_heads * head_dim``
     and are repeated at runtime;
  3. initializes the two DSALT-only predictors so that the first forward is close
     to dense causal attention (``seed_near_dense``), and checks the load with a
     perplexity gate against the original Qwen at short context.

``train.finetune_phase1`` re-initializes the window predictor before fine-tuning.

Usage::

    python -m models.load_qwen_into_dsalt --qwen Qwen/Qwen2.5-3B --train-len 4096 \
        --gate-text some_long_doc.txt

Requires ``transformers``.
"""

from __future__ import annotations

import argparse
import math

import torch

from dsalt.model.config   import DSALTConfig
from dsalt.model.dsalt_lm import DSALTLMHeadModel


# Architecture mapping
def dsalt_config_from_qwen(qwen_cfg, n_min: int, n_max: int, k_lmk: int, train_len: int,
                           rope_base_override: float | None = None) -> DSALTConfig:
    """Build a DSALTConfig with the architecture of a HF Qwen2.5 config.

    Depth, width, heads, GQA kv-heads, FFN size, RoPE base, q/k/v bias and weight
    tying are copied from the backbone. ``max_seq_len`` sizes the RoPE cache.
    """
    # HF attribute names (Qwen2.5 / Llama family).
    n_heads    = qwen_cfg.num_attention_heads
    n_kv_heads = getattr(qwen_cfg, "num_key_value_heads", n_heads)
    # Qwen2.5 uses a RoPE base of 1e6. Most checkpoints store it as `rope_theta`;
    # Qwen2.5-3B nests it inside `rope_scaling`, so both are checked before falling back
    # to 10000. A wrong base breaks the gate.
    def _read_rope_theta(c):
        v = getattr(c, "rope_theta", None)
        if v:
            return float(v), "rope_theta"
        rs = getattr(c, "rope_scaling", None)
        if isinstance(rs, dict) and rs.get("rope_theta"):
            return float(rs["rope_theta"]), "rope_scaling.rope_theta"
        return 10000.0, "default(10000)"

    if rope_base_override is not None:
        rope_base, src = float(rope_base_override), "cli-override"
    else:
        rope_base, src = _read_rope_theta(qwen_cfg)
    print(f"[cfg] rope base = {rope_base:g} (from {src})")
    if rope_base_override is None and rope_base == 10000.0:
        print("[warn] rope_base resolved to 10000, but Qwen2.5 expects 1e6. If the gate "
              "FAILS, re-run with --rope-base 1000000.")
    return DSALTConfig(
        vocab_size  = qwen_cfg.vocab_size,
        d_model     = qwen_cfg.hidden_size,
        n_layers    = qwen_cfg.num_hidden_layers,
        n_heads     = n_heads,
        n_min       = n_min,
        n_max       = n_max,
        k_lmk       = k_lmk,
        max_seq_len = train_len,
        d_ff        = qwen_cfg.intermediate_size,
        dropout     = 0.0,
        # Qwen2.5 3B and larger do not tie embeddings.
        tie_weights = bool(getattr(qwen_cfg, "tie_word_embeddings", False)),
        n_kv_heads  = n_kv_heads,
        qkv_bias    = True,                       # Qwen2.5 has q/k/v biases
        rope_base   = rope_base,
        loss_fn     = "chunked",                  # liger on GPU
    )


# Weight copy into DSALT
@torch.no_grad()
def copy_qwen_weights(dsalt: DSALTLMHeadModel, qwen) -> list[str]:
    """Copy Qwen2.5 weights into the DSALT model in place.

    Returns the DSALT parameter names not filled from the backbone (expected: only
    ``win_gate`` and ``alpha_w``). Raises on any shape mismatch.
    """
    qsd = qwen.state_dict()
    filled: set[str] = set()

    def copy(dst: torch.nn.Parameter, src_key: str, dst_name: str):
        src = qsd[src_key]
        if dst.shape != src.shape:
            raise ValueError(f"shape mismatch {dst_name} {tuple(dst.shape)} <- {src_key} {tuple(src.shape)}")
        dst.copy_(src.to(dst.dtype))
        filled.add(dst_name)

    # embeddings, final norm, lm head
    copy(dsalt.embed_tokens.weight, "model.embed_tokens.weight", "embed_tokens.weight")
    copy(dsalt.final_norm.weight,   "model.norm.weight",         "final_norm.weight")
    if not (dsalt.lm_head.weight is dsalt.embed_tokens.weight):
        copy(dsalt.lm_head.weight, "lm_head.weight", "lm_head.weight")
    else:
        filled.add("lm_head.weight")

    # per-layer blocks
    for i, layer in enumerate(dsalt.layers):
        p = f"model.layers.{i}."
        a = layer.attn
        # attention projections (q/k/v have a bias in Qwen2.5, out_proj does not)
        copy(a.q_proj.weight, p + "self_attn.q_proj.weight", f"layers.{i}.attn.q_proj.weight")
        copy(a.k_proj.weight, p + "self_attn.k_proj.weight", f"layers.{i}.attn.k_proj.weight")
        copy(a.v_proj.weight, p + "self_attn.v_proj.weight", f"layers.{i}.attn.v_proj.weight")
        copy(a.q_proj.bias,   p + "self_attn.q_proj.bias",   f"layers.{i}.attn.q_proj.bias")
        copy(a.k_proj.bias,   p + "self_attn.k_proj.bias",   f"layers.{i}.attn.k_proj.bias")
        copy(a.v_proj.bias,   p + "self_attn.v_proj.bias",   f"layers.{i}.attn.v_proj.bias")
        copy(a.out_proj.weight, p + "self_attn.o_proj.weight", f"layers.{i}.attn.out_proj.weight")
        # SwiGLU FFN
        copy(layer.ffn.gate_proj.weight, p + "mlp.gate_proj.weight", f"layers.{i}.ffn.gate_proj.weight")
        copy(layer.ffn.up_proj.weight,   p + "mlp.up_proj.weight",   f"layers.{i}.ffn.up_proj.weight")
        copy(layer.ffn.down_proj.weight, p + "mlp.down_proj.weight", f"layers.{i}.ffn.down_proj.weight")
        # RMSNorms: input_layernorm before attention, post_attention_layernorm before
        # the FFN
        copy(layer.attn_norm.weight, p + "input_layernorm.weight",          f"layers.{i}.attn_norm.weight")
        copy(layer.ffn_norm.weight,  p + "post_attention_layernorm.weight", f"layers.{i}.ffn_norm.weight")

    missing = [n for n, _ in dsalt.named_parameters() if n not in filled]
    return missing


# Weight copy into the dense baseline
@torch.no_grad()
def copy_qwen_weights_dense(dense, qwen) -> list[str]:
    """Copy Qwen2.5 weights into ``models.model_dense.DenseLM`` (also used for SWA and
    Sparse-T, which share its parameter names).

    Same mapping as :func:`copy_qwen_weights` without the DSALT-only predictors.
    Returns the unfilled parameter names, which should be none.
    """
    qsd = qwen.state_dict()
    filled: set[str] = set()

    def copy(dst, src_key, dst_name):
        src = qsd[src_key]
        if dst.shape != src.shape:
            raise ValueError(f"shape mismatch {dst_name} {tuple(dst.shape)} <- {src_key} {tuple(src.shape)}")
        dst.copy_(src.to(dst.dtype))
        filled.add(dst_name)

    copy(dense.embed_tokens.weight, "model.embed_tokens.weight", "embed_tokens.weight")
    copy(dense.final_norm.weight,   "model.norm.weight",         "final_norm.weight")
    if not (dense.lm_head.weight is dense.embed_tokens.weight):
        copy(dense.lm_head.weight, "lm_head.weight", "lm_head.weight")
    else:
        filled.add("lm_head.weight")

    for i, layer in enumerate(dense.layers):
        p = f"model.layers.{i}."
        a = layer.attn
        copy(a.q_proj.weight, p + "self_attn.q_proj.weight", f"layers.{i}.attn.q_proj.weight")
        copy(a.k_proj.weight, p + "self_attn.k_proj.weight", f"layers.{i}.attn.k_proj.weight")
        copy(a.v_proj.weight, p + "self_attn.v_proj.weight", f"layers.{i}.attn.v_proj.weight")
        copy(a.q_proj.bias,   p + "self_attn.q_proj.bias",   f"layers.{i}.attn.q_proj.bias")
        copy(a.k_proj.bias,   p + "self_attn.k_proj.bias",   f"layers.{i}.attn.k_proj.bias")
        copy(a.v_proj.bias,   p + "self_attn.v_proj.bias",   f"layers.{i}.attn.v_proj.bias")
        copy(a.out_proj.weight, p + "self_attn.o_proj.weight", f"layers.{i}.attn.out_proj.weight")
        copy(layer.ffn.gate_proj.weight, p + "mlp.gate_proj.weight", f"layers.{i}.ffn.gate_proj.weight")
        copy(layer.ffn.up_proj.weight,   p + "mlp.up_proj.weight",   f"layers.{i}.ffn.up_proj.weight")
        copy(layer.ffn.down_proj.weight, p + "mlp.down_proj.weight", f"layers.{i}.ffn.down_proj.weight")
        copy(layer.attn_norm.weight, p + "input_layernorm.weight",          f"layers.{i}.attn_norm.weight")
        copy(layer.ffn_norm.weight,  p + "post_attention_layernorm.weight", f"layers.{i}.ffn_norm.weight")

    return [n for n, _ in dense.named_parameters() if n not in filled]


# Near-dense initialization of the DSALT-only predictors
@torch.no_grad()
def seed_near_dense(dsalt: DSALTLMHeadModel, bias_init: float = 8.0):
    """Initialize ``win_gate`` and ``alpha_w`` so the first forward is close to dense
    causal attention.

    The window is ``w(i) = n_min + σ(f(x_i))·(n_max − n_min)``. A zero ``win_gate``
    weight and a large positive bias give ``σ(f) ≈ 1`` for every token, i.e. the
    largest window. ``alpha_w`` keeps its default σ⁻¹(0.6).
    """
    for layer in dsalt.layers:
        wg = layer.attn.win_gate
        wg.weight.zero_()
        wg.bias.fill_(bias_init)


# Build
def build_dsalt_from_qwen(
    qwen_name: str,
    train_len: int,
    n_min:     int,
    n_max:     int,
    k_lmk:     int,
    dtype:     torch.dtype = torch.bfloat16,
    device:    str = "cuda",
    rope_base_override: float | None = None,
):
    """Load Qwen2.5, build the matching DSALT model, copy the weights and apply
    :func:`seed_near_dense`.

    Returns ``(dsalt_model, tokenizer, dsalt_config)``. ``n_max`` should be close to
    ``train_len`` so that the initial window covers the context.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if n_max < train_len // 2:
        print(f"[warn] n_max={n_max} is small vs train_len={train_len}: the near-dense "
              f"window is capped at n_max, so the gate ppl will be worse than dense. "
              f"Use n_max ≈ train_len for the gate, then keep the trained (contracted) window.")

    tok  = AutoTokenizer.from_pretrained(qwen_name)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_name, torch_dtype=dtype)

    cfg   = dsalt_config_from_qwen(qwen.config, n_min, n_max, k_lmk, train_len, rope_base_override)
    model = DSALTLMHeadModel.from_config(cfg).to(dtype)

    missing = copy_qwen_weights(model, qwen)
    # Only the two DSALT-only predictors per layer may be missing.
    unexpected = [m for m in missing if not (m.endswith("win_gate.weight")
                                             or m.endswith("win_gate.bias")
                                             or m.endswith("alpha_w"))]
    if unexpected:
        raise RuntimeError(f"weights not loaded from backbone (mapping bug): {unexpected}")
    print(f"[ok] copied backbone; DSALT-only params seeded near-dense: {len(missing)} tensors")

    seed_near_dense(model)
    del qwen
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model.to(device), tok, cfg


# Gate text
def load_gate_text(path: str | None, min_chars: int = 200_000) -> str:
    """Return a long natural-language document for the perplexity gate.

    Order of preference: ``path`` if given; WikiText-103 validation; PG19 validation.
    Documents are concatenated until at least ``min_chars`` characters. Raises if no
    source is reachable.
    """
    if path:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        if len(txt) < min_chars:
            print(f"[warn] gate file has only {len(txt)} chars (<{min_chars}); ppl gate may be noisy.")
        return txt

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "no --gate-text given and `datasets` is not installed to fetch one. "
            "Either `pip install datasets` or pass --gate-text path/to/long.txt"
        ) from e

    # Dataset names in `namespace/name` form, without loading scripts.
    candidates = (
        ("Salesforce/wikitext", dict(name="wikitext-103-raw-v1", split="validation"), "text"),
        ("Salesforce/wikitext", dict(name="wikitext-2-raw-v1",   split="validation"), "text"),
        ("emozilla/pg19",       dict(split="validation"),                             "text"),
    )
    for name, kwargs, field in candidates:
        try:
            ds = load_dataset(name, **kwargs)
            buf, n = [], 0
            for row in ds:
                t = row.get(field) if isinstance(row, dict) else row[field]
                if not t:
                    continue
                buf.append(t)
                n += len(t)
                if n >= min_chars:
                    break
            text = "\n".join(buf)
            if len(text) < 2000:
                print(f"[gate] {name} yielded too little text ({len(text)} chars), trying next")
                continue
            print(f"[gate] using {name} ({len(text)} chars)")
            return text
        except Exception as e:
            print(f"[gate] could not load {name}: {e}")
    raise RuntimeError("could not fetch any gate dataset; pass --gate-text instead.")


# Gate
@torch.no_grad()
def gate_perplexity(model, tok, text: str, seq_len: int, device: str = "cuda") -> float:
    """Perplexity of one window of ``seq_len`` tokens of ``text``.

    After :func:`seed_near_dense` it should be close to the perplexity of the original
    Qwen on the same window; a much larger value points to a mapping error (RoPE base,
    norms, GQA shapes).
    """
    model.eval()
    # The gate needs only the logits, so the dense _last_P reconstruction is skipped.
    for layer in model.layers:
        layer.attn._collect_last_P = False
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze()[: seq_len + 1].to(device)
    if ids.numel() < 2:
        raise ValueError("gate text too short for the requested seq_len")
    inp = ids[:-1]
    tgt = ids[1:]
    cu  = torch.tensor([0, inp.numel()], dtype=torch.int32, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out    = model(inp, cu_seqlens=cu, max_seqlen=inp.numel())
        logits = out["logits"]
        nll    = torch.nn.functional.cross_entropy(logits.float(), tgt.long(), reduction="mean")
    ppl = math.exp(nll.item())
    print(f"[gate] DSALT near-dense ppl @ {seq_len} tokens: {ppl:.2f}")
    return ppl


@torch.no_grad()
def dense_reference_ppl(qwen_name: str, tok, text: str, seq_len: int,
                        dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> float:
    """Perplexity of the original dense Qwen on the same text and window, used as the
    gate reference. Loads the backbone a second time.
    """
    from transformers import AutoModelForCausalLM
    qwen = AutoModelForCausalLM.from_pretrained(qwen_name, torch_dtype=dtype).to(device).eval()
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze()[: seq_len + 1].to(device)
    inp, tgt = ids[:-1].unsqueeze(0), ids[1:]
    with torch.amp.autocast("cuda", dtype=dtype):
        logits = qwen(inp).logits.squeeze(0)
        nll    = torch.nn.functional.cross_entropy(logits.float(), tgt.long(), reduction="mean")
    ppl = math.exp(nll.item())
    print(f"[gate] dense Qwen reference ppl @ {seq_len} tokens: {ppl:.2f}")
    del qwen
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl


def main():
    ap = argparse.ArgumentParser(description="Graft Qwen2.5 into DSALT (near-dense seed) and gate it.")
    ap.add_argument("--qwen",      default="Qwen/Qwen2.5-3B")
    ap.add_argument("--train-len", type=int, default=4096)
    ap.add_argument("--n-min",     type=int, default=64)
    ap.add_argument("--n-max",     type=int, default=4096, help="≈ train_len for the gate")
    ap.add_argument("--k-lmk",     type=int, default=32)
    ap.add_argument("--gate-text", default=None,
                    help="optional local long .txt; if omitted, WikiText-103/PG19 is downloaded")
    ap.add_argument("--gate-len",  type=int, default=2048)
    ap.add_argument("--gate-tol",  type=float, default=2.0,
                    help="max ppl gap (DSALT - dense) to PASS the load-correctness gate")
    ap.add_argument("--no-dense-ref", action="store_true",
                    help="skip re-loading dense Qwen for the reference ppl (compare manually)")
    ap.add_argument("--rope-base", type=float, default=None,
                    help="override RoPE base/theta (Qwen2.5 = 1000000); use if auto-read fails")
    args = ap.parse_args()

    model, tok, cfg = build_dsalt_from_qwen(
        args.qwen, args.train_len, args.n_min, args.n_max, args.k_lmk,
        rope_base_override=args.rope_base,
    )
    print(f"[ok] DSALT model built: {model.num_parameters(False)/1e9:.2f}B params, "
          f"{cfg.n_layers}L, {cfg.n_heads}H / {cfg.n_kv_heads}KV, rope_base={cfg.rope_base:g}")

    text = load_gate_text(args.gate_text)
    dsalt_ppl = gate_perplexity(model, tok, text, args.gate_len)

    if args.no_dense_ref:
        print("[gate] dense reference skipped (--no-dense-ref).")
        return
    dense_ppl = dense_reference_ppl(args.qwen, tok, text, args.gate_len)
    gap = dsalt_ppl - dense_ppl
    print(f"\n[gate] DSALT {dsalt_ppl:.2f}  vs  dense {dense_ppl:.2f}  ->  gap {gap:+.2f}")
    if gap <= args.gate_tol:
        print(f"[PASS] gap within tolerance ({args.gate_tol}): weight mapping is correct, "
              f"ready for the short fine-tune (Phase 1).")
    else:
        print(f"[FAIL] gap exceeds tolerance ({args.gate_tol}): likely a mapping bug "
              f"(RoPE base/convention, a swapped norm, or GQA shapes). Do NOT start "
              f"fine-tuning until this gate passes.")


if __name__ == "__main__":
    main()
