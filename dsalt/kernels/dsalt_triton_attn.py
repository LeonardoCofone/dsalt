"""DSALT inference Triton kernel: sparse attention over the window and the landmarks.

Fused attention with online softmax over the local window and the precomputed
landmark indices, for packed sequences. The backward is in ``dsalt_triton_bwd``;
block sizes come from ``autotune``.
"""

import math
import torch
import triton
import triton.language as tl

from .dsalt_triton_bwd import dsalt_triton_backward
from .landmark_tokens_ker import hybrid_scores_per_head
from .autotune import autotune_blocks, get_tuned_config, _heuristic_config
# The selectors live in a Triton-free module so the CPU fallback can import them;
# re-exported here.
from .selectors import (
    _seq_meta,
    _build_seq_block_map,
    _score_block,
    _compute_landmark_indices,
    _build_landmark_kv,
)


@triton.jit
def _dsalt_fwd_kernel(
    Q, K, V, Out, LSE,
    W_sizes, Lmk_K, Lmk_V, Lmk_pos, Lmk_bias, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lset, stride_lseh,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
    stride_lbh, stride_lbb, stride_lbk,
    scale:    tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_LMK:    tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_h  = tl.program_id(1)

    seq_id    = tl.load(Seq_block_map + pid_bm * 2).to(tl.int32)
    block_off = tl.load(Seq_block_map + pid_bm * 2 + 1).to(tl.int32)
    seq_start = tl.load(Cu_seqlens + seq_id).to(tl.int32)
    seq_end   = tl.load(Cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len   = seq_end - seq_start
    m_start   = block_off * BLOCK_M

    offs_m  = tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, HEAD_DIM)
    offs_n  = tl.arange(0, BLOCK_N)
    valid_m = (m_start + offs_m) < seq_len

    q_ptrs = (
        Q
        + (seq_start + m_start + offs_m[:, None]) * stride_qt
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    w_sizes     = tl.load(W_sizes + seq_start + m_start + offs_m, mask=valid_m, other=1).to(tl.int32)
    w_sizes     = tl.maximum(w_sizes, 1)
    w_max_block = tl.max(w_sizes, axis=0)
    i_abs       = m_start + offs_m

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    window_start = tl.maximum(0, m_start - w_max_block + 1)
    window_end   = m_start + BLOCK_M
    n_start      = window_start - (window_start % BLOCK_N)

    for _ in range(0, (window_end - n_start + BLOCK_N - 1) // BLOCK_N):
        n_end_blk    = n_start + BLOCK_N
        blk_in_range = (n_end_blk > window_start) & (n_start < window_end) & (n_start < seq_len)
        valid_n      = ((offs_n + n_start) < seq_len) & blk_in_range

        k_blk = tl.load(
            K + (seq_start + n_start + offs_n[None, :]) * stride_kt
              + pid_h * stride_kh
              + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        )
        v_blk = tl.load(
            V + (seq_start + n_start + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh
              + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        )

        qk    = tl.dot(q, k_blk) * scale
        j_abs = n_start + offs_n
        in_win = (
            (j_abs[None, :] >= i_abs[:, None] - w_sizes[:, None] + 1) &
            (j_abs[None, :] <= i_abs[:, None])
        )
        final  = in_win & valid_n[None, :] & valid_m[:, None]
        qk     = tl.where(final, qk, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        p      = tl.where(final, tl.exp(qk - m_new[:, None]), 0.0)
        # When m_i is still -inf (no valid token yet for this row), avoid
        # -inf-(-inf)=nan.
        l_corr = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        l_i    = l_i * l_corr + tl.sum(p, axis=1)
        acc    = acc * l_corr[:, None] + tl.dot(p.to(k_blk.dtype), tl.trans(v_blk))
        m_i     = m_new
        n_start += BLOCK_N

    offs_lk = tl.arange(0, K_LMK)
    lk_k = tl.load(
        Lmk_K + pid_h * stride_lkh + seq_id * stride_lkb
              + offs_lk[None, :] * stride_lks + offs_d[:, None] * stride_lkd
    )
    lk_v = tl.load(
        Lmk_V + pid_h * stride_lvh + seq_id * stride_lvb
              + offs_lk[None, :] * stride_lvs + offs_d[:, None] * stride_lvd
    )
    lmk_pos = tl.load(
        Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk
    ).to(tl.int32)

    lmk_valid  = (lmk_pos >= 0) & (lmk_pos < seq_len)
    win_lo     = i_abs[:, None] - w_sizes[:, None] + 1
    causal_lmk = (lmk_pos[None, :] <= i_abs[:, None]) & (lmk_pos[None, :] < win_lo)
    valid_lmk  = causal_lmk & valid_m[:, None] & lmk_valid[None, :]

    lmk_bias = tl.load(
        Lmk_bias + pid_h * stride_lbh + seq_id * stride_lbb + offs_lk * stride_lbk
    ).to(tl.float32)
    qk_lmk = tl.dot(q, lk_k) * scale + lmk_bias[None, :]
    qk_lmk = tl.where(valid_lmk, qk_lmk, float("-inf"))

    m_new   = tl.maximum(m_i, tl.max(qk_lmk, axis=1))
    p_lmk   = tl.where(valid_lmk, tl.exp(qk_lmk - m_new[:, None]), 0.0)
    l_corr  = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
    l_i     = l_i * l_corr + tl.sum(p_lmk, axis=1)
    acc     = acc * l_corr[:, None] + tl.dot(p_lmk.to(lk_v.dtype), tl.trans(lk_v))
    m_i = m_new

    out_val = tl.where(
        l_i[:, None] > 1e-9,
        acc / l_i[:, None],
        tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
    )
    tl.store(
        Out
        + (seq_start + m_start + offs_m[:, None]) * stride_ot
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od,
        out_val.to(Out.dtype.element_ty),
        mask=valid_m[:, None],
    )

    lse_val = tl.where(l_i > 1e-9, m_i + tl.log(l_i), float("-inf"))
    tl.store(
        LSE + (seq_start + m_start + offs_m) * stride_lset + pid_h * stride_lseh,
        lse_val,
        mask=valid_m,
    )


def _resolve_blocks(head_dim: int, device: torch.device) -> dict:
    """Config (BLOCK_M/BLOCK_N/num_warps/num_stages) for ``(head_dim, GPU)``.

    Uses the values produced by the autotune if already available, otherwise the
    dynamic heuristics (which also serve as the tuning seed/fallback).
    """
    tuned = get_tuned_config(head_dim, device)
    if tuned is not None:
        return tuned
    return _heuristic_config(head_dim, device)


def _launch_fwd(q_c, k_c, v_c, out, lse, w_int, lmk_K, lmk_V, lmk_pos,
                lmk_bias_c, cu_int, seq_block_map, total_blk, n_heads,
                scale, HEAD_DIM_C, k_lmk, cfg):
    """Launch the forward kernel with a given block config; used by both the normal run
    and the autotune benchmark.
    """
    _dsalt_fwd_kernel[(total_blk, n_heads)](
        q_c, k_c, v_c, out, lse,
        w_int, lmk_K, lmk_V, lmk_pos, lmk_bias_c, cu_int, seq_block_map,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
        lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
        lmk_pos.stride(0), lmk_pos.stride(1), lmk_pos.stride(2),
        lmk_bias_c.stride(0), lmk_bias_c.stride(1), lmk_bias_c.stride(2),
        scale=scale,
        HEAD_DIM=HEAD_DIM_C,
        K_LMK=k_lmk,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def _maybe_autotune(head_dim, device, q_c, k_c, v_c, lmk_indices, lmk_bias,
                    w_sizes, cu_seqlens, n_heads, scale, HEAD_DIM_C, k_lmk):
    """Run the fwd+bwd autotune once per (head_dim, GPU); the result is cached in
    ``autotune._TUNED_CONFIG``.
    """
    if get_tuned_config(head_dim, device) is not None:
        return

    cu_int      = cu_seqlens.to(torch.int32).contiguous()
    lmk_pos     = lmk_indices.to(torch.int32).contiguous()
    lmk_bias_c  = lmk_bias.detach().to(torch.float32).contiguous()
    w_int       = w_sizes.clamp(min=1).long().contiguous()
    total_len   = q_c.shape[0]

    with torch.no_grad():
        lmk_K, lmk_V = _build_landmark_kv(
            k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
        )

    def make_runner(cfg):
        # seq_block_map depends on BLOCK_M → rebuilt for every candidate.
        seq_block_map, total_blk = _build_seq_block_map(cu_seqlens, cfg["BLOCK_M"], device)
        out = torch.zeros_like(q_c)
        lse = torch.empty((total_len, n_heads), device=device, dtype=torch.float32)

        def run():
            _launch_fwd(
                q_c, k_c, v_c, out, lse, w_int, lmk_K, lmk_V, lmk_pos,
                lmk_bias_c, cu_int, seq_block_map, total_blk, n_heads,
                scale, HEAD_DIM_C, k_lmk, cfg,
            )
            grad_out = torch.ones_like(out)
            dsalt_triton_backward(
                grad_out, q_c, k_c, v_c, out, lse,
                lmk_K, lmk_V, lmk_pos, lmk_bias_c,
                w_sizes, cu_seqlens, scale,
                seq_block_map, total_blk,
                cfg["BLOCK_M"], cfg["BLOCK_N"], HEAD_DIM_C,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
            )
        return run

    autotune_blocks(head_dim, device, make_runner, verbose=True)


class DSALTAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens):
        total_len, n_heads, head_dim = q.shape
        device     = q.device
        scale      = 1.0 / math.sqrt(head_dim)
        HEAD_DIM_C = triton.next_power_of_2(head_dim)
        k_lmk      = lmk_indices.shape[-1]

        in_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q_c   = q.contiguous().to(in_dtype)
        k_c   = k.contiguous().to(in_dtype)
        v_c   = v.contiguous().to(in_dtype)

        # Autotune only once per (head_dim, GPU), on the first real batch.
        _maybe_autotune(
            head_dim, device, q_c, k_c, v_c, lmk_indices, lmk_bias,
            w_sizes, cu_seqlens, n_heads, scale, HEAD_DIM_C, k_lmk,
        )
        cfg     = _resolve_blocks(head_dim, device)
        BLOCK_M = cfg["BLOCK_M"]
        BLOCK_N = cfg["BLOCK_N"]

        out   = torch.zeros_like(q_c)
        lse   = torch.empty((total_len, n_heads), device=device, dtype=torch.float32)
        w_int = w_sizes.clamp(min=1).long().contiguous()

        with torch.no_grad():
            lmk_K, lmk_V = _build_landmark_kv(
                k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
            )
            seq_block_map, total_blk = _build_seq_block_map(cu_seqlens, BLOCK_M, device)

        cu_int      = cu_seqlens.to(torch.int32).contiguous()
        lmk_pos     = lmk_indices.to(torch.int32).contiguous()
        lmk_bias_c  = lmk_bias.detach().to(torch.float32).contiguous()

        _launch_fwd(
            q_c, k_c, v_c, out, lse, w_int, lmk_K, lmk_V, lmk_pos,
            lmk_bias_c, cu_int, seq_block_map, total_blk, n_heads,
            scale, HEAD_DIM_C, k_lmk, cfg,
        )

        ctx.save_for_backward(q_c, k_c, v_c, out, lse, lmk_K, lmk_V, lmk_pos, lmk_bias_c)
        ctx.w_sizes       = w_sizes
        ctx.cu_seqlens    = cu_seqlens
        ctx.seq_block_map = seq_block_map
        ctx.total_blk     = total_blk
        ctx.scale         = scale
        ctx.BLOCK_M       = BLOCK_M
        ctx.BLOCK_N       = BLOCK_N
        ctx.HEAD_DIM_C    = HEAD_DIM_C
        ctx.out_dtype     = q.dtype

        return out.float()

    @staticmethod
    def backward(ctx, grad_out):
        q_c, k_c, v_c, out, lse, lmk_K, lmk_V, lmk_pos, lmk_bias_c = ctx.saved_tensors

        dq, dk, dv, d_bias = dsalt_triton_backward(
            grad_out, q_c, k_c, v_c, out, lse,
            lmk_K, lmk_V, lmk_pos, lmk_bias_c,
            ctx.w_sizes, ctx.cu_seqlens, ctx.scale,
            ctx.seq_block_map, ctx.total_blk,
            ctx.BLOCK_M, ctx.BLOCK_N, ctx.HEAD_DIM_C,
        )

        return (
            dq.to(ctx.out_dtype), dk.to(ctx.out_dtype), dv.to(ctx.out_dtype),
            None, d_bias.to(ctx.out_dtype), None, None,
        )

# The custom autograd Function is opaque to torch.compile. Identity on torch builds
# without `torch._dynamo`.
def _dynamo_opaque(fn):
    disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(fn) if disable is not None else fn


@_dynamo_opaque
def dsalt_triton_attention(
    q:           torch.Tensor,
    k:           torch.Tensor,
    v:           torch.Tensor,
    lmk_indices: torch.Tensor,
    lmk_bias:    torch.Tensor,
    w_sizes:     torch.Tensor,
    cu_seqlens:  torch.Tensor,
) -> torch.Tensor:
    return DSALTAttentionFunction.apply(
        q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens,
    )