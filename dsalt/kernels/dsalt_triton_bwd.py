"""Backward of the DSALT inference kernel.

Accumulates the key/value gradients with ``atomic_add``, which caps the backward key
tile (``_BWD_MAX_BLOCK_N``) below the forward's.
"""

import torch
import triton
import triton.language as tl


# Cap on BLOCK_N in the backward, which accumulates with atomic_add and cannot use tiles
# as wide as the forward.
_BWD_MAX_BLOCK_N = 32


@triton.jit
def _dsalt_bwd_preprocess(
    Out, DO,
    Delta,
    stride_ot, stride_oh, stride_od,
    stride_dot, stride_doh, stride_dod,
    stride_dt, stride_dh,
    total_len: tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    BLOCK_M:   tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    valid  = offs_m < total_len

    o = tl.load(
        Out + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :] * stride_od,
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_dot + pid_h * stride_doh + offs_d[None, :] * stride_dod,
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta + offs_m * stride_dt + pid_h * stride_dh,
        delta,
        mask=valid,
    )


@triton.jit
def _dsalt_bwd_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    LSE, W_sizes, Lmk_K, Lmk_V, Lmk_pos, Lmk_bias, D_bias,
    Delta, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_dot, stride_doh, stride_dod,
    stride_dqt, stride_dqh, stride_dqd,
    stride_dkt, stride_dkh, stride_dkd,
    stride_dvt, stride_dvh, stride_dvd,
    stride_lset, stride_lseh,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
    stride_lbh, stride_lbb, stride_lbk,
    stride_dbbh, stride_dbbb, stride_dbbk,
    stride_dt,  stride_dh,
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
    offs_lk = tl.arange(0, K_LMK)
    valid_m = (m_start + offs_m) < seq_len
    i_abs   = m_start + offs_m

    q_base = seq_start + m_start + offs_m[:, None]
    q = tl.load(
        Q + q_base * stride_qt + pid_h * stride_qh + offs_d[None, :] * stride_qd,
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + q_base * stride_dot + pid_h * stride_doh + offs_d[None, :] * stride_dod,
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    delta = tl.load(
        Delta + (seq_start + m_start + offs_m) * stride_dt + pid_h * stride_dh,
        mask=valid_m, other=0.0,
    ).to(tl.float32)
    lse = tl.load(
        LSE + (seq_start + m_start + offs_m) * stride_lset + pid_h * stride_lseh,
        mask=valid_m, other=float("-inf"),
    ).to(tl.float32)

    w_sizes     = tl.load(W_sizes + seq_start + m_start + offs_m, mask=valid_m, other=1).to(tl.int32)
    w_sizes     = tl.maximum(w_sizes, 1)
    w_max_block = tl.max(w_sizes, axis=0)

    row_ok = valid_m & (lse > float("-inf"))

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    lk = tl.load(
        Lmk_K + pid_h * stride_lkh + seq_id * stride_lkb
              + offs_lk[None, :] * stride_lks + offs_d[:, None] * stride_lkd,
    ).to(tl.float32)
    lv = tl.load(
        Lmk_V + pid_h * stride_lvh + seq_id * stride_lvb
              + offs_lk[None, :] * stride_lvs + offs_d[:, None] * stride_lvd,
    ).to(tl.float32)
    lmk_pos = tl.load(
        Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk,
    ).to(tl.int32)

    lmk_valid  = (lmk_pos >= 0) & (lmk_pos < seq_len)
    win_lo     = i_abs[:, None] - w_sizes[:, None] + 1
    causal_lmk = (lmk_pos[None, :] <= i_abs[:, None]) & (lmk_pos[None, :] < win_lo)
    valid_lmk  = causal_lmk & row_ok[:, None] & lmk_valid[None, :]

    lmk_bias = tl.load(
        Lmk_bias + pid_h * stride_lbh + seq_id * stride_lbb + offs_lk * stride_lbk
    ).to(tl.float32)
    qk_lmk = tl.dot(q, lk) * scale + lmk_bias[None, :]
    p_lmk  = tl.where(valid_lmk, tl.exp(qk_lmk - lse[:, None]), 0.0)

    dp_lmk = tl.dot(do, lv)
    recip  = tl.where(valid_lmk, p_lmk * (dp_lmk - delta[:, None]), 0.0)
    ds_lmk = recip * scale

    dq += tl.dot(ds_lmk, tl.trans(lk))

    dlk = tl.dot(tl.trans(ds_lmk), q)
    dlv = tl.dot(tl.trans(p_lmk), do)

    dbias_s   = tl.sum(recip, axis=0)
    dbias_ptr = D_bias + pid_h * stride_dbbh + seq_id * stride_dbbb + offs_lk * stride_dbbk
    tl.atomic_add(dbias_ptr, dbias_s)

    lmk_abs = seq_start + lmk_pos
    lmk_abs_safe = seq_start + tl.maximum(lmk_pos, 0)
    dlk_ptrs = DK + lmk_abs_safe[:, None] * stride_dkt + pid_h * stride_dkh + offs_d[None, :] * stride_dkd
    dlv_ptrs = DV + lmk_abs_safe[:, None] * stride_dvt + pid_h * stride_dvh + offs_d[None, :] * stride_dvd
    tl.atomic_add(dlk_ptrs, dlk, mask=lmk_valid[:, None])
    tl.atomic_add(dlv_ptrs, dlv, mask=lmk_valid[:, None])

    window_start = tl.maximum(0, m_start - w_max_block + 1)
    window_end   = m_start + BLOCK_M
    n_start_win  = window_start - (window_start % BLOCK_N)
    n_iter       = (window_end - n_start_win + BLOCK_N - 1) // BLOCK_N

    n_start = n_start_win
    for _ in range(0, n_iter):
        valid_n = ((offs_n + n_start) < seq_len) & (n_start < window_end)
        j_abs   = n_start + offs_n
        in_win  = (
            (j_abs[None, :] >= i_abs[:, None] - w_sizes[:, None] + 1) &
            (j_abs[None, :] <= i_abs[:, None])
        )
        final = in_win & valid_n[None, :] & row_ok[:, None]

        k_blk = tl.load(
            K + (seq_start + n_start + offs_n[None, :]) * stride_kt
              + pid_h * stride_kh + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)
        v_blk = tl.load(
            V + (seq_start + n_start + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, k_blk) * scale
        p  = tl.where(final, tl.exp(qk - lse[:, None]), 0.0)

        dp = tl.dot(do, v_blk)
        ds = tl.where(final, p * (dp - delta[:, None]) * scale, 0.0)

        dq += tl.dot(ds, tl.trans(k_blk))

        dk_blk = tl.dot(tl.trans(ds), q)
        dv_blk = tl.dot(tl.trans(p), do)

        n_tok   = n_start + offs_n
        store_n = (n_tok < seq_len) & (n_tok < window_end)
        abs_n   = seq_start + n_tok
        dk_ptrs = DK + abs_n[:, None] * stride_dkt + pid_h * stride_dkh + offs_d[None, :] * stride_dkd
        dv_ptrs = DV + abs_n[:, None] * stride_dvt + pid_h * stride_dvh + offs_d[None, :] * stride_dvd
        tl.atomic_add(dk_ptrs, dk_blk, mask=store_n[:, None])
        tl.atomic_add(dv_ptrs, dv_blk, mask=store_n[:, None])

        n_start += BLOCK_N

    tl.store(
        DQ + (seq_start + m_start + offs_m[:, None]) * stride_dqt
           + pid_h * stride_dqh + offs_d[None, :] * stride_dqd,
        dq.to(DQ.dtype.element_ty),
        mask=valid_m[:, None],
    )


def dsalt_triton_backward(
    grad_out:      torch.Tensor,
    q:             torch.Tensor,
    k:             torch.Tensor,
    v:             torch.Tensor,
    out:           torch.Tensor,
    lse:           torch.Tensor,
    lmk_K:         torch.Tensor,
    lmk_V:         torch.Tensor,
    lmk_pos:       torch.Tensor,
    lmk_bias:      torch.Tensor,
    w_sizes:       torch.Tensor,
    cu_seqlens:    torch.Tensor,
    scale:         float,
    seq_block_map: torch.Tensor,
    total_blk:     int,
    BLOCK_M:       int,
    BLOCK_N:       int,
    HEAD_DIM_C:    int,
    num_warps:     int | None = None,
    num_stages:    int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_len, n_heads, head_dim = q.shape
    device   = q.device
    num_seqs = cu_seqlens.shape[0] - 1
    k_lmk    = lmk_K.shape[2]
    d_bias = torch.zeros(n_heads, num_seqs, k_lmk, device=device, dtype=torch.float32)

    in_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
    q_f   = q.contiguous().to(in_dtype)
    k_f   = k.contiguous().to(in_dtype)
    v_f   = v.contiguous().to(in_dtype)
    out_f = out.contiguous().to(torch.float32)
    do_f  = grad_out.contiguous().to(in_dtype)

    dq = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)
    dk = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)
    dv = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)

    delta = torch.empty(total_len, n_heads, device=device, dtype=torch.float32)

    grid_pre = (triton.cdiv(total_len, BLOCK_M), n_heads)
    _dsalt_bwd_preprocess[grid_pre](
        out_f, do_f, delta,
        out_f.stride(0), out_f.stride(1), out_f.stride(2),
        do_f.stride(0),  do_f.stride(1),  do_f.stride(2),
        delta.stride(0), delta.stride(1),
        total_len=total_len,
        HEAD_DIM=HEAD_DIM_C,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    # num_warps/num_stages: explicit values first (autotune benchmark), then the config
    # tuned for this GPU, then the per-head_dim heuristic. BLOCK_M/BLOCK_N must match
    # the forward, since seq_block_map is shared.
    if num_warps is None or num_stages is None:
        from .autotune import get_tuned_config
        tuned = get_tuned_config(head_dim, device)
        if num_warps is None:
            num_warps = tuned["num_warps"] if tuned is not None else (4 if head_dim <= 64 else 2)
        if num_stages is None:
            num_stages = tuned["num_stages"] if tuned is not None else 2
    num_stages_bwd = num_stages
    cu_int    = cu_seqlens.to(torch.int32).contiguous()
    lse_c     = lse.contiguous().to(torch.float32)

    lmk_K_f   = lmk_K.to(in_dtype).contiguous()
    lmk_V_f   = lmk_V.to(in_dtype).contiguous()
    lmk_pos_i = lmk_pos.to(torch.int32).contiguous()
    lmk_bias_f = lmk_bias.to(torch.float32).contiguous()
    w_int     = w_sizes.clamp(min=1).to(torch.int32).contiguous()

    # The backward needs more registers and shared memory than the forward, so BLOCK_N
    # is capped at _BWD_MAX_BLOCK_N.
    BLOCK_N = min(BLOCK_N, _BWD_MAX_BLOCK_N)

    _dsalt_bwd_kernel[(total_blk, n_heads)](
        q_f, k_f, v_f, do_f,
        dq, dk, dv,
        lse_c, w_int, lmk_K_f, lmk_V_f, lmk_pos_i, lmk_bias_f, d_bias,
        delta, cu_int, seq_block_map,
        q_f.stride(0),  q_f.stride(1),  q_f.stride(2),
        k_f.stride(0),  k_f.stride(1),  k_f.stride(2),
        v_f.stride(0),  v_f.stride(1),  v_f.stride(2),
        do_f.stride(0), do_f.stride(1), do_f.stride(2),
        dq.stride(0),   dq.stride(1),   dq.stride(2),
        dk.stride(0),   dk.stride(1),   dk.stride(2),
        dv.stride(0),   dv.stride(1),   dv.stride(2),
        lse_c.stride(0), lse_c.stride(1),
        lmk_K_f.stride(0), lmk_K_f.stride(1), lmk_K_f.stride(2), lmk_K_f.stride(3),
        lmk_V_f.stride(0), lmk_V_f.stride(1), lmk_V_f.stride(2), lmk_V_f.stride(3),
        lmk_pos_i.stride(0), lmk_pos_i.stride(1), lmk_pos_i.stride(2),
        lmk_bias_f.stride(0), lmk_bias_f.stride(1), lmk_bias_f.stride(2),
        d_bias.stride(0), d_bias.stride(1), d_bias.stride(2),
        delta.stride(0), delta.stride(1),
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_C,
        K_LMK=lmk_K.shape[2],
        num_warps=num_warps,
        num_stages=num_stages_bwd,
    )

    return dq, dk, dv, d_bias