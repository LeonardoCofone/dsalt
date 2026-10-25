"""DSALT training Triton kernel: sparse attention with differentiable selectors.

The inference kernel (``dsalt_triton_attn``) computes ``A(i) = W(i) ∪ L(i)`` with
hard selectors. Training also needs gradients for the two predictors (``win_gate``
and ``alpha``) without materializing the dense ``[N,H,L,L]`` attention.

Two differentiable terms inside the sparse kernel:

* soft window edge (trains win_gate): inside the band the logit gets
  ``log σ((w̃(i) − d)/τ_win)`` with ``d = i − j``, exactly 0 in the core
  (``d ≤ w̃ − B``). The kernel returns ``d_w̃``, using
  ``∂ logσ(z)/∂w̃ = (1−σ(z))·(1/τ)``.
* soft landmark weight (trains alpha): each landmark logit gets
  ``log σ(s_j(α)/τ_lmk)``, passed in as ``Lmk_logw`` and computed in PyTorch. The
  kernel returns ``d_logw``, which autograd carries back to α.

The kernel returns ``dq, dk, dv, d_w̃, d_logw``.
"""

import math
import torch
import triton
import triton.language as tl

from .dsalt_triton_attn import (
    _build_seq_block_map,
    _build_landmark_kv,
    _seq_meta,
)
from .landmark_tokens_ker import hybrid_scores_per_head
from .autotune import get_tuned_config, _heuristic_config

# Fallback backward key tile, used only when the resolved config has no tuned
# ``BLOCK_N_BWD``. The autotune picks it per device (32/64/128, filtered by shared
# memory); 32 is the safe value at small head_dim.
_BWD_MAX_BLOCK_N = 32

# On sm_75 every tl.dot dimension must be at least 16. With k_lmk < 16 the two landmark
# dots violate this, so the landmark axis is padded with invalid landmarks (pos=-1,
# logw=0) that the kernel masks out.
_MIN_DOT = 16


def _pad_landmarks(lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, k_lmk):
    """Pad the landmark axis to at least _MIN_DOT so tl.dot is legal on sm_75.

    Returns (lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, k_lmk_pad). Padded slots get pos=-1
    (masked out) and logw=0, kv=0.
    """
    if k_lmk >= _MIN_DOT:
        return lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, k_lmk
    pad = _MIN_DOT - k_lmk
    n_heads, num_seqs = lmk_pos_i.shape[0], lmk_pos_i.shape[1]
    hd = lmk_K.shape[-1]
    dev = lmk_K.device
    lmk_K = torch.cat(
        [lmk_K, torch.zeros(n_heads, num_seqs, pad, hd, device=dev, dtype=lmk_K.dtype)], dim=2
    ).contiguous()
    lmk_V = torch.cat(
        [lmk_V, torch.zeros(n_heads, num_seqs, pad, hd, device=dev, dtype=lmk_V.dtype)], dim=2
    ).contiguous()
    lmk_pos_i = torch.cat(
        [lmk_pos_i, torch.full((n_heads, num_seqs, pad), -1, device=dev, dtype=lmk_pos_i.dtype)], dim=2
    ).contiguous()
    lmk_logw_c = torch.cat(
        [lmk_logw_c, torch.zeros(n_heads, num_seqs, pad, device=dev, dtype=lmk_logw_c.dtype)], dim=2
    ).contiguous()
    return lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, _MIN_DOT


# --------------------------------------------------------------------------- #
#  forward
# --------------------------------------------------------------------------- #
@triton.jit
def _train_fwd_kernel(
    Q, K, V, Out, LSE,
    W_cont, Lmk_K, Lmk_V, Lmk_pos, Lmk_logw, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lset, stride_lseh,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
    stride_lwh, stride_lwb, stride_lwk,
    scale:    tl.constexpr,
    tau_win:  tl.constexpr,
    win_edge: tl.constexpr,
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
        Q + (seq_start + m_start + offs_m[:, None]) * stride_qt
          + pid_h * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    # continuous per-query window w̃(i)
    w_cont = tl.load(W_cont + seq_start + m_start + offs_m, mask=valid_m, other=1.0).to(tl.float32)
    w_max_block = tl.max(tl.maximum(tl.ceil(w_cont), 1.0), axis=0).to(tl.int32)
    i_abs = m_start + offs_m

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
              + pid_h * stride_kh + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        )
        v_blk = tl.load(
            V + (seq_start + n_start + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        )

        qk    = tl.dot(q, k_blk) * scale
        j_abs = n_start + offs_n
        d     = (i_abs[:, None] - j_abs[None, :]).to(tl.float32)              # i - j
        # band: 0 ≤ d, and d ≤ w̃ (soft tail handled by the bias below)
        causal = (j_abs[None, :] <= i_abs[:, None]) & (d <= w_cont[:, None])
        final  = causal & valid_n[None, :] & valid_m[:, None]
        # soft window edge: 0 in the hard core, log σ((w̃-d)/τ) on the boundary
        z       = (w_cont[:, None] - d) / tau_win
        logedge = -tl.log(1.0 + tl.exp(-z))                                   # log σ(z)
        in_core = d <= (w_cont[:, None] - win_edge)
        bias    = tl.where(in_core, 0.0, logedge)
        qk      = tl.where(final, qk + bias, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        p      = tl.where(final, tl.exp(qk - m_new[:, None]), 0.0)
        # When m_i is still -inf (nothing accumulated yet for this row), m_i - m_new is
        # nan; l_corr = 0 is the correct online-softmax value there.
        l_corr = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        l_i    = l_i * l_corr + tl.sum(p, axis=1)
        acc    = acc * l_corr[:, None] + tl.dot(p.to(k_blk.dtype), tl.trans(v_blk))
        m_i     = m_new
        n_start += BLOCK_N

    # landmarks: hard selection, soft α-weight via Lmk_logw
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
    win_lo     = i_abs[:, None].to(tl.float32) - w_cont[:, None]              # j < i - w̃ ⇒ outside band
    causal_lmk = (lmk_pos[None, :] <= i_abs[:, None]) & (lmk_pos[None, :].to(tl.float32) < win_lo)
    valid_lmk  = causal_lmk & valid_m[:, None] & lmk_valid[None, :]

    lmk_logw = tl.load(
        Lmk_logw + pid_h * stride_lwh + seq_id * stride_lwb + offs_lk * stride_lwk
    ).to(tl.float32)
    qk_lmk = tl.dot(q, lk_k) * scale + lmk_logw[None, :]
    qk_lmk = tl.where(valid_lmk, qk_lmk, float("-inf"))

    m_new   = tl.maximum(m_i, tl.max(qk_lmk, axis=1))
    p_lmk   = tl.where(valid_lmk, tl.exp(qk_lmk - m_new[:, None]), 0.0)
    l_corr  = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
    l_i     = l_i * l_corr + tl.sum(p_lmk, axis=1)
    acc     = acc * l_corr[:, None] + tl.dot(p_lmk.to(lk_v.dtype), tl.trans(lk_v))
    m_i = m_new

    out_val = tl.where(
        l_i[:, None] > 1e-9, acc / l_i[:, None],
        tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
    )
    tl.store(
        Out + (seq_start + m_start + offs_m[:, None]) * stride_ot
            + pid_h * stride_oh + offs_d[None, :] * stride_od,
        out_val.to(Out.dtype.element_ty),
        mask=valid_m[:, None],
    )
    lse_val = tl.where(l_i > 1e-9, m_i + tl.log(l_i), float("-inf"))
    tl.store(
        LSE + (seq_start + m_start + offs_m) * stride_lset + pid_h * stride_lseh,
        lse_val, mask=valid_m,
    )


# --------------------------------------------------------------------------- #
#  backward
# --------------------------------------------------------------------------- #
@triton.jit
def _train_bwd_preprocess(
    Out, DO, Delta,
    stride_ot, stride_oh, stride_od,
    stride_dot, stride_doh, stride_dod,
    stride_dt, stride_dh,
    total_len: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    valid  = offs_m < total_len
    o = tl.load(Out + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :] * stride_od,
                mask=valid[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_dot + pid_h * stride_doh + offs_d[None, :] * stride_dod,
                 mask=valid[:, None], other=0.0).to(tl.float32)
    tl.store(Delta + offs_m * stride_dt + pid_h * stride_dh, tl.sum(o * do, axis=1), mask=valid)


@triton.jit
def _train_bwd_kernel(
    Q, K, V, DO, DQ, DK, DV, D_w,
    LSE, W_cont, Lmk_K, Lmk_V, Lmk_pos, Lmk_logw, D_logw,
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
    stride_lwh, stride_lwb, stride_lwk,
    stride_dlwh, stride_dlwb, stride_dlwk,
    stride_dwt, stride_dwh,
    stride_dt, stride_dh,
    scale:    tl.constexpr,
    tau_win:  tl.constexpr,
    win_edge: tl.constexpr,
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
    q  = tl.load(Q + q_base * stride_qt + pid_h * stride_qh + offs_d[None, :] * stride_qd,
                 mask=valid_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + q_base * stride_dot + pid_h * stride_doh + offs_d[None, :] * stride_dod,
                 mask=valid_m[:, None], other=0.0).to(tl.float32)
    delta = tl.load(Delta + (seq_start + m_start + offs_m) * stride_dt + pid_h * stride_dh,
                    mask=valid_m, other=0.0).to(tl.float32)
    lse = tl.load(LSE + (seq_start + m_start + offs_m) * stride_lset + pid_h * stride_lseh,
                  mask=valid_m, other=float("-inf")).to(tl.float32)

    w_cont = tl.load(W_cont + seq_start + m_start + offs_m, mask=valid_m, other=1.0).to(tl.float32)
    w_max_block = tl.max(tl.maximum(tl.ceil(w_cont), 1.0), axis=0).to(tl.int32)
    row_ok = valid_m & (lse > float("-inf"))

    dq   = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dw_i = tl.zeros([BLOCK_M], dtype=tl.float32)                              # ∂loss/∂w̃(i)

    # ---- landmarks ----
    lk = tl.load(Lmk_K + pid_h * stride_lkh + seq_id * stride_lkb
                       + offs_lk[None, :] * stride_lks + offs_d[:, None] * stride_lkd).to(tl.float32)
    lv = tl.load(Lmk_V + pid_h * stride_lvh + seq_id * stride_lvb
                       + offs_lk[None, :] * stride_lvs + offs_d[:, None] * stride_lvd).to(tl.float32)
    lmk_pos = tl.load(Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk).to(tl.int32)

    lmk_valid  = (lmk_pos >= 0) & (lmk_pos < seq_len)
    win_lo     = i_abs[:, None].to(tl.float32) - w_cont[:, None]
    causal_lmk = (lmk_pos[None, :] <= i_abs[:, None]) & (lmk_pos[None, :].to(tl.float32) < win_lo)
    valid_lmk  = causal_lmk & row_ok[:, None] & lmk_valid[None, :]

    lmk_logw = tl.load(Lmk_logw + pid_h * stride_lwh + seq_id * stride_lwb + offs_lk * stride_lwk).to(tl.float32)
    qk_lmk = tl.dot(q, lk) * scale + lmk_logw[None, :]
    p_lmk  = tl.where(valid_lmk, tl.exp(qk_lmk - lse[:, None]), 0.0)

    dp_lmk = tl.dot(do, lv)
    recip  = tl.where(valid_lmk, p_lmk * (dp_lmk - delta[:, None]), 0.0)
    ds_lmk = recip * scale
    dq += tl.dot(ds_lmk, tl.trans(lk))
    dlk = tl.dot(tl.trans(ds_lmk), q)
    dlv = tl.dot(tl.trans(p_lmk), do)

    # gradient to the landmark log-weight (= classic d_bias): ∂loss/∂logw_k = Σ_i recip_ik
    dlogw_s   = tl.sum(recip, axis=0)
    tl.atomic_add(D_logw + pid_h * stride_dlwh + seq_id * stride_dlwb + offs_lk * stride_dlwk, dlogw_s)

    lmk_abs_safe = seq_start + tl.maximum(lmk_pos, 0)
    tl.atomic_add(DK + lmk_abs_safe[:, None] * stride_dkt + pid_h * stride_dkh + offs_d[None, :] * stride_dkd,
                  dlk, mask=lmk_valid[:, None])
    tl.atomic_add(DV + lmk_abs_safe[:, None] * stride_dvt + pid_h * stride_dvh + offs_d[None, :] * stride_dvd,
                  dlv, mask=lmk_valid[:, None])

    # ---- window band ----
    window_start = tl.maximum(0, m_start - w_max_block + 1)
    window_end   = m_start + BLOCK_M
    n_start      = window_start - (window_start % BLOCK_N)
    n_iter       = (window_end - n_start + BLOCK_N - 1) // BLOCK_N

    for _ in range(0, n_iter):
        valid_n = ((offs_n + n_start) < seq_len) & (n_start < window_end)
        j_abs   = n_start + offs_n
        d       = (i_abs[:, None] - j_abs[None, :]).to(tl.float32)
        causal  = (j_abs[None, :] <= i_abs[:, None]) & (d <= w_cont[:, None])
        final   = causal & valid_n[None, :] & row_ok[:, None]

        k_blk = tl.load(K + (seq_start + n_start + offs_n[None, :]) * stride_kt
                          + pid_h * stride_kh + offs_d[:, None] * stride_kd,
                        mask=valid_n[None, :], other=0.0).to(tl.float32)
        v_blk = tl.load(V + (seq_start + n_start + offs_n[None, :]) * stride_vt
                          + pid_h * stride_vh + offs_d[:, None] * stride_vd,
                        mask=valid_n[None, :], other=0.0).to(tl.float32)

        z       = (w_cont[:, None] - d) / tau_win
        sig     = tl.sigmoid(z)
        logedge = -tl.log(1.0 + tl.exp(-z))
        in_core = d <= (w_cont[:, None] - win_edge)
        bias    = tl.where(in_core, 0.0, logedge)

        qk = tl.dot(q, k_blk) * scale + bias
        p  = tl.where(final, tl.exp(qk - lse[:, None]), 0.0)

        dp = tl.dot(do, v_blk)
        ds_raw = p * (dp - delta[:, None])                                    # ∂loss/∂logit (pre-scale)
        ds = tl.where(final, ds_raw * scale, 0.0)

        dq += tl.dot(ds, tl.trans(k_blk))

        # ∂loss/∂w̃ from the soft edge: ∂bias/∂w̃ = (1-σ(z))·(1/τ), only on the boundary
        dbias_dw = tl.where(in_core, 0.0, (1.0 - sig) / tau_win)
        dw_contrib = tl.where(final, ds_raw * dbias_dw, 0.0)
        dw_i += tl.sum(dw_contrib, axis=1)
        # The band dk/dv are computed key-parallel in _train_bwd_dkdv_kernel without
        # atomics; this kernel computes dq, dw and the landmark dk/dv.
        n_start += BLOCK_N

    tl.store(DQ + (seq_start + m_start + offs_m[:, None]) * stride_dqt
                + pid_h * stride_dqh + offs_d[None, :] * stride_dqd,
             dq.to(DQ.dtype.element_ty), mask=valid_m[:, None])
    # accumulate ∂loss/∂w̃(i) across heads (one D_w per token)
    tl.atomic_add(D_w + (seq_start + m_start + offs_m) * stride_dwt + pid_h * stride_dwh,
                  dw_i, mask=valid_m)


@triton.jit
def _train_bwd_dkdv_kernel(
    Q, K, V, DO, DK, DV,
    LSE, W_cont, Delta, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_dot, stride_doh, stride_dod,
    stride_dkt, stride_dkh, stride_dkd,
    stride_dvt, stride_dvh, stride_dvd,
    stride_lset, stride_lseh,
    stride_dt, stride_dh,
    scale:    tl.constexpr,
    tau_win:  tl.constexpr,
    win_edge: tl.constexpr,
    N_MAX:    tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Key-parallel dk/dv for the window band, without atomics.

    Each program owns one key block ``[kn, kn+BLOCK_N)`` of one head and loops over the
    query blocks that can attend to it (query i attends key j iff ``j ≤ i ≤ j + w̃(i)``,
    so i ≤ j + N_MAX). dk/dv are accumulated in registers and stored once.

    Must run before the landmark dk/dv atomics in _train_bwd_kernel, which add on top
    of the band contribution stored here.
    """
    pid_kn = tl.program_id(0)
    pid_h  = tl.program_id(1)

    seq_id    = tl.load(Seq_block_map + pid_kn * 2).to(tl.int32)
    block_off = tl.load(Seq_block_map + pid_kn * 2 + 1).to(tl.int32)
    seq_start = tl.load(Cu_seqlens + seq_id).to(tl.int32)
    seq_end   = tl.load(Cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len   = seq_end - seq_start
    n_start   = block_off * BLOCK_N

    offs_n  = tl.arange(0, BLOCK_N)
    offs_d  = tl.arange(0, HEAD_DIM)
    offs_m  = tl.arange(0, BLOCK_M)
    j_abs   = n_start + offs_n
    valid_n = j_abs < seq_len

    k_blk = tl.load(K + (seq_start + j_abs[:, None]) * stride_kt + pid_h * stride_kh
                      + offs_d[None, :] * stride_kd, mask=valid_n[:, None], other=0.0).to(tl.float32)
    v_blk = tl.load(V + (seq_start + j_abs[:, None]) * stride_vt + pid_h * stride_vh
                      + offs_d[None, :] * stride_vd, mask=valid_n[:, None], other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Queries that can see this key block satisfy j ≤ i ≤ j + w̃(i) ≤ j + N_MAX, so only
    # [n_start, n_start + BLOCK_N + N_MAX) is scanned.
    m_lo  = n_start
    m_hi  = tl.minimum(n_start + BLOCK_N + N_MAX, seq_len)
    m_start = m_lo - (m_lo % BLOCK_M)
    m_iter = (m_hi - m_start + BLOCK_M - 1) // BLOCK_M

    for _ in range(0, m_iter):
        i_abs   = m_start + offs_m
        valid_m = (i_abs < seq_len) & (i_abs >= n_start)
        q  = tl.load(Q + (seq_start + i_abs[:, None]) * stride_qt + pid_h * stride_qh
                       + offs_d[None, :] * stride_qd, mask=valid_m[:, None], other=0.0).to(tl.float32)
        do = tl.load(DO + (seq_start + i_abs[:, None]) * stride_dot + pid_h * stride_doh
                        + offs_d[None, :] * stride_dod, mask=valid_m[:, None], other=0.0).to(tl.float32)
        lse = tl.load(LSE + (seq_start + i_abs) * stride_lset + pid_h * stride_lseh,
                      mask=valid_m, other=float("-inf")).to(tl.float32)
        delta = tl.load(Delta + (seq_start + i_abs) * stride_dt + pid_h * stride_dh,
                        mask=valid_m, other=0.0).to(tl.float32)
        w_cont = tl.load(W_cont + seq_start + i_abs, mask=valid_m, other=1.0).to(tl.float32)
        row_ok = valid_m & (lse > float("-inf"))

        # logits for this (query block i, key block j): [BLOCK_M, BLOCK_N]
        d      = (i_abs[:, None] - j_abs[None, :]).to(tl.float32)
        causal = (j_abs[None, :] <= i_abs[:, None]) & (d <= w_cont[:, None])
        final  = causal & valid_n[None, :] & row_ok[:, None]
        z       = (w_cont[:, None] - d) / tau_win
        logedge = -tl.log(1.0 + tl.exp(-z))
        in_core = d <= (w_cont[:, None] - win_edge)
        bias    = tl.where(in_core, 0.0, logedge)

        qk = tl.dot(q, tl.trans(k_blk)) * scale + bias
        p  = tl.where(final, tl.exp(qk - lse[:, None]), 0.0)                   # [BM,BN]
        dp = tl.dot(do, tl.trans(v_blk))                                       # [BM,BN]
        ds = tl.where(final, p * (dp - delta[:, None]) * scale, 0.0)          # [BM,BN]

        dk += tl.dot(tl.trans(ds), q)                                          # [BN,HEAD_DIM]
        dv += tl.dot(tl.trans(p), do)                                          # [BN,HEAD_DIM]
        m_start += BLOCK_M

    tl.store(DK + (seq_start + j_abs[:, None]) * stride_dkt + pid_h * stride_dkh
                + offs_d[None, :] * stride_dkd, dk.to(DK.dtype.element_ty), mask=valid_n[:, None])
    tl.store(DV + (seq_start + j_abs[:, None]) * stride_dvt + pid_h * stride_dvh
                + offs_d[None, :] * stride_dvd, dv.to(DV.dtype.element_ty), mask=valid_n[:, None])


# --------------------------------------------------------------------------- #
#  python wrappers
# --------------------------------------------------------------------------- #
# Tuned training-kernel configs, keyed by (head_dim, sm_major, sm_minor), separate from
# the inference autotune because the training fwd+bwd has a different register and
# shared-memory profile.
_TRAIN_TUNED: dict = {}


def _resolve_cfg(head_dim: int, device: torch.device) -> dict:
    major, minor = torch.cuda.get_device_capability(device)
    key = (int(head_dim), int(major), int(minor))
    if key in _TRAIN_TUNED:
        return _TRAIN_TUNED[key]
    # fall back to the inference-tuned config, then the heuristic
    tuned = get_tuned_config(head_dim, device)
    return tuned if tuned is not None else _heuristic_config(head_dim, device)


def _train_candidates(head_dim: int, device: torch.device) -> list[dict]:
    """Candidate configs for the training kernel: the inference sweep
    (`_candidate_configs`) with ``with_bwd_tile=True``, so the backward key tile
    ``BLOCK_N_BWD`` is also tuned per device (32/64/128, filtered by shared memory).
    """
    from .autotune import _candidate_configs
    return _candidate_configs(head_dim, device, with_bwd_tile=True)


def _maybe_autotune_train(head_dim, device, q, k, v, lmk_pos, lmk_logw, w_cont,
                          cu_seqlens, tau_win, win_edge, n_max, cu_list):
    """Tune the training kernel once per (head_dim, GPU), measuring a full fwd+bwd."""
    major, minor = torch.cuda.get_device_capability(device)
    key = (int(head_dim), int(major), int(minor))
    if key in _TRAIN_TUNED:
        return

    from .autotune import _bench_step, _is_main_process, _print_table

    total_len, n_heads, hd = q.shape
    in_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
    q_c = q.detach().contiguous().to(in_dtype)
    k_c = k.detach().contiguous().to(in_dtype)
    v_c = v.detach().contiguous().to(in_dtype)
    w_d = w_cont.detach()
    lp  = lmk_pos.detach()
    lw  = lmk_logw.detach()

    # Fresh leaves on each call, with requires_grad on q/k/v, so .backward() has a graph
    # to run.
    def run():
        with torch.enable_grad():
            qg = q_c.clone().requires_grad_(True)
            kg = k_c.clone().requires_grad_(True)
            vg = v_c.clone().requires_grad_(True)
            wd = w_d.clone().requires_grad_(True)
            o = DSALTTrainFunction.apply(
                qg, kg, vg, lp, lw, wd, cu_seqlens, tau_win, win_edge, n_max, cu_list,
            )
            o.sum().backward()

    results = []
    for cfg in _train_candidates(head_dim, device):
        _TRAIN_TUNED[key] = cfg  # so the inner .apply uses this candidate
        try:
            ms = _bench_step(run)
            results.append((cfg, ms, None))
        except Exception as e:
            # keep a short slice of the error message
            msg = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:40]}" if str(e).strip() else type(e).__name__
            results.append((cfg, None, msg))
        finally:
            del _TRAIN_TUNED[key]

    valid = [(c, t) for (c, t, e) in results if t is not None]
    best_cfg = min(valid, key=lambda x: x[1])[0] if valid else _heuristic_config(head_dim, device)
    best_ms  = min((t for _, t, _ in results if t is not None), default=None)

    # Phase 2: with the forward axes fixed, search the backward key tile BLOCK_N_BWD per
    # device; 32 is measured again in the same state for a fair comparison.
    if valid and "BLOCK_N_BWD" in best_cfg:
        from .autotune import _bwd_tile_candidates
        for cand in _bwd_tile_candidates(best_cfg, head_dim, device):
            _TRAIN_TUNED[key] = cand
            try:
                ms = _bench_step(run)
                results.append((cand, ms, None))
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:40]}" if str(e).strip() else type(e).__name__
                results.append((cand, None, msg))
            finally:
                del _TRAIN_TUNED[key]
        valid = [(c, t) for (c, t, e) in results if t is not None]
        best_cfg = min(valid, key=lambda x: x[1])[0]
        best_ms  = min((t for _, t, _ in results if t is not None), default=None)

    _TRAIN_TUNED[key] = best_cfg
    if _is_main_process(device):
        print("  [DSALT train-kernel autotune]")
        _print_table(head_dim, device, results, best_cfg, best_ms)


class DSALTTrainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens,
                tau_win, win_edge, n_max, cu_list=None):
        total_len, n_heads, head_dim = q.shape
        device     = q.device
        scale      = 1.0 / math.sqrt(head_dim)
        HEAD_DIM_C = triton.next_power_of_2(head_dim)
        k_lmk      = lmk_pos.shape[-1]

        # Autotune the training kernel once per (head_dim, GPU) on the first batch.
        _maybe_autotune_train(
            head_dim, device, q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens,
            float(tau_win), float(win_edge), int(n_max), cu_list,
        )

        in_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q_c = q.contiguous().to(in_dtype)
        k_c = k.contiguous().to(in_dtype)
        v_c = v.contiguous().to(in_dtype)

        cfg     = _resolve_cfg(head_dim, device)
        BLOCK_M = cfg["BLOCK_M"]
        BLOCK_N = cfg["BLOCK_N"]
        # Backward key tile from the autotune, or the fallback for configs without it.
        BLOCK_N_BWD = cfg.get("BLOCK_N_BWD", min(BLOCK_N, _BWD_MAX_BLOCK_N))

        out = torch.zeros_like(q_c)
        lse = torch.empty((total_len, n_heads), device=device, dtype=torch.float32)
        w_c = w_cont.contiguous().to(torch.float32)

        with torch.no_grad():
            lmk_K, lmk_V = _build_landmark_kv(k_c, v_c, lmk_pos, cu_seqlens, k_lmk, n_heads, head_dim)
            seq_block_map, total_blk = _build_seq_block_map(cu_seqlens, BLOCK_M, device, cu_list)

        cu_int    = cu_seqlens.to(torch.int32).contiguous()
        lmk_pos_i = lmk_pos.to(torch.int32).contiguous()
        lmk_logw_c = lmk_logw.detach().to(torch.float32).contiguous()

        # Pad the landmark axis to at least 16 (see _MIN_DOT).
        lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, k_lmk_pad = _pad_landmarks(
            lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, k_lmk
        )

        _train_fwd_kernel[(total_blk, n_heads)](
            q_c, k_c, v_c, out, lse,
            w_c, lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, cu_int, seq_block_map,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
            lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
            lmk_pos_i.stride(0), lmk_pos_i.stride(1), lmk_pos_i.stride(2),
            lmk_logw_c.stride(0), lmk_logw_c.stride(1), lmk_logw_c.stride(2),
            scale=scale, tau_win=float(tau_win), win_edge=float(win_edge),
            HEAD_DIM=HEAD_DIM_C, K_LMK=k_lmk_pad,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )

        ctx.save_for_backward(q_c, k_c, v_c, out, lse, lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, w_c)
        ctx.k_lmk_pad     = k_lmk_pad
        ctx.n_max         = int(n_max)
        ctx.cu_list       = cu_list
        ctx.cu_seqlens    = cu_seqlens
        ctx.seq_block_map = seq_block_map
        ctx.total_blk     = total_blk
        ctx.scale         = scale
        ctx.BLOCK_M       = BLOCK_M
        ctx.BLOCK_N       = BLOCK_N
        ctx.BLOCK_N_BWD   = BLOCK_N_BWD
        ctx.HEAD_DIM_C    = HEAD_DIM_C
        ctx.k_lmk         = k_lmk
        ctx.tau_win       = float(tau_win)
        ctx.win_edge      = float(win_edge)
        ctx.out_dtype     = q.dtype
        return out.float()

    @staticmethod
    def backward(ctx, grad_out):
        q_c, k_c, v_c, out, lse, lmk_K, lmk_V, lmk_pos_i, lmk_logw_c, w_c = ctx.saved_tensors
        total_len, n_heads, head_dim = q_c.shape
        device   = q_c.device
        num_seqs = ctx.cu_seqlens.shape[0] - 1

        in_dtype = q_c.dtype if q_c.dtype in (torch.float16, torch.bfloat16) else torch.float16
        do_f  = grad_out.contiguous().to(in_dtype)
        out_f = out.contiguous().to(torch.float32)

        dq = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)
        dk = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)
        dv = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)
        d_w   = torch.zeros(total_len, n_heads, device=device, dtype=torch.float32)
        d_logw = torch.zeros(n_heads, num_seqs, ctx.k_lmk_pad, device=device, dtype=torch.float32)
        delta = torch.empty(total_len, n_heads, device=device, dtype=torch.float32)

        BLOCK_M = ctx.BLOCK_M
        # Backward key tile chosen by the autotune.
        BLOCK_N = ctx.BLOCK_N_BWD
        cu_int  = ctx.cu_seqlens.to(torch.int32).contiguous()

        from .autotune import get_tuned_config
        tuned = get_tuned_config(head_dim, device)
        num_warps  = tuned["num_warps"] if tuned is not None else (4 if head_dim <= 64 else 2)
        num_stages = tuned["num_stages"] if tuned is not None else 2

        grid_pre = (triton.cdiv(total_len, BLOCK_M), n_heads)
        _train_bwd_preprocess[grid_pre](
            out_f, do_f, delta,
            out_f.stride(0), out_f.stride(1), out_f.stride(2),
            do_f.stride(0),  do_f.stride(1),  do_f.stride(2),
            delta.stride(0), delta.stride(1),
            total_len=total_len, HEAD_DIM=ctx.HEAD_DIM_C, BLOCK_M=BLOCK_M, num_warps=4,
        )

        # Band dk/dv, key-parallel and without atomics. Must run before the main
        # backward kernel, which atomic_adds the landmark dk/dv on top. The block_map
        # must match BLOCK_N_BWD.
        kn_block_map, total_kn = _build_seq_block_map(ctx.cu_seqlens, BLOCK_N, device, ctx.cu_list)
        _train_bwd_dkdv_kernel[(total_kn, n_heads)](
            q_c, k_c, v_c, do_f, dk, dv,
            lse, w_c, delta, cu_int, kn_block_map,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            do_f.stride(0), do_f.stride(1), do_f.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            lse.stride(0), lse.stride(1),
            delta.stride(0), delta.stride(1),
            scale=ctx.scale, tau_win=ctx.tau_win, win_edge=ctx.win_edge,
            N_MAX=ctx.n_max,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=ctx.HEAD_DIM_C,
            num_warps=num_warps, num_stages=num_stages,
        )

        _train_bwd_kernel[(ctx.total_blk, n_heads)](
            q_c, k_c, v_c, do_f, dq, dk, dv, d_w,
            lse, w_c, lmk_K.to(in_dtype), lmk_V.to(in_dtype), lmk_pos_i, lmk_logw_c, d_logw,
            delta, cu_int, ctx.seq_block_map,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            do_f.stride(0), do_f.stride(1), do_f.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            lse.stride(0), lse.stride(1),
            lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
            lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
            lmk_pos_i.stride(0), lmk_pos_i.stride(1), lmk_pos_i.stride(2),
            lmk_logw_c.stride(0), lmk_logw_c.stride(1), lmk_logw_c.stride(2),
            d_logw.stride(0), d_logw.stride(1), d_logw.stride(2),
            d_w.stride(0), d_w.stride(1),
            delta.stride(0), delta.stride(1),
            scale=ctx.scale, tau_win=ctx.tau_win, win_edge=ctx.win_edge,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=ctx.HEAD_DIM_C, K_LMK=ctx.k_lmk_pad,
            num_warps=num_warps, num_stages=num_stages,
        )

        # d_w is summed over heads inside the kernel: one gradient per token.
        d_w_tok = d_w.sum(dim=1)                                              # [total_len]
        # Trim the padded landmark slots back to k_lmk.
        d_logw = d_logw[:, :, :ctx.k_lmk]
        od = ctx.out_dtype
        return (
            dq.to(od), dk.to(od), dv.to(od),
            None,                 # lmk_pos (no grad)
            d_logw.to(od),        # to alpha via log σ(s/τ)
            d_w_tok.to(od),       # to win_gate via w̃
            None, None, None,     # cu_seqlens, tau_win, win_edge
            None,                 # n_max (constexpr, no grad)
            None,                 # cu_list (host helper, no grad)
        )


# The custom autograd Function and its Triton launches are opaque to torch.compile,
# which fuses the code around them. Identity on torch builds without `_dynamo`.
def _dynamo_opaque(fn):
    disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(fn) if disable is not None else fn


@_dynamo_opaque
def dsalt_triton_train_attention(q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens,
                                 tau_win=1.0, win_edge=4.0, n_max=256, cu_list=None):
    """Differentiable sparse DSALT attention for training.

    ``lmk_logw`` (per-landmark log σ(s/τ), differentiable in α) and ``w_cont``
    (continuous window, differentiable in win_gate) receive ``d_logw`` and ``d_w̃``
    through autograd. ``n_max`` bounds the key-parallel dk/dv scan in the backward.
    Opaque to torch.compile.
    """
    return DSALTTrainFunction.apply(
        q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens, tau_win, win_edge, n_max, cu_list,
    )
