"""PyTorch selector helpers shared by the Triton kernels and the dense path: window
sizing, landmark top-k, packed metadata and landmark KV gather. No Triton code, so
the CPU fallback can import them.
"""

import torch

from .landmark_tokens_ker import hybrid_scores_per_head


_SEQ_BLOCK_MAP_CACHE: dict = {}
_SEQ_META_CACHE: dict = {}


def _seq_meta(cu_seqlens: torch.Tensor, total: int, device: torch.device, cu_list: list | None = None):
    # Cache key from the host-side cu_list when available (no device-to-host sync).
    if cu_list is not None:
        key = (int(cu_list[-1]), len(cu_list) - 1)
    else:
        key = (int(cu_seqlens[-1]), cu_seqlens.shape[0] - 1)
    if key in _SEQ_META_CACHE:
        return _SEQ_META_CACHE[key]
    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total, device=device) - starts[seq_ids]
    if cu_list is not None:
        max_len = max(cu_list[b + 1] - cu_list[b] for b in range(len(cu_list) - 1))
    else:
        max_len = int(lens.max())
    val = (num_seqs, lens, starts, seq_ids, seq_off, max_len)
    _SEQ_META_CACHE[key] = val
    return val


def _build_seq_block_map(
    cu_seqlens: torch.Tensor,
    block_m:    int,
    device:     torch.device,
    cu_list:    list | None = None,
) -> tuple[torch.Tensor, int]:
    """Map each ``BLOCK_M`` query block to its ``(sequence, block-in-sequence)`` pair.

    Returns ``(seq_block_map [total_blocks, 2], total_blocks)``, cached per
    ``(total_len, num_seqs, block_m)``. Consumed by both the forward and backward
    kernels so they share the packed block layout.
    """
    num_seqs  = cu_seqlens.shape[0] - 1
    # Cache key from the host-side cu_list when available.
    total_len = int(cu_list[-1]) if cu_list is not None else int(cu_seqlens[-1])
    key = (total_len, num_seqs, block_m)
    if key in _SEQ_BLOCK_MAP_CACHE:
        return _SEQ_BLOCK_MAP_CACHE[key]

    cu_cpu     = torch.tensor(cu_list, dtype=torch.long) if cu_list is not None else cu_seqlens.detach().to("cpu")
    lens       = cu_cpu[1:] - cu_cpu[:-1]
    blocks_per = (lens + block_m - 1) // block_m
    total_blks = int(blocks_per.sum())
    seq_col    = torch.repeat_interleave(torch.arange(lens.shape[0], dtype=torch.int32), blocks_per)
    blk_col    = (
        torch.arange(total_blks, dtype=torch.int32)
        - torch.repeat_interleave(blocks_per.cumsum(0) - blocks_per, blocks_per).int()
    )
    result = torch.stack([seq_col, blk_col], dim=1).contiguous().to(device)
    _SEQ_BLOCK_MAP_CACHE[key] = (result, total_blks)
    return result, total_blks


def _score_block(x: torch.Tensor, W_V: torch.Tensor, alpha: torch.Tensor, n_heads: int, dh: int):
    # The score formula is defined once, in kernels.landmark_tokens_ker.
    return hybrid_scores_per_head(x, W_V, alpha, n_heads, dh)


def _compute_landmark_indices(
    x:          torch.Tensor,
    W_V:        torch.Tensor,
    alpha:      torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    k_lmk:      int,
    n_min:      int,
    total_len:  int,
    cu_list:    list | None = None,
    head_dim:   int | None = None,
) -> torch.Tensor:
    """Hard top-k landmark indices per (head, sequence), excluding in-window tokens.

    Scores every token with the hybrid score, standardized over the whole sequence,
    and returns the top-``k_lmk`` indices ``[H, num_seqs, k_lmk]`` (``-1`` where fewer
    candidates exist) and the signals ``z_x`` / ``z_v``. The selection is detached.

    ``head_dim`` must be given under GQA, where ``W_V`` has only ``n_kv_heads`` head
    rows.
    """
    device   = x.device
    total    = x.shape[0]
    num_seqs, lens, starts, seq_ids, seq_off, max_len = _seq_meta(cu_seqlens, total, device, cu_list)

    n_heads = alpha.shape[0]
    dh      = head_dim if head_dim is not None else W_V.shape[0] // n_heads

    scores, z_x, z_v = _score_block(x, W_V, alpha, n_heads, dh)

    covered    = seq_off < w_sizes.long()
    covered_h  = covered.unsqueeze(1).expand(-1, n_heads)
    scores_fil = scores.masked_fill(covered_h, float("-inf"))

    # Uniform-length detection from the host-side cu_list when available.
    if cu_list is not None:
        seq_lens = [cu_list[b + 1] - cu_list[b] for b in range(len(cu_list) - 1)]
        uniform  = len(seq_lens) > 0 and all(s == seq_lens[0] for s in seq_lens)
        L        = seq_lens[0] if uniform else 0
    else:
        uniform = bool((lens == lens[0]).all())
        L = int(lens[0]) if uniform else 0
    if uniform:
        sp = scores_fil.T.view(n_heads, num_seqs, L)
        k_eff = min(k_lmk, L)
        top_val, top_lc = torch.topk(sp, k_eff, dim=2, sorted=False)
        out = torch.full((n_heads, num_seqs, k_lmk), -1, dtype=torch.long, device=device)
        valid = torch.isfinite(top_val)
        out[:, :, :k_eff] = torch.where(valid, top_lc, torch.full_like(top_lc, -1))
        return out, z_x, z_v

    score_pad = torch.full((n_heads, num_seqs, max_len), float("-inf"), device=device)
    score_pad[:, seq_ids, seq_off] = scores_fil.T
    k_eff           = min(k_lmk, max_len)
    top_val, top_lc = torch.topk(score_pad, k_eff, dim=2, sorted=False)
    out = torch.full((n_heads, num_seqs, k_lmk), -1, dtype=torch.long, device=device)
    valid = torch.isfinite(top_val)
    top_lc = torch.where(valid, top_lc, torch.full_like(top_lc, -1))
    out[:, :, :k_eff] = top_lc
    return out, z_x, z_v


def _build_landmark_kv(
    K:           torch.Tensor,
    V:           torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
    k_lmk:       int,
    n_heads:     int,
    head_dim:    int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the landmark keys/values at ``lmk_indices`` into ``[H, num_seqs, k, D]``.

    Invalid (``-1``) landmark slots are zero-filled, so the kernel can mask them out.
    """
    starts   = cu_seqlens[:-1].to(K.device)
    safe_idx = lmk_indices.clamp(min=0)
    abs_idx  = starts[None, :, None] + safe_idx
    h_idx    = torch.arange(n_heads, device=K.device)[:, None, None]
    lmk_K    = K[abs_idx, h_idx, :]
    lmk_V    = V[abs_idx, h_idx, :]
    invalid  = (lmk_indices < 0).unsqueeze(-1)
    lmk_K    = lmk_K.masked_fill(invalid, 0.0)
    lmk_V    = lmk_V.masked_fill(invalid, 0.0)
    return lmk_K.contiguous(), lmk_V.contiguous()
