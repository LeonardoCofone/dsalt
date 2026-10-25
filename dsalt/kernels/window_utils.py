import torch


def compute_window_sizes(
    x_prev: torch.Tensor,
    n_min:  int,
    n_max:  int,
) -> torch.Tensor:
    """Constant window size ``(n_min+n_max)//2`` for every token, aligned with ``x_prev``.

    The adaptive window of the model is predicted by ``win_gate`` in
    :class:`DSALTAttention`; this helper is a non-adaptive utility.
    """
    T = x_prev.shape[0]
    w = (n_min + n_max) // 2
    return torch.full((T,), float(w), device=x_prev.device, dtype=torch.float32)


def build_local_window_mask(
    seq_len:      int,
    window_sizes: torch.Tensor,
    device:       torch.device,
) -> torch.Tensor:
    """Boolean causal local-window mask ``[T, T]`` for one sequence.

    Row ``i`` attends to keys in ``[i - w(i) + 1, i]``, built with a difference array
    and a cumsum. ``True`` means attend.
    """
    rows = torch.arange(seq_len, device=device)
    w    = window_sizes.clamp(min=1, max=seq_len).long()
    lo   = (rows - w + 1).clamp(min=0)

    ones  = torch.ones(seq_len, 1, dtype=torch.int8, device=device)
    delta = torch.zeros(seq_len, seq_len + 1, dtype=torch.int8, device=device)
    delta.scatter_add_(1, lo.unsqueeze(1), ones)
    delta.scatter_add_(1, (rows + 1).unsqueeze(1), -ones)
    return delta[:, :seq_len].cumsum(dim=1).bool()


def build_local_window_mask_packed(
    cu_seqlens:   torch.Tensor,
    window_sizes: torch.Tensor,
    total_len:    int,
    device:       torch.device,
) -> torch.Tensor:
    """Boolean causal local-window mask ``[T, T]`` for packed sequences: a key is
    attended only if it is inside the query's window and in the same sequence.
    """
    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)

    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.arange(total_len, device=device) - starts[seq_ids]

    w      = window_sizes.clamp(min=1).long()
    lo_rel = (seq_off - w + 1).clamp(min=0)
    hi_rel = seq_off

    abs_lo = (starts[seq_ids] + lo_rel).clamp(min=0)
    abs_hi = (starts[seq_ids] + hi_rel).clamp(max=total_len - 1)

    col_idx = torch.arange(total_len, device=device)
    lo2d = abs_lo.unsqueeze(1)   # [T, 1]
    hi2d = abs_hi.unsqueeze(1)   # [T, 1]
    c    = col_idx.unsqueeze(0)  # [1, T]

    same_seq = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)  # [T, T]
    in_win   = (c >= lo2d) & (c <= hi2d)                     # [T, T]
    return same_seq & in_win


@torch.compile(dynamic=False)
def apply_rotary_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding (RoPE) to ``q`` and ``k``.

    Uses the half-split rotation convention with the precomputed ``cos``/``sin``
    from :func:`build_rope_cache`. Returns the rotated ``(q, k)``.
    """
    half  = q.shape[-1] // 2
    def _rot(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return q * cos + _rot(q) * sin, k * cos + _rot(k) * sin


def build_rope_cache(
    seq_len:  int,
    head_dim: int,
    device:   torch.device,
    dtype:    torch.dtype = torch.float32,
    base:     float       = 10000.0,
    scale:    float       = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the RoPE ``(cos, sin)`` cache of shape ``[seq_len, head_dim]``.

    ``scale`` is the RoPE/YaRN positional scaling (positions are divided by it),
    ``base`` the rotary base frequency. Consumed by :func:`apply_rotary_emb`.
    """
    theta     = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype) / scale
    emb       = torch.cat([torch.outer(positions, theta)] * 2, dim=-1)
    return emb.cos(), emb.sin()