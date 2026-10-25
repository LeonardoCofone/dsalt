"""Hybrid-energy landmark score, the single implementation of the DSALT formula.

The attention module and the Triton kernels both call
:func:`hybrid_scores_per_head`, so the formula cannot diverge between paths.

For token ``j``, layer ``l``, head ``h``::

    s = alpha * z(||x_j W_V||_2) + (1 - alpha) * z(||x_j||_2)

where ``z(·)`` standardizes over the candidate tokens, ``alpha = sigmoid(alpha_raw)``
is learned per head and layer, ``||x_j W_V||_2`` is computed per head and
``||x_j||_2`` is shared across heads.
"""

import torch
import torch.nn as nn


def hybrid_scores_per_head(
    x:       torch.Tensor,
    W_V:     torch.Tensor,
    alpha:   torch.Tensor,
    n_heads: int,
    dh:      int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head hybrid score and the standardized signals.

    Args:
        x:       Hidden states ``[T, d]``.
        W_V:     Value projection ``[n_kv_heads*dh, d]`` (``v_proj.weight``); ``[d, d]``
                 under MHA.
        alpha:   Per-head balance ``[n_heads]`` in ``(0, 1)``.
        n_heads: Number of query heads, the width of the returned score.
        dh:      Per-head dimension.

    Returns:
        ``(scores, z_x, z_v)``: ``scores`` ``[T, n_heads]``, the standardized hidden-state
        norm ``z_x`` ``[T]`` and the standardized per-head value norm ``z_v``
        ``[T, n_heads]``, returned so the alpha gradient can be formed without
        recomputing the norms.

    Under GQA the value norm is computed for the ``n_kv_heads`` groups and repeated to
    the query heads of each group; ``alpha`` stays per query head.
    """
    T = x.shape[0]

    x_norm = x.norm(dim=-1).float()
    z_x    = (x_norm - x_norm.mean()) / x_norm.std().clamp(min=1e-6)

    # value head rows in W_V (n_heads for MHA, fewer under GQA)
    n_kv = W_V.shape[0] // dh
    xwv   = (x @ W_V.T).view(T, n_kv, dh).norm(dim=-1).float()                  # [T, n_kv]
    mu_v  = xwv.mean(0, keepdim=True)
    std_v = xwv.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv - mu_v) / std_v                                                # [T, n_kv]
    if n_kv != n_heads:                                                         # GQA: share per kv group
        n_rep = n_heads // n_kv
        z_v   = z_v.repeat_interleave(n_rep, dim=1)                             # [T, n_heads]

    scores = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(1)
    return scores, z_x, z_v


def compute_hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Hybrid score with the historical signature ``(x, W_V, alpha)``.

    A scalar ``alpha`` computes the value norm over the whole vector and returns
    ``[T]``; a vector ``[n_heads]`` delegates to :func:`hybrid_scores_per_head` and
    returns ``[T, n_heads]``.
    """
    if alpha.ndim == 0:
        x_norm      = x.norm(dim=-1)
        xwv         = (x @ W_V.T).norm(dim=-1)
        mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
        mu_v, std_v = xwv.mean(),    xwv.std().clamp(min=1e-6)
        return alpha * (xwv - mu_v) / std_v + (1.0 - alpha) * (x_norm - mu_x) / std_x

    n_heads = alpha.shape[0]
    dh      = W_V.shape[0] // n_heads
    scores, _, _ = hybrid_scores_per_head(x, W_V, alpha, n_heads, dh)
    return scores


def select_landmarks(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k landmarks from the scores, optionally excluding the local window."""
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))

    n_valid  = int((scores != float("-inf")).sum())
    k_actual = min(k, n_valid)

    if k_actual == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    _, indices = torch.topk(scores, k=k_actual, dim=-1, sorted=False)
    return indices


def soft_landmark_weights(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
    temperature:  float = 1.0,
) -> torch.Tensor:
    """Temperature-scaled softmax weights over the candidates."""
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))
    weights = torch.softmax(scores / temperature, dim=-1)
    return weights


class HybridEnergyLandmarkSelector(nn.Module):
    """Landmark selector with a learnable ``alpha`` ``[n_layers, n_heads]``, initialized
    at ``sigmoid^{-1}(0.6)`` as in :class:`~dsalt.modules.dsalt_attention.DSALTAttention`.
    """

    def __init__(self, n_layers: int, n_heads: int):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((n_layers, n_heads), torch.logit(torch.tensor(0.6)).item())
        )

    def forward(
        self,
        x:           torch.Tensor,
        W_V:         torch.Tensor,
        layer_idx:   int,
        head_idx:    int,
        k:           int,
        window_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha    = torch.sigmoid(self.alpha[layer_idx, head_idx])
        scores   = compute_hybrid_scores(x, W_V, alpha)
        in_win   = window_mask.any(dim=0)

        indices  = select_landmarks(scores, k=k, exclude_mask=in_win)
        weights  = soft_landmark_weights(scores, k=k, exclude_mask=in_win)

        return indices, weights
