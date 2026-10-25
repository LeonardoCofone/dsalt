import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q:         torch.Tensor,
    k:         torch.Tensor,
    v:         torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training:  bool  = False,
) -> torch.Tensor:
    """Masked SDPA for the batched ``[B, H, T, D]`` path.

    Turns the boolean ``attn_mask`` (True = attend) into an additive ``-inf`` mask
    and runs ``F.scaled_dot_product_attention``. The fallback used when Triton is
    unavailable.
    """
    additive = torch.zeros(attn_mask.shape, dtype=q.dtype, device=q.device)
    additive.masked_fill_(~attn_mask, float("-inf"))
    if additive.dim() == 3:
        additive = additive.unsqueeze(0)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )

def sparse_attention_forward_packed(
    q:         torch.Tensor,
    k:         torch.Tensor,
    v:         torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training:  bool  = False,
) -> torch.Tensor:
    """Masked SDPA for the packed ``[T, H, D]`` (single-batch) path.

    Lifts the packed tensors to ``[1, H, T, D]``, applies the additive ``-inf``
    mask, and returns the result back in ``[T, H, D]``. Fallback used when Triton
    is unavailable.
    """
    q_ = q.transpose(0, 1).unsqueeze(0)   # [1, H, T, D]
    k_ = k.transpose(0, 1).unsqueeze(0)
    v_ = v.transpose(0, 1).unsqueeze(0)

    # Build the additive mask in q's dtype to avoid a cast inside SDPA.
    additive = attn_mask.to(dtype=q.dtype)              # [T, T]
    additive = additive.masked_fill(~attn_mask, float("-inf"))
    additive = additive.unsqueeze(0).unsqueeze(0)       # [1, 1, T, T]

    out = F.scaled_dot_product_attention(
        q_, k_, v_,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()  # [T, H, D]