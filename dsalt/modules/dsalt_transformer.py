import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..kernels.RMSENorm  import RMSENorm
from .dsalt_attention    import DSALTAttention


class SwiGLUFFN(nn.Module):
    """Gated SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DSALTTransformerBlock(nn.Module):
    """One pre-norm Transformer block: RMSNorm → DSALTAttention → residual, then
    RMSNorm → SwiGLUFFN → residual. Optionally uses activation checkpointing.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        n_min:       int,
        n_max:       int,
        k_lmk:       int,
        max_seq_len: int,
        d_ff:        int,
        dropout:     float = 0.0,
        yarn_scale:  float = 1.0,
        layer_idx:   int   = 0,
        n_kv_heads:  int | None = None,
        qkv_bias:    bool   = False,
        rope_base:   float  = 10000.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSENorm(d_model)
        self.ffn_norm  = RMSENorm(d_model)
        self.attn = DSALTAttention(
            d_model=d_model, n_heads=n_heads, n_min=n_min, n_max=n_max,
            k_lmk=k_lmk, max_seq_len=max_seq_len, dropout=dropout,
            yarn_scale=yarn_scale, layer_idx=layer_idx,
            n_kv_heads=n_kv_heads, qkv_bias=qkv_bias, rope_base=rope_base,
        )
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x:                      torch.Tensor,
        cu_seqlens:             torch.Tensor | None = None,
        max_seqlen:             int | None          = None,
        gradient_checkpointing: bool                = False,
        rope_cs:                tuple | None         = None,
        cu_list:                list | None          = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gradient_checkpointing and self.training:
            def custom_attn(h):
                attn_out, layer_aux = self.attn(h, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, rope_cs=rope_cs, cu_list=cu_list)
                nonlocal aux
                aux = layer_aux
                return attn_out

            aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            attn_normed = self.attn_norm(x)
            x_attn = torch.utils.checkpoint.checkpoint(custom_attn, attn_normed, use_reentrant=False)
            x = x + x_attn
            ffn_normed = self.ffn_norm(x)
            x_ffn = torch.utils.checkpoint.checkpoint(self.ffn, ffn_normed, use_reentrant=False)
            x = x + x_ffn
        else:
            x_attn, aux = self.attn(self.attn_norm(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, rope_cs=rope_cs, cu_list=cu_list)
            x = x + x_attn
            x_ffn = self.ffn(self.ffn_norm(x))
            x = x + x_ffn

        return x, aux