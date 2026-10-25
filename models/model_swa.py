from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsalt.kernels.RMSENorm import RMSENorm
from dsalt.kernels.window_utils import build_rope_cache, apply_rotary_emb
from dsalt.modules.dsalt_transformer import SwiGLUFFN
from dsalt.model.dsalt_lm import _chunked_cross_entropy, _liger_cross_entropy, _LIGER_OK

try:
    import os as _os
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    # Without torch.compile, flex_attention materializes the full [L, L] scores and runs
    # out of memory at long context. Compiling it is safe only when the surrounding
    # model is eager, as in the evaluation scripts: inside the trainer's torch.compile
    # the nested compile produces a broken Triton kernel. Set DSALT_COMPILE_FLEX=1 for
    # long-context evaluation and leave it unset for training.
    if _os.environ.get("DSALT_COMPILE_FLEX") == "1":
        flex_attention = torch.compile(flex_attention, dynamic=True)
    _FLEX_OK = True
except Exception:
    flex_attention = None
    create_block_mask = None
    _FLEX_OK = False


# Sliding-window attention (SWA) baseline. Same blocks, normalization, FFN, RoPE, loss
# and GQA handling as the dense model; each query attends to itself and the previous
# ``window - 1`` keys, with no adaptive width and no landmarks.

# Mean adaptive window of the DSALT config in utils/utils_dsalt.py: (64 + 256) / 2.
WINDOW_FROM_DSALT = 160

# Additive masks and flex BlockMasks depend only on the length, window and device, so
# they are cached once for all layers.
_MASK_CACHE: dict = {}

_BLOCK_MASK_CACHE: dict = {}


def _make_sliding_window_mask_mod(window: int):
    """flex_attention ``mask_mod`` for a causal sliding window of ``window`` positions.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window)
    return mask_mod


def _get_block_mask(L: int, window: int, device):
    key = (L, window, str(device))
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is None:
        # _compile=True builds the mask block by block; in eager mode create_block_mask
        # evaluates it at full [L, L] resolution and runs out of memory at long context.
        bm = create_block_mask(_make_sliding_window_mask_mod(window),
                               B=None, H=None, Q_LEN=L, KV_LEN=L, device=device,
                               _compile=True)
        _BLOCK_MASK_CACHE[key] = bm
    return bm


@torch._dynamo.disable
def _flex_call(qb, kb, vb, block_mask):
    """Run flex_attention outside the surrounding torch.compile graph.

    If dynamo traces this call inside the trainer's compiled model, it nests a second
    inductor pass that emits a broken Triton kernel. Disabling dynamo here keeps the
    rest of the model compiled.
    """
    return flex_attention(qb, kb, vb, block_mask=block_mask)


def _repeat_kv_packed(t: torch.Tensor, n_kv_heads: int, n_rep: int, n_heads: int) -> torch.Tensor:
    """Expand grouped kv heads to full n_heads for the [T, n_kv_heads, D] layout."""
    if n_rep == 1:
        return t
    T, _, D = t.shape
    return t.unsqueeze(2).expand(T, n_kv_heads, n_rep, D).reshape(T, n_heads, D)


def _sliding_window_mask(L: int, window: int, device, dtype) -> torch.Tensor:
    """Additive [L, L] mask: 0 where query i may attend to key j, -inf otherwise.

    Allowed iff j <= i and i - j < window, so a query sees itself and the
    ``window - 1`` preceding tokens. The mask is cached per (L, window, device).
    """
    idx = torch.arange(L, device=device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)          # [L,L], diff[i,j] = i - j
    allowed = (diff >= 0) & (diff < window)             # causal and inside the window
    mask = torch.zeros(L, L, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class SWAAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0, yarn_scale=1.0,
                 n_kv_heads=None, qkv_bias=False, rope_base=10000.0,
                 window=WINDOW_FROM_DSALT):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # GQA as in the dense model.
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.dropout = dropout
        self.window = int(window)
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        cos, sin = build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"), base=rope_base, scale=yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._last_P: torch.Tensor | None = None

    def _get_mask(self, L, device, dtype):
        key = (L, self.window, str(device), dtype)
        m = _MASK_CACHE.get(key)
        if m is None:
            m = _sliding_window_mask(L, self.window, device, dtype)
            _MASK_CACHE[key] = m
        return m

    def forward(self, x, cu_seqlens, total_len, rope_cs, n_seqs, seq_len, uniform):
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(total_len, H, D)
        k = self.k_proj(x).view(total_len, self.n_kv_heads, D)
        v = self.v_proj(x).view(total_len, self.n_kv_heads, D)
        cos, sin = rope_cs
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        k = _repeat_kv_packed(k, self.n_kv_heads, self.n_rep, H)
        v = _repeat_kv_packed(v, self.n_kv_heads, self.n_rep, H)
        self._last_P = None

        if uniform:
            B, L = n_seqs, seq_len
            qb = q.view(B, L, H, D).transpose(1, 2)
            kb = k.view(B, L, H, D).transpose(1, 2)
            vb = v.view(B, L, H, D).transpose(1, 2)
            if _FLEX_OK and x.device.type == "cuda":
                # flex_attention computes only the blocks inside the window. SDPA with
                # an additive mask cannot use the flash backend and was about 3x slower
                # at 48 layers.
                block_mask = _get_block_mask(L, self.window, x.device)
                ob = _flex_call(qb, kb, vb, block_mask)
            else:
                # CPU or old-torch fallback: dense additive mask with SDPA.
                mask = self._get_mask(L, x.device, qb.dtype)      # [L,L] additive
                ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask)
            out = ob.transpose(1, 2).reshape(total_len, H, D)
            if not self.training and getattr(self, "_collect_last_P", True):
                self._store_last_P(qb[:1], kb[:1], L, D, x.device)
            return self.out_proj(out.reshape(total_len, H * D))

        out = torch.empty_like(q)
        for b in range(n_seqs):
            s = int(cu_seqlens[b])
            e = int(cu_seqlens[b + 1])
            Lb = e - s
            qb = q[s:e].transpose(0, 1).unsqueeze(0)
            kb = k[s:e].transpose(0, 1).unsqueeze(0)
            vb = v[s:e].transpose(0, 1).unsqueeze(0)
            if _FLEX_OK and x.device.type == "cuda":
                # The dense [Lb, Lb] mask below runs out of memory at long context, so
                # the GPU path uses the block-sparse mask.
                block_mask = _get_block_mask(Lb, self.window, x.device)
                ob = _flex_call(qb, kb, vb, block_mask)
            else:
                mask = self._get_mask(Lb, x.device, qb.dtype)
                ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask)
            out[s:e] = ob.squeeze(0).transpose(0, 1)
            if not self.training and b == 0 and getattr(self, "_collect_last_P", True):
                self._store_last_P(qb, kb, Lb, D, x.device)
        return self.out_proj(out.reshape(total_len, H * D))

    @torch.no_grad()
    def _store_last_P(self, qb, kb, L, D, device):
        # Attention matrix of the first sequence, [H, T, T], in the same layout as the
        # dense model. Masked entries are exactly 0 after the softmax.
        scale = D ** -0.5
        q0 = qb[0]                                                  # [H,T,D]
        k0 = kb[0]                                                  # [H,T,D]
        scores = torch.matmul(q0, k0.transpose(-2, -1)) * scale     # [H,T,T]
        mask = self._get_mask(L, device, scores.dtype)             # [T,T] additive
        scores = scores + mask
        self._last_P = torch.softmax(scores, dim=-1).detach()      # [H,T,T]


class SWABlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.0, yarn_scale=1.0,
                 n_kv_heads=None, qkv_bias=False, rope_base=10000.0,
                 window=WINDOW_FROM_DSALT):
        super().__init__()
        self.attn_norm = RMSENorm(d_model)
        self.ffn_norm = RMSENorm(d_model)
        self.attn = SWAAttention(d_model, n_heads, max_seq_len, dropout, yarn_scale,
                                 n_kv_heads=n_kv_heads, qkv_bias=qkv_bias,
                                 rope_base=rope_base, window=window)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x, cu_seqlens=None, total_len=None, rope_cs=None,
                n_seqs=None, seq_len=None, uniform=None, max_seqlen=None):
        # As in DenseBlock, the packed metadata is rebuilt when only cu_seqlens is
        # given.
        if rope_cs is None:
            if total_len is None:
                total_len = x.shape[0]
            if cu_seqlens is None:
                cu_seqlens = torch.tensor([0, total_len], device=x.device, dtype=torch.int32)
            rope_cs, n_seqs, seq_len, uniform = self._rebuild_meta(cu_seqlens, total_len)
        x = x + self.attn(self.attn_norm(x), cu_seqlens, total_len, rope_cs, n_seqs, seq_len, uniform)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    @torch._dynamo.disable
    def _rebuild_meta(self, cu_seqlens, total_len):
        attn0 = self.attn
        device = cu_seqlens.device
        num_seqs = cu_seqlens.shape[0] - 1
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
        starts = cu_seqlens[:-1].to(device)
        seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
        pos_ids = torch.arange(total_len, device=device) - starts[seq_ids]
        rope_cs = (attn0.rope_cos[pos_ids], attn0.rope_sin[pos_ids])
        lens_list = lens.tolist()
        seq_len = lens_list[0] if lens_list else total_len
        uniform = all(l == seq_len for l in lens_list)
        return rope_cs, num_seqs, seq_len, uniform


class SWALM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        yarn_scale: float = 1.0,
        tie_weights: bool = True,
        loss_fn: str = "chunked",
        lm_head_chunk_size: int = 2048,
        n_kv_heads: int | None = None,
        qkv_bias: bool = False,
        rope_base: float = 10000.0,
        window: int = WINDOW_FROM_DSALT,
    ):
        super().__init__()
        if d_ff is None:
            hidden = int(8 / 3 * d_model)
            d_ff = ((hidden + 127) // 128) * 128
        if loss_fn == "liger" and not _LIGER_OK:
            raise RuntimeError("loss_fn='liger' requires the dsalt fused cross_entropy kernel; use 'chunked'.")

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.loss_fn = loss_fn
        self.lm_head_chunk_size = lm_head_chunk_size
        self.window = int(window)

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            SWABlock(d_model, n_heads, d_ff, max_seq_len, dropout, yarn_scale,
                     n_kv_heads=n_kv_heads, qkv_bias=qkv_bias, rope_base=rope_base,
                     window=window)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSENorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if any(k in name for k in ("out_proj", "down_proj")) else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    @torch._dynamo.disable
    def _to_autocast_dtype(x: torch.Tensor) -> torch.Tensor:
        dev = x.device.type
        try:
            enabled = torch.is_autocast_enabled(dev)
            dtype = torch.get_autocast_dtype(dev)
        except TypeError:
            enabled = dev == "cuda" and torch.is_autocast_enabled()
            dtype = torch.get_autocast_gpu_dtype() if enabled else x.dtype
        return x.to(dtype) if enabled and x.dtype != dtype else x

    @staticmethod
    @torch._dynamo.disable
    def _packed_meta(cu_seqlens, total_len, attn0):
        device = cu_seqlens.device
        num_seqs = cu_seqlens.shape[0] - 1
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
        starts = cu_seqlens[:-1].to(device)
        seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
        pos_ids = torch.arange(total_len, device=device) - starts[seq_ids]
        rope_cs = (attn0.rope_cos[pos_ids], attn0.rope_sin[pos_ids])
        l0 = int(lens[0].item())
        uniform = bool((lens == l0).all().item())
        seq_len = l0 if uniform else 0
        return rope_cs, num_seqs, seq_len, uniform

    def _compute_loss(self, x, labels):
        flat_x = x.view(-1, self.d_model)
        flat_labels = labels.view(-1).long()
        if self.loss_fn == "liger":
            return _liger_cross_entropy(flat_x, self.lm_head.weight, flat_labels)
        return _chunked_cross_entropy(flat_x, self.lm_head.weight, flat_labels, self.lm_head_chunk_size)

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, labels=None,
                gradient_checkpointing=False):
        total_len = input_ids.shape[0]
        if cu_seqlens is None and input_ids.dim() == 2:
            B, T = input_ids.shape
            cu_seqlens = torch.arange(0, (B + 1) * T, T, device=input_ids.device, dtype=torch.int32)
            input_ids = input_ids.reshape(-1)
            total_len = input_ids.shape[0]
        elif cu_seqlens is None:
            cu_seqlens = torch.tensor([0, total_len], device=input_ids.device, dtype=torch.int32)

        x = self.embed_dropout(self.embed_tokens(input_ids))
        x = self._to_autocast_dtype(x)
        rope_cs, n_seqs, seq_len, uniform = self._packed_meta(cu_seqlens, total_len, self.layers[0].attn)
        for layer in self.layers:
            if gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cu_seqlens, total_len, rope_cs, n_seqs, seq_len, uniform,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cu_seqlens, total_len, rope_cs, n_seqs, seq_len, uniform)
        x = self.final_norm(x)

        if labels is not None:
            loss = self._compute_loss(x, labels)
            return {"loss": loss, "logits": None}
        return {"loss": None, "logits": self.lm_head(x)}

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)
