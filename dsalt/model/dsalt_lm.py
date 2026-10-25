import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm          import RMSENorm
from ..modules.dsalt_transformer import DSALTTransformerBlock
from .config                     import DSALTConfig

# Liger needs Triton; imported only when loss_fn="liger" is requested.
try:
    from ..kernels.cross_entropy import LigerFusedLinearCrossEntropyFunction
    _LIGER_OK = True
except Exception:
    LigerFusedLinearCrossEntropyFunction = None
    _LIGER_OK = False


def _chunked_cross_entropy(
    x:          torch.Tensor,
    weight:     torch.Tensor,
    labels:     torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    total     = x.shape[0]
    loss_acc  = torch.zeros((), device=x.device, dtype=torch.float32)
    valid_acc = torch.zeros((), device=x.device, dtype=torch.float32)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        y_c = labels[start:end]
        # F.cross_entropy computes the log_softmax in fp32 internally, so the logits are
        # passed in their own dtype without an fp32 copy.
        logits = F.linear(x[start:end], weight)
        loss = F.cross_entropy(logits, y_c, ignore_index=-100, reduction="sum")
        loss_acc  += loss
        valid_acc += (y_c != -100).sum()

    return loss_acc / valid_acc.clamp(min=1.0)


def _liger_cross_entropy(
    x:      torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    loss, *_ = LigerFusedLinearCrossEntropyFunction.apply(
        x.contiguous(),
        weight,
        labels.contiguous(),
    )
    return loss


# Valid loss_fn values. "auto" is resolved per (device, vocab) at the first forward by
# loss_autotune and cached.
_LOSS_FN = {
    "auto":    None,
    "liger":   _liger_cross_entropy,
    "chunked": _chunked_cross_entropy,
}


class DSALTLMHeadModel(nn.Module):
    """Causal language model based on DSALT.

    A stack of :class:`~dsalt.modules.dsalt_transformer.DSALTTransformerBlock` with token
    embeddings, a final RMSNorm and an LM head (optionally tied to the embedding).

    The input is in packed format (concatenated sequences and ``cu_seqlens``); the
    forward also accepts ``[B, T]`` tensors for inference. Build it preferably with
    :meth:`from_config` and a :class:`~dsalt.model.config.DSALTConfig`.

    Args:
        vocab_size:         Vocabulary size.
        d_model:            Model dimension, divisible by ``n_heads``.
        n_layers:           Number of blocks.
        n_heads:            Number of attention heads.
        n_min, n_max:       Bounds of the adaptive local window.
        k_lmk:              Number of landmark tokens.
        max_seq_len:        Length of the RoPE cache.
        d_ff:               FFN dimension; ``None`` gives about 8/3·d_model, rounded.
        dropout:            Dropout probability.
        yarn_scale:         RoPE/YaRN positional scaling factor.
        tie_weights:        Tie the LM head to the embedding.
        padding_idx:        Padding index of the embedding.
        lm_head_chunk_size: Chunk size of the "chunked" cross-entropy.
        loss_fn:            ``"chunked"`` (default), ``"liger"`` (fused Triton kernel)
                            or ``"auto"`` (measured per GPU, by speed only; a large
                            chunk materializes ``[chunk, vocab]`` fp32 logits).
        aux_loss_weight:    Unused, kept for signature compatibility: the window and
                            landmark predictors are trained in the main forward.
                            Leave at 0.0.
    """

    def __init__(
        self,
        vocab_size:         int,
        d_model:            int,
        n_layers:           int,
        n_heads:            int,
        n_min:              int,
        n_max:              int,
        k_lmk:              int,
        max_seq_len:        int,
        d_ff:               int | None = None,
        dropout:            float      = 0.0,
        yarn_scale:         float      = 1.0,
        tie_weights:        bool       = True,
        padding_idx:        int | None = None,
        lm_head_chunk_size: int        = 2048,
        loss_fn:            str        = "chunked",
        aux_loss_weight:    float      = 0.0,
        n_kv_heads:         int | None = None,
        qkv_bias:           bool       = False,
        rope_base:          float      = 10000.0,
    ):
        super().__init__()
        assert loss_fn in _LOSS_FN, f"loss_fn must be one of {list(_LOSS_FN)}"
        if loss_fn == "liger" and not _LIGER_OK:
            raise RuntimeError(
                "loss_fn='liger' requires the Triton kernel (fused cross_entropy), "
                "not available in this environment. Use loss_fn='chunked'."
            )

        self.d_model            = d_model
        self.n_layers           = n_layers
        self.vocab_size         = vocab_size
        self.lm_head_chunk_size = lm_head_chunk_size
        self.loss_fn            = loss_fn
        self.aux_loss_weight    = aux_loss_weight

        if d_ff is None:
            hidden_dim  = int(8 / 3 * d_model)
            multiple_of = 128
            d_ff        = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of

        self.embed_tokens  = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DSALTTransformerBlock(
                d_model=d_model, n_heads=n_heads, n_min=n_min, n_max=n_max,
                k_lmk=k_lmk, max_seq_len=max_seq_len, d_ff=d_ff,
                dropout=dropout, yarn_scale=yarn_scale, layer_idx=i,
                n_kv_heads=n_kv_heads, qkv_bias=qkv_bias, rope_base=rope_base,
            )
            for i in range(n_layers)
        ])

        self.final_norm = RMSENorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    @classmethod
    def from_config(cls, config: "DSALTConfig") -> "DSALTLMHeadModel":
        """Instantiate the model from a :class:`~dsalt.model.config.DSALTConfig`."""
        return cls(**config.to_dict())

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if any(k in name for k in ("out_proj", "down_proj")) else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _resolve_loss_fn(self, flat_x: torch.Tensor, flat_labels: torch.Tensor) -> tuple[str, int]:
        """Resolve ``loss_fn="auto"`` to a concrete (loss_fn, chunk_size), measured once per
        ``(device, vocab)`` and cached. Explicit ``"chunked"`` and ``"liger"`` pass through.
        """
        if self.loss_fn != "auto":
            return self.loss_fn, self.lm_head_chunk_size
        from ..kernels.loss_autotune import autotune_loss
        choice = autotune_loss(
            flat_x, self.lm_head.weight, flat_labels,
            vocab=self.vocab_size,
            chunked_fn=_chunked_cross_entropy,
            liger_fn=_liger_cross_entropy if _LIGER_OK else None,
            liger_ok=_LIGER_OK,
            default_chunk=self.lm_head_chunk_size,
        )
        return choice["loss_fn"], (choice["chunk_size"] or self.lm_head_chunk_size)

    def _compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_x      = x.view(-1, self.d_model)
        # F.cross_entropy needs int64 targets, which torch.compile enforces.
        flat_labels = labels.view(-1).long()
        loss_fn, chunk = self._resolve_loss_fn(flat_x, flat_labels)
        if loss_fn == "liger":
            return _liger_cross_entropy(flat_x, self.lm_head.weight, flat_labels)
        return _chunked_cross_entropy(flat_x, self.lm_head.weight, flat_labels, chunk)

    @staticmethod
    @torch._dynamo.disable
    def _to_autocast_dtype(x: torch.Tensor) -> torch.Tensor:
        """Cast ``x`` to the active autocast compute dtype (no-op without autocast).

        Opaque to torch.compile: ``torch.is_autocast_enabled`` would be evaluated at trace
        time, and the traced state can differ from the runtime one. Supports the
        ``device_type`` API (PyTorch ≥ 2.4) and the older no-argument form.
        """
        dev = x.device.type
        try:
            enabled = torch.is_autocast_enabled(dev)
            dtype   = torch.get_autocast_dtype(dev)
        except TypeError:  # PyTorch < 2.4: CUDA-only no-argument API
            enabled = dev == "cuda" and torch.is_autocast_enabled()
            dtype   = torch.get_autocast_gpu_dtype() if enabled else x.dtype
        return x.to(dtype) if enabled and x.dtype != dtype else x

    @staticmethod
    @torch._dynamo.disable
    def _packed_rope_meta(cu_seqlens: torch.Tensor, total_len: int, attn0):
        """Per-step packed metadata: RoPE cos/sin gather and a host copy of cu_seqlens.

        Opaque to torch.compile: the positions depend on the data and the ``.tolist()`` for
        ``cu_list`` yields ints that change every step, which would force a re-trace. The
        one device-to-host sync happens once per step here, not once per layer.
        """
        device   = cu_seqlens.device
        num_seqs = cu_seqlens.shape[0] - 1
        lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
        starts   = cu_seqlens[:-1].to(device)
        seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
        pos_ids  = torch.arange(total_len, device=device) - starts[seq_ids]
        rope_cs  = (attn0.rope_cos[pos_ids], attn0.rope_sin[pos_ids])
        cu_list  = cu_seqlens.detach().to("cpu").tolist()
        return rope_cs, cu_list

    def forward(
        self,
        input_ids:              torch.Tensor,
        cu_seqlens:             torch.Tensor | None     = None,
        max_seqlen:             int | None              = None,
        labels:                 torch.Tensor | None     = None,
        gradient_checkpointing: bool                    = False,
        padding_mask:           torch.BoolTensor | None = None,
    ) -> dict:
        if max_seqlen is None:
            max_seqlen = input_ids.shape[-1]

        x = self.embed_dropout(self.embed_tokens(input_ids))
        # Embedding outputs are not cast by autocast; cast x to the compute dtype so the
        # residual stream has one dtype.
        x = self._to_autocast_dtype(x)
        aux_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)

        rope_cs = None
        cu_list = None
        if cu_seqlens is not None:
            rope_cs, cu_list = self._packed_rope_meta(
                cu_seqlens, input_ids.shape[0], self.layers[0].attn,
            )

        for layer in self.layers:
            x, layer_aux = layer(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gradient_checkpointing=gradient_checkpointing,
                rope_cs=rope_cs,
                cu_list=cu_list,
            )
            aux_loss = aux_loss + layer_aux

        x = self.final_norm(x)

        if labels is not None:
            main_loss = self._compute_loss(x, labels)
            loss      = main_loss + self.aux_loss_weight * aux_loss
            return {"loss": loss, "logits": None, "aux_loss": aux_loss.detach()}

        return {"loss": None, "logits": self.lm_head(x), "aux_loss": None}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())