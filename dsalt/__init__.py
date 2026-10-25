"""DSALT: sparse attention with adaptive windows and content-selected landmarks.

Main components:
- :class:`DSALTConfig` / :class:`DSALTLMHeadModel`: configuration and language model;
- :class:`DSALTAttention`, :class:`DSALTTransformerBlock`, :class:`SwiGLUFFN`: blocks;
- :func:`hybrid_scores_per_head`: the landmark score, shared by the SDPA path and
  the Triton kernels;
- utilities for the adaptive window and RoPE;
- :class:`DSALTTrainer`: training loop with DDP and checkpointing.

The Triton kernels (sparse attention, fused cross-entropy) are optional; without
Triton the library uses the SDPA fallback.

Example::

    from dsalt import DSALTConfig, DSALTLMHeadModel
    cfg   = DSALTConfig(vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
                        n_min=64, n_max=256, k_lmk=16, max_seq_len=1024)
    model = DSALTLMHeadModel.from_config(cfg)
"""

from dsalt.kernels import (
    dsalt_triton_attention,
    RMSENorm,
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
    hybrid_scores_per_head,
    compute_hybrid_scores,
    select_landmarks,
    soft_landmark_weights,
    HybridEnergyLandmarkSelector,
    sparse_attention_forward,
    sparse_attention_forward_packed,
    LigerFusedLinearCrossEntropyFunction,
)

from dsalt.modules import (
    DSALTAttention,
    DSALTTransformerBlock,
    SwiGLUFFN,
)

from dsalt.model import DSALTConfig, DSALTLMHeadModel
from dsalt.training.trainer import DSALTTrainer

__all__ = [
    "LigerFusedLinearCrossEntropyFunction",
    "DSALTTrainer",
    "dsalt_triton_attention",
    "RMSENorm",
    "compute_window_sizes",
    "build_local_window_mask",
    "build_local_window_mask_packed",
    "apply_rotary_emb",
    "build_rope_cache",
    "hybrid_scores_per_head",
    "compute_hybrid_scores",
    "select_landmarks",
    "soft_landmark_weights",
    "HybridEnergyLandmarkSelector",
    "sparse_attention_forward",
    "sparse_attention_forward_packed",
    "DSALTAttention",
    "DSALTTransformerBlock",
    "SwiGLUFFN",
    "DSALTConfig",
    "DSALTLMHeadModel",
]