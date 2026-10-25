# `dsalt/`: the DSALT library

The implementation of DSALT as an installable library (also on PyPI as `dsalt`). It
imports nothing from `models/`, `train/`, `utils/` or `results/`.

Each token attends over `A(i) = W(i) ∪ L(i)`: an adaptive local window `W(i)` whose
width is predicted per token in `[n_min, n_max]`, and the `k_lmk` landmark tokens
selected by the hybrid score that precede the window. The window predictor and the
per-head balance of the score are trained end to end. Landmarks are selected once per
sequence with whole-sequence statistics (see the main [README](../README.md)).

## Quick use

```python
from dsalt import DSALTConfig, DSALTLMHeadModel

cfg = DSALTConfig(vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
                  n_min=64, n_max=256, k_lmk=16, max_seq_len=1024)
model = DSALTLMHeadModel.from_config(cfg)
```

The public API is re-exported at package level; see `__init__.py`.

## Layout

```
dsalt/
├── model/       configuration and language model
├── modules/     attention and transformer blocks
├── kernels/     Triton kernels, selectors, RoPE and window utilities, fused loss
└── training/    training loop, DDP setup, logging
```

### `model/`

| File | Contents |
| --- | --- |
| `config.py` | `DSALTConfig`, the serializable configuration. Besides the usual fields it has `n_min` / `n_max` / `k_lmk`, and `n_kv_heads`, `qkv_bias` and `rope_base` for GQA backbones such as Qwen2.5. Saved next to every checkpoint as `model_config.json`. |
| `dsalt_lm.py` | `DSALTLMHeadModel`: embedding, block stack, LM head, and the loss paths (`chunked` in PyTorch, or the fused `liger` kernel). |

### `modules/`

| File | Contents |
| --- | --- |
| `dsalt_attention.py` | `DSALTAttention`. Builds the differentiable selectors (a soft window edge that trains `win_gate`, a soft landmark weight that trains the per-head `alpha`) and dispatches to the Triton kernels on CUDA or to the masked-SDPA fallback. |
| `dsalt_transformer.py` | `DSALTTransformerBlock` (pre-norm decoder block) and `SwiGLUFFN`. |

### `kernels/`

| File | Contents |
| --- | --- |
| `landmark_tokens_ker.py` | The hybrid landmark score. The SDPA path and the Triton kernels both call `hybrid_scores_per_head`. |
| `selectors.py` | PyTorch selector helpers (window sizing, landmark top-k, packed metadata, landmark KV gather), importable without Triton. |
| `sparse_attn.py` | Masked-SDPA implementation of the operator, used without Triton. |
| `window_utils.py` | Window masks (single and packed sequences) and RoPE. |
| `dsalt_triton_attn.py` | Inference kernel: fused sparse attention with online softmax over the window and the landmarks. |
| `dsalt_triton_bwd.py` | Backward of the inference kernel. |
| `dsalt_triton_train.py` | Training kernel with differentiable selectors, so gradients reach `win_gate` and `alpha` without the dense `[N,H,L,L]` attention. |
| `cross_entropy.py` | Fused linear + cross-entropy, adapted from Liger-Kernel (BSD 2-Clause notice in the file header). |
| `autotune.py` | Block-size tuning once per `(head_dim, compute capability)`. |
| `loss_autotune.py` | Chooses between the `chunked` and fused loss on the current device. |
| `RMSENorm.py` | RMSNorm. |
| `_verify_train_kernel.py` | GPU check of the Triton training kernel against the dense SDPA reference: `python -m dsalt.kernels._verify_train_kernel`. |

### `training/`

| File | Contents |
| --- | --- |
| `trainer.py` | `DSALTTrainer`: DDP, gradient accumulation, mixed precision, gradient checkpointing, `torch.compile`, checkpointing, and periodic diagnostics (Dobrushin ρ, effective rank, window and alpha statistics). |
| `gpu_auto.py` | DDP setup, device selection, VRAM statistics. |
| `logging_config.py` | Step formatter and throughput timer. |

## Triton is optional

Without Triton or CUDA the package still imports and the Triton symbols are `None`;
`DSALTAttention` then uses the masked-SDPA path, which computes the same operator with
more memory. The memory and latency results of the paper require the kernels.
