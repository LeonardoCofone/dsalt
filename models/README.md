# `models/`: baselines and the Qwen2.5 graft

The models DSALT is compared against, and the code that loads a pretrained Qwen2.5
backbone into each of them. The DSALT model is in [`dsalt/`](../dsalt/).

## Baselines

The baselines use the same blocks, normalization, SwiGLU FFN, RoPE, GQA handling and
loss as DSALT (imported from `dsalt/`), and differ only in the attention operator.

| Model | Each query attends to |
| --- | --- |
| `DenseLM` | all earlier tokens |
| `SWALM` | a fixed window of the last `window` tokens |
| `SparseTLM` | a fixed window plus summary columns every `stride` tokens |
| DSALT | an adaptive window plus landmarks chosen by content |

Defaults: `window = 160`, the mean of DSALT's `n_min = 64` and `n_max = 256`; `stride =
128`, which gives `4096 / 128 = 32` summary columns, as many as DSALT's landmarks. On the
3B grafts the window is 312, the mean DSALT window on that model. The evaluation loaders
read `window` and `stride` back from each checkpoint.

| File | Contents |
| --- | --- |
| `model_dense.py` | `DenseLM`, full causal attention with SDPA / FlashAttention, or with the attention matrix materialized for the memory comparison. |
| `model_swa.py` | `SWALM`, sliding-window attention. |
| `model_spatra.py` | `SparseTLM`, Sparse Transformer (Child et al., 2019), fixed factorized pattern. |
| `load_qwen_into_dsalt.py` | Loads a Qwen2.5 backbone into DSALT and, with `copy_qwen_weights_dense`, into the baselines. |

## `flex_attention` and `DSALT_COMPILE_FLEX`

`SWALM` and `SparseTLM` use `flex_attention` mask mods. Without `torch.compile`,
`flex_attention` materializes the full `[L, L]` scores and runs out of memory at long
context. Compiling it is safe only when the surrounding model is eager:

- training: leave `DSALT_COMPILE_FLEX` unset (the trainer compiles the whole model, and
  a nested compile produces a broken Triton kernel);
- long-context evaluation (`results/context_scaling.py`, `results/counterfactual.py`,
  `results/bench.py`): set `DSALT_COMPILE_FLEX=1`.

Without `flex_attention` (old PyTorch, CPU) both models fall back to SDPA with a dense
mask.

## Loading a Qwen2.5 backbone

The window predictor `win_gate` and the per-head balance `alpha_w` do not exist in a
dense model, so a grafted model needs a short fine-tune. `load_qwen_into_dsalt.py`:

1. builds a DSALT model with the architecture of the backbone (depth, `d_model`, heads,
   GQA kv-heads, SwiGLU, RMSNorm, RoPE base, q/k/v bias, untied LM head);
2. copies the Qwen weights by name;
3. initializes the two DSALT-only predictors so that the first forward is close to
   dense attention, and checks the load: the perplexity at short context must be close
   to that of the original Qwen.

```bash
python -m models.load_qwen_into_dsalt --qwen Qwen/Qwen2.5-3B --train-len 4096
```

The fine-tune is [`train/finetune_phase1.py`](../train/README.md), which
re-initializes the window predictor so that the window starts mid-range. Requires
`transformers`.
