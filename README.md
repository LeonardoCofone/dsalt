# DSALT: Sparse Attention with Adaptive Windows and Content-Selected Landmarks

Code and raw results for the paper *DSALT: Sparse Attention with Adaptive Windows and
Content-Selected Landmarks* by Leonardo Cofone (arXiv link to be added on release).

In DSALT each token attends to an adaptive local window `W(i)`, whose width is
predicted per token, and to `k` landmark tokens `L(i)` chosen by a score that combines
the norm of the value vector and the norm of the hidden state. Attention is computed
over `A(i) = W(i) ∪ L(i)`, so the cost per token is `O((w + k) d)` instead of `O(L d)`.

This repository contains the reference implementation, the baselines it is compared
against (dense, sliding window, fixed-stride Sparse Transformer), the training and
evaluation scripts, and the JSON results behind the numbers and figures of the paper.

**Causality of the landmark selection.** The library selects landmarks once per
sequence and head: the score is standardized over the whole sequence and the top-k is
taken over all positions. A query never attends to a later token, but which earlier
tokens are available as landmarks depends on later tokens, so this selection is not
causal in the strict sense. `results/causal_landmark_check.py` evaluates the trained
48-layer models with a block-causal selection, in which the landmarks of each block of
512 queries come only from earlier positions. The two selections differ by at most
2.3% in perplexity (at 131,072 tokens), and the paper reports the block-causal numbers
for the 48-layer models (Section 3).

## Layout

| Path | Contents |
| --- | --- |
| [`dsalt/`](dsalt/) | The DSALT library: model, attention module, Triton kernels, trainer. |
| [`models/`](models/) | The baselines (dense, SWA, Sparse Transformer) and the Qwen2.5 graft. |
| [`train/`](train/) | Training scripts: from-scratch models and the fine-tune of the grafted 3B models. |
| [`utils/`](utils/) | Run configuration, shard dataset and loaders, data preparation. |
| [`results/`](results/) | Evaluation scripts, the result JSONs and the figure script. |

Each directory has its own README.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch 2.4 or newer is needed for `flex_attention` in the baselines. Triton is
optional: without it `dsalt` falls back to a masked-SDPA implementation of the same
operator (see [`dsalt/README.md`](dsalt/README.md)). Reproducing the results needs a
CUDA GPU; the models in the paper were trained on a single A100 80GB.

## Reproducing the paper

All entry points are module paths, run from the repository root.

1. **Data.** Pre-tokenized shards in the format of `utils/data_utils.py`. The
   from-scratch models are trained on shards from `python -m utils.prepare_train_shards`
   (PG19, arXiv and FineWeb-Edu, GPT-2 tokenizer; since windows are cut within
   documents, about 56%, 15% and 29% of the training windows come from the three
   sources). The grafts are fine-tuned on PG19 shards from
   `python -m utils.prepare_pg19_qwen`. The long-context validation sets come from
   `python -m utils.prepare_longctx_val`, once per tokenizer.
2. **Train.** From scratch: `train.train_script` (DSALT), `train.train_dense`,
   `train.train_swa` and `train.train_spatra`, all reading `utils/utils_dsalt.py::CFG`.
   Grafts: `train.finetune_phase1 --arch {dsalt,dense,swa,spatra}`.
3. **Evaluate.** `results.paper_eval` and `results.eval_phase1_3b` for the per-layer
   metrics (Dobrushin coefficient, effective ranks, entropy, sink),
   `results.context_scaling` for perplexity, latency and memory against context
   length, `results.counterfactual` for the test that replaces the distant context,
   `results.causal_landmark_check` for the block-causal evaluation of the 48-layer
   DSALT models.
4. **Figures.** `python -m results.make_figures` regenerates the figures of the paper
   from the JSONs, without a GPU.

Every script opens with a docstring giving its command lines.
[`results/README.md`](results/README.md) maps each experiment to its script and JSON.

### Checkpoints and data

Checkpoints and tokenized shards are not included because of their size; the
checkpoints are available on request. The result JSONs in `results/` are included,
and the figure and analysis scripts run on them alone.

## License

Apache 2.0, see [`LICENSE`](LICENSE). `dsalt/kernels/cross_entropy.py` is adapted from
[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) and keeps its BSD 2-Clause
notice.

The library is also distributed on PyPI as [`dsalt`](https://pypi.org/project/dsalt/)
([source](https://github.com/LeonardoCofone/dsalt-library)).

## Citation

```bibtex
@misc{cofone2026dsalt,
  author = {Cofone, Leonardo},
  title  = {{DSALT}: Sparse Attention with Adaptive Windows and Content-Selected Landmarks},
  year   = {2026},
  note   = {arXiv preprint}
}
```
