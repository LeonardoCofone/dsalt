# `results/`: evaluation scripts and raw results

The evaluation scripts, the JSON files they produced, and the script that draws the
figures of the paper from those files.

## Experiments

| Experiment | Script | Measures |
| --- | --- | --- |
| Per-layer metrics | `paper_eval.py` (from scratch), `eval_phase1_3b.py` (3B grafts) | Dobrushin coefficient ρ, effective ranks, attention entropy, attention sink, noise propagation, per layer |
| Context scaling | `context_scaling.py` | perplexity, peak memory, latency and out-of-memory against context length |
| Counterfactual context | `counterfactual.py` | change in perplexity when the distant context is replaced by unrelated text |
| Block-causal evaluation | `causal_landmark_check.py` | context sweep and counterfactual of the 48-layer DSALT models with landmarks selected only from earlier positions, against the whole-sequence selection |
| Other benchmarks | `bench.py` | BABILong, recall probes, long-range LM utilization; not reported in the paper |

Helpers: `analyze_paper_eval.py` prints a `paper_eval` JSON as columns,
`add_depth_aggregates.py` adds the mean over layers to the `paper_eval` JSONs (it loads
no model), and `make_figures.py` draws the figures.

## Result files

| File | Produced by | Contents |
| --- | --- | --- |
| `paper_eval_results_{70M,0_3B}.json` | `paper_eval.py` | 70M/14L and 0.3B/18L, DSALT and dense |
| `paper_eval_results_48L_full.json` | `paper_eval.py` | 170M/48L, seed 1, four operators |
| `paper_eval_results_48L_{999,6767}seed.json` | `paper_eval.py` | 170M/48L, seeds 999 and 6767, DSALT and dense |
| `paper_eval_results_3b.json` | `eval_phase1_3b.py` | Qwen2.5-3B grafts, four operators |
| `context_scaling_proto{70,18}.json` | `context_scaling.py --mode proto` | 70M and 0.3B |
| `context_scaling_proto48.json` | `context_scaling.py --mode proto` | 170M/48L, seed 1, four operators |
| `context_scaling_proto48_seed{999,6767}.json` | `context_scaling.py --mode proto` | 170M/48L, seeds 999 and 6767 |
| `context_scaling_3b.json` | `context_scaling.py --mode 3b` | Qwen2.5-3B grafts, four operators |
| `counterfactual_proto48.json` | `counterfactual.py --mode proto` | counterfactual test, 170M/48L, seed 1 |
| `causal_landmark_check.json` | `causal_landmark_check.py` | 170M/48L DSALT, three seeds: whole-sequence and block-causal perplexity (sweep and counterfactual) |
| `bench_3b_4way.json` | `bench.py`, `counterfactual.py` | 3B grafts: `babilong`, `lost_in_middle`, `math`, `lm_utilization`, `counterfactual` |

The paper reports the 48-layer DSALT perplexities and counterfactual values from
`causal_landmark_check.json` (block-causal selection). The DSALT entries of
`context_scaling_proto48*.json` and `counterfactual_proto48.json` use the whole-sequence
selection; their perplexities are within 2.3% of the block-causal ones. All other DSALT
results use the whole-sequence selection.

`paper_eval` JSONs hold one dict per model (`dsalt` / `dense` / `swa` / `spatra`), each a
`{metric: {"mean", "std", "n"}}` summary plus the per-layer series (`dobrushin_L{i}` and
so on). `context_scaling` JSONs carry a `config` block with the run settings and one
entry per model; dense appears twice, as `dense_flash` (FlashAttention) and
`dense_naive` (attention matrix materialized), because their memory differs.

In the `counterfactual` section of `bench_3b_4way.json`, the dense and Sparse-T entries
disagree with the context sweep on the same models (dense: 96.8 against 8.55 at 16k
tokens) for a reason not yet found; the paper does not use them. The DSALT and SWA
entries agree with the sweep.

The sections of `bench_3b_4way.json` come from separate runs merged with
`--merge-into`, which keeps the sections not recomputed in a run.

## Running the evaluations

Two checkpoint formats, selected with `--mode`:

- `3b`: graft directories, rebuilt from the `model_config.json` next to each checkpoint;
- `proto`: from-scratch models. DSALT has a `model_config.json`; the baselines store
  their config inside the `.pt` (`ckpt["cfg"]`), including `window` and `stride`.

Checkpoint and shard paths given as relative paths are resolved from `results/`; use
absolute paths when in doubt.

```bash
# per-layer metrics (from scratch: edit MODELS, OUT_PATH and the batch counts at the top)
python -m results.paper_eval
python -m results.eval_phase1_3b --dsalt-dir ... --dense-dir ... --out paper_eval_3b.json

# context scaling, 3B, four operators
DSALT_COMPILE_FLEX=1 python -m results.context_scaling --mode 3b \
    --dsalt-dir ./checkpoints_phase1_dsalt --dense-dir ./checkpoints_phase1_dense \
    --swa-dir ./checkpoints_phase1_swa --spatra-dir ./checkpoints_phase1_spatra \
    --shards ./data_shards_pg19_qwen --lengths 2048,4096,8192,16384,32768 \
    --rope-mode ntk --out context_scaling_3b.json

# counterfactual test, merged into the 3B JSON
DSALT_COMPILE_FLEX=1 python -m results.counterfactual \
    --dsalt-dir ... --dense-dir ... --swa-dir ... --spatra-dir ... \
    --lengths 16384,32768,65536,131072 \
    --merge-into bench_3b_4way.json --out bench_3b_4way.json

# analysis, no GPU
python -m results.analyze_paper_eval results/paper_eval_results_48L_full.json
python -m results.add_depth_aggregates --write
```

`DSALT_COMPILE_FLEX=1` is required when SWA or Sparse-T are evaluated at long context:
without it `flex_attention` materializes the full `[L, L]` scores (see
[`models/README.md`](../models/README.md)).

Evaluation choices in `context_scaling.py`: for every length the windows are anchored
to the ends of the same documents, so a change across lengths is not a change of text;
the per-window spread is reported next to the token-weighted perplexity; every model
gets the same RoPE extension (`--rope-mode {ntk,linear,none}`), relative to its own
base.

## Figures

```bash
python -m results.make_figures
```

writes the figures of the paper to `images/` from the JSONs alone; the 48-layer DSALT
curves come from `causal_landmark_check.json`. The attention-pattern
figure needs a forward pass through the checkpoints; by default it is cropped from the
committed render `images/figure2_attention_patterns.png`, and `--attention` renders it
again (GPU, checkpoints and `--lm-shards` required, see the script's docstring).

## Not included

Checkpoints and tokenized shards (see the main [README](../README.md)).
