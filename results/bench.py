"""Long-context benchmarks: BABILong, recall probes, and long-range LM utilization.

None of these benchmarks is reported in the paper except the counterfactual test
(results/counterfactual.py), which merges its section into the same JSON.

  1. BABILong (RMT-team/babilong): bAbI facts buried in long PG19 text at 4k-128k,
     scored by comparing log p(answer) across the closed answer set of each task,
     so no generation is needed. Tasks qa1, qa2, qa3.
  2. Lost-in-the-middle and math: a fact or an equation a*b=c is placed at a chosen
     position in long text and recalled at the end; scored by perplexity.
  3. Long-range LM utilization: perplexity of the final ``last_k`` tokens of a long
     document as the left context grows.

All models get the same NTK RoPE extension.

Run (3B grafts, four operators)::

    DSALT_COMPILE_FLEX=1 python -m results.bench \
        --dsalt-dir ./checkpoints_phase1_dsalt --dense-dir ./checkpoints_phase1_dense \
        --swa-dir ./checkpoints_phase1_swa --spatra-dir ./checkpoints_phase1_spatra \
        --tokenizer Qwen/Qwen2.5-3B \
        --babilong-tasks qa1,qa2,qa3 --lengths 4k,16k,64k,128k --n-examples 100 \
        --lm-shards ./data_shards_longctx_qwen --lm-lengths 4096,16384,65536,131072 --last-k 512 \
        --out bench_3b_4way.json

DSALT_COMPILE_FLEX=1 is required with --swa-dir / --spatra-dir at long context.

Run (from-scratch, 48 layers; GPT-2 tokenizer and a separate output file)::

    python -m results.bench --mode proto \
        --dsalt-dir checkpoints_dsalt_48L --dense-ckpt checkpoints_dense_48L/dense_step5000.pt \
        --swa-ckpt checkpoints_swa_48L/swa_step5000.pt \
        --spatra-ckpt checkpoints_spatra_48L/spatra_step5000.pt \
        --skip-babilong --skip-mid \
        --lm-shards ./data_shards_proto --lm-lengths 4096,16384,65536,131072 \
        --mid-lengths 4096,16384,65536,131072 --n-examples 50 \
        --out bench_proto48.json

--merge-into loads earlier results and overwrites only the sections recomputed in
this run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the top-level packages importable from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dsalt.kernels.window_utils import build_rope_cache

L_TRAIN = 4096


# device / rope
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype(device):
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _patch_rope(model, L, device, rope_mode="ntk", cache_len=None):
    """NTK (default), linear or no RoPE extension, the same for every model.

    ``L`` sets the extension factor (L / L_TRAIN) and stays the nominal benchmark length;
    ``cache_len`` sets how many positions the cache holds and must cover the longest
    sequence actually fed to the model, which for BABILong can exceed L.
    """
    head_dim = model.layers[0].attn.head_dim
    base0 = float(getattr(model.layers[0].attn, "rope_base", 10000.0))
    s = max(1.0, L / L_TRAIN) if rope_mode != "none" else 1.0
    if rope_mode == "ntk":
        base, pos_scale = base0 * (s ** (head_dim / (head_dim - 2))), 1.0
    elif rope_mode == "linear":
        base, pos_scale = base0, s
    else:
        base, pos_scale = base0, 1.0
    cos, sin = build_rope_cache(max(L, cache_len or L), head_dim, device,
                               base=base, scale=pos_scale)
    for layer in model.layers:
        layer.attn.rope_cos = cos
        layer.attn.rope_sin = sin


# model and scoring
def load_model(src, kind, device, mode="3b", vocab=50257):
    """Two checkpoint formats, as in context_scaling: "3b" (graft directory with
    model_config.json) and "proto" (DSALT directory with config, baselines as .pt with
    cfg inside, loaded with the context_scaling loaders).
    """
    if mode == "proto":
        import results.context_scaling as cs
        if kind == "dsalt":
            return cs.load_dsalt(src, device, "proto", vocab)
        return cs.load_iso_baseline(kind, src, device, vocab)
    from results.eval_phase1_3b import load_grafted
    model, arch = load_grafted(str(Path(src)), device)
    assert arch == kind, f"{src} is arch={arch}, expected {kind}"
    model.eval()
    for layer in model.layers:
        layer.attn._collect_last_P = False
    return model


class _TailLMHead:
    """Temporarily replace ``model.lm_head`` so that the vocabulary projection runs only
    on the final ``keep_last`` rows.

    At 64k-128k tokens and a 151665-token vocabulary, the full [T, V] logits do not fit
    in memory. The returned logits are [keep_last, V]; index them from the end.
    """

    def __init__(self, model, keep_last):
        self.model = model
        self.keep_last = keep_last
        self._orig = model.lm_head

    def __enter__(self):
        orig = self._orig
        k = self.keep_last

        class _Wrap(torch.nn.Module):
            def forward(self, x):
                return orig(x[-k:])

        self.model.lm_head = _Wrap()
        return self

    def __exit__(self, *exc):
        self.model.lm_head = self._orig
        return False


@torch.no_grad()
def _logits(model, kind, ids, device, amp, keep_last=None):
    """Forward returning logits; with ``keep_last`` only the final rows are projected
    (see _TailLMHead).
    """
    T = ids.shape[0]
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    k = T if keep_last is None else min(keep_last, T)
    with _TailLMHead(model, k):
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            out = model(ids, cu_seqlens=cu, max_seqlen=T) if kind == "dsalt" \
                else model(ids, cu_seqlens=cu)
    return out["logits"]


def _safe(fn, *a, device=None, label="", **kw):
    """Run a benchmark step; on CUDA OOM free the cache and return ``{"oom": True}``.

    Only OOM is caught. Other CUDA errors, such as a device-side assert from a RoPE
    cache shorter than the sequence, leave the context unusable and must propagate.
    """
    try:
        return fn(*a, **kw)
    except torch.cuda.OutOfMemoryError as e:
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [SKIP {label}] CUDA OOM: {str(e).splitlines()[0]}", flush=True)
        return {"oom": True, "accuracy": float("nan"), "margin": float("nan"),
                "ppl": float("nan"), "ppl_correct": float("nan"), "ppl_wrong": float("nan")}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  [SKIP {label}] OOM (RuntimeError): {str(e).splitlines()[0]}", flush=True)
            return {"oom": True, "accuracy": float("nan"), "margin": float("nan"),
                    "ppl": float("nan"), "ppl_correct": float("nan"), "ppl_wrong": float("nan")}
        raise


@torch.no_grad()
def _answer_logprob(model, kind, prompt_ids, answer_ids, device, amp):
    """Length-normalized log p(answer | prompt) in one forward over [prompt, answer].

    Only the answer rows are passed through log_softmax.
    """
    seq = torch.cat([prompt_ids, answer_ids]).to(device)
    a = answer_ids.shape[0]
    logits = _logits(model, kind, seq, device, amp, keep_last=a + 1)  # [a+1, V]
    rows = logits[-(a + 1):-1].float()                       # [a, V], aligned to the answer
    idx = torch.arange(a, device=device)
    logp = F.log_softmax(rows, dim=-1)
    return logp[idx, answer_ids.to(device)].mean().item()


# BABILong, scored against the closed answer set of each task
def _babilong_split(L_tag):
    """Map the short length tag to the dataset split name."""
    return {"4k": "4k", "16k": "16k", "32k": "32k", "64k": "64k",
            "128k": "128k", "1k": "1k", "2k": "2k"}.get(L_tag, L_tag)


def _babilong_L(L_tag):
    return {"1k": 1024, "2k": 2048, "4k": 4096, "16k": 16384,
            "32k": 32768, "64k": 65536, "128k": 131072}[L_tag]


def load_babilong(task, L_tag, n, tokenizer, max_len):
    """Return (examples, answer_set, real_max_len).

    The answer set is every distinct target in the task slice, the candidates the
    scoring ranks against. The split names are nominal (sized in GPT-2 tokens), so with
    the Qwen tokenizer a prompt can exceed L: prompts are truncated to ``max_len`` from
    the left, keeping the question at the end, and the real maximum length is returned
    so the RoPE cache can be sized to it.
    """
    from datasets import load_dataset
    ds = load_dataset("RMT-team/babilong", task, split=_babilong_split(L_tag), streaming=True)
    rows = []
    answers = set()
    for ex in ds:
        rows.append({"input": ex["input"], "question": ex["question"],
                     "target": ex["target"].strip()})
        answers.add(ex["target"].strip())
        if len(rows) >= n:
            break
    answer_set = sorted(answers)
    examples = []
    n_trunc = 0
    for r in rows:
        prompt = r["input"] + "\n" + r["question"].strip() + "\nAnswer:"
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > max_len:
            ids = ids[-max_len:]
            n_trunc += 1
        examples.append({"prompt_ids": torch.tensor(ids, dtype=torch.long),
                         "target": r["target"]})
    real_max = max((e["prompt_ids"].numel() for e in examples), default=1)
    if n_trunc:
        print(f"  [babilong {task} {L_tag}] truncated {n_trunc}/{len(examples)} prompts "
              f"to {max_len} tokens (nominal split length is a word count, not Qwen ids)",
              flush=True)
    return examples, answer_set, real_max


@torch.no_grad()
def run_babilong(model, kind, tokenizer, task, L_tag, n, device, amp):
    """One forward per example: bAbI answers are single words, so the first token of
    every candidate is scored from the next-token distribution at the end of the prompt.
    """
    L = _babilong_L(L_tag)
    # Truncate to the nominal L, so every model scores the same prompt.
    examples, answer_set, real_max = load_babilong(task, L_tag, n, tokenizer, max_len=L)
    # first token of each candidate answer
    cand_first = {}
    for a in answer_set:
        ids = tokenizer(" " + a, add_special_tokens=False)["input_ids"]
        cand_first[a] = ids[0] if ids else tokenizer.eos_token_id
    _patch_rope(model, L, device, "ntk", cache_len=real_max)
    correct = halluc = 0
    for ex in examples:
        assert ex["prompt_ids"].numel() <= real_max, "prompt exceeds the rope cache"
        logits = _logits(model, kind, ex["prompt_ids"].to(device), device, amp, keep_last=1)
        last = logits[-1].float()
        logp = F.log_softmax(last, dim=-1)
        scores = {a: logp[cand_first[a]].item() for a in answer_set}
        pred = max(scores, key=scores.get)
        ok = (pred == ex["target"])
        correct += int(ok); halluc += int(not ok)
    n_ex = max(1, len(examples))
    return {"accuracy": correct / n_ex, "hallucination": halluc / n_ex,
            "n": len(examples), "n_candidates": len(answer_set)}


# Recall probes scored by perplexity: a fact or an equation is placed at a chosen
# position in long real text and recalled at the end. No answer is generated.
@torch.no_grad()
def _ppl_of_span(model, kind, ctx_ids, span_ids, device, amp, rope_len):
    """Perplexity of span_ids appended after ctx_ids; only the span is scored.

    ``rope_len`` must be at least len(ctx) + len(span), since the forward indexes
    rope_cos[:T].
    """
    seq = torch.cat([ctx_ids, span_ids]).to(device)
    assert seq.shape[0] <= rope_len, f"seq {seq.shape[0]} > rope_len {rope_len}"
    s = span_ids.shape[0]
    # Rows -(s+1) .. -2 predict span[0] .. span[s-1]; row -1 predicts the token after
    # the span.
    logits = _logits(model, kind, seq, device, amp, keep_last=s + 1)  # [s+1, V]
    rows = logits[-(s + 1):-1].float()                       # [s, V], aligned to the span
    idx = torch.arange(s, device=device)
    logp = F.log_softmax(rows, dim=-1)
    nll = -logp[idx, span_ids.to(device)].mean().item()
    return math.exp(min(nll, 20.0))


def _haystack(stream, L_tokens, i):
    """A deterministic slice of L_tokens of real text, different for each example index
    ``i``.
    """
    n = int(stream.numel())
    span = max(1, n - L_tokens - 1)
    start = (i * 2654435761) % span
    return stream[start:start + L_tokens]


_LIM_COLORS = ["red", "blue", "green", "amber", "violet", "silver"]


@torch.no_grad()
def run_lost_in_middle(model, kind, tokenizer, stream, L, positions, n, device, amp):
    """Lost-in-the-middle probe: an invented fact (a colour) is placed at each fractional
    position of L tokens of real text and recalled at the end. Metric: accuracy of the
    gold colour against the other colours.

    Every colour is a single Qwen token, so one forward per example scores all of them.
    """
    _patch_rope(model, L, device, "ntk")
    cons, vow = "bcdfgklmnprstvz", "aeiou"
    # first token of each candidate colour
    cand = {c: tokenizer(f" {c}", add_special_tokens=False)["input_ids"] for c in _LIM_COLORS}
    max_ans = max(len(v) for v in cand.values())
    cand_first = {c: ids[0] for c, ids in cand.items()}
    res = {}
    for pos in positions:
        correct = 0
        for i in range(n):
            name = "".join(cons[(i * 7 + k) % len(cons)] + vow[(i * 3 + k) % len(vow)]
                           for k in range(3)).capitalize()
            colour = _LIM_COLORS[i % len(_LIM_COLORS)]
            fact = tokenizer(f" The banner of {name} is {colour}.", add_special_tokens=False)["input_ids"]
            recall = tokenizer(f"\nThe banner of {name} is", add_special_tokens=False)["input_ids"]
            # leave room so that context plus answer fit in L
            hay = _haystack(stream, L - len(fact) - len(recall) - max_ans - 1, i).tolist()
            ins = int(pos * len(hay))
            ctx_ids = torch.tensor(hay[:ins] + fact + hay[ins:] + recall, dtype=torch.long)
            logits = _logits(model, kind, ctx_ids.to(device), device, amp, keep_last=1)
            logp = F.log_softmax(logits[-1].float(), dim=-1)
            scores = {c: logp[t].item() for c, t in cand_first.items()}
            correct += int(max(scores, key=scores.get) == colour)
        res[f"pos{int(pos*100)}"] = correct / max(1, n)
    return res


def run_math(model, kind, tokenizer, stream, L, n, device, amp, n_digits=3, positions=(0.5,)):
    """Math recall probe: an equation a*b=c is placed in real text and recalled; the
    perplexity of the correct product is compared with that of a wrong one.
    margin = ppl(wrong) - ppl(correct) > 0 means the equation was kept.

    ``positions`` sets where the equation is placed (fraction of the haystack). Returns
    ppl, margin and accuracy per position, keyed pos0 / pos50 / pos100; with a single
    position the keys have no suffix.
    """
    _patch_rope(model, L, device, "ntk")
    import random as _r
    out = {}
    single = len(positions) == 1
    for pos in positions:
        rng = _r.Random(1234)
        correct_ppls, wrong_ppls = [], []
        for i in range(n):
            a = rng.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)
            b = rng.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)
            c = a * b
            c_wrong = c + rng.choice([1, -1, 7, -7])
            fact = tokenizer(f" The code is {a} times {b} equals {c}.", add_special_tokens=False)["input_ids"]
            recall = tokenizer(f"\nThe code is {a} times {b} equals", add_special_tokens=False)["input_ids"]
            ok = tokenizer(f" {c}", add_special_tokens=False)["input_ids"]
            bad = tokenizer(f" {c_wrong}", add_special_tokens=False)["input_ids"]
            ans_room = max(len(ok), len(bad))
            hay = _haystack(stream, L - len(fact) - len(recall) - ans_room - 1, i).tolist()
            ins = int(pos * len(hay))
            ctx_ids = torch.tensor(hay[:ins] + fact + hay[ins:] + recall, dtype=torch.long)
            correct_ppls.append(_ppl_of_span(model, kind, ctx_ids, torch.tensor(ok, dtype=torch.long), device, amp, L))
            wrong_ppls.append(_ppl_of_span(model, kind, ctx_ids, torch.tensor(bad, dtype=torch.long), device, amp, L))
        cp = sum(correct_ppls) / max(1, len(correct_ppls))
        wp = sum(wrong_ppls) / max(1, len(wrong_ppls))
        acc = sum(1 for x, y in zip(correct_ppls, wrong_ppls) if x < y) / max(1, len(correct_ppls))
        sfx = "" if single else f"_pos{int(pos * 100)}"
        out[f"ppl_correct{sfx}"] = cp
        out[f"ppl_wrong{sfx}"] = wp
        out[f"margin{sfx}"] = wp - cp
        out[f"accuracy{sfx}"] = acc
    return out


# Long-range LM utilization: perplexity of the final tokens against left context
def _load_stream(shards_dir, device):
    import utils.data_utils as d
    d.SHARDS_DIR = Path(shards_dir)
    from utils.data_utils import LongformShardDataset
    val_ds = LongformShardDataset("val_shard")
    parts, all_bounds, offset = [], [], 0
    for sp in sorted(Path(shards_dir).glob("val_shard_*.pt")):
        if sp.name.endswith(".bounds.pt"):
            continue
        toks = torch.load(sp, map_location="cpu").view(-1).long()
        bp = sp.with_suffix(".bounds.pt")
        bounds = (torch.load(bp, map_location="cpu").long()
                  if bp.exists() else torch.tensor([toks.numel()], dtype=torch.long))
        parts.append(toks)
        all_bounds.append(bounds + offset)
        offset += toks.numel()
    return torch.cat(parts).to(device), torch.cat(all_bounds)


def _doc_starts(bounds, L, n_windows):
    """Up to n_windows windows of length L, each inside one document, longest docs first."""
    ends = bounds.tolist()
    spans, prev = [], 0
    for e in ends:
        if e - prev >= L:
            spans.append((prev, e))
        prev = e
    spans.sort(key=lambda se: se[1] - se[0], reverse=True)
    return [e - L for (s, e) in spans[:n_windows]]


@torch.no_grad()
def run_lm_utilization(model, kind, stream, bounds, L, last_k, n_windows, device, amp):
    """Perplexity of the final last_k tokens with L - last_k tokens of left context; the
    final tokens are the same at every L.
    """
    _patch_rope(model, L, device, "ntk")
    starts = _doc_starts(bounds, L, n_windows)
    if not starts:
        return {"ppl": float("nan"), "n_windows": 0}
    tot_nll, tot_tok = 0.0, 0
    for s in starts:
        seg = stream[s:s + L].contiguous()
        labels = seg.clone()
        labels[:-1] = seg[1:]
        labels[-1] = -100
        labels[: max(0, L - last_k)] = -100
        cu = torch.tensor([0, L], device=device, dtype=torch.int32)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            out = model(seg, cu_seqlens=cu, max_seqlen=L, labels=labels) if kind == "dsalt" \
                else model(seg, cu_seqlens=cu, labels=labels)
        ntok = int((labels != -100).sum().item())
        tot_nll += out["loss"].item() * ntok
        tot_tok += ntok
    return {"ppl": math.exp(min(tot_nll / max(1, tot_tok), 20.0)), "n_windows": len(starts)}


# main
def main():
    global L_TRAIN
    ap = argparse.ArgumentParser(description="Long-context benchmarks (BABILong + LM utilization), DSALT vs dense.")
    ap.add_argument("--mode", choices=["3b", "proto"], default="3b",
                    help="checkpoint format: 3b = Qwen-grafted dirs (model_config.json); "
                         "proto = from-scratch prototypes (48L etc.), same loaders as "
                         "context_scaling --mode proto")
    ap.add_argument("--dsalt-dir", required=True,
                    help="DSALT checkpoint dir (both modes have model_config.json)")
    ap.add_argument("--dense-dir", default=None, help="[3b] dense grafted dir")
    ap.add_argument("--dense-ckpt", default=None, help="[proto] dense .pt with cfg inside")
    ap.add_argument("--swa-dir", default=None,
                    help="[3b] optional SWA grafted dir (model_config.json + checkpoint); "
                         "omit to skip the SWA column. NOTE: at 64k/128k run with "
                         "DSALT_COMPILE_FLEX=1, or eager flex materialises [L,L] and OOMs.")
    ap.add_argument("--spatra-dir", default=None,
                    help="[3b] optional Sparse-Transformer grafted dir; omit to skip")
    ap.add_argument("--swa-ckpt", default=None, help="[proto] SWA .pt (window inside)")
    ap.add_argument("--spatra-ckpt", default=None, help="[proto] SparseT .pt (window+stride inside)")
    ap.add_argument("--vocab", type=int, default=50257, help="[proto] baseline vocab size")
    ap.add_argument("--tokenizer", default=None,
                    help="default: Qwen/Qwen2.5-3B in 3b mode, gpt2 in proto mode "
                         "(MUST match the vocab the models and the shards were built on)")
    ap.add_argument("--l-train", type=int, default=4096)
    ap.add_argument("--babilong-tasks", default="qa1,qa2,qa3")
    ap.add_argument("--lengths", default="4k,16k,64k,128k", help="babilong length tags")
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--skip-babilong", action="store_true")
    ap.add_argument("--mid-lengths", default="4096,16384,65536,131072",
                    help="lengths for lost-in-the-middle + math (ppl-based)")
    ap.add_argument("--positions", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--skip-mid", action="store_true", help="skip lost-in-the-middle")
    ap.add_argument("--skip-math", action="store_true", help="skip math")
    ap.add_argument("--math-positions", default="0.5",
                    help="where to bury the equation, as a fraction of the haystack. The "
                         "default 0.5 (middle) reproduces the single-column math table; "
                         "pass e.g. 0.0,0.25,0.5,0.75,1.0 to get the lost-in-the-middle "
                         "curve on a task the models can actually do (chance = 0.5).")
    ap.add_argument("--lm-shards", default="./data_shards_longctx_qwen")
    ap.add_argument("--lm-lengths", default="4096,16384,65536,131072")
    ap.add_argument("--last-k", type=int, default=512)
    ap.add_argument("--lm-windows", type=int, default=8)
    ap.add_argument("--skip-lm", action="store_true")
    ap.add_argument("--merge-into", default=None,
                    help="path of an existing bench JSON to use as the base: sections NOT "
                         "recomputed in this run (because of --skip-*) are carried over "
                         "instead of being emptied. Typically the same file as --out.")
    ap.add_argument("--out", default="bench_3b.json")
    args = ap.parse_args()

    L_TRAIN = args.l_train
    # Variable prompt lengths would trigger torch.compile recompiles, so run eager,
    # unless DSALT_COMPILE_FLEX=1: then flex_attention is compiled with dynamic=True and
    # dynamo must stay enabled.
    import os
    if os.environ.get("DSALT_COMPILE_FLEX") != "1":
        torch._dynamo.config.disable = True
    device = _device()
    amp = _amp_dtype(device)

    from transformers import AutoTokenizer
    tok_name = args.tokenizer or ("gpt2" if args.mode == "proto" else "Qwen/Qwen2.5-3B")
    tok = AutoTokenizer.from_pretrained(tok_name)

    babilong_tasks = [t for t in args.babilong_tasks.split(",") if t]
    bl_lengths = [t for t in args.lengths.split(",") if t]
    lm_lengths = [int(x) for x in args.lm_lengths.split(",") if x]
    mid_lengths = [int(x) for x in args.mid_lengths.split(",") if x]
    positions = [float(x) for x in args.positions.split(",") if x]
    math_positions = [float(x) for x in args.math_positions.split(",") if x]

    # The LM stream is the haystack for lost-in-the-middle and math and the corpus for
    # LM utilization.
    stream = bounds = None
    if not (args.skip_lm and args.skip_mid and args.skip_math):
        stream, bounds = _load_stream(args.lm_shards, device)
        print(f"[bench] lm stream {stream.numel():,} tokens | {bounds.numel():,} docs")
        # Fail fast if the shards were tokenized for a different vocabulary.
        if args.mode == "proto" and int(stream.max()) >= args.vocab:
            raise SystemExit(f"[bench] shard token id {int(stream.max())} >= vocab "
                             f"{args.vocab}: --lm-shards points at shards tokenised for a "
                             f"different vocabulary (use the proto shards, e.g. ./data_shards_proto)")

    # 3b: graft directories, DSALT and dense required. proto: DSALT directory and
    # baseline .pt checkpoints, each optional.
    if args.mode == "proto":
        model_dirs = [("dsalt", args.dsalt_dir)]
        if args.dense_ckpt:
            model_dirs.append(("dense", args.dense_ckpt))
        if args.swa_ckpt:
            model_dirs.append(("swa", args.swa_ckpt))
        if args.spatra_ckpt:
            model_dirs.append(("spatra", args.spatra_ckpt))
    else:
        if not args.dense_dir:
            ap.error("--dense-dir is required in 3b mode")
        model_dirs = [("dsalt", args.dsalt_dir), ("dense", args.dense_dir)]
        if args.swa_dir:
            model_dirs.append(("swa", args.swa_dir))
        if args.spatra_dir:
            model_dirs.append(("spatra", args.spatra_dir))
    tags = [t for t, _ in model_dirs]
    hdr = {"dsalt": "DSALT", "dense": "DENSE", "swa": "SWA", "spatra": "SparseT"}

    # Number of forward passes, printed before any model is loaded.
    n_fwd = 0
    if not args.skip_babilong:
        n_fwd += args.n_examples * len(babilong_tasks) * len(bl_lengths)
    if not args.skip_mid:
        n_fwd += args.n_examples * len(positions) * len(mid_lengths)
    if not args.skip_math:
        # two forwards per example (correct and wrong span) at every position
        n_fwd += args.n_examples * 2 * len(math_positions) * len(mid_lengths)
    if not args.skip_lm:
        n_fwd += args.lm_windows * len(lm_lengths)
    print(f"[bench] budget: ~{n_fwd:,} forwards per model, {n_fwd * len(model_dirs):,} total "
          f"({len(model_dirs)} models)")

    # With --merge-into, start from the previous results so that a partial run refreshes
    # only its own sections.
    results = {"babilong": {}, "lost_in_middle": {}, "math": {}, "lm_utilization": {}}
    if args.merge_into:
        base = json.loads(Path(args.merge_into).read_text(encoding="utf-8"))
        for section in results:
            if base.get(section):
                results[section] = base[section]
        kept = [s for s in results if results[s]]
        print(f"[bench] merging into {args.merge_into} | carried over: {', '.join(kept) or 'nothing'}")

    for kind, ckpt in model_dirs:
        print(f"\n[bench] ===== {kind} =====")
        model = load_model(ckpt, kind, device, mode=args.mode, vocab=args.vocab)

        if not args.skip_babilong:
            results["babilong"][kind] = {}
            for task in babilong_tasks:
                results["babilong"][kind][task] = {}
                for Ltag in bl_lengths:
                    r = _safe(run_babilong, model, kind, tok, task, Ltag, args.n_examples,
                              device, amp, device=device, label=f"babilong {task} {Ltag}")
                    results["babilong"][kind][task][Ltag] = r
                    print(f"  babilong {task} {Ltag:>4s} | acc {r['accuracy']:.3f} "
                          f"({r['n']} ex, {r['n_candidates']} cand)", flush=True)

        if not args.skip_mid:
            results["lost_in_middle"][kind] = {}
            for L in mid_lengths:
                r = _safe(run_lost_in_middle, model, kind, tok, stream, L, positions,
                          args.n_examples, device, amp, device=device, label=f"lost-in-mid L={L}")
                results["lost_in_middle"][kind][str(L)] = r
                if r.get("oom"):
                    print(f"  lost-in-mid L={L:>7d} | skipped (OOM)", flush=True)
                else:
                    row = " ".join(f"{k}={v:6.2f}" for k, v in r.items())
                    print(f"  lost-in-mid L={L:>7d} | recall-ppl {row}", flush=True)

        if not args.skip_math:
            results["math"][kind] = {}
            for L in mid_lengths:
                r = _safe(run_math, model, kind, tok, stream, L, args.n_examples,
                          device, amp, positions=math_positions,
                          device=device, label=f"math L={L}")
                results["math"][kind][str(L)] = r
                if r.get("oom"):
                    print(f"  math L={L:>7d} | skipped (OOM)", flush=True)
                elif len(math_positions) == 1:
                    print(f"  math L={L:>7d} | acc {r['accuracy']:.3f} margin {r['margin']:+.3f} "
                          f"(ppl ok {r['ppl_correct']:.2f} vs wrong {r['ppl_wrong']:.2f})", flush=True)
                else:
                    cells = " ".join(f"pos{int(p*100)}={r[f'accuracy_pos{int(p*100)}']:.2f}"
                                     for p in math_positions)
                    print(f"  math L={L:>7d} | acc by position: {cells}", flush=True)

        if not args.skip_lm:
            results["lm_utilization"][kind] = {}
            for L in lm_lengths:
                r = _safe(run_lm_utilization, model, kind, stream, bounds, L, args.last_k,
                          args.lm_windows, device, amp, device=device, label=f"lm-util L={L}")
                results["lm_utilization"][kind][str(L)] = r
                if r.get("oom"):
                    print(f"  lm-util L={L:>7d} | skipped (OOM)", flush=True)
                else:
                    print(f"  lm-util L={L:>7d} | last{args.last_k} ppl {r['ppl']:8.3f} "
                          f"({r['n_windows']}w)", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    out = Path(args.out)
    # Keep the previous file when merging in place.
    if args.merge_into and out.exists():
        bak = out.with_suffix(out.suffix + ".bak")
        bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[bench] backed up previous results -> {bak}")
    out.write_text(json.dumps(_clean(results), indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nsaved -> {args.out}\n")

    # Summary tables, one column per model. A section is printed when it has data,
    # computed now or merged in.
    def _cell(x, fmt="8.3f", w=8):
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return f"{'OOM':>{w}s}"
        return f"{x:{fmt}}"

    def _get(section, tag, *keys):
        """Nested lookup tolerating a model/length absent from a merged-in section."""
        node = results[section].get(tag) or {}
        for k in keys[:-1]:
            node = node.get(k) or {}
        return node.get(keys[-1]) if isinstance(node, dict) else None

    def _cols(section):
        """Model columns that actually have data in this section (keeps `tags` order)."""
        return [t for t in tags if results[section].get(t)]

    def _lengths(section, tag, sub=None):
        """Length keys present for `tag`, numerically sorted (str keys in the JSON)."""
        node = results[section].get(tag) or {}
        if sub is not None:
            node = node.get(sub) or {}
        return list(node.keys())

    bl = _cols("babilong")
    if bl:
        ref = bl[0]
        print("=== BABILong accuracy ===")
        print(f"{'task':6s} {'L':>5s}" + "".join(f"{hdr[t]:>9s}" for t in bl))
        for task in _lengths("babilong", ref):
            for Ltag in _lengths("babilong", ref, task):
                row = "".join(" " + _cell(_get("babilong", t, task, Ltag, "accuracy")) for t in bl)
                print(f"{task:6s} {Ltag:>5s}{row}")

    lim = _cols("lost_in_middle")
    if lim:
        ref = lim[0]
        Ls = sorted(_lengths("lost_in_middle", ref), key=int)
        pos_keys = [k for k in (results["lost_in_middle"][ref].get(Ls[0]) or {})
                    if k.startswith("pos")] if Ls else []
        print("\n=== Lost-in-the-middle: accuracy vs fact position (higher=found it) ===")
        print(f"{'L':>8s} {'pos':>6s}" + "".join(f"{hdr[t]:>9s}" for t in lim))
        for L in Ls:
            for pk in pos_keys:
                row = "".join(" " + _cell(_get("lost_in_middle", t, L, pk), "8.2f") for t in lim)
                print(f"{int(L):>8d} {pk[3:]:>5s}%{row}")

    mt = _cols("math")
    if mt:
        Ls = sorted(_lengths("math", mt[0]), key=int)
        row0 = (results["math"][mt[0]].get(Ls[0]) or {}) if Ls else {}
        pos_sfx = sorted((k[len("accuracy"):] for k in row0 if k.startswith("accuracy_pos")),
                         key=lambda s: int(s[4:]))
        if pos_sfx:
            print("\n=== Math lost-in-the-middle: accuracy vs where the equation is buried ===")
            print("    (chance = 0.500; the equation's digits are re-derivable, so a model")
            print("     that kept the context scores ~1.0 -- this is the noise-accumulation probe)")
            print(f"{'L':>8s} {'pos':>5s}" + "".join(f"{hdr[t] + ' acc':>13s}" for t in mt)
                  + "".join(f"{hdr[t] + ' marg':>14s}" for t in mt))
            for L in Ls:
                for sfx in pos_sfx:
                    accs = "".join(" " + _cell(_get("math", t, L, f"accuracy{sfx}"), "12.3f", 12) for t in mt)
                    margs = "".join(" " + _cell(_get("math", t, L, f"margin{sfx}"), "+13.3f", 13) for t in mt)
                    print(f"{int(L):>8d} {sfx[4:]:>4s}%{accs}{margs}")
        else:
            print("\n=== Math: kept the equation? (accuracy = ppl(correct) < ppl(wrong)) ===")
            print(f"{'L':>8s}"
                  + "".join(f"{hdr[t] + ' acc':>13s}" for t in mt)
                  + "".join(f"{hdr[t] + ' marg':>14s}" for t in mt))
            for L in Ls:
                accs = "".join(" " + _cell(_get("math", t, L, "accuracy"), "12.3f", 12) for t in mt)
                margs = "".join(" " + _cell(_get("math", t, L, "margin"), "+13.3f", 13) for t in mt)
                print(f"{int(L):>8d}{accs}{margs}")

    lmu = _cols("lm_utilization")
    if lmu:
        print("\n=== Long-range LM utilization: final-token ppl vs left context ===")
        print(f"{'L':>8s}" + "".join(f"{hdr[t] + ' ppl':>13s}" for t in lmu))
        for L in sorted(_lengths("lm_utilization", lmu[0]), key=int):
            row = "".join(" " + _cell(_get("lm_utilization", t, L, "ppl"), "12.3f", 12) for t in lmu)
            print(f"{int(L):>8d}{row}")


if __name__ == "__main__":
    main()
