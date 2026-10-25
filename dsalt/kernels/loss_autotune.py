"""Per-device choice of the language-model loss.

* ``chunked`` (PyTorch): materializes ``[chunk, vocab]`` logits in fp32 and runs a
  dense ``cross_entropy`` per chunk. Works everywhere.
* ``liger`` (Triton): fuses ``lm_head @ x`` with the cross-entropy and never
  materializes the logits. Faster on recent GPUs, slower on older ones.

With ``loss_fn="auto"`` both are measured, ``chunked`` over chunk sizes derived
from the token count, once per ``(device, vocab)``, and the faster one is cached.
"""

import os

import torch

# Winner per (device_name, vocab_size).
# Value: dict { "loss_fn": "chunked"|"liger", "chunk_size": int|None }.
_LOSS_TUNED: dict = {}


def _key(device: torch.device, vocab: int) -> tuple:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return (name, int(vocab))


def _is_main_process(device: torch.device | None = None) -> bool:
    """True only on the main rank (same logic as :func:`autotune._is_main_process`)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    for var in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val) == 0
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return int(idx) == 0


def _chunk_candidates(n_tok: int, vocab: int) -> list[int]:
    """Candidate chunk sizes for the chunked loss: fractions of the token count (1/16 to
    all), with a floor. Ascending, so the sweep can stop at the first OOM.
    """
    cands: list[int] = []
    for div in (16, 8, 4, 2, 1):
        c = max(256, (n_tok + div - 1) // div)
        c = min(c, n_tok)
        if c not in cands:
            cands.append(c)
    return cands


def _bench(loss_call, warmup: int = 2, iters: int = 5) -> float:
    """Forward-only CUDA time (ms) of a loss strategy; ``inf`` on OOM or failure.

    Timed under ``torch.no_grad()`` on detached views, so it allocates almost no extra
    memory. The forward is enough to rank the two strategies, since the difference is
    whether the logits are materialized. ``empty_cache()`` runs after an OOM.
    """
    try:
        with torch.no_grad():
            for _ in range(warmup):
                loss_call()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                loss_call()
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("inf")
    except Exception:
        return float("inf")


@torch._dynamo.disable
def autotune_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab: int,
    chunked_fn,
    liger_fn,
    liger_ok: bool,
    default_chunk: int,
) -> dict:
    """Measure the loss strategies once per ``(device, vocab)`` and cache the winner.

    Opaque to torch.compile: it runs inside the first compiled forward but is a one-shot
    eager routine (CUDA event timing, prints, ``empty_cache``).

    ``x`` is the flat ``[n_tok, d_model]`` hidden state, ``weight`` the lm_head weight,
    ``labels`` the flat labels. Returns ``{"loss_fn", "chunk_size"}``.
    """
    device = x.device
    key = _key(device, vocab)
    if key in _LOSS_TUNED:
        return _LOSS_TUNED[key]

    # Without Triton only the chunked loss is available.
    if device.type != "cuda" or not liger_ok or liger_fn is None:
        choice = {"loss_fn": "chunked", "chunk_size": default_chunk}
        _LOSS_TUNED[key] = choice
        return choice

    n_tok = x.shape[0]
    results: list[tuple[str, int | None, float]] = []
    main = _is_main_process(device)

    # Detached views of the live tensors, not clones: the autotune runs inside the first
    # forward, when memory is already in use. Forward-only timing under no_grad
    # allocates only each candidate's logits.
    x_d = x.detach()
    w_d = weight.detach()
    lbl = labels.detach()

    # One progress line per candidate, on the main rank.
    chunk_cands = _chunk_candidates(n_tok, vocab)
    n_cands = len(chunk_cands) + 1  # + liger
    if main:
        name = torch.cuda.get_device_name(device)
        print(f"\n  DSALT loss autotune | vocab={vocab} | n_tok={n_tok} | {name}")
        print(f"  measuring {n_cands} candidate(s) (forward-only, no clones)…")

    # chunked: chunk sizes from small to large, stopping at the first OOM (a larger
    # chunk needs more memory).
    for i, cs in enumerate(chunk_cands, start=1):
        ms = _bench(lambda cs=cs: chunked_fn(x_d, w_d, lbl, cs))
        results.append(("chunked", cs, ms))
        if main:
            ms_s = "OOM/fail" if ms == float("inf") else f"{ms:8.4f} ms"
            print(f"    [{i}/{n_cands}] chunked  chunk={cs:<8} {ms_s}")
        if ms == float("inf"):
            for cs_skip in chunk_cands[i:]:
                results.append(("chunked", cs_skip, float("inf")))
            break

    # liger: one fused config.
    ms = _bench(lambda: liger_fn(x_d, w_d, lbl))
    results.append(("liger", None, ms))
    if main:
        ms_s = "OOM/fail" if ms == float("inf") else f"{ms:8.4f} ms"
        print(f"    [{n_cands}/{n_cands}] liger    chunk={'-':<8} {ms_s}")

    # Free the candidates' scratch before training starts.
    torch.cuda.empty_cache()

    valid = [r for r in results if r[2] != float("inf")]
    if not valid:
        # Every candidate ran out of memory: fall back to chunked.
        choice = {"loss_fn": "chunked", "chunk_size": default_chunk}
    else:
        best = min(valid, key=lambda r: r[2])
        choice = {"loss_fn": best[0], "chunk_size": best[1]}

    if _is_main_process(device):
        _print_table(device, vocab, results, choice)

    _LOSS_TUNED[key] = choice
    return choice


def _print_table(device, vocab, results, choice) -> None:
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    print("\n" + "─" * 68)
    print(f"  DSALT loss autotune  |  vocab={vocab}  |  {name} (sm_{major}{minor})")
    print("─" * 68)
    print(f"  {'loss_fn':>10} {'chunk':>10} {'ms/step':>12}")
    print("  " + "-" * 64)
    best_ms = min((m for *_, m in results if m != float("inf")), default=None)

    def _sort_key(r):
        ms = r[2]
        return (ms == float("inf"), ms if ms != float("inf") else 0.0)

    for fn, cs, ms in sorted(results, key=_sort_key):
        cs_s = "-" if cs is None else str(cs)
        if ms == float("inf"):
            ms_s = f"{'OOM/fail':>12}"
            mark = ""
        else:
            ms_s = f"{ms:12.4f}"
            mark = "  ←best" if ms == best_ms else ""
        print(f"  {fn:>10} {cs_s:>10} {ms_s}{mark}")

    print("  " + "-" * 64)
    cs_s = "-" if choice["chunk_size"] is None else str(choice["chunk_size"])
    best_str = f" ({best_ms:.4f} ms/step)" if best_ms is not None else ""
    print(f"  choice: loss_fn={choice['loss_fn']} chunk_size={cs_s}{best_str}")
    print("─" * 68 + "\n")
