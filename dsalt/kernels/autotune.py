"""Autotuning of the DSALT kernel block sizes.

Unlike ``@triton.autotune``, which re-tunes for every new shape key (and with
variable-length packing would re-tune continuously), tuning runs once per
``(head_dim, compute capability)`` and fixes the configuration for the run.

Kernel constraints:

* ``BLOCK_M`` is shared by forward and backward, which reuse the same
  ``seq_block_map``.
* The inference backward caps ``BLOCK_N`` at 32 (atomic_add), so ``BLOCK_N > 32``
  only affects the forward.
* ``HEAD_DIM``, ``K_LMK`` and ``scale`` are fixed by the shapes.

The tuned parameters are ``BLOCK_M``, ``BLOCK_N``, ``num_warps`` and ``num_stages``;
the cost measured is a full forward + backward step on a real batch.
"""

import os

import torch
import triton

# Tuned configs, indexed by (head_dim, sm_major, sm_minor).
# Value: dict with BLOCK_M, BLOCK_N, num_warps, num_stages.
_TUNED_CONFIG: dict = {}


def _is_main_process(device: torch.device | None = None) -> bool:
    """True only on the main rank (or always, without DDP).

    Every rank benchmarks its own GPU, but the table is printed once. The rank is read
    from ``torch.distributed`` or the ``torchrun`` environment variables. Under
    ``mp.spawn`` the autotune can run before ``init_process_group``; then only
    ``cuda:0`` counts as main.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    for var in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val) == 0
    # Without torch.distributed or an env rank, use the device ordinal, so mp.spawn
    # workers do not each print before init_process_group.
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return int(idx) == 0


def _device_key(head_dim: int, device: torch.device) -> tuple:
    major, minor = torch.cuda.get_device_capability(device)
    return (int(head_dim), int(major), int(minor))


def _heuristic_config(head_dim: int, device: torch.device) -> dict:
    """Safe configuration, used as a candidate and as the fallback if tuning fails.
    ``num_stages`` is 3 on sm_80+ and 2 on smaller GPUs.
    """
    if head_dim <= 64:
        block_m, block_n, num_warps = 64, 64, 4
    elif head_dim <= 128:
        block_m, block_n, num_warps = 32, 32, 2
    elif head_dim <= 256:
        block_m, block_n, num_warps = 16, 16, 2
    else:
        block_m, block_n, num_warps = 16, 8, 2

    major, _ = torch.cuda.get_device_capability(device)
    num_stages = 3 if major >= 8 else 2
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _candidate_configs(head_dim: int, device: torch.device,
                       with_bwd_tile: bool = False) -> list[dict]:
    """Candidates for phase 1 of the autotune.

    ``BLOCK_M`` takes ½/1/2/4× the heuristic value within [16, 128], ``num_warps``
    2/4/8 and ``num_stages`` 2/3 on sm_80+. ``BLOCK_N`` follows ``BLOCK_M`` up to 64.
    Configs are filtered by estimated shared memory.

    With ``with_bwd_tile`` (training only) every candidate starts from
    ``BLOCK_N_BWD = 32``; the backward tile is searched in phase 2
    (`_bwd_tile_candidates`) once the forward axes are fixed.
    """
    base = _heuristic_config(head_dim, device)
    major, _ = torch.cuda.get_device_capability(device)

    bm0 = base["BLOCK_M"]
    block_ms = sorted({max(16, bm0 // 2), bm0, min(128, bm0 * 2), min(128, bm0 * 4)})

    stage_opts = (2, 3) if major >= 8 else (2,)
    warp_opts = (2, 4, 8)

    seen = set()
    configs = []
    for bm in block_ms:
        bn = min(bm, 64)
        for nw in warp_opts:
            for ns in stage_opts:
                cfg = {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "num_stages": ns}
                key = (bm, bn, nw, ns)
                if key in seen:
                    continue
                if not _fits_shared_memory(cfg, head_dim, device):
                    continue
                if with_bwd_tile:
                    cfg["BLOCK_N_BWD"] = 32
                seen.add(key)
                configs.append(cfg)

    # Ensures the heuristic is always among the candidates.
    base_key = (base["BLOCK_M"], base["BLOCK_N"], base["num_warps"], base["num_stages"])
    if base_key not in seen:
        if with_bwd_tile:
            base = {**base, "BLOCK_N_BWD": 32}
        configs.append(base)
    return configs


def _bwd_tile_candidates(cfg: dict, head_dim: int, device: torch.device) -> list[dict]:
    """Phase-2 candidates: ``BLOCK_N_BWD`` in {32, 64, 128} around the winning phase-1
    config, filtered by the backward shared-memory budget of this device. Always
    includes 32.
    """
    bm = cfg["BLOCK_M"]
    tiles = [n for n in (32, 64, 128)
             if n >= 16 and _bwd_tile_fits_smem(bm, n, head_dim, device)]
    if 32 not in tiles:
        tiles = [32] + tiles
    out = []
    for n in tiles:
        out.append({**cfg, "BLOCK_N_BWD": n})
    return out


def _smem_budget(device: torch.device) -> int:
    """Usable per-block shared memory in bytes: 75% of the device maximum, leaving room
    for the compiler's own scratch.
    """
    try:
        props = torch.cuda.get_device_properties(device)
        smem_max = getattr(props, "shared_memory_per_block_optin", None) \
            or props.shared_memory_per_block
    except Exception:
        smem_max = 48 * 1024  # conservative default
    return int(0.75 * smem_max)


def _fits_shared_memory(cfg: dict, head_dim: int, device: torch.device) -> bool:
    """Conservative estimate of the forward shared-memory footprint:
    ``(BM + 2*BN) * D`` elements at 4 bytes, against the per-block limit.
    """
    head_dim_c = triton.next_power_of_2(head_dim)
    est_bytes = (cfg["BLOCK_M"] + 2 * cfg["BLOCK_N"]) * head_dim_c * 4
    return est_bytes <= _smem_budget(device)


def _bwd_tile_fits_smem(block_m: int, block_n_bwd: int, head_dim: int,
                        device: torch.device) -> bool:
    """Whether a backward key tile of ``block_n_bwd`` fits in shared memory.

    The dk/dv kernel keeps ``q/do[BM, D]`` and ``k/v[BN, D]`` in shared memory, so the
    estimate is ``(2*BM + 2*BN) * D`` fp32 elements. This only drops configs that cannot
    fit; the benchmark discards those that still fail.
    """
    head_dim_c = triton.next_power_of_2(head_dim)
    est_bytes = (2 * block_m + 2 * block_n_bwd) * head_dim_c * 4
    return est_bytes <= _smem_budget(device)


def _bench_step(run_fwd_bwd, n_warmup: int = 3, n_iter: int = 10) -> float:
    """Measure the average time (ms) of a fwd+bwd step with CUDA events."""
    for _ in range(n_warmup):
        run_fwd_bwd()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        run_fwd_bwd()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def autotune_blocks(head_dim: int, device: torch.device, make_runner, verbose: bool = True) -> dict:
    """Find the best config for ``(head_dim, GPU)`` and cache it.

    Args:
        head_dim:   Per-head dimension (determines the tiles).
        device:     CUDA device to benchmark on.
        make_runner: Callable ``(cfg) -> (run_fn | None)``. Takes a config and
            returns a zero-argument function that runs one fwd+bwd with that
            config, or ``None`` if the config cannot be instantiated.
        verbose:    If ``True``, print the results table (debug).

    Returns:
        The winning config (dict with BLOCK_M/BLOCK_N/num_warps/num_stages).
    """
    key = _device_key(head_dim, device)
    if key in _TUNED_CONFIG:
        return _TUNED_CONFIG[key]

    configs = _candidate_configs(head_dim, device)
    results = []  # (cfg, ms | None, err | None)

    for cfg in configs:
        runner = make_runner(cfg)
        if runner is None:
            results.append((cfg, None, "skip"))
            continue
        try:
            ms = _bench_step(runner)
            results.append((cfg, ms, None))
        except Exception as e:  # config that overruns smem/registers → discarded
            msg = type(e).__name__
            results.append((cfg, None, msg))

    valid = [(c, t) for (c, t, e) in results if t is not None]
    if valid:
        best_cfg, best_ms = min(valid, key=lambda x: x[1])
    else:
        # Heuristic fallback: no config measurable on this GPU.
        best_cfg, best_ms = _heuristic_config(head_dim, device), None

    _TUNED_CONFIG[key] = best_cfg

    # Every DDP rank benchmarks its own GPU; only the main one prints.
    if verbose and _is_main_process(device):
        _print_table(head_dim, device, results, best_cfg, best_ms)

    return best_cfg


def _print_table(head_dim, device, results, best_cfg, best_ms):
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    # Show the backward key tile column only when it is being tuned (training).
    has_bwd = any("BLOCK_N_BWD" in c for c, _, _ in results)
    print("\n" + "─" * 68)
    print(f"  DSALT autotune  |  head_dim={head_dim}  |  {name} (sm_{major}{minor})")
    print("─" * 68)
    bwd_hdr = f" {'bwdN':>6}" if has_bwd else ""
    print(f"  {'BLOCK_M':>8} {'BLOCK_N':>8}{bwd_hdr} {'warps':>6} {'stages':>7} {'ms/step':>10}")
    print("  " + "-" * 64)

    def _sort_key(r):
        _, t, _ = r
        return (t is None, t if t is not None else 0.0)

    for cfg, ms, err in sorted(results, key=_sort_key):
        is_best = cfg == best_cfg and ms is not None
        mark = "  ←best" if is_best else ""
        time_str = f"{ms:10.4f}" if ms is not None else f"{err:>10}"
        bwd_col = f" {cfg.get('BLOCK_N_BWD', ''):>6}" if has_bwd else ""
        print(f"  {cfg['BLOCK_M']:>8} {cfg['BLOCK_N']:>8}{bwd_col} {cfg['num_warps']:>6} "
              f"{cfg['num_stages']:>7} {time_str}{mark}")

    print("  " + "-" * 64)
    if best_ms is not None:
        bwd_str = f" BLOCK_N_BWD={best_cfg['BLOCK_N_BWD']}" if "BLOCK_N_BWD" in best_cfg else ""
        print(f"  choice: BLOCK_M={best_cfg['BLOCK_M']} BLOCK_N={best_cfg['BLOCK_N']}{bwd_str} "
              f"num_warps={best_cfg['num_warps']} num_stages={best_cfg['num_stages']} "
              f"({best_ms:.4f} ms/step)")
    else:
        print(f"  no measurable config → heuristic fallback: {best_cfg}")
    print("─" * 68 + "\n")


def get_tuned_config(head_dim: int, device: torch.device) -> dict | None:
    """Return the already-tuned config for ``(head_dim, GPU)``, or ``None``."""
    return _TUNED_CONFIG.get(_device_key(head_dim, device))
