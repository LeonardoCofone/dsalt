"""Hardware / distributed helpers: DDP setup, device selection, VRAM stats."""

import os
import torch
import torch.distributed as dist


def setup_ddp(backend: str = "nccl") -> tuple[int, int, int]:
    """Read the distributed identity from the environment and init the process group.

    Returns ``(rank, local_rank, world_size)``. When ``world_size > 1`` it binds the
    CUDA device, initialises the process group (once), and enables TF32 matmul on
    Ampere+ (``sm_80+``). A no-op identity ``(0, 0, 1)`` on a single process.
    """
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))

        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision('high')

    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int = 0) -> torch.device:
    """Return the CUDA device for ``local_rank`` if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def get_gpu_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """VRAM stats (GB) for ``device``: allocated, peak, reserved, total, utilization.

    Returns an empty dict on CPU-only environments.
    """
    if not torch.cuda.is_available():
        return {}

    dev = device or torch.device(f"cuda:{torch.cuda.current_device()}")

    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved  = torch.cuda.memory_reserved(dev)  / 1024**3
    peak      = torch.cuda.max_memory_allocated(dev) / 1024**3
    total     = torch.cuda.get_device_properties(dev).total_memory / 1024**3

    return {
        "allocated_gb":    round(allocated, 3),
        "peak_gb":         round(peak, 3),
        "reserved_gb":     round(reserved, 3),
        "total_gb":        round(total, 3),
        "utilization_pct": round(100.0 * allocated / max(total, 1e-6), 1),
    }


def is_main_process(rank: int = 0) -> bool:
    """True on the main rank (rank 0)."""
    return rank == 0


def barrier(rank: int, world_size: int) -> None:
    """Distributed barrier (no-op outside DDP)."""
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
