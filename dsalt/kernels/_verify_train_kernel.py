"""GPU correctness check: Triton training kernel against the dense SDPA reference.

Run on a GPU before trusting the kernel:

    python -m dsalt.kernels._verify_train_kernel

The same q, k, v, x go through the differentiable Triton kernel and the dense SDPA
fallback, which share the selectors and the log-bias formula; the attention output
and all gradients must match up to fp16 rounding.
"""

import math
import torch


def _attn_out(attn, q, k, v, x, cu, total_len, device, force_triton):
    """Run only the sparse attention core (no out_proj, no RoPE)."""
    import dsalt.modules.dsalt_attention as M
    saved = M._TRITON_TRAIN_OK
    M._TRITON_TRAIN_OK = force_triton
    try:
        cu_list = cu.detach().to("cpu").tolist()
        if force_triton:
            out = attn._packed_train_triton(q, k, v, x, cu, total_len, device, cu_list)
        else:
            lens = [cu_list[b + 1] - cu_list[b] for b in range(len(cu_list) - 1)]
            out = attn._packed_train(q, k, v, x, cu_list, lens, total_len, device, cu)
    finally:
        M._TRITON_TRAIN_OK = saved
    return out                                                                  # [total_len,H,D]


def _run(attn, q0, k0, v0, x0, cu, total_len, device, force_triton):
    attn.zero_grad(set_to_none=True)
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    x = x0.detach().clone().requires_grad_(True)
    out = _attn_out(attn, q, k, v, x, cu, total_len, device, force_triton)
    loss = out.float().pow(2).mean()
    loss.backward()
    return {
        "out":   out.detach().float(),
        "q":     q.grad.detach().float().clone(),
        "k":     k.grad.detach().float().clone(),
        "v":     v.grad.detach().float().clone(),
        "win":   attn.win_gate.weight.grad.detach().float().clone(),
        "alpha": attn.alpha_w.grad.detach().float().clone(),
    }


def main():
    assert torch.cuda.is_available(), "A CUDA GPU is required."
    dev = "cuda"
    torch.manual_seed(0)
    from dsalt.modules.dsalt_attention import DSALTAttention, _TRITON_TRAIN_OK
    print("Triton train disponibile:", _TRITON_TRAIN_OK)

    H, D, L = 4, 16, 96

    def stats(a, b):
        amax = (a - b).abs().max().item()
        rel  = (a - b).norm().item() / (b.norm().item() + 1e-9)
        return amax, rel

    overall_ok = True
    # k_lmk=16 needs no landmark padding, k_lmk=8 does; testing both separates the two
    # code paths.
    for k_lmk in (16, 8):
        torch.manual_seed(0)
        attn = DSALTAttention(d_model=H * D, n_heads=H, n_min=8, n_max=48,
                              k_lmk=k_lmk, max_seq_len=256, layer_idx=0).to(dev)
        attn.train()
        x = torch.randn(L, H * D, device=dev) * 0.5
        cu = torch.tensor([0, L], dtype=torch.int32, device=dev)
        # same q, k, v for both paths, in fp16
        q = (torch.randn(L, H, D, device=dev) * 0.3).half().float()
        k = (torch.randn(L, H, D, device=dev) * 0.3).half().float()
        v = (torch.randn(L, H, D, device=dev) * 0.3).half().float()

        g_tri = _run(attn, q, k, v, x, cu, L, dev, force_triton=True)
        g_ref = _run(attn, q, k, v, x, cu, L, dev, force_triton=False)

        pad = "no-pad" if k_lmk >= 16 else "PAD active"
        print(f"\n=== k_lmk={k_lmk} ({pad}) ===")
        print(f"{'tensor':6s} | {'max|Δ|':>11} | {'relΔ':>9} | {'ref‖·‖':>10}")
        for key in ("out", "q", "k", "v", "win", "alpha"):
            amax, rel = stats(g_tri[key], g_ref[key])
            print(f"{key:6s} | {amax:11.3e} | {rel:9.3e} | {g_ref[key].norm().item():10.3e}")

        ok = True
        for key, tol in (("out", 5e-2), ("q", 5e-2), ("k", 5e-2), ("v", 5e-2),
                         ("win", 5e-3), ("alpha", 5e-4)):
            amax, _ = stats(g_tri[key], g_ref[key])
            if amax > tol:
                ok = False
                print(f"  ✗ {key} oltre tolleranza assoluta ({amax:.3e} > {tol})")
        print("  →", "OK ✓" if ok else "✗ DISCREPANZA")
        overall_ok = overall_ok and ok

    print("\nRISULTATO:", "OK ✓ kernel coerente" if overall_ok else "✗ DISCREPANZA")


if __name__ == "__main__":
    main()
