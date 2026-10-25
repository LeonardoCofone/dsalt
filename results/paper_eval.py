from __future__ import annotations

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

from utils.data_utils import make_loaders

VOCAB_SIZE = 50257

# Models to evaluate, as (kind, checkpoint path); the first entry is the reference
# column.
# - "dsalt": rebuilt from model_config.json next to the checkpoint.
# - "dense", "swa", "spatra": cfg stored inside the .pt (ckpt["cfg"]), one shared loader
# and forward path.
# Set a path to None to skip it.
MODELS = [
    ("dsalt",  "checkpoints_dsalt_48L_6767seed/checkpoint_best.pt"),
    ("dense",  "checkpoints_dense_48L_6767seed/dense_best.pt"),
    #("swa",    "checkpoints_SWA_48L/swa_best.pt"),
    #("spatra", "checkpoints_SPATRA_48L/spatra_best.pt"),
]
# Used only to resolve the dense checkpoint when its path is "AUTO".
DENSE_DIR = "checkpoints_dense_48L_6767seed"
MAX_BATCHES = 100
HEAVY_BATCHES = 20
OUT_PATH = "paper_eval_results_48L_6767seed.json"
DOBRUSHIN_ROWS = 256

# Baselines with the dense interface (class -> module), built by load_iso().
_ISO_ARCH = {
    "dense":  ("models.model_dense",  "DenseLM"),
    "swa":    ("models.model_swa",    "SWALM"),
    "spatra": ("models.model_spatra", "SparseTLM"),
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype(device):
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _abs(path):
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def _resolve_dense_best():
    ckpt_dir = _abs(DENSE_DIR)
    candidates = sorted(ckpt_dir.glob("dense_step*.pt")) + [ckpt_dir / "dense_final.pt"]
    candidates = [p for p in candidates if p.exists()]
    forced_ckpt = ckpt_dir / "dense_step15000.pt"
    if forced_ckpt.exists():
        candidates = [forced_ckpt]
    if not candidates:
        raise FileNotFoundError(f"no dense checkpoints in {ckpt_dir}")

    hist_path = ckpt_dir / "dense_history.json"
    if not hist_path.exists():
        print(f"[paper_eval] no dense_history.json, using {candidates[-1].name}")
        return candidates[-1]

    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    by_step = dict(zip(hist.get("val_steps", []), hist.get("val_ppl", [])))
    last_step = max(by_step) if by_step else None

    def ckpt_ppl(p):
        if p.name == "dense_final.pt":
            step = last_step
        else:
            try:
                step = int(p.stem.replace("dense_step", ""))
            except ValueError:
                return float("inf")
        return by_step.get(step, float("inf"))

    best = min(candidates, key=ckpt_ppl)
    print(f"[paper_eval] dense best = {best.name} (val_ppl={ckpt_ppl(best):.4f})")
    return best


def _eff_rank(sv):
    svn = sv / (sv.sum() + 1e-9)
    return torch.exp(-(svn * (svn + 1e-9).log()).sum()).item()


def _dobrushin(P):
    Pm = P.mean(dim=0)
    T = Pm.shape[0]
    lo = T // 2
    step = max(1, (T - lo) // DOBRUSHIN_ROWS)
    Psub = Pm[lo::step]
    overlaps = []
    for a in range(Psub.shape[0]):
        mins = torch.minimum(Psub[a].unsqueeze(0), Psub).sum(dim=-1)
        mins[a] = 1.0
        overlaps.append(mins.min().item())
    return 1.0 - min(overlaps)


class WelfordSet:
    def __init__(self):
        self.store = {}

    def add(self, key, val):
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            return
        if key not in self.store:
            self.store[key] = []
        self.store[key].append(float(val))

    def summary(self):
        out = {}
        for k, vals in self.store.items():
            t = torch.tensor(vals)
            out[k] = {
                "mean": t.mean().item(),
                "std": t.std(correction=0).item() if t.numel() > 1 else 0.0,
                "n": t.numel(),
            }
        return out


def load_dsalt(ckpt_path, device):
    from dsalt import DSALTConfig, DSALTLMHeadModel
    ckpt_path = _abs(ckpt_path)
    cfg = DSALTConfig.load(ckpt_path.parent / "model_config.json")
    cfg.loss_fn = "chunked"
    model = DSALTLMHeadModel.from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[paper_eval] dsalt loaded from {ckpt_path.name} | params={n:,} | step={ckpt.get('step', '?')}")
    return model, "dsalt"


def load_iso(kind, ckpt_path, device):
    """Loader for the baselines (dense / swa / spatra).

    The cfg is stored inside the .pt (ckpt["cfg"]); SWA and Sparse-T also read their
    ``window`` (and Sparse-T its ``stride``) from the checkpoint, so the training-time
    attention geometry is reproduced.
    """
    import importlib
    mod_name, cls_name = _ISO_ARCH[kind]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    ckpt_path = _abs(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg["seq_len"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        tie_weights=cfg.get("tie_weights", True),
        loss_fn="chunked",
    )
    if kind in ("swa", "spatra") and ckpt.get("window") is not None:
        kwargs["window"] = int(ckpt["window"])
    if kind == "spatra" and ckpt.get("stride") is not None:
        kwargs["stride"] = int(ckpt["stride"])
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    extra = ""
    if kind in ("swa", "spatra") and ckpt.get("window") is not None:
        extra = f" | window={ckpt['window']}"
        if kind == "spatra" and ckpt.get("stride") is not None:
            extra += f" stride={ckpt['stride']}"
    print(f"[paper_eval] {kind} loaded from {ckpt_path.name} | params={n:,} | "
          f"step={ckpt.get('step', '?')}{extra}")
    return model, kind


@torch.no_grad()
def _forward_hiddens_dsalt(model, ids, cu, device):
    x = model.embed_tokens(ids)
    hiddens = [x.clone()]
    rope_cs, cu_list = model._packed_rope_meta(cu, ids.shape[0], model.layers[0].attn)
    T = int((cu[1] - cu[0]).item())
    for layer in model.layers:
        out = layer(x, cu_seqlens=cu, max_seqlen=T, rope_cs=rope_cs, cu_list=cu_list)
        x = out[0] if isinstance(out, tuple) else out
        hiddens.append(x.clone())
    return hiddens


@torch.no_grad()
def _forward_hiddens_iso(model, ids, cu, device):
    """Forward pass shared by the baselines."""
    total_len = ids.shape[0]
    x = model._to_autocast_dtype(model.embed_tokens(ids))
    rope_cs, n_seqs, seq_len, uniform = model._packed_meta(cu, total_len, model.layers[0].attn)
    hiddens = [x.clone()]
    for layer in model.layers:
        x = layer(x, cu, total_len, rope_cs, n_seqs, seq_len, uniform)
        hiddens.append(x.clone())
    return hiddens


@torch.no_grad()
def eval_model(model, kind, loader, device, amp, max_batches, acc):
    import time
    fwd = _forward_hiddens_dsalt if kind == "dsalt" else _forward_hiddens_iso

    t_start = time.time()
    for bi, (ids, labels, cu, _) in enumerate(loader):
        if bi >= max_batches:
            break
        t_batch = time.time()
        ids = ids.to(device)
        labels = labels.to(device)
        cu = cu.to(device)

        T0 = int((cu[1] - cu[0]).item())
        ids0 = ids[:T0]
        cu0 = torch.tensor([0, T0], device=device, dtype=torch.int32)

        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            if kind == "dsalt":
                out = model(ids, cu_seqlens=cu, max_seqlen=int((cu[1:] - cu[:-1]).max().item()), labels=labels)
            else:
                out = model(ids, cu_seqlens=cu, labels=labels)
        loss = out["loss"].item()
        nvalid = int((labels != -100).sum().item())
        acc.add("val_nll", loss)
        acc.add("val_tokens", nvalid)

        if bi < HEAVY_BATCHES:
            hiddens = fwd(model, ids0, cu0, device)

            h0_norm = hiddens[0].float().norm().item()
            for li, h in enumerate(hiddens[1:]):
                hf = h.float()
                sv = torch.linalg.svdvals(hf)
                acc.add(f"hidden_eff_rank_L{li}", _eff_rank(sv))
                xm = hf.mean(dim=0, keepdim=True)
                acc.add(f"res_norm_L{li}", (hf - xm).norm().item() / (h0_norm + 1e-9))

            hf_last = hiddens[-1].float()
            sv_last = torch.linalg.svdvals(hf_last)
            acc.add("hidden_eff_rank_final", _eff_rank(sv_last))

            last_attn = model.layers[-1].attn
            if getattr(last_attn, "_last_P", None) is not None:
                P = last_attn._last_P.float()
                P_avg = P.mean(dim=0)
                sv_a = torch.linalg.svdvals(P_avg)
                acc.add("attn_eff_rank", _eff_rank(sv_a))
                acc.add("sigma2", sv_a[1].item() if sv_a.shape[0] > 1 else 0.0)
                Ps = P.clamp(min=1e-9)
                acc.add("attn_entropy", -(Ps * Ps.log()).sum(dim=-1).mean().item())
                eh = -(Ps * Ps.log()).sum(dim=-1).mean(dim=-1)
                acc.add("head_spec_std", eh.std(correction=0).item())
                acc.add("attn_sink", P[:, :, 0].mean().item())

            for li, layer in enumerate(model.layers):
                attn = layer.attn
                if getattr(attn, "_last_P", None) is not None:
                    Pl = attn._last_P.float()
                    acc.add(f"dobrushin_L{li}", _dobrushin(Pl))
                    sv_l = torch.linalg.svdvals(Pl.mean(dim=0))
                    acc.add(f"attn_eff_rank_L{li}", _eff_rank(sv_l))
                    Pls = Pl.clamp(min=1e-9)
                    acc.add(f"entropy_L{li}", -(Pls * Pls.log()).sum(dim=-1).mean().item())
                    acc.add(f"sink_L{li}", Pl[:, :, 0].mean().item())

            min_dist = 64
            n_pairs = min(64, T0 - min_dist)
            if n_pairs > 0:
                pi = torch.randint(min_dist, T0, (n_pairs,), device=device)
                pj = pi - min_dist
                acc.add("token_dist", (hf_last[pi] - hf_last[pj]).norm(dim=-1).mean().item())

            res_final = (hf_last - hf_last.mean(dim=0, keepdim=True)).norm().item() / (h0_norm + 1e-9)
            if T0 > 128 and math.isfinite(res_final) and res_final < 200.0:
                inject = T0 // 4
                ids_p = ids0.clone()
                ids_p[inject] = torch.randint(0, VOCAB_SIZE, (1,), device=device).item()
                hp = fwd(model, ids_p, cu0, device)
                fs = inject + 64
                if T0 > fs:
                    base_last = hf_last[fs:T0].norm(dim=-1).mean().item()
                    raw_last = (hp[-1].float()[fs:T0] - hf_last[fs:T0]).norm(dim=-1).mean().item()
                    acc.add("noise_norm", raw_last)
                    acc.add("noise_rel", raw_last / (base_last + 1e-9))
                    for li, (ho, hpp) in enumerate(zip(hiddens[1:], hp[1:])):
                        hof = ho.float()[fs:T0]
                        raw_l = (hpp.float()[fs:T0] - hof).norm(dim=-1).mean().item()
                        acc.add(f"noise_L{li}", raw_l)
                        acc.add(f"noise_rel_L{li}", raw_l / (hof.norm(dim=-1).mean().item() + 1e-9))

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t_batch
        tag = "heavy" if bi < HEAVY_BATCHES else "ppl  "
        print(f"[{kind}] batch {bi + 1}/{max_batches} ({tag}) | {dt:5.2f}s/batch | "
              f"elapsed {time.time() - t_start:6.1f}s", flush=True)

    if "val_nll" in acc.store:
        tot_nll = sum(a * b for a, b in zip(acc.store["val_nll"], acc.store["val_tokens"]))
        tot_tok = sum(acc.store["val_tokens"])
        acc.store["_val_ppl_token_weighted"] = [math.exp(min(tot_nll / max(1, tot_tok), 20.0))]


def _resolve_models(device):
    """MODELS as a list of (kind, checkpoint path), skipping None and resolving the
    dense checkpoint when its path is "AUTO".
    """
    resolved = []
    for kind, path in MODELS:
        if path is None:
            continue
        if kind == "dense" and path == "AUTO":
            path = _resolve_dense_best()
        else:
            p = _abs(path)
            if not p.exists():
                raise FileNotFoundError(f"{kind} checkpoint not found: {p}")
        resolved.append((kind, path))
    if not resolved:
        raise RuntimeError("no models to evaluate; set at least one path in MODELS")
    return resolved


def main():
    device = _device()
    amp = _amp_dtype(device)
    torch.manual_seed(42)

    _, val_loader, _ = make_loaders(limit=None)

    models = _resolve_models(device)
    ref_kind = models[0][0]

    results = {}
    for kind, path in models:
        if kind == "dsalt":
            model, k = load_dsalt(path, device)
        else:
            model, k = load_iso(kind, path, device)
        acc = WelfordSet()
        torch.manual_seed(42)
        eval_model(model, k, val_loader, device, amp, MAX_BATCHES, acc)
        results[kind] = acc.summary()
        if "_val_ppl_token_weighted" in acc.store:
            results[kind]["val_ppl"] = {"mean": acc.store["_val_ppl_token_weighted"][0], "std": 0.0, "n": 1}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(OUT_PATH).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved → {OUT_PATH}\n")

    kinds = [k for k, _ in models]

    def _row(key, getf):
        cells = "".join(f"{getf(k):>16.4f}" for k in kinds)
        print(f"{key:24s}{cells}")

    def _hdr():
        cells = "".join(f"{k.upper():>16s}" for k in kinds)
        print(f"{'metric':24s}{cells}")
        print("-" * (24 + 16 * len(kinds)))

    def _scalar(k, key):
        return results[k].get(key, {}).get("mean", float("nan"))

    keys = ["val_ppl", "hidden_eff_rank_final", "attn_eff_rank", "sigma2",
            "attn_entropy", "head_spec_std", "attn_sink", "noise_norm",
            "noise_rel", "token_dist"]
    _hdr()
    for key in keys:
        _row(key, lambda k, key=key: _scalar(k, key))

    def _per_layer(prefix, title):
        n = sum(1 for kk in results[ref_kind] if kk.startswith(prefix))
        if n == 0:
            return
        print(f"\n=== {title} ===")
        head = "".join(f"{k.upper():>12s}" for k in kinds)
        print(f"{'layer':8s}{head}")
        for li in range(n):
            cells = "".join(
                f"{results[k].get(f'{prefix}{li}', {}).get('mean', float('nan')):12.4f}"
                for k in kinds)
            print(f"L{li:<7d}{cells}")

    _per_layer("dobrushin_L", "Dobrushin rho per layer (1 = the layer does not contract)")
    _per_layer("hidden_eff_rank_L", "Hidden eff_rank per layer (T x d)")
    _per_layer("noise_rel_L", "Relative noise per layer (perturbation / hidden norm)")


if __name__ == "__main__":
    main()
