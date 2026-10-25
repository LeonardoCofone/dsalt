"""Print a paper_eval JSON as columns: headline scalars, per-layer Dobrushin rho, and
the differences between DSALT and each baseline.

The file holds one dict per model (dsalt / dense / swa / spatra), each a
{metric: {"mean", "std", "n"}} summary plus per-layer "dobrushin_L{i}" entries.

Usage:
    python -m results.analyze_paper_eval paper_eval_results_48L_full.json
"""
import sys
import json
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_eval_results_48L_999seed.json")
data = json.loads(path.read_text(encoding="utf-8"))

# Column order: DSALT first, then the baselines present in the file.
ORDER = ["dsalt", "dense", "swa", "spatra"]
HDR = {"dsalt": "DSALT", "dense": "DENSE", "swa": "SWA", "spatra": "SparseT"}
tags = [t for t in ORDER if t in data]
if not tags:
    sys.exit(f"no known model keys in {path}; found: {list(data)}")
ref = tags[0]


def val(tag, key):
    v = data[tag].get(key)
    if isinstance(v, dict):
        return v.get("mean")
    return v


# headline scalars
KEYS = ["val_ppl", "hidden_eff_rank_final", "attn_eff_rank", "attn_entropy",
        "head_spec_std", "attn_sink", "noise_norm", "noise_rel"]

w = 13
print(f"\n=== headline metrics ({path.name}) ===")
print(f"{'metric':22s}" + "".join(f"{HDR[t]:>{w}s}" for t in tags))
print("-" * (22 + w * len(tags)))
for k in KEYS:
    if not any(val(t, k) is not None for t in tags):
        continue
    cells = ""
    for t in tags:
        x = val(t, k)
        cells += f"{'--':>{w}s}" if x is None else f"{x:>{w}.4f}"
    print(f"{k:22s}{cells}")

# Dobrushin rho per layer
n_layers = sum(1 for k in data[ref] if k.startswith("dobrushin_L"))
if n_layers:
    print(f"\n=== Dobrushin rho per layer  (1 = the layer does not contract) ===")
    print(f"{'layer':7s}" + "".join(f"{HDR[t]:>{w}s}" for t in tags))
    print("-" * (7 + w * len(tags)))
    means = {t: [] for t in tags}
    for li in range(n_layers):
        row = f"L{li:<6d}"
        for t in tags:
            x = val(t, f"dobrushin_L{li}")
            if x is not None:
                means[t].append(x)
            row += f"{'--':>{w}s}" if x is None else f"{x:>{w}.4f}"
        print(row)
    print("-" * (7 + w * len(tags)))
    avg = "avg    "
    for t in tags:
        m = sum(means[t]) / len(means[t]) if means[t] else float("nan")
        avg += f"{m:>{w}.4f}"
    print(avg)

# differences between DSALT and each baseline
print(f"\n=== DSALT advantage vs each baseline ===")
for t in tags:
    if t == ref:
        continue
    dp = val(ref, "val_ppl")
    bp = val(t, "val_ppl")
    dr = sum(means[ref]) / len(means[ref]) if n_layers and means[ref] else None
    br = sum(means[t]) / len(means[t]) if n_layers and means.get(t) else None
    line = f"  {HDR[ref]} vs {HDR[t]:8s}"
    if dp is not None and bp is not None:
        line += f" | val_ppl {dp:.3f} vs {bp:.3f}  (Δ {bp - dp:+.3f})"
    if dr is not None and br is not None:
        line += f" | rho {dr:.4f} vs {br:.4f}  (Δ {dr - br:+.4f})"
    print(line)
print()
