"""Add depth-aggregated statistics to existing paper_eval_results*.json files.

``paper_eval.py`` reports headline scalars such as ``attn_eff_rank`` from the last
layer only. The per-layer series are already stored in the JSON, so the mean over
layers can be computed offline: no model is loaded and no evaluation is re-run.

For each model and each per-layer prefix ``P`` (attn_eff_rank, dobrushin, entropy,
hidden_eff_rank, noise, noise_rel, res_norm, sink) it writes::

    "<P>_depth_mean": {"mean", "std", "n", "n_layers", "aggregate", "source"}

where ``mean`` is the mean over layers of the per-layer means, ``std`` the std of
those means across layers (not batch noise), ``n`` the per-layer batch count when it
is the same for every layer; ``median``, ``min`` and ``max`` are stored too.

Existing keys are kept unless --force is given, and a file is rewritten only when
something changed.

Usage::

    python -m results.add_depth_aggregates                 # dry run, prints a report
    python -m results.add_depth_aggregates --write         # write into the JSONs
    python -m results.add_depth_aggregates --write --force # recompute existing keys
    python -m results.add_depth_aggregates --glob "**/paper_eval_results*.json"

Files are matched relative to the repository root.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

# A metric key is per-layer when it looks like "<prefix>_L<digits>".
LAYER_RX = re.compile(r"^(?P<prefix>.+)_L(?P<idx>\d+)$")
SUFFIX = "_depth_mean"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _collect_series(kind_block: dict) -> dict[str, list[tuple[int, dict]]]:
    """Group per-layer metric entries by prefix -> [(layer_index, entry), ...]."""
    series: dict[str, list[tuple[int, dict]]] = {}
    for key, val in kind_block.items():
        if key.endswith(SUFFIX) or not isinstance(val, dict) or "mean" not in val:
            continue
        m = LAYER_RX.match(key)
        if not m:
            continue
        series.setdefault(m.group("prefix"), []).append((int(m.group("idx")), val))
    for prefix in series:
        series[prefix].sort(key=lambda t: t[0])
    return series


def _aggregate(entries: list[tuple[int, dict]], prefix: str) -> dict | None:
    """Depth statistics over the per-layer means. None when nothing is usable."""
    idxs = [i for i, _ in entries]
    means = [e["mean"] for _, e in entries
             if isinstance(e.get("mean"), (int, float)) and math.isfinite(e["mean"])]
    if not means:
        return None

    n_layers = len(means)
    mu = sum(means) / n_layers
    # Population std across layers: the spread of the depth profile, not a sampling
    # error.
    var = sum((v - mu) ** 2 for v in means) / n_layers
    ordered = sorted(means)
    mid = n_layers // 2
    median = ordered[mid] if n_layers % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])

    # Carry the batch count through only when every layer agrees on it.
    ns = {e.get("n") for _, e in entries if isinstance(e.get("n"), int)}
    n = ns.pop() if len(ns) == 1 else None

    # Layers should be contiguous 0..n-1; gaps are flagged.
    contiguous = idxs == list(range(len(idxs)))

    agg = {
        "mean": mu,
        "std": math.sqrt(var),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "n_layers": n_layers,
        "aggregate": "mean over layers of per-layer means; std is across layers",
        "source": f"{prefix}_L0..{prefix}_L{idxs[-1]}",
    }
    if n is not None:
        agg["n"] = n
    if not contiguous:
        agg["warning"] = f"non-contiguous layer indices: {idxs}"
    if len(means) != len(entries):
        agg["warning_nonfinite"] = f"{len(entries) - len(means)} layer(s) dropped as non-finite"
    return agg


def process_file(path: Path, force: bool) -> tuple[dict, list[str]]:
    """Return (updated_document, list_of_added_keys). Does not touch disk."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    added: list[str] = []

    for kind, block in doc.items():
        if not isinstance(block, dict):
            continue
        for prefix, entries in _collect_series(block).items():
            out_key = prefix + SUFFIX
            if out_key in block and not force:
                continue
            agg = _aggregate(entries, prefix)
            if agg is None:
                continue
            block[out_key] = agg
            added.append(f"{kind}/{out_key}")
    return doc, added


def _fmt(v: float) -> str:
    return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"


def report(path: Path, doc: dict, focus: str) -> None:
    """Print last-layer vs depth-mean for one metric, plus pairwise deltas."""
    kinds = [k for k, v in doc.items() if isinstance(v, dict) and focus + SUFFIX in v]
    if not kinds:
        return
    print(f"\n  {focus}:  last-layer  ->  depth-mean  (n_layers)")
    vals: dict[str, tuple[float, float]] = {}
    for k in kinds:
        depth = doc[k][focus + SUFFIX]
        last = doc[k].get(focus, {}).get("mean")
        vals[k] = (last, depth["mean"])
        last_s = _fmt(last) if isinstance(last, (int, float)) else "n/a"
        print(f"    {k:<8s} {last_s:>10s}  ->  {_fmt(depth['mean']):>10s}"
              f"   (+/-{_fmt(depth['std'])} across {depth['n_layers']} layers)")
    if "dsalt" in vals and "dense" in vals:
        (l_ds, d_ds), (l_de, d_de) = vals["dsalt"], vals["dense"]
        if all(isinstance(x, (int, float)) and x for x in (l_ds, l_de, d_ds, d_de)):
            print(f"    dsalt vs dense: last-layer {100 * (l_ds / l_de - 1):+.1f}%"
                  f"   depth-mean {100 * (d_ds / d_de - 1):+.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="**/paper_eval_results*.json",
                    help="glob relative to the repo root")
    ap.add_argument("--write", action="store_true",
                    help="persist changes (default is a dry run)")
    ap.add_argument("--force", action="store_true",
                    help="recompute aggregates that already exist")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copy written before the first modification")
    ap.add_argument("--focus", default="attn_eff_rank",
                    help="metric shown in the comparison report")
    args = ap.parse_args()

    root = _repo_root()
    paths = sorted(p for p in root.glob(args.glob) if p.is_file())
    if not paths:
        print(f"no files matched {args.glob!r} under {root}")
        return 1

    total = 0
    for path in paths:
        rel = path.relative_to(root)
        try:
            doc, added = process_file(path, args.force)
        except json.JSONDecodeError as e:
            print(f"[skip] {rel}: invalid JSON ({e})")
            continue

        print(f"\n===== {rel}")
        if not added:
            print("  nothing to add (aggregates already present; use --force to recompute)")
        else:
            print(f"  {len(added)} aggregate key(s): "
                  f"{', '.join(sorted({a.split('/')[1] for a in added}))}")
        report(path, doc, args.focus)

        if added and args.write:
            if not args.no_backup:
                backup = path.with_suffix(path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(path, backup)
            # allow_nan=False keeps the file valid JSON, as in paper_eval.py.
            path.write_text(json.dumps(doc, indent=2, allow_nan=False),
                            encoding="utf-8")
            print(f"  written -> {rel}")
        total += len(added)

    if not args.write:
        print(f"\nDRY RUN: {total} key(s) would be added. Re-run with --write to persist.")
    else:
        print(f"\nDone: {total} key(s) added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
