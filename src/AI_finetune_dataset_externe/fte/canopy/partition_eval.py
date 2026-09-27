"""Grid evaluation of the canopy partition on the 2 example tiles, scored exactly as `vineyard eval-examples`.

    python -m fte.canopy.partition_eval [--base-run complete-v4] [--contact-dir work/preds/canopy/v1/c1]

Writes work/reports/partition_grid.{json,md}; prints the best config (mean canopy.score, every tile
>= base - tile_tol).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import shapely

from fte.canopy.merge_fix import MergeParams, merge_fragments
from fte.canopy.partition import ContactFn, PartitionParams, partition_canopies, png_contact_loader
from fte.paths import AI_RUNS, AI_WORK, BASE_RUN, EXAMPLE_TILES, PREDS, WORK

REPORTS: Final = WORK / "reports"
DEFAULT_CONTACT_DIR: Final = PREDS / "canopy" / "v1" / "c1"
REFERENCE_REF: Final = "LATEST_REFERENCE"
WIDTH_ALPHAS: Final = (0.25, 0.30, 0.35, 0.40, 0.45, 0.55)
NN_ALPHAS: Final = (0.35, 0.45, 0.55)
T_CS: Final = (0.4, 0.5, 0.6)
MIN_LENS: Final = (1.6, 2.0)
MERGE_GAPS: Final = (0.08, 0.11, 0.14)
MERGE_LENS: Final = (2.5, 3.0)
TOP_K_FOR_MERGE: Final = 3
TILE_TOL: Final = 0.002
METRICS: Final = ("canopy.score", "canopy.iou", "canopy.f1", "canopy.tp", "canopy.n_pred", "canopy.n_ref")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExampleData:
    base: Any  # AnnSet restricted to the example tiles
    ref: Any  # reference AnnSet
    eval_cfg: Any  # vineyard EvalConfig


def load_examples(base_run: str = BASE_RUN, tiles: Sequence[str] = EXAMPLE_TILES) -> ExampleData:
    from vineyard.annset.io import read_annset, resolve_run_dir
    from vineyard.config.loader import load_config

    base = read_annset(resolve_run_dir(AI_WORK, base_run) / "annset").for_tiles(tiles)
    ref = read_annset(resolve_run_dir(AI_WORK, REFERENCE_REF) / "annset")
    return ExampleData(base, ref, load_config().eval)


def _union_delta(before: Any, after: Any) -> dict[str, float]:
    out = {}
    for tile_id in sorted(set(before["tile_id"])):
        a = shapely.union_all(before[before["tile_id"] == tile_id].geometry.values).area
        b = shapely.union_all(after[after["tile_id"] == tile_id].geometry.values).area
        out[tile_id] = float(abs(b - a))
    return out


def score_canopies(data: ExampleData, canopies: Any, label: str) -> dict[str, Any]:
    """Official per-tile canopy metrics of `data.base` with its canopies replaced by `canopies`."""
    from vineyard.eval.report import evaluate_annsets

    pred = data.base.with_layer("canopies", canopies)
    report = evaluate_annsets(pred, data.ref, list(data.base.meta.tile_ids), data.eval_cfg)
    per_tile = {t: {k: report.per_tile[t].get(k) for k in METRICS} for t in report.tiles}
    return {"label": label, "per_tile": per_tile, "mean_score": report.mean.get("canopy.score"),
            "union_delta_m2": _union_delta(data.base.canopies, canopies)}


def grid_params(has_contact: bool) -> list[PartitionParams]:
    grid = [PartitionParams(mode="width", alpha=a, min_len_m=m) for a in WIDTH_ALPHAS for m in MIN_LENS]
    if has_contact:
        grid += [PartitionParams(mode="nn", t_c=t) for t in T_CS]
        grid += [PartitionParams(mode="width+nn", alpha=a, t_c=t) for a in NN_ALPHAS for t in T_CS]
    return grid


def _label(p: PartitionParams) -> str:
    alpha = "" if p.mode == "nn" else f" alpha={p.alpha}"
    t_c = "" if p.mode == "width" else f" t_c={p.t_c}"
    return f"{p.mode}{alpha}{t_c} min_len={p.min_len_m}"


def run_grid(data: ExampleData, contact: ContactFn | None, has_contact: bool) -> list[dict[str, Any]]:
    rows = [score_canopies(data, data.base.canopies, "identity") | {"params": None, "merge": None}]
    for params in grid_params(has_contact):
        t0 = time.monotonic()
        cut = partition_canopies(data.base.canopies, data.base.row_pieces, params, contact)
        row = score_canopies(data, cut, _label(params)) | {"params": params.to_dict(), "merge": None}
        rows.append(row | {"seconds": round(time.monotonic() - t0, 2)})
        _log.info("%s -> mean %.4f", row["label"], row["mean_score"])
    return rows


def run_merge_grid(data: ExampleData, rows: list[dict[str, Any]], contact: ContactFn | None) -> list[dict[str, Any]]:
    """Merge-fix after the TOP_K_FOR_MERGE best partitions (and after no partition)."""
    ranked = sorted((r for r in rows[1:]), key=lambda r: -r["mean_score"])[:TOP_K_FOR_MERGE]
    out = []
    for prow in [rows[0], *ranked]:
        params = PartitionParams.from_mapping(prow["params"]) if prow["params"] else None
        cut = (partition_canopies(data.base.canopies, data.base.row_pieces, params, contact) if params
               else data.base.canopies)
        for gap in MERGE_GAPS:
            for length in MERGE_LENS:
                mp = MergeParams(max_gap_m=gap, max_len_m=length)
                merged = merge_fragments(cut, data.base.row_pieces, mp, contact)
                label = f"{prow['label']} + merge gap={gap} len={length}"
                out.append(score_canopies(data, merged, label) | {"params": prow["params"], "merge": mp.to_dict()})
                _log.info("%s -> mean %.4f", label, out[-1]["mean_score"])
    return out


def pick_best(rows: list[dict[str, Any]], tol: float = TILE_TOL) -> dict[str, Any]:
    base = rows[0]["per_tile"]

    def ok(row: dict[str, Any]) -> bool:
        return all(row["per_tile"][t]["canopy.score"] >= base[t]["canopy.score"] - tol for t in base)

    return max((r for r in rows if ok(r)), key=lambda r: r["mean_score"])


def render_md(rows: list[dict[str, Any]], best: dict[str, Any], best_partition: dict[str, Any]) -> str:
    tiles = sorted(rows[0]["per_tile"])
    head = ["config", *[f"{t} score/iou/f1/n_pred" for t in tiles], "mean score", "max union delta m2"]
    lines = ["# Partition grid (example tiles)", "", "| " + " | ".join(head) + " |",
             "|---" * len(head) + "|"]
    for r in rows:
        cells = [r["label"]]
        for t in tiles:
            m = r["per_tile"][t]
            cells.append(f"{m['canopy.score']:.4f} / {m['canopy.iou']:.4f} / {m['canopy.f1']:.4f} / "
                         f"{m['canopy.n_pred']}")
        cells += [f"{r['mean_score']:.4f}", f"{max(r['union_delta_m2'].values()):.2e}"]
        lines.append("| " + " | ".join(cells) + " |")
    for name, row in (("Best partition-only", best_partition), ("Best overall", best)):
        lines += ["", f"**{name}:** `{row['label']}` mean {row['mean_score']:.4f}", "", "```json",
                  json.dumps({"partition": row["params"], "merge": row["merge"]}), "```"]
    lines.append("")
    return "\n".join(lines)


def _has_example_maps(contact_dir: Path | None) -> bool:
    return contact_dir is not None and all((contact_dir / f"{t}.png").is_file() for t in EXAMPLE_TILES)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-run", default=BASE_RUN)
    ap.add_argument("--contact-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    ap.add_argument("--no-merge", action="store_true", help="skip the merge-fix grid")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    contact_dir = args.contact_dir or (DEFAULT_CONTACT_DIR if _has_example_maps(DEFAULT_CONTACT_DIR) else None)
    has_contact = _has_example_maps(contact_dir)
    if args.contact_dir is not None and not has_contact:
        _log.warning("contact maps for the example tiles missing in %s: nn modes skipped", args.contact_dir)
    if not (AI_RUNS / args.base_run).exists() and "/" not in args.base_run:
        raise SystemExit(f"base run not found: {args.base_run}")
    data = load_examples(args.base_run)
    contact = png_contact_loader(contact_dir) if has_contact else None
    rows = run_grid(data, contact, has_contact)
    best_partition = pick_best(rows)
    rows = rows + ([] if args.no_merge else run_merge_grid(data, rows, contact))
    best = pick_best(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"base_run": args.base_run, "contact_dir": str(contact_dir) if has_contact else None,
           "rows": rows, "best": best, "best_partition_only": best_partition}
    (args.out_dir / "partition_grid.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (args.out_dir / "partition_grid.md").write_text(render_md(rows, best, best_partition), encoding="utf-8")
    for name, row in (("best partition-only", best_partition), ("best overall", best)):
        print(f"{name}: {row['label']} mean={row['mean_score']:.4f}")
        for t, m in row["per_tile"].items():
            print(f"  {t}: score={m['canopy.score']:.4f} iou={m['canopy.iou']:.4f} f1={m['canopy.f1']:.4f} "
                  f"n_pred={m['canopy.n_pred']}")
        print(f"  --partition '{json.dumps(row['params'])}'" + (f" --merge '{json.dumps(row['merge'])}'"
                                                                if row["merge"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
