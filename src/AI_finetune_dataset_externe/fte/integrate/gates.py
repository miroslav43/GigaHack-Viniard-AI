"""Acceptance gates of an fte run vs its base run: example-tile metrics and whole-set invariants.

    python -m fte.integrate.gates --run <out-run> --base complete-v4 [--cover-override]

Writes work/reports/gates_<run>.{json,md}; exit code 1 when any gate fails.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import shapely

from fte.paths import AI_RUNS, EXAMPLE_TILES, WORK

REPORTS: Final = WORK / "reports"
CANOPY_MEAN_MIN: Final = 0.875
CANOPY_TILE_TOL: Final = 0.002
NOT_WORSE_KEYS: Final = ("attributes.score", "rows.f1", "interrow.iou", "interrow.f1", "grouping.f1")
UNION_REL_MAX: Final = 1e-3
UNION_ABS_OK_M2: Final = 0.02  # tiny tiles: a few merge-fix bridges are always acceptable
COUNT_RATIO: Final = (0.8, 1.5)
COUNT_RATIO_MIN_BASE: Final = 20  # the ratio of a 4-canopy tile is noise (4 -> 3 = 0.75)
WASTE_PER_TILE_MAX: Final = 5
WASTE_TOTAL_MAX: Final = 400
EPS: Final = 1e-9

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    value: Any
    threshold: Any
    detail: str = ""


def _eval_doc(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics" / "eval_examples.json"
    if not path.is_file():
        raise FileNotFoundError(f"eval report missing: {path} (run --eval first)")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_gates(base: dict[str, Any], out: dict[str, Any]) -> list[Gate]:
    mean = out["mean"].get("canopy.score")
    gates = [Gate("canopy.mean", mean is not None and mean >= CANOPY_MEAN_MIN, mean, CANOPY_MEAN_MIN)]
    for tile in EXAMPLE_TILES:
        b, o = base["per_tile"][tile]["canopy.score"], out["per_tile"][tile]["canopy.score"]
        gates.append(Gate(f"canopy.tile.{tile}", o >= b - CANOPY_TILE_TOL - EPS, o, round(b - CANOPY_TILE_TOL, 6)))
    for key in NOT_WORSE_KEYS:
        b, o = base["mean"].get(key), out["mean"].get(key)
        ok = b is None or (o is not None and o >= b - EPS)
        gates.append(Gate(f"not_worse.{key}", ok, o, b))
    return gates


def _union_areas(canopies: Any) -> pd.Series:
    groups = canopies.groupby("tile_id").geometry
    return groups.apply(lambda g: shapely.union_all(g.values).area)


def canopy_invariants(base: Any, out: Any) -> list[Gate]:
    ub, uo = _union_areas(base.canopies), _union_areas(out.canopies)
    tiles = sorted(set(ub.index) | set(uo.index))
    delta = {t: abs(uo.get(t, 0.0) - ub.get(t, 0.0)) for t in tiles}
    rel = {t: (0.0 if delta[t] < UNION_ABS_OK_M2 else delta[t] / max(ub.get(t, 0.0), EPS)) for t in tiles}
    worst = max(rel, key=rel.get) if rel else ""
    new_tiles = sorted(set(uo.index) - set(ub.index))
    nb, no = base.canopies["tile_id"].value_counts(), out.canopies["tile_id"].value_counts()
    ratio = {t: no.get(t, 0) / nb[t] for t in nb.index if nb[t] >= COUNT_RATIO_MIN_BASE}
    bad_ratio = sorted(t for t, r in ratio.items() if not COUNT_RATIO[0] <= r <= COUNT_RATIO[1])
    return [
        Gate("canopy.union_rel_delta.max", rel.get(worst, 0.0) < UNION_REL_MAX, rel.get(worst, 0.0), UNION_REL_MAX,
             f"{worst} (abs {delta.get(worst, 0.0):.4f} m2; < {UNION_ABS_OK_M2} m2 always ok)"),
        Gate("canopy.no_new_tiles", not new_tiles, len(new_tiles), 0, ",".join(new_tiles[:5])),
        Gate("canopy.count_ratio", not bad_ratio, [round(min(ratio.values()), 3), round(max(ratio.values()), 3)]
             if ratio else None, list(COUNT_RATIO), ",".join(bad_ratio[:5])),
    ]


def _layer_equal(a: Any, b: Any, key: str, attrs: Sequence[str]) -> tuple[bool, int]:
    if len(a) != len(b):
        return False, abs(len(a) - len(b))
    sa, sb = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    if not (sa[key] == sb[key]).all():
        return False, int((sa[key] != sb[key]).sum())
    geom_diff = ~np.asarray(shapely.equals_exact(sa.geometry.values, sb.geometry.values, 1e-9))
    attr_diff = np.zeros(len(sa), dtype=bool)
    for col in attrs:
        attr_diff |= (sa[col].astype(str) != sb[col].astype(str)).to_numpy()
    n = int((geom_diff | attr_diff).sum())
    return n == 0, n


def structure_invariants(base: Any, out: Any, cover_override: bool) -> list[Gate]:
    rows_ok, rows_n = _layer_equal(base.row_pieces, out.row_pieces, "piece_id", ("row_id", "row_structure"))
    ir_attrs = () if cover_override else ("interrow_cover",)
    ir_ok, ir_n = _layer_equal(base.interrow_pieces, out.interrow_pieces, "piece_id", ir_attrs)
    gates = [Gate("rows.unchanged", rows_ok, rows_n, 0), Gate("interrows.unchanged", ir_ok, ir_n, 0)]
    if cover_override and len(base.interrow_pieces) == len(out.interrow_pieces):
        sb = base.interrow_pieces.sort_values("piece_id")["interrow_cover"].to_numpy()
        so = out.interrow_pieces.sort_values("piece_id")["interrow_cover"].to_numpy()
        frac = float((sb != so).mean()) if len(sb) else 0.0
        gates.append(Gate("interrows.cover_changed_frac", frac <= 0.15, round(frac, 4), 0.15))
    return gates


def waste_invariants(out: Any) -> list[Gate]:
    per_tile = out.waste["tile_id"].value_counts()
    on_examples = int(out.waste["tile_id"].isin(EXAMPLE_TILES).sum())
    worst = int(per_tile.max()) if len(per_tile) else 0
    return [
        Gate("waste.examples_zero", on_examples == 0, on_examples, 0),
        Gate("waste.per_tile_max", worst <= WASTE_PER_TILE_MAX, worst, WASTE_PER_TILE_MAX),
        Gate("waste.total_max", len(out.waste) <= WASTE_TOTAL_MAX, len(out.waste), WASTE_TOTAL_MAX),
    ]


def evaluate_gates(run_dir: Path, base_dir: Path, *, cover_override: bool, with_metrics: bool = True) -> list[Gate]:
    from vineyard.annset.io import read_annset

    base, out = read_annset(base_dir / "annset"), read_annset(run_dir / "annset")
    gates = metric_gates(_eval_doc(base_dir), _eval_doc(run_dir)) if with_metrics else []
    return gates + canopy_invariants(base, out) + structure_invariants(base, out, cover_override) + \
        waste_invariants(out)


def render_md(run: str, base: str, gates: list[Gate]) -> str:
    lines = [f"# Gates: {run} vs {base}", "", "| gate | status | value | threshold | detail |", "|---|---|---|---|---|"]
    lines += [f"| {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.value} | {g.threshold} | {g.detail} |"
              for g in gates]
    verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
    return "\n".join([*lines, "", f"**Overall: {verdict}**", ""])


def write_gates(run: str, base: str, gates: list[Gate], out_dir: Path = REPORTS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"run": run, "base": base, "passed": all(g.passed for g in gates), "gates": [asdict(g) for g in gates]}
    path = out_dir / f"gates_{run}.json"
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (out_dir / f"gates_{run}.md").write_text(render_md(run, base, gates), encoding="utf-8")
    return path


def run_gates(run: str, base: str, *, cover_override: bool, with_metrics: bool = True) -> bool:
    gates = evaluate_gates(AI_RUNS / run, AI_RUNS / base, cover_override=cover_override, with_metrics=with_metrics)
    path = write_gates(run, base, gates)
    for g in gates:
        _log.info("%s %s value=%s threshold=%s %s", "PASS" if g.passed else "FAIL", g.name, g.value, g.threshold,
                  g.detail)
    _log.info("gates written to %s", path)
    return all(g.passed for g in gates)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="complete-v4")
    ap.add_argument("--cover-override", action="store_true")
    ap.add_argument("--no-metrics", action="store_true", help="invariants only (no eval report needed)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ok = run_gates(args.run, args.base, cover_override=args.cover_override, with_metrics=not args.no_metrics)
    print("GATES PASS" if ok else "GATES FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
