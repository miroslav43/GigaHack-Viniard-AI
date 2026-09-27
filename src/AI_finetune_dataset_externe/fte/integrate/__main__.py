"""End-to-end integration of the fte edits into a new src/AI run.

    python -m fte.integrate --base-run complete-v4 --out-run complete-v4-fte1 \
        [--waste-csv /abs/waste_confirmed.csv [--waste-mode direct|pipeline]] \
        [--partition '{"mode":"width","alpha":0.35,"min_len_m":2.0}' [--merge '{"max_gap_m":0.11,"max_len_m":3.0}']
         | --partition-best [--partition-only]] [--contact-dir DIR] \
        [--cover-csv cover.csv] [--eval] [--export] [--overwrite]

Steps: copy run -> (pipeline waste re-run) -> annset edits -> eval-examples -> export-cvat + validate -> gates.
`--waste-mode direct` (default) rebuilds the final waste layer from the CSV with vineyard's own merge
(no pipeline re-run). `pipeline` runs `vineyard run --from waste --until assemble`, which re-assembles the
AnnSet from the SHARED tile cache work/cache/{canopy,interrow,row_attrs}: only safe when that cache still
holds the base run's results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from fte.canopy.merge_fix import MergeParams
from fte.canopy.partition import PartitionParams
from fte.integrate.annset_edit import EditSpec, apply_edits
from fte.integrate.gates import run_gates
from fte.integrate.run_copy import copy_run, update_marker
from fte.integrate.vineyard_cli import run_vineyard
from fte.paths import AI_RUNS, WORK

GRID_REPORT: Final = WORK / "reports" / "partition_grid.json"
UPLOAD_GLOB: Final = "exports/marcaj_upload/*.zip"
ALL_TILES_GLOB: Final = "siret3_r*"

_log = logging.getLogger("fte.integrate")


def _canopy_edits(args: argparse.Namespace) -> tuple[PartitionParams | None, MergeParams | None]:
    """(partition, merge) from --partition/--merge JSON or the best row of the grid report."""
    if (args.partition or args.merge) and args.partition_best:
        raise SystemExit("--partition/--merge and --partition-best are exclusive")
    if args.partition_best:
        if not GRID_REPORT.is_file():
            raise SystemExit(f"{GRID_REPORT} missing: run python -m fte.canopy.partition_eval first")
        doc = json.loads(GRID_REPORT.read_text(encoding="utf-8"))
        best = doc["best_partition_only"] if args.partition_only else doc["best"]
        part, merge = best.get("params"), best.get("merge")
        return (PartitionParams.from_mapping(part) if part else None,
                MergeParams(**merge) if merge else None)
    part = PartitionParams.from_json(args.partition) if args.partition else None
    merge = MergeParams(**json.loads(args.merge)) if args.merge else None
    return part, merge


def _abs_file(path: Path | None, what: str) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{what} not found: {resolved}")
    return resolved


def _restore_link(link: Path, target: str | None) -> None:
    if target is None or not link.is_symlink() or os.readlink(link) == target:
        return
    tmp = link.with_name(link.name + ".fte-restore")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp, target_is_directory=True)
    os.replace(tmp, link)
    _log.warning("restored %s -> %s", link, target)


def _pipeline_waste(out_run: str, csv_path: Path, log_path: Path) -> None:
    """`vineyard run --from waste --until assemble` on the copy; LATEST_MODEL is left untouched."""
    _log.warning("pipeline waste re-run assembles from the shared tile cache (see module doc)")
    link = AI_RUNS / "LATEST_MODEL"
    before = os.readlink(link) if link.is_symlink() else None
    try:
        # a (match-all) tile filter keeps the runner from repointing LATEST_MODEL
        run_vineyard(["run", "--run-id", out_run, "--from", "waste", "--until", "assemble",
                      "--tiles", ALL_TILES_GLOB,
                      "--set", f"paths.waste_confirmed={json.dumps(str(csv_path))}"], log_path=log_path)
    finally:
        _restore_link(link, before)


def _export(run_dir: Path, log_path: Path) -> list[str]:
    run_vineyard(["export-cvat", "--annset", str(run_dir)], log_path=log_path)
    zips = sorted(str(p) for p in run_dir.glob(UPLOAD_GLOB))
    if not zips:
        raise SystemExit(f"no upload ZIPs under {run_dir / 'exports'}")
    run_vineyard(["cvat", "validate", *zips], log_path=log_path)
    return zips


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m fte.integrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-run", default="complete-v4")
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--overwrite", action="store_true", help="replace an out-run made by fte.integrate")
    ap.add_argument("--waste-csv", type=Path)
    ap.add_argument("--waste-mode", choices=("direct", "pipeline"), default="direct")
    ap.add_argument("--partition", help="PartitionParams as JSON")
    ap.add_argument("--merge", help="MergeParams as JSON (merge-fix after the partition; default off)")
    ap.add_argument("--partition-best", action="store_true", help="best params of work/reports/partition_grid.json")
    ap.add_argument("--partition-only", action="store_true", help="with --partition-best: ignore the merge-fix")
    ap.add_argument("--contact-dir", type=Path)
    ap.add_argument("--cover-csv", type=Path)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--no-gates", action="store_true")
    return ap


def run(args: argparse.Namespace) -> int:
    partition, merge = _canopy_edits(args)
    waste_csv, cover_csv = _abs_file(args.waste_csv, "waste CSV"), _abs_file(args.cover_csv, "cover CSV")
    run_dir = copy_run(args.base_run, args.out_run, overwrite=args.overwrite)
    log_path = run_dir / "logs" / "fte_integrate.log"
    if waste_csv is not None and args.waste_mode == "pipeline":
        _pipeline_waste(args.out_run, waste_csv, log_path)
    spec = EditSpec(partition=partition, merge=merge, contact_dir=args.contact_dir, cover_csv=cover_csv,
                    waste_csv=waste_csv if args.waste_mode == "direct" else None)
    summary = apply_edits(run_dir, spec)
    update_marker(run_dir, partition=partition.to_dict() if partition else None,
                  merge=merge.to_dict() if merge else None, waste_csv=str(waste_csv or ""),
                  waste_mode=args.waste_mode, cover_csv=str(cover_csv or ""),
                  contact_dir=str(args.contact_dir or ""), edits=summary.to_dict())
    print(f"edited {run_dir}: {summary.to_dict()}")
    if args.eval:
        run_vineyard(["eval-examples", "--annset", str(run_dir)], log_path=log_path)
        print((run_dir / "metrics" / "eval_examples.md").read_text(encoding="utf-8").split("\n## Gates")[0][:1500])
    if args.export:
        zips = _export(run_dir, log_path)
        print(f"exported + validated {len(zips)} ZIPs")
    if args.no_gates:
        return 0
    ok = run_gates(args.out_run, args.base_run, cover_override=cover_csv is not None, with_metrics=args.eval)
    print("GATES PASS" if ok else "GATES FAIL")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
