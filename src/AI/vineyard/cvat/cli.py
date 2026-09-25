"""`vineyard cvat ...` sub-app: validate upload ZIPs, build the Marcaj format-test ZIP, round-trip the
examples (XML bytes, AnnSet counts/sums, AnnSet -> CVAT vertex deviation). Messages are Romanian.
"""

from __future__ import annotations

import hashlib
import importlib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final

import numpy as np
import typer

from vineyard.cli_options import CommonOptions, fail, load_cli_config, with_config_options
from vineyard.config import AppConfig
from vineyard.contracts.enums import Source
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.cvat.model import CvatDocument
from vineyard.cvat.reader import parse_xml
from vineyard.cvat.testzip import default_test_zip_path, make_test_zip
from vineyard.cvat.to_annset import document_to_annset
from vineyard.cvat.writer import serialize_document
from vineyard.errors import VineyardError
from vineyard.geo.tiling import TILE_PX, existing_tile_ids, tile_ref

EXIT_INVALID: Final = 1
IMAGES_PREFIX: Final = "images/"
TILE_SUFFIX: Final = ".tif"
TILE_INDEX_FILE: Final = "tile_index.parquet"
EXAMPLES_XML: Final = "annotations.xml"
TO_CVAT_MODULE: Final = "vineyard.cvat.to_cvat"
# Oracles measured on the organizers' examples (design 01 §0): counts and vector sums.
EXAMPLE_COUNTS: Final[Mapping[str, tuple[int, int, int]]] = {
    "siret3_r021_c012": (399, 25, 24), "siret3_r006_c004": (251, 26, 25)}
EXAMPLE_SUMS: Final[Mapping[str, tuple[float, float, float]]] = {
    "siret3_r021_c012": (237.119, 910.104, 2068.031), "siret3_r006_c004": (299.056, 1031.451, 1996.361)}
SUM_TOL: Final = 0.01
_HASH_CHUNK: Final = 1 << 20

app = typer.Typer(help="CVAT: validate, make-test-zip, roundtrip-examples.", no_args_is_help=True,
                  add_completion=False)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


# ---------------------------------------------------------------- sha256 of the original tiles


def zip_tile_ids(zip_paths: Sequence[Path]) -> tuple[str, ...]:
    """Tile ids of the `images/<tile>.tif` entries of upload ZIPs (sorted, unique)."""
    ids: set[str] = set()
    for path in zip_paths:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                stem = PurePosixPath(name).name.removesuffix(TILE_SUFFIX)
                if name.startswith(IMAGES_PREFIX) and name.endswith(TILE_SUFFIX) and is_valid_id(IdKind.TILE, stem):
                    ids.add(stem)
    return tuple(sorted(ids))


def _index_shas(cfg: AppConfig) -> dict[str, str]:
    path = Path(cfg.paths.work_dir) / TILE_INDEX_FILE
    if not path.is_file():
        return {}
    from vineyard.geo.vector_io import read_layer

    index = read_layer(path, "tile_index", validate=False)
    return dict(zip(index["tile_id"], index["sha256"], strict=True))


def _zip_member_sha(zf: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with zf.open(name) as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256(cfg: AppConfig, tile_ids: Sequence[str]) -> dict[str, str]:
    """sha256 of the organizers' tiles: tile_index when present, else hashed from the tile ZIPs."""
    known = _index_shas(cfg)
    wanted = {f"{t}{TILE_SUFFIX}": t for t in tile_ids if t not in known}
    for zip_path in sorted(Path(cfg.paths.data_root).glob(cfg.paths.tiles_zip_glob)):
        if not wanted:
            break
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                tile = wanted.pop(PurePosixPath(member).name, None)
                if tile is not None:
                    known[tile] = _zip_member_sha(zf, member)
    return {t: known[t] for t in sorted(tile_ids) if t in known}


# ---------------------------------------------------------------- round-trip of the examples


def _annset_checks(doc: CvatDocument, cfg: AppConfig) -> tuple[list[Check], object]:
    refs = {img.tile_id: tile_ref(img.tile_id) for img in doc.images}
    annset, qa = document_to_annset(doc, tile_refs=refs, source=Source.REFERENCE, run_id="roundtrip",
                                    model_version="reference-examples@roundtrip", cfg=cfg.import_,
                                    canopy_cfg=cfg.canopy)
    checks = [Check("qa_issues_empty", len(qa) == 0, f"{len(qa)} probleme")]
    for tile in sorted(EXAMPLE_COUNTS):
        layers = [annset.canopies, annset.row_pieces, annset.interrow_pieces]
        subs = [g[g["tile_id"] == tile] for g in layers]
        counts = tuple(len(s) for s in subs)
        sums = (subs[0]["area_m2"].sum(), subs[1]["length_m"].sum(), subs[2]["area_m2"].sum())
        sums_ok = all(abs(a - b) <= SUM_TOL for a, b in zip(sums, EXAMPLE_SUMS[tile], strict=True))
        checks.append(Check(f"counts_{tile}", counts == EXAMPLE_COUNTS[tile], f"{counts}"))
        checks.append(Check(f"sums_{tile}", sums_ok, ", ".join(f"{v:.3f}" for v in sums)))
    return checks, annset


def _deviation(a: np.ndarray, b: np.ndarray, closed: bool) -> float:
    """Max vertex deviation up to ring rotation (polygons) or direction (polylines); inf if sizes differ."""
    if a.shape != b.shape:
        return float("inf")
    if not closed:
        return float(min(np.abs(a - b).max(), np.abs(a - b[::-1]).max()))
    return float(min(np.abs(a - np.roll(b, k, axis=0)).max() for k in range(len(b))))


def _to_cvat_check(doc: CvatDocument, annset: object, cfg: AppConfig) -> Check:
    try:
        to_cvat = importlib.import_module(TO_CVAT_MODULE)
    except ImportError as exc:
        return Check("annset_to_cvat_dev_px", False, f"to_cvat indisponibil: {exc}")
    images, _ = to_cvat.annset_to_images(annset, [tile_ref(i.tile_id) for i in doc.images], cfg.export.cvat)
    worst = 0.0
    for img in images:
        orig = doc.image(img.name)
        for label in ("row", "interrow_area", "vineyard", "waste"):
            new = [s for s in img.shapes if s.label == label]
            old = [s for s in orig.shapes if s.label == label]
            if len(new) != len(old):
                return Check("annset_to_cvat_dev_px", False, f"{img.name} {label}: {len(new)} != {len(old)}")
            for s_old, s_new in zip(old, new, strict=True):
                dev = _deviation(np.asarray(s_old.points), np.asarray(s_new.points), s_old.tag == "polygon")
                worst = max(worst, dev)
    limit = cfg.export.cvat.roundtrip_max_dev_px
    return Check("annset_to_cvat_dev_px", worst <= limit, f"max {worst:.3f} px (limită {limit})")


def roundtrip_checks(cfg: AppConfig) -> tuple[Check, ...]:
    """XML bytes identical, AnnSet(reference) counts/sums, AnnSet -> CVAT vertex deviation."""
    xml = (Path(cfg.paths.examples_dir) / EXAMPLES_XML).read_bytes()
    doc, issues = parse_xml(xml, source_name=EXAMPLES_XML)
    same = serialize_document(doc, decimals=cfg.export.cvat.coord_decimals) == xml
    checks = [Check("xml_byte_identical", same and not issues, f"{len(xml)} B, {len(issues)} probleme")]
    annset_checks, annset = _annset_checks(doc, cfg)
    return (*checks, *annset_checks, _to_cvat_check(doc, annset, cfg))


# ---------------------------------------------------------------- commands

ZipsArg = Annotated[list[Path], typer.Argument(help="ZIP-uri de upload de verificat.")]
FullSetOpt = Annotated[bool, typer.Option("--full-set", help="Verifică și setul complet de 311 tile-uri.")]
OutOpt = Annotated[Path | None, typer.Option("--out", help="Calea ZIP-ului (implicit work/exports/marcaj_test).")]
WasteVidOpt = Annotated[str | None, typer.Option(
    "--waste-vid", help="vineyard_id pentru cutia waste (implicit blocul cel mai apropiat).")]


@app.command("validate")
@with_config_options
def validate(opts: CommonOptions, zips: ZipsArg, full_set: FullSetOpt = False) -> None:
    """Verifică ZIP-urile de upload (structură, sha256, XML, geometrie, atribute)."""
    from vineyard.cvat.validator_zip import validate_upload_set, validate_zip

    try:
        cfg = load_cli_config(opts)
        paths = [Path(z) for z in zips]
        shas = expected_sha256(cfg, zip_tile_ids([p for p in paths if p.is_file()]))
        if full_set:
            report = validate_upload_set(paths, expected_tiles=existing_tile_ids(), expected_sha256=shas,
                                         cfg=cfg.export.cvat, tile_px=TILE_PX)
        else:
            reports = [validate_zip(p, expected_sha256=shas, cfg=cfg.export.cvat, tile_px=TILE_PX) for p in paths]
            report = reports[0]
            for other in reports[1:]:
                report = report.merge(other)
    except (VineyardError, zipfile.BadZipFile) as exc:
        raise fail(exc if isinstance(exc, VineyardError) else VineyardError(str(exc))) from exc
    for line in report.summary_lines():
        typer.echo(line)
    typer.echo(f"{report.n_errors} erori, {report.n_warnings} avertismente în {len(zips)} ZIP-uri")
    raise typer.Exit(0 if report.ok else EXIT_INVALID)


@app.command("make-test-zip")
@with_config_options
def make_test_zip_cmd(opts: CommonOptions, out: OutOpt = None, waste_vid: WasteVidOpt = None) -> None:
    """ZIP de test pentru Marcaj: 2 tile-uri exemplu + 1 cutie waste + 1 tile gol."""
    try:
        cfg = load_cli_config(opts)
        path = make_test_zip(out or default_test_zip_path(cfg), cfg, waste_vineyard_id=waste_vid)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"{path} ({path.stat().st_size} B)")


@app.command("roundtrip-examples")
@with_config_options
def roundtrip_examples(opts: CommonOptions) -> None:
    """Exemple: XML identic octet cu octet, AnnSet(reference), AnnSet -> CVAT."""
    try:
        checks = roundtrip_checks(load_cli_config(opts))
    except VineyardError as exc:
        raise fail(exc) from exc
    for check in checks:
        typer.echo(f"{'OK  ' if check.ok else 'EȘEC'} {check.name}: {check.detail}")
    raise typer.Exit(0 if all(c.ok for c in checks) else EXIT_INVALID)
