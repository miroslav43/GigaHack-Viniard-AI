# Design: Foundation, CVAT I/O and Evaluation (project root `/Users/maleticimiroslav/Vin Gigahack/src/AI/`)

Unless a path is absolute, it is relative to `/Users/maleticimiroslav/Vin Gigahack/src/AI/`.

## 0. Facts measured during exploration (read-only) that shape this design

- **Tile ZIPs.** Each ZIP is flat, with no folders, and every entry is STORED (compress_type 0). Tile counts are 74/71/78/76/12, for 311 unique names. Grid rows run r005–r039 and columns c000–c033. Total tif bytes: 385,142,048. Tile size: 154,398–1,654,532 B.
- **Example tiles are the challenge tiles.** They are byte-identical: r006_c004 is in part1 and r021_c012 is in part2.
- **Example ZIP layout.**
  - `annotations.xml` comes first and is deflated (259,785 → 52,218 B), followed by `images/*.tif` STORED. There are no directory entries.
  - The XML image order is **not sorted**: r021 is id 0, r006 is id 1.
- **Example XML conventions.**
  - Each shape is on one line.
  - Shape order per image: `row` polylines, then `interrow_area`, then `vineyard`.
  - Attributes on each shape element: `label, source, occluded, points, z_order`. `source="manual"` on all 750 shapes.
  - Row IDs have **2 digits** (`V01-R01`).
  - All 699 polygons have a negative shoelace area in px, i.e. they are CCW in UTM.
- **Canopy vertices are all integers, maximum 2047.0, never 2048.0.** That is the raw `findContours` index convention. Interrows and rows do reach 2048.0.
  - This contradicts the claim in contract §1.2 that canopies clipped by the tile edge have vertices exactly on 0.0/2048.0.
  - It matters for canopy IoU (see §8 R1).
- **Reference sums** (used as test oracles):

  | Tile | Canopies (count, Σ area) | Interrows (count, Σ area) | Rows (count, Σ length) |
  |---|---|---|---|
  | r021_c012 | 399, 237.119 m² | 24, 2068.031 m² | 25, 910.104 m |
  | r006_c004 | 251, 299.056 m² | 25, 1996.361 m² | 26, 1031.451 m |

  - Disrupted rows in r006: R06, R07, R08, R09, R23.
  - Ordering rows by n·c (descending) reproduces R01…Rn on both tiles.
- **Packing simulation at ≤ 85,000,000 B.** Greedy packing gives **5 parts** whether the XML is 26, 45 or 80 KB per tile (e.g. 65/63/67/69/47 tiles). Balanced packing gives about 79–82 MB per part. The ~7 parts in arch §4.13 came from the old ~45-tile split, and the count is now derived.
- **Python on this machine.**
  - Homebrew python3.14 has a broken `pyexpat` (dyld symbol error), so stdlib `xml.etree` fails there.
  - python3.12.9 works, and conda python is 3.11 without shapely.
  - Consequence: parse XML with lxml and pin a uv-managed 3.12.

---

## 1. Files

**Project files**

| Path | Responsibility | ~Lines |
|---|---|---|
| `pyproject.toml` | pins, hatchling `packages=["vineyard"]`, uv torch-cpu source on Linux, `[tool.uv] environments=[darwin-arm64, linux]`, pytest/coverage/ruff settings | 110 |
| `.python-version` | `3.12` | 1 |
| `uv.lock`, `requirements.lock` | generated and committed | — |
| `Makefile` | setup, lock, doctor, test, cov, lint, ingest, prep, preannotate, reference, eval, test-zip, export, validate-zips, final, post, publish, web, bench, docker-build, docker-push; `unexport PROJ_DATA PROJ_LIB GDAL_DATA`; thread env vars; all paths quoted (the data folder name contains a space and `&`) | 110 |
| `Dockerfile` | 2-stage uv build (arch §4.15), `CMD ["make","final"]` | 45 |
| `.dockerignore` | excludes `work/ .venv/ *.tif models/ .pytest_cache/` | 8 |
| `.gitignore` | `work/ .pytest_cache/ .coverage* htmlcov/ .ruff_cache/ dist/ *.egg-info/`; the repo root already ignores `models/`, `.venv/`, `*.pt` | 10 |
| `configs/default.yaml` | the single merged config | 240 |
| `configs/local.example.yaml` | override example | 15 |
| `/Users/maleticimiroslav/Vin Gigahack/.github/workflows/ai-tests.yml` | CI: `working-directory: src/AI`, `uv sync --locked`, `pytest -m "not needs_tiles"` | 30 |

**`vineyard/` package — core**

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/__init__.py` | `__version__`; imports `_env` first; no numpy import | 10 |
| `vineyard/_env.py` | thread-env defaults (arch §4.1); drops PROJ_DATA/PROJ_LIB/GDAL_DATA that point into conda (with a warning) | 50 |
| `vineyard/cli.py` | typer app with all contract §8 commands plus `all`, `final`, `from-marcaj`, `doctor`, `config show`; lazy stage and sub-app loading; Romanian messages | 320 |
| `vineyard/cli_options.py` | shared options (`--config --set --run-id --tiles --workers --force --force-all --allow-failures --annset`) → `RunContext` | 90 |
| `vineyard/doctor.py` | environment checks: py 3.12, imports, GDAL/PROJ versions, MPS, conda variables, data paths, free disk | 100 |
| `vineyard/errors.py` | `VineyardError`, `ConfigError`, `SchemaError`, `IngestError`, `StageError`, `ExportBlocked`, `CvatFormatError`, each carrying context | 60 |
| `vineyard/logging_setup.py` | rich console + JSONL handler, `log_event`, `log_failure` | 170 |

**Config** (`vineyard/config/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/config/__init__.py` | re-exports `AppConfig`, `load_config`, `cfg_hash` | 20 |
| `vineyard/config/loader.py` | YAML deep-merge, `--set` overrides, path resolution against the project root | 150 |
| `vineyard/config/hashing.py` | canonical JSON, `cfg_hash`, `cfg_subtree` | 50 |
| `vineyard/config/model.py` | `AppConfig` root (frozen, `extra=forbid`) | 70 |
| `vineyard/config/sections_core.py` | project, paths, grid, runtime, nodata, veg, logging (**mine**) | 160 |
| `vineyard/config/sections_io.py` | export, export.cvat, import, eval (**mine**) | 150 |
| `vineyard/config/sections_perception.py` | rows, orchard, blocks, canopy, interrow, row_structure. I seed it from the docs; the perception subsystem owns it | 180 |
| `vineyard/config/sections_ml.py` | nn, waste. Seeded; NN/waste subsystems own it | 120 |
| `vineyard/config/sections_post.py` | targets, route, measure, web. Seeded; post-Marcaj subsystem owns it | 130 |

**Contracts** (`vineyard/contracts/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/contracts/__init__.py` | `CONTRACT_VERSION = "1.1"` | 10 |
| `vineyard/contracts/enums.py` | enums, `LABEL_GEOMETRY`, `LABEL_ATTRIBUTES`, `ANNSET_LAYER_OF_LABEL`, `QA_CODES` | 110 |
| `vineyard/contracts/ids.py` | regexes, formatters, parsers, strict/relaxed validation | 190 |
| `vineyard/contracts/ordering.py` | angle conventions, canonical normal, R001 ordering, V01 sort key (contract §1.5, §2.4) | 90 |
| `vineyard/contracts/schema_defs.py` | `LAYER_SCHEMAS` table for every layer in contract §2.5 | 320 |
| `vineyard/contracts/schemas.py` | `validate_layer`, `coerce_layer`, `empty_layer` | 200 |
| `vineyard/contracts/tile_grid.txt` | the 311 tile_ids (package data, generated once from the ZIPs) | 311 |

**Geo** (`vineyard/geo/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/geo/tiling.py` | **CONTRACT** grid and pixel↔UTM functions | 160 |
| `vineyard/geo/raster.py` | read_tile, valid_mask, mask↔polygon, rasterize, resize, mask PNG I/O | 260 |
| `vineyard/geo/vector_io.py` | GeoParquet and GeoJSON (32635 with `crs` member / 4326), rounding, atomic writes | 240 |
| `vineyard/geo/ops.py` | clip, orient, make_valid, notch_holes, split_multi, dedupe | 240 |

**AnnSet** (`vineyard/annset/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/annset/model.py` | `AnnSet`, `AnnSetMeta` (frozen) | 120 |
| `vineyard/annset/io.py` | read/write AnnSet, `resolve_run_dir`, LATEST_* symlinks | 160 |

`annset/assemble.py` and `annset/derive.py` belong to other subsystems.

**Pipeline** (`vineyard/pipeline/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/pipeline/registry.py` | `StageSpec`, `STAGE_MODULES` (lazy), stage orders, `load_stage`, `select_stages` | 170 |
| `vineyard/pipeline/cache.py` | `cache_key`, `is_fresh`, `write_key` | 80 |
| `vineyard/pipeline/atomic.py` | atomic write helpers | 80 |
| `vineyard/pipeline/parallel.py` | `parallel_map` (spawn Pool), `init_worker`, `ItemOutcome` | 150 |
| `vineyard/pipeline/context.py` | `RunPaths`, `RunContext`, `make_run_id`, `new_run_context` | 200 |
| `vineyard/pipeline/runner.py` | `run_stages`, `run_tile_stage`, `StageResult`, `RunReport`, run.json, LATEST links, exit codes | 300 |
| `vineyard/pipeline/timings.py` | rusage (CPU, peak RSS normalised for macOS/Linux), `aggregate_timings(jsonl)` → timings.json | 120 |
| `vineyard/pipeline/tile_index.py` | `read_tile_index`, `tile_refs` (joins `tile_valid` when present) | 80 |

**Ingest and tile preparation**

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/ingest/tiles.py` | iterate ZIPs, extract with sha256, verify names, count, tags vs grid | 220 |
| `vineyard/ingest/route_inputs.py` | `02_route` → `in_*` layers plus sanity checks | 140 |
| `vineyard/perception/vegmask.py` | port of prototype `veg_mask` (Lab a*) | 70 |
| `vineyard/pipeline/stages/ingest.py` | STAGE wrapper | 110 |
| `vineyard/pipeline/stages/tile_prep.py` | STAGE: per-tile worker plus collect into `tile_valid`; mask path/load accessors | 200 |
| `vineyard/pipeline/stages/import_reference.py` | STAGE: examples → AnnSet(reference) | 100 |
| `vineyard/pipeline/stages/export_cvat.py` | STAGE: AnnSet(model) → validated upload ZIPs (thin wrapper; the pre-annotation lead may co-own it) | 130 |
| `vineyard/pipeline/stages/evaluate.py` | STAGE `eval`: eval-examples plus regression gate | 140 |

**CVAT I/O** (`vineyard/cvat/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/cvat/template.py` | `META_BLOCK` verbatim, XML header/footer, `LABEL_SPECS` parsed from META | 90 |
| `vineyard/cvat/model.py` | `CvatShape`, `CvatImage`, `CvatDocument` (frozen) | 80 |
| `vineyard/cvat/writer.py` | byte-exact serializer | 170 |
| `vineyard/cvat/reader.py` | tolerant lxml parser for ZIP or XML, multi-file merge | 240 |
| `vineyard/cvat/rle.py` | CVAT `<mask>` RLE → polygon (import only) | 60 |
| `vineyard/cvat/normalize.py` | enum and ID normalization, case collisions | 140 |
| `vineyard/cvat/to_cvat.py` | AnnSet → per-tile CvatImages (geometric cleaning, UTM→px) | 250 |
| `vineyard/cvat/to_annset.py` | CvatDocument → AnnSet plus qa_issues (px→UTM, make_valid, CCW, IDs) | 260 |
| `vineyard/cvat/report.py` | `Issue`, `ValidationReport` | 80 |
| `vineyard/cvat/validator_doc.py` | label/attribute/enum/geometry/overlap checks | 260 |
| `vineyard/cvat/validator_zip.py` | whitelist, name bijection, sha, size, id sequence, meta, global set | 200 |
| `vineyard/cvat/packer.py` | `plan_parts` (greedy count + balanced contiguous split), deterministic `write_upload_zip` | 200 |
| `vineyard/cvat/selfcheck.py` | round-trip written ZIPs vs in-memory document and AnnSet | 120 |
| `vineyard/cvat/export.py` | `export_upload` orchestrator: staging dir, manifests, id_registry, atomic publish | 220 |
| `vineyard/cvat/testzip.py` | Marcaj format-test ZIP (examples + 1 waste box + 1 empty tile) | 90 |
| `vineyard/cvat/cli.py` | sub-app `vineyard cvat validate / make-test-zip / roundtrip-examples` | 110 |

**Evaluation** (`vineyard/eval/`)

| Path | Responsibility | ~Lines |
|---|---|---|
| `vineyard/eval/matching.py` | STRtree candidate pairs, polygon/box IoU, greedy and Hungarian one-to-one matching | 130 |
| `vineyard/eval/metrics.py` | official metrics (canopy, rows, interrow, attributes, grouping, counts, waste, FP penalty) | 300 |
| `vineyard/eval/report.py` | `evaluate_annsets` → `EvalReport` (per tile, mean and pooled), JSON/Markdown, baseline compare, gates | 170 |

**Tests** (`tests/`)

- `tests/conftest.py` (~160 lines): fixtures `cfg`, `data_root`, `examples_xml`, `example_tif`, `tmp_work`, `make_geotiff` (rasterio JPEG GTiff), `make_zip`. Marker `needs_tiles` skips when the 5 tile ZIPs are absent (they are gitignored). Examples and `02_route` are committed, so CI can use them.
- Test files:
  - Contracts and config: `test_tiling.py test_ids.py test_ordering.py test_enums.py test_schemas.py test_config.py test_logging.py`
  - Geo and AnnSet: `test_vector_io.py test_raster.py test_ops.py test_annset_io.py`
  - Pipeline: `test_cache.py test_parallel.py test_registry.py test_runner.py`
  - Ingest and tile prep: `test_ingest.py test_route_inputs.py test_tile_prep.py test_vegmask_parity.py`
  - CVAT: `test_cvat_template.py test_cvat_xml.py test_cvat_reader_marcaj.py test_cvat_normalize.py test_cvat_convert.py test_cvat_validator.py test_cvat_packer.py test_cvat_export.py test_import_reference.py`
  - Evaluation and CLI: `test_metrics.py test_eval_examples.py test_cli.py`
  - Roughly 60–250 lines each.

---

## 2. Public interfaces (**[C]** = CONTRACT, frozen; changing one needs a `CONTRACT_VERSION` bump)

### 2.1 contracts
```python
# contracts/__init__.py [C]
CONTRACT_VERSION: Final[str] = "1.1"   # 1.0 + applied arch-wins resolutions (85,000,000 B, derived parts, spacing 1.8–3.8)

# contracts/enums.py [C]  (StrEnum, lowercase values exactly as contract §2.3)
class Label(StrEnum): vineyard, waste, row, interrow_area
class RowStructure(StrEnum): regular, disrupted, unassessable
class InterrowCover(StrEnum): bare_soil, vegetation, mixed, unassessable
class Source(StrEnum): model, marcaj, reference
class TargetKind(StrEnum): row_gap, row_end_short, missing_row, waste, other   # + see §8 R9
class TileStatus(StrEnum): ok, empty_nodata, no_vineyard, failed
class Severity(StrEnum): error, warning, info
class EdgeKind(StrEnum): interrow_centerline, passage_centerline, connector, target_spur
class GeomKind(StrEnum): polygon, polyline, box
LABEL_GEOMETRY: Mapping[Label, GeomKind]
LABEL_ATTRIBUTES: Mapping[Label, tuple[str, ...]]   # ordered: row=(vineyard_id,row_id,row_structure) ...
ANNSET_LAYER_OF_LABEL: Mapping[Label, str]          # vineyard→canopies, row→row_pieces, interrow_area→interrow_pieces, waste→waste
QA_CODES: frozenset[str]                            # contract §2.5.14 codes + tile_failed, invalid_after_rounding

# contracts/ids.py [C]
class IdKind(StrEnum): tile, block, row, interrow, row_piece, interrow_piece, canopy, row_candidate, waste, target
ID_PATTERNS: Mapping[IdKind, re.Pattern[str]]       # regexes exactly as contract §2.4
def tile_id_from_file_name(file_name: str) -> str
def file_name_from_tile_id(tile_id: str) -> str
def parse_tile_id(tile_id: str) -> tuple[int, int]                 # (grid_row, grid_col)
def format_tile_id(grid_row: int, grid_col: int) -> str
def format_vineyard_id(index: int, n_blocks: int) -> str           # 2 digits, 3 if n_blocks > 99
def format_row_id(vineyard_id: str, row_index: int) -> str         # V03-R017
def format_interrow_id(vineyard_id: str, k: int) -> str            # V03-I017
def format_row_piece_id(row_id: str, tile_id: str, dup: int = 1) -> str   # ...@tile[#2]
def format_interrow_piece_id(interrow_id: str, tile_id: str, dup: int = 1) -> str
def format_marcaj_interrow_piece_id(tile_id: str, k: int) -> str   # siret3_r021_c012:I001
def format_canopy_id(tile_id: str, k: int) -> str                  # :C0001
def format_row_candidate_id(tile_id: str, k: int) -> str           # :K07
def format_waste_id(k: int, part: Literal["", "a", "b"] = "") -> str
def format_target_id(kind: TargetKind, k: int) -> str              # T-GAP-0001
def is_valid_id(kind: IdKind, value: str, *, strict: bool = True) -> bool  # relaxed = non-empty stripped
def row_index_of(row_id: str) -> int | None                        # parses -R(\d+)$ (accepts R01 and R001)

# contracts/ordering.py [C]
def angle_deg_utm(p0: tuple[float, float], p1: tuple[float, float]) -> float   # [0,180)
def angle_px_to_utm(angle_px_deg: float) -> float
def canonical_normal(angle_deg: float) -> tuple[float, float]      # n_y>0; |n_y|<0.01 → n_x>0
def order_by_normal(angle_deg: float, centroids: np.ndarray) -> np.ndarray   # indices, R001 = max n·c
def block_sort_key(x: float, y: float) -> tuple[int, int]          # (-round(y), round(x))

# contracts/schemas.py [C]
@dataclass(frozen=True)
class ColumnSpec: name: str; kind: Literal["str","int8","int16","int32","int64","float32","float64","bool"]
                  nullable: bool = False; enum: type[StrEnum] | None = None; id_kind: IdKind | None = None
@dataclass(frozen=True)
class LayerSchema: name: str; geom_types: tuple[str, ...]; pk: tuple[str, ...]
                   columns: tuple[ColumnSpec, ...]; provenance: bool = True; annset: bool = False
LAYER_SCHEMAS: Mapping[str, LayerSchema]
PROVENANCE_COLUMNS: tuple[ColumnSpec, ...]   # source, run_id, model_version, confidence(float32), qa_flags
def validate_layer(gdf: gpd.GeoDataFrame, name: str, *, strict_ids: bool | None = None) -> None  # raises SchemaError
def coerce_layer(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame   # new frame: exact dtypes and column order
def empty_layer(name: str) -> gpd.GeoDataFrame
```

### 2.2 geo
```python
# geo/tiling.py [C] — signatures exactly as contract §8
GSD_M = 0.025; TILE_PX = 2048; TILE_M = 51.2; GRID_ORIGIN_X = 628992.0; GRID_ORIGIN_Y = 5221222.4
@dataclass(frozen=True)
class TileRef:
    tile_id: str; grid_row: int; grid_col: int; x0: float; y0: float
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...
def tile_ref_from_grid(row: int, col: int) -> TileRef
def tile_ref_from_tags(path: str | Path) -> TileRef          # rasterio transform/CRS vs grid, |Δ|<1e-6 else IngestError
def px_to_utm(t: TileRef, uv: np.ndarray) -> np.ndarray      # (N,2)
def utm_to_px(t: TileRef, xy: np.ndarray) -> np.ndarray      # no clip
def index_to_px(ij: np.ndarray) -> np.ndarray                # +0.5
def tiles_for_bounds(minx: float, miny: float, maxx: float, maxy: float) -> list[str]
def mosaic_px(t: TileRef, uv: np.ndarray) -> np.ndarray
# non-contract helpers in the same module
def tile_ref(tile_id: str) -> TileRef
def tile_box(t: TileRef) -> Polygon
def existing_tile_ids() -> frozenset[str]                    # from contracts/tile_grid.txt

# geo/raster.py
def read_tile(path: str | Path) -> np.ndarray                # (2048,2048,3) uint8 RGB, C-contiguous
def valid_mask(rgb: np.ndarray, cfg: NodataConfig) -> np.ndarray          # bool, True = image
def valid_polygon(valid: np.ndarray, t: TileRef, *, approx_eps_px: float) -> MultiPolygon  # UTM
def mask_to_polygons(mask: np.ndarray, t: TileRef, *, approx_eps_px: float, min_area_px: float,
                     pixel_offset: float = 0.5, outset_px: float = 0.0) -> list[Polygon]  # UTM, CCW, holes kept
def rasterize_px(polys_uv: Sequence[np.ndarray], shape: tuple[int, int] = (2048, 2048), value: int = 1) -> np.ndarray
def rasterize_utm(geoms: Sequence[BaseGeometry], t: TileRef, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray
def resize_aligned(arr: np.ndarray, out_px: int, *, mode: Literal["down", "up"]) -> np.ndarray  # AREA / LINEAR
def write_mask_png(path: Path, mask: np.ndarray) -> None     # 1-bit (IMWRITE_PNG_BILEVEL), atomic
def read_mask_png(path: Path) -> np.ndarray                  # bool

# geo/vector_io.py
def write_layer(gdf: gpd.GeoDataFrame, name: str, path: Path) -> Path   # coerce → validate → atomic GeoParquet
def read_layer(path: Path, name: str | None = None, *, validate: bool = True) -> gpd.GeoDataFrame
def write_geojson(gdf: gpd.GeoDataFrame, path: Path, *, decimals: int = 3, id_column: str | None = None) -> Path
    # EPSG:32635 with "crs": urn:ogc:def:crs:EPSG::32635; pure json; deterministic bytes
def write_geojson_4326(gdf: gpd.GeoDataFrame, path: Path, *, decimals: int = 7, id_column: str | None = None) -> Path
def read_geojson(path: Path, *, expect_epsg: int = 32635) -> gpd.GeoDataFrame
def round_geometry(geom: BaseGeometry, decimals: int) -> BaseGeometry

# geo/ops.py  (unit-agnostic)
def clip_polygonal(geom: BaseGeometry, clip: BaseGeometry) -> list[Polygon]
def clip_line(line: LineString, clip: BaseGeometry) -> LineString | None   # one polyline per row per tile (§1.6)
def clip_box(xyxy: tuple[float, float, float, float], bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None
def orient_ccw(poly: Polygon) -> Polygon
def make_valid_polygonal(geom: BaseGeometry) -> list[Polygon]   # make_valid(method="structure"), polygonal parts only
def split_multi(geom: BaseGeometry) -> list[BaseGeometry]
def notch_holes(poly: Polygon, notch_width: float) -> list[Polygon]
def drop_consecutive_duplicates(coords: np.ndarray, *, closed: bool) -> np.ndarray
```

### 2.3 annset, config, logging
```python
# annset/model.py [C]
ANNSET_LAYERS: Final = ("canopies", "row_pieces", "interrow_pieces", "waste")
@dataclass(frozen=True)
class AnnSetMeta: contract_version: str; source: Source; run_id: str; model_version: str
                  created_at: str; n_tiles: int; counts: Mapping[str, int]; inputs: tuple[str, ...]
@dataclass(frozen=True)
class AnnSet:  # GeoDataFrames are never mutated; with_layer/for_tiles return new objects
    meta: AnnSetMeta; canopies: gpd.GeoDataFrame; row_pieces: gpd.GeoDataFrame
    interrow_pieces: gpd.GeoDataFrame; waste: gpd.GeoDataFrame
    def layer(self, name: str) -> gpd.GeoDataFrame
    def with_layer(self, name: str, gdf: gpd.GeoDataFrame) -> "AnnSet"
    def for_tiles(self, tile_ids: Iterable[str]) -> "AnnSet"
    def tile_ids(self) -> frozenset[str]
def empty_annset(meta: AnnSetMeta) -> AnnSet
# annset/io.py [C]
def write_annset(annset: AnnSet, annset_dir: Path) -> Path        # validates each layer; annset.json written last
def read_annset(annset_dir: Path, *, validate: bool = True) -> AnnSet
def resolve_run_dir(work_dir: Path, ref: str) -> Path              # run_id | LATEST_MODEL | LATEST_MARCAJ | LATEST_REFERENCE | path
def update_latest_link(work_dir: Path, source: Source, run_dir: Path) -> None

# config [C for load_config / cfg_hash]
class AppConfig(BaseModel):  # ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    contract_version: str; project; paths; grid; runtime; nodata; veg; nn; rows; orchard; blocks
    canopy; interrow; row_structure; waste; targets; route; export
    import_: ImportConfig = Field(alias="import"); measure; web; eval; logging
DEFAULT_CONFIG: Final[Path]   # src/AI/configs/default.yaml
def load_config(paths: Sequence[Path] = (DEFAULT_CONFIG,), overrides: Sequence[str] = (), *,
                project_root: Path | None = None) -> AppConfig
def cfg_hash(cfg: AppConfig, keys: Sequence[str]) -> str          # sha1 of canonical JSON of the dotted subtrees
def cfg_subtree(cfg: AppConfig, key: str) -> Any
def resolved_config_dict(cfg: AppConfig) -> dict[str, Any]

# logging_setup.py
def setup_logging(*, level: str, console: Literal["rich", "plain"], jsonl_path: Path | None, run_id: str, tz: str) -> None
def get_logger(name: str) -> logging.Logger                        # "vineyard.<module>"
def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO,
              stage: str | None = None, tile_id: str | None = None, **fields: Any) -> None
def log_failure(logger: logging.Logger, event: str, exc: BaseException, **fields: Any) -> None  # always with traceback
```

### 2.4 pipeline (every stage owner builds on this)
```python
# pipeline/registry.py [C]
StageRun = Callable[["RunContext"], "StageResult"]
@dataclass(frozen=True)
class StageSpec: name: str; version: str; scope: Literal["global", "tile", "block"]
                 cfg_keys: tuple[str, ...]; requires: tuple[str, ...]; run: StageRun; description: str = ""
PRE_STAGES: Final = ("ingest","tile_prep","nn_infer","rows_detect","rows_link","blocks","canopy",
                     "interrow","row_attrs","waste","assemble")
POST_STAGES: Final = ("derive","targets","passable","route","measure","web_bundle")
STAGE_MODULES: Final[Mapping[str, str]]   # name → "vineyard.pipeline.stages.<name>"; each module defines STAGE: StageSpec
def load_stage(name: str) -> StageSpec     # lazy importlib; StageNotImplemented if the module is missing
def select_stages(order: Sequence[str], *, from_: str | None, until: str | None) -> tuple[str, ...]

# pipeline/cache.py, atomic.py [C]
def cache_key(stage: str, stage_version: str, cfg_digest: str, input_keys: Iterable[str]) -> str   # sorted inputs
def is_fresh(artifact: Path, key: str) -> bool
def write_key(artifact: Path, key: str) -> None                    # "<artifact>.key", written last
@contextmanager
def atomic_path(path: Path) -> Iterator[Path]                      # tmp "<name>.tmp-<pid>" → os.replace
def atomic_write_bytes(path: Path, data: bytes) -> None
def atomic_write_json(path: Path, obj: Any) -> None                # sort_keys, deterministic

# pipeline/parallel.py [C]
@dataclass(frozen=True)
class ItemOutcome(Generic[R]): key: str; ok: bool; value: R | None; error: str | None
                               traceback: str | None; duration_s: float
def parallel_map(fn: Callable[[T], R], items: Sequence[T], *, key: Callable[[T], str], workers: int,
                 chunksize: int, maxtasksperchild: int) -> Iterator[ItemOutcome[R]]   # workers<=1 → inline
def init_worker() -> None    # cv2.setNumThreads(1), ocl off; never imports torch

# pipeline/context.py [C]
@dataclass(frozen=True)
class RunPaths: project_root: Path; work_dir: Path; tiles_dir: Path; cache_dir: Path; static_layers_dir: Path
                runs_dir: Path; run_dir: Path; layers_dir: Path; annset_dir: Path; exports_dir: Path
                qa_dir: Path; metrics_dir: Path; logs_dir: Path
    def tile_cache(self, stage: str, tile_id: str, ext: str) -> Path   # work/cache/<stage>/<tile_id>.<ext>
@dataclass(frozen=True)
class RunContext: run_id: str; source: Source; cfg: AppConfig; paths: RunPaths
                  tile_filter: tuple[str, ...]; force: frozenset[str]; force_all: bool; workers: int
                  allow_failures: bool; git_sha: str; package_version: str; annset_ref: str | None
    def selected_tiles(self, all_ids: Iterable[str]) -> tuple[str, ...]   # fnmatch globs, sorted
    def should_force(self, stage: str) -> bool
    def stage_cfg_digest(self, spec: StageSpec) -> str
    def model_version(self, *, nn: str = "none", waste: str = "none") -> str
def make_run_id(now: datetime, source: Source, digest: str) -> str   # 20260926T0310-model-a1b2c3
def new_run_context(cfg: AppConfig, *, source: Source, run_id: str | None = None, tiles: Sequence[str] = (),
                    force: Sequence[str] = (), force_all: bool = False, workers: int | None = None,
                    allow_failures: bool | None = None, annset_ref: str | None = None) -> RunContext

# pipeline/runner.py [C]
@dataclass(frozen=True)
class StageResult: stage: str; n_items: int; n_cached: int; n_failed: int; failed: tuple[str, ...] = ()
                   outputs: tuple[Path, ...] = (); metrics: Mapping[str, float] = field(default_factory=dict)
@dataclass(frozen=True)
class RunReport: run_id: str; results: tuple[StageResult, ...]; exit_code: int
@dataclass(frozen=True)
class TileTask: tile_id: str; tif_path: Path; key: str; cfg: BaseModel; inputs: Mapping[str, Path]; outputs: Mapping[str, Path]
def run_stages(ctx: RunContext, names: Sequence[str]) -> RunReport
def run_tile_stage(ctx: RunContext, spec: StageSpec, *, make_task: Callable[[str], TileTask],
                   tile_fn: Callable[[TileTask], Mapping[str, Any]],   # top-level, picklable; writes its outputs atomically
                   input_keys: Callable[[str], Sequence[str]], tile_ids: Sequence[str] | None = None) -> StageResult
# pipeline/tile_index.py
def read_tile_index(ctx: RunContext) -> gpd.GeoDataFrame
def tile_refs(ctx: RunContext) -> Mapping[str, TileRef]

# pipeline/stages/tile_prep.py (accessors used by rows/canopy/interrow/nn/waste) [C]
@dataclass(frozen=True)
class TileStats: tile_id: str; valid_frac: float; nodata_frac: float; veg_frac: float; status: TileStatus
def veg_mask_path(paths: RunPaths, tile_id: str) -> Path
def valid_mask_path(paths: RunPaths, tile_id: str) -> Path
def load_veg_mask(paths: RunPaths, tile_id: str) -> np.ndarray      # bool
def load_valid_mask(paths: RunPaths, tile_id: str) -> np.ndarray
def load_tile_stats(paths: RunPaths, tile_id: str) -> TileStats
def tile_prep_key(paths: RunPaths, tile_id: str) -> str              # for downstream cache keys (contract §3.2)
# perception/vegmask.py
def neg_a_star(rgb: np.ndarray) -> np.ndarray                         # float32, -(a*-128)
def veg_mask_lab_a(rgb: np.ndarray, *, blur_sigma_px: float, threshold: float) -> np.ndarray   # bool
```

### 2.5 cvat
```python
# template.py [C]
META_BLOCK: Final[str]; XML_HEADER: Final[str]; XML_FOOTER: Final[str]
@dataclass(frozen=True)
class AttrSpec: name: str; input_type: Literal["text", "select"]; values: tuple[str, ...]; default: str
@dataclass(frozen=True)
class LabelSpec: name: str; cvat_type: Literal["polygon", "rectangle", "polyline"]
                 shape_tag: Literal["polygon", "box", "polyline"]; attributes: tuple[AttrSpec, ...]
LABEL_SPECS: Mapping[str, LabelSpec]   # parsed from META_BLOCK at import
# model.py [C]
@dataclass(frozen=True)
class CvatShape: tag: Literal["polygon", "polyline", "box"]; label: str
                 points: tuple[tuple[float, float], ...]      # box: ((xtl,ytl),(xbr,ybr))
                 attributes: tuple[tuple[str, str], ...]; source: str = "manual"
                 occluded: int = 0; z_order: int = 0; ref_id: str | None = None   # ref_id is not serialized
@dataclass(frozen=True)
class CvatImage: id: int; name: str; width: int; height: int; shapes: tuple[CvatShape, ...]
@dataclass(frozen=True)
class CvatDocument: images: tuple[CvatImage, ...]; meta_xml: str | None = None; version: str = "1.1"
# writer / reader / normalize
def serialize_document(doc: CvatDocument, *, decimals: int = 1) -> bytes
def format_coord(v: float, decimals: int) -> str               # never "-0.0"
def parse_xml(data: bytes, *, source_name: str) -> tuple[CvatDocument, tuple[Issue, ...]]
def read_cvat_files(paths: Sequence[Path], *, duplicate_policy: Literal["last_wins", "error"]) -> tuple[CvatDocument, tuple[Issue, ...]]
def normalize_enum(attr: str, raw: str, *, synonyms: Mapping[str, str], accept_synonyms: bool) -> tuple[str, Issue | None]
def normalize_id(raw: str | None) -> str
def find_case_collisions(values: Iterable[str]) -> tuple[tuple[str, ...], ...]
# conversion [C]
@dataclass(frozen=True)
class TileWriteStats: tile_id: str; n_in: Mapping[str, int]; n_out: Mapping[str, int]; n_dropped: Mapping[str, int]; n_split: int
def annset_to_images(annset: AnnSet, tiles: Sequence[TileRef], cfg: CvatExportConfig) -> tuple[tuple[CvatImage, ...], tuple[TileWriteStats, ...]]
def document_to_annset(doc: CvatDocument, *, tile_refs: Mapping[str, TileRef], source: Source, run_id: str,
                       model_version: str, cfg: ImportConfig, canopy_cfg: CanopyConfig) -> tuple[AnnSet, gpd.GeoDataFrame]  # (annset, qa_issues)
# validation / packing / export
@dataclass(frozen=True)
class Issue: severity: Severity; code: str; message: str; zip_name: str = ""; tile_id: str = ""; object_ref: str = ""
@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...]
    @property
    def ok(self) -> bool: ...
    def merge(self, other: "ValidationReport") -> "ValidationReport": ...
def validate_document(doc: CvatDocument, *, cfg: CvatExportConfig, tile_px: int,
                      waste_block_dist_m: Mapping[str, float] | None = None) -> ValidationReport
def validate_zip(path: Path, *, expected_sha256: Mapping[str, str], cfg: CvatExportConfig, tile_px: int) -> ValidationReport
def validate_upload_set(zip_paths: Sequence[Path], *, expected_tiles: frozenset[str],
                        expected_sha256: Mapping[str, str], cfg: CvatExportConfig, tile_px: int) -> ValidationReport
@dataclass(frozen=True)
class PartPlan: index: int; tile_ids: tuple[str, ...]; est_bytes: int
def plan_parts(sizes: Sequence[tuple[str, int]], *, max_bytes: int, n_parts: int | None, balance: bool) -> tuple[PartPlan, ...]
def write_upload_zip(out: Path, xml_bytes: bytes, images: Sequence[tuple[str, Path]], *, deflate_level: int) -> Mapping[str, str]  # name→sha256
@dataclass(frozen=True)
class UploadResult: out_dir: Path; zips: tuple[Path, ...]; report: ValidationReport; manifest: Path; empty_tiles: Path; id_registry: Path
def export_upload(annset: AnnSet, tile_index: gpd.GeoDataFrame, cfg: AppConfig, out_dir: Path,
                  tile_status: gpd.GeoDataFrame | None = None) -> UploadResult   # raises ExportBlocked on any error
def make_test_zip(out: Path, cfg: AppConfig) -> Path
```

### 2.6 eval
```python
@dataclass(frozen=True)
class CanopyScore: iou: float; f1: float; score: float; tp: int; n_pred: int; n_ref: int
@dataclass(frozen=True)
class RowScore: f1: float; tp: int; n_pred: int; n_ref: int; matches: tuple[tuple[int, int], ...]
@dataclass(frozen=True)
class AttrScore: accuracy: float; macro_f1: float; score: float; n_ref: int
def class_iou(pred: Sequence[Polygon], ref: Sequence[Polygon]) -> float
def match_polygons(pred: Sequence[Polygon], ref: Sequence[Polygon], *, min_iou: float) -> tuple[tuple[int, int, float], ...]
def canopy_metrics(pred, ref, *, match_iou: float, w_iou: float, w_f1: float) -> CanopyScore
def mutual_cover(a: LineString, b: LineString, tol_m: float) -> tuple[float, float]
def row_axis_f1(pred: Sequence[LineString], ref: Sequence[LineString], *, tol_m: float, min_cover: float) -> RowScore
def attribute_scores(y_ref: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]) -> AttrScore
def grouping_consistency(ref_ids: Sequence[str], pred_ids: Sequence[str]) -> float    # pair-counting F1
def value_score(pred: float, ref: float, tol: float) -> float
def waste_f1(pred: Sequence[Polygon], ref: Sequence[Polygon], *, match_iou: float) -> RowScore
def fp_canopy_penalty(pred: Sequence[Polygon], tile_area_m2: float, *, factor: float) -> float
def evaluate_annsets(pred: AnnSet, ref: AnnSet, tiles: Sequence[str], cfg: EvalConfig) -> EvalReport
def compare_to_baseline(report: EvalReport, baseline: Mapping[str, Any], *, max_drop: float) -> tuple[str, ...]
```

### 2.7 Layers and files read and written

| Producer | Writes | Reads |
|---|---|---|
| ingest | `work/tiles/*.tif`; `work/tile_index.parquet` (tile_index: tile_id, file_name, grid_row, grid_col, x0, y0, x1, y1, gsd_m, width_px, height_px, src_zip, path, sha256, file_size, nodata_frac=NaN, valid_area_m2=NaN; the last two are filled via a join by `read_tile_index`); `work/layers/in_passages\|in_forbidden\|in_study_area\|in_start.parquet` (fid, type, name, source) | `01_tiles/*.zip`, `02_route/*.geojson` |
| tile_prep | `work/cache/valid/<t>.png`, `work/cache/veg/<t>.png`, `work/cache/tile_stats/<t>.json` (+ `.key` files); `runs/<id>/layers/tile_valid.parquet` (tile_id, valid_frac, geometry) | tiles, tile_index |
| import_reference | `runs/<ts>-reference-<h6>/annset/{canopies,row_pieces,interrow_pieces,waste}.parquet + annset.json`; `qa/qa_issues.parquet`; link `runs/LATEST_REFERENCE` | examples `annotations.xml` |
| export_cvat | `runs/<id>/exports/marcaj_upload/siret3_upload_KKofNN.zip`, `upload_manifest.csv` (zip_name, image_id, tile_id, sha256, n_vineyard, n_row, n_interrow_area, n_waste, xml_bytes), `empty_tiles.csv` (tile_id, zip_name, image_id, reason), `id_registry.json`, `validation_report.json`, `upload_summary.json` | AnnSet(model): canopies (canopy_id, tile_id, vineyard_id), row_pieces (piece_id, row_id, vineyard_id, tile_id, row_structure), interrow_pieces (piece_id, vineyard_id, tile_id, interrow_cover), waste (waste_id, tile_id, vineyard_id, dist_block_m); tile_index; tile_status (optional) |
| eval | `runs/<model_id>/metrics/eval_examples.json` + `.md` | AnnSet(model), AnnSet(reference) |
| all stages | `runs/<id>/logs/pipeline.jsonl`, `run.json`, `metrics/timings.json` | — |

---

## 3. Algorithm notes (only where the docs leave a choice open, or where the port needs care)

1. **Grid values** (contract §1.3, §8).
   - Compute `x0 = round(GRID_ORIGIN_X + TILE_M*col, 6)`, and likewise for y0, so values equal the tag doubles.
   - `GridConfig` validates that the YAML grid equals the tiling constants; the constants are the contract.
   - `tiles_for_bounds` works on the grid with strict interior overlap: `floor((min-OX)/TM + 1e-9) … ceil((max-OX)/TM - 1e-9)-1`, intersected with `tile_grid.txt`. That is O(1) with no I/O setup.
2. **Pixel conventions** (contract §1.2/§1.4).
   - `rasterize_px` uses `fillPoly(round((uv-0.5)*16), shift=4)`.
   - `mask_to_polygons` runs `findContours(RETR_CCOMP, CHAIN_APPROX_NONE)` → `approxPolyDP(eps)` → `+pixel_offset` → Polygon with holes → `buffer(outset_px)` → UTM → `make_valid(method="structure")` → CCW.
   - `pixel_offset` and `outset_px` are parameters: the reference uses offset 0 and outset 0 (§0), while the contract default is 0.5/0.5. See R1.
3. **Prototype ports.**

   | Prototype | New home | What changes |
   |---|---|---|
   | `axes.veg_mask` | `perception/vegmask.veg_mask_lab_a` | Same ops. The RGB now comes from rasterio (GDAL libjpeg) instead of tifffile, so decoded values may differ by ±1 DN. Returns bool. |
   | `fit.corridor`, `attrs.band/ir_mask` | used by other subsystems | Must call `rasterize_px`; the prototype's `np.round(p*16)` without −0.5 is a 1.25 cm shift. |
   | `fit.score` | `eval.canopy_metrics` | Vector (shapely UTM) instead of a raster built with `np.round(p)` fillPoly, which is biased. |
   | `axes.row_f1`/`cover` | `eval.row_axis_f1` | Exact `a.intersection(b.buffer(0.4)).length/a.length` instead of 200-point sampling; optimal one-to-one via `scipy.optimize.linear_sum_assignment` on feasible pairs instead of first-fit greedy; any polyline, per tile. |
   | `axes.clip_to_tile` | not ported here | It is the snap-to-edge extension and belongs to rows (extend to the **clip** edge, contract §1.6). `geo.ops.clip_line` is the plain geometric clip. |
   | `fit.pts` | `cvat.reader` | Point parsing. |

   Because of the canopy change, re-baseline the gates on vector numbers (the prototype's 0.82/0.855 were raster).
4. **Metric definitions left open by the brief.**
   - Canopy: class IoU on union areas per tile. With disjoint reference polygons, IoU ≥ 0.5 implies a unique match, so greedy by IoU descending is exact. STRtree `query(predicate="intersects")` supplies candidate pairs. F1 = 2TP/(n_pred + n_ref).
   - Interrow attribute matching: one-to-one at IoU ≥ 0.5.
   - Row attributes ride on the row F1 matching.
   - A missing prediction becomes `y_pred=None`: it counts as wrong for accuracy and as FN for the true class. Macro-F1 averages over enum values present in `y_ref ∪ y_pred`.
   - `score = (acc + macroF1)/2` per attribute, and the reported attribute score is the mean of the two attributes.
   - vineyard_id consistency: pair-counting F1 over matched objects.
   - Reporting: per tile, the mean of tiles, and pooled.
5. **nodata** (contract §1.7).
   - `nd = max(R,G,B) ≤ 10` → `connectedComponentsWithStats` → keep only components touching the border with area ≥ 800 px (0.5 m²) → close 5 px → dilate 4 px.
   - The veg mask is multiplied by `valid` eroded by 4 more px.
   - `valid_frac < 0.02` gives status `empty_nodata`.
6. **Ingest.**
   - Stream each ZIP member to `work/tiles/<name>.tmp-<pid>`, hashing sha256 while copying (zipfile checks the CRC), then `os.replace`.
   - Skip when the file exists and its `.key` equals `sha1(name‖CRC)`.
   - Verify:
     - the name matches the regex;
     - the name is in `tile_grid.txt`;
     - no duplicates across ZIPs;
     - count equals `grid.expected_tiles`;
     - `tile_ref_from_tags`: CRS 32635, transform `(0.025, 0, x0, 0, -0.025, y0)` with |Δ| < 1e-6, 2048², 3×uint8, `AREA_OR_POINT=Area`.
   - Any deviation raises an `IngestError` naming the tile.
   - Route inputs checks:
     - `crs` member equal to `urn:ogc:def:crs:EPSG::32635` (error otherwise);
     - `make_valid`, logging any repair;
     - START inside passages (warning otherwise);
     - START lies in r018_c010 at (28.0, 2002.0);
     - study_area area = 815,267.84 m² (311 × 2621.44).
7. **CVAT writer** (arch §4.13.1, contract §4.1).
   - Hand-built strings mirroring the example byte for byte:
     - `XML_HEADER` + `META_BLOCK` + `\n`;
     - one shape per line;
     - empty image written as `<image …>\n</image>`;
     - empty attribute written as `<attribute name="vineyard_id"></attribute>`.
   - Escaping: `xml.sax.saxutils.escape` for text; for attribute values the same plus `"` → `&quot;`.
   - Shape order: row, interrow_area, vineyard, waste. Within a label, sort by PK.
   - Geometry cleaning in px:
     - intersect with box(0, 0, 2048, 2048) → `make_valid_polygonal` → explode;
     - drop parts < 16 px²;
     - `notch_holes`;
     - simplify 0.5 px with `preserve_topology`;
     - orientation: CCW in UTM, i.e. negative shoelace in px;
     - drop the closing vertex; round to 1 decimal; drop consecutive duplicates;
     - clip coordinates to [0, 2048] before formatting (prevents "-0.0").
   - Validity loop: if the polygon is invalid after rounding, run `make_valid` in px and re-round once. If it is still invalid, drop it and raise a qa warning `invalid_after_rounding`.
   - Polylines use `clip_line` (one per row per tile). Boxes use `clip_box` (min/max, 1 decimal).
8. **notch_holes.**
   - For each hole, find the nearest point on the exterior to the hole ring and subtract a rectangle of width `notch_width_px` (0.2 px) along that shortest segment. The result is a valid polygon with no interiors.
   - If the difference produces several parts (the hole spans the full width), keep all parts with the same attributes (contract §4.1).
9. **Packer.**
   - Estimate per-tile bytes as tif size + 76 + 2·len(`images/<name>`) + the deflate-9 size of that tile's `<image>` fragment compressed alone. That is a conservative estimate.
   - n = the greedy count at `max_bytes`, or `n_parts` if set.
   - If `balance_parts`, do a contiguous linear partition into n parts that minimises the maximum part (binary search on capacity).
   - Deterministic ZIPs: `annotations.xml` first (DEFLATED level 9), then `images/<name>` in sorted order (STORED). Every entry gets `date_time=(1980,1,1,0,0,0)`, `external_attr=0o100644<<16`, `create_system=3`. No directory entries.
   - After writing, `os.path.getsize` is authoritative. If a part is over the limit, repack with n+1 (at most 2 retries), then raise `ExportBlocked`.
10. **Export flow** (atomic).
    - Build all 311 images (empty ones included).
    - Write ZIPs into `marcaj_upload.staging-<pid>/`.
    - Run `validate_zip` on each, then `validate_upload_set`, then the self-check:
      - (a) re-parse vs the in-memory document: counts, attributes, vertices ≤ 0.06 px;
      - (b) `document_to_annset` vs source AnnSet: per tile and label, union IoU ≥ 0.98 and equal attribute multisets;
      - (c) sha256 of every image;
      - (d) 311 unique names.
    - Only then `os.replace` into `marcaj_upload/`. On failure, rename to `marcaj_upload.FAILED-<ts>` and exit with code 2.
    - Export refuses to run if AnnSet(model) has qa errors (contract §2.7) unless `--allow-qa-errors`.
    - `id_registry.json` = `{contract_version, run_id, last_vineyard, blocks: {V01: {last_row, last_interrow}}, manual_row_start: 900}`. R900+ follows arch §3.3, which wins over contract "next free".
11. **Reader tolerance** (arch §4.13.8, contract §4.3).
    - lxml `iterparse(tag="image", resolve_entities=False, huge_tree=True)`.
    - Match by NFC basename.
    - Ignore `subset`, `task_id`, `group_id`, `source` and the image `id`. Accept 2-decimal coordinates, `<tag>`, and empty images.
    - `<mask>` → RLE → polygon with a warning. `<track>` is an error. An unknown label or wrong shape for its label is a qa error and the object is skipped.
    - With several files, the last one wins, with a warning.
12. **document_to_annset.**
    - px→UTM → `make_valid` → CCW.
    - IDs:
      - canopy_id in XML order;
      - row piece_id `<row_id>@<tile>` with `#k` for duplicates;
      - interrow piece_id `<tile>:I001`;
      - waste_id sorted by (tile, ytl, xtl).
    - Canopy `row_id` = nearest row piece in the tile within 0.5 m, using `STRtree.query_nearest(max_distance)`; ties go to the lowest piece_id.
    - `is_clump` = area > `canopy.clump_area_m2`.
    - Relaxed ID regexes for marcaj and reference.
    - Provenance: `model_version = "<source>-export@<sha8>"`, `confidence = 1.0`.
13. **Runner, cache and logging.**
    - Tile key = `sha1(f"{stage}:{VERSION}" ‖ cfg_digest ‖ sorted(input_keys))`.
    - Workers write their artifacts; the main process writes `.key` only after a successful outcome.
    - Workers return small dicts; only the main process writes `pipeline.jsonl`, which avoids concurrent appends.
    - A failed tile gets a `tile.failed` event with its traceback, a `tile_failed` qa issue and `status=failed`. Exit code ≠ 0 unless `--allow-failures`.
    - Outcomes are sorted by key, so outputs are deterministic.
    - `LATEST_<SOURCE>` is updated only when no `--tiles` filter was used.
    - `workers: auto` = cpu_count − 3 = 8 on an M3 Pro.
    - `STAGE_MODULES` is lazy, so importing the registry never imports torch.
14. **CLI.**
    - `import vineyard` sets the thread environment before numpy/cv2 are imported.
    - Sub-apps `nn`, `waste` and `web` are loaded from `vineyard.<pkg>.cli` if present, otherwise a placeholder exits with code 3 ("neimplementat"). Rule for owners: no torch import at module top.

---

## 4. Config keys I own (`configs/default.yaml`)

| YAML path | Default | Source |
|---|---|---|
| `contract_version` | `"1.1"` | contract §0/§13 (bump) |
| `project.name` / `project.crs` | `siret3` / `EPSG:32635` | contract §9 |
| `paths.data_root` | `"../../data & info"` | §9 adapted to the layout decision |
| `paths.tiles_zip_glob` | `01_tiles/siret3_challenge_tiles_part*of5.zip` | §9 |
| `paths.route_dir` / `paths.examples_dir` | `02_route` / `05_examples/siret3_examples_cvat` | §9 |
| `paths.work_dir` | `work` | §9 / user |
| `paths.models_dir` | `../../models` (already holds sam3.1, gitignored) | §9 adapted |
| `paths.repo_root` / `paths.web_data_dir` | `../..` / `../Web/data` | user decision |
| `grid.{gsd_m,tile_px,tile_m,origin_x,origin_y}` | 0.025, 2048, 51.2, 628992.0, 5221222.4 (must equal the tiling constants) | §9, §1.3 |
| `grid.expected_tiles` / `grid.tiepoint_tol_m` | 311 / 1.0e-6 | §1.3 |
| `runtime.workers` / `workers_auto_reserve` / `workers_sweep` | auto / 3 / [5, 8, 10] | arch §3.6, §4.1 |
| `runtime.chunksize` / `maxtasksperchild` | 2 / 50 | arch §3.6 |
| `runtime.seed` / `allow_failures` | 0 / false | §9 |
| `runtime.thread_env` | {OMP, OPENBLAS, VECLIB, GDAL_NUM_THREADS: "1", GDAL_CACHEMAX: "256"} | arch §4.1 |
| `nodata.{max_rgb,min_area_m2,close_px,dilate_px,veg_erode_px,min_valid_frac}` | 10, 0.5, 5, 4, 4, 0.02 | §1.7, §9 |
| `nodata.approx_eps_px` | 2.0 | §1.7 step 4 |
| `veg.{method,blur_sigma_px,threshold,morph_close_px}` | lab_a, 2.5, 4.0, 0 | §9 = arch vegmask |
| `veg.otsu_fallback_veg_frac` / `veg.texture_fallback` | [0.10, 0.90] / `v_std_0.25m` (consumed by canopy/rows) | arch §3.6 |
| `export.min_row_piece_m` / `min_interrow_piece_m2` | 0.5 / 0.25 (applied in assemble) | §9 |
| `export.cvat.coord_decimals` / `shape_source` | 1 / `manual` | §9, user |
| `export.cvat.max_zip_bytes` / `n_parts` / `balance_parts` | 85000000 / null (derived) / true | user, arch §4.13 |
| `export.cvat.zip_name` / `verify_roundtrip` | `siret3_upload_{i:02d}of{n:02d}.zip` / true | §9 |
| `export.cvat.min_polygon_px2` / `min_polyline_px` / `simplify_px` | 16 / 4 / 0.5 | arch §3.6, §4.13 |
| `export.cvat.notch_width_px` | 0.2 | this design (§4.1 notches) |
| `export.cvat.max_canopy_interrow_overlap_m2` | 0.05 | arch §4.13 step 2 |
| `export.cvat.waste_empty_vid_min_dist_m` | 10.0 | arch §4.13, §4.9 |
| `export.cvat.roundtrip_max_dev_px` / `selfcheck_min_union_iou` | 0.06 / 0.98 | §4.1 / this design |
| `export.cvat.xml_deflate_level` / `manual_row_start` | 9 / 900 | arch §4.13 / §3.3 |
| `import.accept_enum_synonyms` / `duplicate_policy` | true / last_wins | §9 |
| `import.enum_synonyms` | {"bare soil": bare_soil, bare: bare_soil, grass: vegetation, veg: vegetation, unknown: unassessable} | §4.4 |
| `import.canopy_row_assign_max_m` / `expected_images` | 0.5 / 311 | §2.5.7 / arch §4.13.8 |
| `eval.example_tiles` | [siret3_r006_c004, siret3_r021_c012] | README |
| `eval.{canopy_match_iou,canopy_w_iou,canopy_w_f1}` | 0.5, 0.6, 0.4 | brief |
| `eval.{row_tol_m,row_min_cover}` | 0.4, 0.8 | brief |
| `eval.{waste_match_iou,count_tol,length_tol,fp_canopy_factor}` | 0.3, 0.15, 0.10, 0.5 | brief |
| `eval.interrow_match_iou` | 0.5 | this design |
| `eval.regression_max_drop` | 0.01 | §10 |
| `eval.gates` | {canopy_score: 0.80, row_f1: 0.95, interrow_iou: 0.95, attributes: 0.90} | arch §5 |
| `logging.{level,jsonl,console,tz}` | INFO, true, rich, Europe/Chisinau | §9, §10 |

**Seeded for other owners** (contract §9 structure; arch §3.6 values win on conflict):

- `rows.filter.spacing_range_m: [1.8, 3.8]`.
- `rows.link.perp_tol_m: 0.3` (arch `link_offset_max_m`); `rows.link.gap_max_tiles: 1`.
- `rows.detect.peak_min_dist_factor: 0.6` replaces the fixed 0.9 m; `peak_prominence_frac` is replaced by `periodicity_min_snr: 3.0`.
- `interrow.shadow_v_max: 50` (arch) instead of 35.
- `waste.min_area_m2: 0.015` (arch) instead of 0.05, plus the arch `auto_*` keys.
- `route.*` uses arch keys (`candidate_radius_m`, `tsp_time_s`, `max_outside_share: 0.005`) plus contract `start_file` and `walking_speed_kmh`.
- New proposal: `canopy.vector_offset_px: 0.5` (see R1).
- `blocks.id_prefix/id_width/row_id_width` are kept as Literal values (V, 2, 3), because the ID regexes are contract.

---

## 5. Tests to write first (TDD)

**`test_tiling.py`**
- `tile_ref_from_grid(6,4)` gives x0 = 629196.8, y0 = 5220915.2, bounds (629196.8, 5220864.0, 629248.0, 5220915.2).
- `tile_ref_from_grid(21,12)` gives (629606.4, 5220147.2).
- `px_to_utm(r018_c010, [[28.0, 2002.0]])` ≈ (629504.70, 5220250.75) to 1e-6, and the inverse holds.
- Random round-trip error ≤ 1e-9.
- `index_to_px([[0,0]]) == [[0.5,0.5]]`.
- `mosaic_px(r006_c004, [[0,0]]) == [[8192, 12288]]`.
- `tiles_for_bounds(*tile_box(r006_c004).bounds) == ["siret3_r006_c004"]` (strict interior); a box straddling 4 tiles returns 4 names, filtered by existence.
- `tile_ref_from_tags` on both example tifs equals `from_grid`.
- `[needs_tiles]` all 311 match; `tile_grid.txt` equals the ZIP names.

**`test_ids.py` / `test_ordering.py`**
- Strict regexes accept V01, V123, V01-R001, V01-I001, `siret3_r021_c012:C0001`, W0001, W0001a, T-GAP-0001, `V01-R001@siret3_r021_c012#2`.
- Strict regexes reject v01, V1 and V01-R01; relaxed mode accepts V01-R01.
- `format_vineyard_id(5, 12) == "V05"`; `format_vineyard_id(5, 120) == "V005"`.
- `row_index_of("V01-R01") == 1`.
- `canonical_normal(90.0) == (1.0, 0.0)` (up to sign rule); n_y > 0 for 133.3°.
- `order_by_normal` on the example axes reproduces R01..R25 and R01..R26 exactly (measured True).

**`test_enums.py`**
- Exact lowercase values.
- `LABEL_ATTRIBUTES` equals the attribute order parsed from `META_BLOCK`.

**`test_schemas.py`**
- `empty_layer(n)` passes `validate_layer` for every n.
- Each of these raises `SchemaError` naming the column or object: missing column; "Regular"; duplicate PK; EPSG:4326; bowtie geometry; a 0.04 m LineString; `V01-R01` with `source=model`.
- `V01-R01` with `source=reference` passes.
- `coerce_layer` converts int64 → int16.

**`test_config.py`**
- `default.yaml` loads; `canopy.corridor_half_m == 0.30`; `rows.filter.spacing_range_m == [1.8, 3.8]`; `export.cvat.max_zip_bytes == 85000000`; `shape_source == "manual"`.
- An unknown YAML key and a typo in `--set` both fail.
- `--set canopy.min_area_m2=0.25` gives a float; `--set export.cvat.shape_source=auto` works.
- Assigning to a frozen field raises.
- `cfg_hash(["canopy"])` is stable, changes when canopy changes, and is unchanged when route changes.
- `data_root` resolves to `/Users/maleticimiroslav/Vin Gigahack/data & info`.
- A grid value ≠ the tiling constant fails.

**`test_logging.py`**
- A JSONL line has ts with a +03:00 or +02:00 offset, plus level, run_id, stage, tile_id, event and extra fields.
- `log_failure` includes the traceback.

**`test_vector_io.py`**
- Parquet round-trip keeps dtypes and CRS; no `.tmp-*` files are left.
- GeoJSON 32635 has the exact `crs` member and 3 decimals; `route.geojson` style uses 2 decimals.
- 4326 output has no `crs` member and 7 decimals.
- Writing twice gives identical bytes.
- Reading `02_route/*` gives 1 feature each; START = (629504.7, 5220250.75).

**`test_raster.py`**
- `rasterize_px` of the square (10,10)–(20,20) sets exactly 100 px (rows and cols 10..19).
- `mask_to_polygons` of that mask: area 100 ± 2 px² with offset 0.5 and outset 0.5, and 81 px² with offset 0 and outset 0 (reference style); CCW in UTM.
- `valid_mask`: a black 60×60 corner touching the border becomes nodata; an interior black 30×30 stays valid; a 1–10 DN ring is removed; valid is dilated by 4 px.
- `resize_aligned` gives 2048 → 1024 → 2048 shapes; pixel k covers u ∈ [2k, 2k+2).
- The 1-bit PNG round-trips exactly.

**`test_ops.py`**
- `clip_polygonal` across the tile edge gives the inner part; a clip producing a MultiPolygon gives 2 polygons.
- `clip_line` across a nodata hole gives one LineString from first to last vertex.
- `orient_ccw` makes the exterior CCW and holes CW.
- A bowtie gives 2 triangles.
- `notch_holes` on a square with a hole gives 1 valid polygon without interiors, with area loss ≤ notch_width × distance; a full-width hole gives 2 polygons.

**`test_cvat_template.py`**
- `META_BLOCK.encode()` equals the example bytes from `<meta>` to `</meta>`.
- `LABEL_SPECS`: waste is `rectangle`/`box`; row selects (regular, disrupted, unassessable) with default regular; interrow default bare_soil.

**`test_cvat_xml.py`**
- `parse(example)`:
  - r021: 25 polylines, 24 interrow, 399 vineyard, 0 box;
  - r006: 26/25/251/0;
  - all `source=manual`.
- `serialize(parse(example)) == example bytes` (byte-identical, original order kept).
- Exact expected strings for a box line, an empty attribute and an empty image.
- Escaping round-trips `A&B"<`.
- `format_coord(-0.04, 1) == "0.0"`.

**`test_cvat_reader_marcaj.py`**
- A synthetic Marcaj export parses:
  - image attributes id, name `images/siret3_r021_c012.tif`, subset, task_id;
  - 2 decimals; `source=file`; group_id; `<tag>`;
  - it is matched by basename.
- `<track>` is an error; unknown label `vine` is a qa error and the object is skipped; `<mask>` becomes a polygon with a warning.
- The same image in 2 files: last wins, with a warning.

**`test_cvat_normalize.py`**
- `" Regular "` → regular with no issue.
- `bare soil` → bare_soil with a warning.
- `grass` → vegetation.
- `Foo` → `bad_enum` error, raw value kept.
- `" V01 "` → V01.
- `{V03, v03}` → `id_case_collision`.
- A row without `row_id` → `missing_attr`.

**`test_cvat_convert.py`** (round-trip)
- Example → AnnSet(reference):
  - counts 399/251, 25/26, 24/25, 0;
  - disrupted = {V02-R06, R07, R08, R09, R23}; covers 21 bare_soil + 4 mixed;
  - Σ canopy area 237.119 / 299.056 m² (±0.01);
  - Σ interrow area 2068.031 / 1996.361;
  - row length 910.104 / 1031.451 m;
  - every polygon CCW, no closing vertex;
  - ≥ 99% of canopies get a `row_id`.
- AnnSet → document: IoU ≥ 0.999 and Hausdorff < 0.1 px against the originals; identical attributes.
- Writer cleaning cases: polygon outside the tile; bowtie; < 16 px² part dropped; hole notched; order row → interrow → vineyard → waste.

**`test_cvat_validator.py`**
- A clean example document has 0 errors.
- Each of these is an error with its own code: wrong attribute name; missing or extra attribute; "Regular"; bowtie; 2048.5 or −0.1; 2 distinct vertices; 10 px²; 3 px polyline; xtl ≥ xbr; duplicated closing vertex; positive shoelace; label `vine`; row as polygon; empty vineyard_id on vineyard; canopy∩interrow of 96 px² (0.06 m²).
- 64 px² of overlap is OK.
- Empty waste vineyard_id: OK at dist 12 m, error at 5 m, warning when the distance is unknown.
- ZIP checks, each an error: `.DS_Store`, `debug.xml`, `__MACOSX/`, orphan image, orphan XML name, case mismatch, NFD name, `.TIF`, sha mismatch, image ids not 0..n−1 in sorted order, modified meta.
- A size over a test limit of 10,000 B is an error; so is a missing or duplicate tile across parts.

**`test_cvat_packer.py`**
- Plans are deterministic, contiguous and sorted; each part is ≤ the limit; n is minimal; balanced parts differ by at most one tile size.
- `[needs_tiles]` real sizes with a 26–80 KB XML estimate give **n = 5** and every written ZIP is ≤ 85,000,000 B.
- ZIP entry order is xml first; tif STORED, xml DEFLATED; date 1980; no directory entries; extracted sha matches; rerunning gives a byte-identical ZIP.

**`test_cvat_export.py`**
- An empty AnnSet over 3 synthetic tiles produces:
  - manifest columns in contract order;
  - `empty_tiles.csv` listing all 3;
  - `id_registry.json` keys;
  - staging → final rename.
- An injected invalid object raises `ExportBlocked` and leaves the FAILED directory.
- `make_test_zip` contains 2 example tiles + 1 empty tile + 1 waste box and passes the validator.

**`test_ingest.py` / `test_route_inputs.py`**
- Synthetic ZIPs with 3 real-named GeoTIFFs (expected_tiles=3) produce tile_index with sha256, file_size, src_zip, grid_row and grid_col.
- Tiepoint off by 0.1 m → `IngestError` naming the tile; duplicate across ZIPs → error; `foo.tif` → error; wrong count → error.
- The second run is fully cached (mtime unchanged).
- Route: START within passages; study_area = 815,267.84 m².

**`test_tile_prep.py` / `test_vegmask_parity.py`**
- A synthetic green stripe on brown soil gives veg_frac equal to the stripe fraction ±2%.
- A black border gives the expected valid_frac; < 0.02 gives `empty_nodata`.
- The key changes with `veg.threshold`.
- On the example tiles the veg mask agrees ≥ 99.5% with the prototype (JPEG decoder tolerance).

**`test_cache.py` / `test_parallel.py` / `test_registry.py` / `test_runner.py`**
- `cache_key` is sensitive to version, config and inputs, and ignores input order.
- An exception inside `atomic_path` leaves no file and no key.
- `parallel_map` with workers=2 (spawn) returns all outcomes; workers=1 runs inline.
- `select_stages(from_="canopy", until="assemble") == (canopy, interrow, row_attrs, waste, assemble)`.
- With 3 tiles where 1 raises: `n_failed=1`, `tile.failed` event with traceback, qa `tile_failed`, exit ≠ 0 (0 with `--allow-failures`); a rerun gives cached=2 and retries 1.
- `--force` recomputes.
- After `import vineyard.pipeline.stages.tile_prep`, `"torch" not in sys.modules`.

**`test_metrics.py`**
- Canopy:
  - ref box(0,0,1,1) vs pred box(0.2,0,1.2,1): IoU 0.6667, match → score 0.8;
  - pred box(0.5,0,1.5,1): IoU 1/3, no match → score 0.2;
  - 10% of canopies dropped → F1 = 0.947.
- Rows, with ref (0,0)–(10,0):
  - pred at y=0.3 matches;
  - pred at y=0.5 does not;
  - pred (0,0.3)–(5,0.3) does not (mutual 80% fails).
- Attributes: ref [regular, disrupted, regular], pred [regular, regular, None] → acc 0.3333, macro-F1 0.25, score 0.2917.
- `value_score(11, 10, 0.15) == 0.3333`; `value_score(12, 10, 0.15) == 0`.
- Waste boxes at IoU 0.35 match; at 0.25 they do not.
- Grouping: identical partitions = 1.0; swapped labels = 1.0.
- `fp_canopy_penalty` of 262.144 m² on a 2621.44 m² tile = 0.05.

**`test_eval_examples.py`** — reference vs reference: canopy 1.0, row F1 1.0 (25/25, 26/26), interrow IoU 1.0, attributes 1.0; rows shifted 0.5 m give row F1 0.

**`test_import_reference.py`** — the stage is idempotent (same run reused); `LATEST_REFERENCE` is set.

**`test_cli.py`** — `--help` lists ingest, run, all, export-cvat, import-marcaj, from-marcaj, import-reference, derive, targets, route, measure, post, final, publish, eval-examples, qa, nn, web, bench, cvat, config, doctor. A missing stage prints a Romanian message and exits 3.

---

## 6. Dependencies

**What I provide to others**
- Every subsystem: `AppConfig`/`load_config`/`cfg_hash`, `RunContext`, `run_tile_stage`, `parallel_map`, `StageSpec`, atomic writes, `log_event`/`log_failure`, `validate_layer`/`write_layer`/`read_layer`, `LAYER_SCHEMAS`, ids/ordering, the `tiling` contract functions.
- Perception (rows, canopy, interrow, row_attrs): `load_veg_mask`, `load_valid_mask`, `load_tile_stats`, `tile_prep_key`, `read_tile`, `rasterize_px`/`rasterize_utm`, `mask_to_polygons`, `clip_polygonal`/`clip_line`/`orient_ccw`/`make_valid_polygonal`, `tile_refs`, `read_tile_index`, `in_*` layers.
- NN: `read_tile`, `resize_aligned`, `load_valid_mask`, `evaluate_annsets` for ablation A–F.
- Waste: `clip_box`, `format_waste_id`, empty `waste` schema, `in_*` layers.
- Post-Marcaj:
  - `read_cvat_files` + `document_to_annset(source=marcaj)` for `import_marcaj`;
  - `read_annset`/`resolve_run_dir`;
  - `write_geojson(decimals=2)` for `route.geojson`;
  - `write_geojson_4326` for web;
  - `order_by_normal` and `format_interrow_id` for derive;
  - `paths.repo_root` and `paths.web_data_dir`.

**What I need from others**
- The pre-annotation lead's `assemble` stage must write AnnSet(model) to `runs/<id>/annset/`, conforming to the schemas, with the semantic minimums already applied (`export.min_*`, `canopy.min_area_m2`, overrides/forced-empty tiles). It should also write `tile_status` (for `empty_tiles.csv` reasons) and `qa/qa_issues.parquet`.
- Each stage owner delivers `pipeline/stages/<name>.py` with `STAGE: StageSpec`. Sub-app owners deliver `vineyard/{nn,waste,web}/cli.py` exposing `app` with no top-level torch import.
- Post-Marcaj owns `import_marcaj`, `derive`, `targets`, `passable`, `route`, `measure`, `web_bundle` and `publish`. The CLI only wires them.

---

## 7. Work packages

Each WP ends with `make test` green, then commit and push to `origin main`.

**WP-A: Skeleton, environment, config, logging, CLI shell** (≈2.5 h, starts T0 = 22:30)
- In: arch §4.1/§4.15, contract §8–§10.
- Out: pyproject, `uv.lock`, `requirements.lock`, Makefile, Dockerfile, `.gitignore`, `.dockerignore`, CI workflow, `_env`, `errors`, config package with all sections seeded plus `default.yaml`, `logging_setup`, `doctor`, `cli.py`/`cli_options.py` with every command wired lazily.
- Acceptance:
  - `uv sync --locked` succeeds;
  - `uv run vineyard doctor` passes, including rasterio, pyogrio, pyproj, cv2, torch MPS, and GPKG and Parquet round-trips;
  - `vineyard --help` is complete;
  - `test_config`, `test_logging`, `test_cli` pass.
- Ordering: the config models are frozen by **23:15** (others import them).

**WP-B: Contracts, geo, AnnSet** (≈3 h, starts T0)
- Out: `contracts/*` (including `tile_grid.txt`), `geo/{tiling,raster,vector_io,ops}`, `annset/{model,io}`, `pipeline/atomic.py`.
- Acceptance: `test_tiling`, `test_ids`, `test_ordering`, `test_enums`, `test_schemas`, `test_vector_io`, `test_raster`, `test_ops`, `test_annset_io` pass with ≥ 85% coverage.
- Ordering: push `tiling.py`, `enums.py`, `ids.py` and `schema_defs.py` by **23:15**; everything else by 01:30.

**WP-C: Pipeline core, ingest, tile_prep** (≈3.5 h, starts 23:15 after the A/B freezes; stubs allowed earlier)
- Out: `pipeline/{registry,cache,parallel,context,runner,timings,tile_index}`, `ingest/*`, `perception/vegmask.py`, stages `ingest` and `tile_prep`, `run --from/--until/--tiles/--force` wiring.
- Acceptance:
  - `vineyard ingest` on the real data verifies 311 tiles and writes `tile_index` + `in_*` in < 60 s;
  - `vineyard run --until tile_prep` produces 311 masks with 0 failures and timings in JSONL;
  - a rerun is fully cached in < 5 s;
  - the vegmask parity test passes.
- Deadline: ingest by **00:30** and tile_prep on all 311 by **01:30**, because perception needs them.

**WP-D: CVAT I/O, export, reference** (≈4 h, starts T0 with template/model/writer/reader, which have no dependencies)
- Out: `cvat/*`, stages `export_cvat` and `import_reference`, `vineyard cvat …` sub-app.
- Milestones:
  - **M1 01:00**: byte-identical example round-trip plus `make_test_zip` passing the validator, then the team uploads the test ZIP to Marcaj (arch §4.13.5).
  - **M2 02:30**: `export-cvat` on an empty AnnSet for all 311 tiles gives about 5 ZIPs, each ≤ 85,000,000 B, passing validation and self-check.
  - **M3 06:00–07:00**: final export on the real AnnSet(model).
- Acceptance: all `test_cvat_*` and `test_import_reference` pass.
- Needs B's `tiling`/`ops`/`annset` (by 01:00).

**WP-E: Evaluation** (≈2.5 h, starts 23:30)
- Out: `eval/{matching,metrics,report}`, stage `eval` and `vineyard eval-examples [--baseline] [--write-baseline]`.
- Acceptance:
  - `test_metrics` passes;
  - reference vs reference = 1.0;
  - with a prototype-equivalent AnnSet (reference axes → a* ∩ corridor) the canopy score is about 0.85 ± 0.03 on vector metrics;
  - gates report pass/fail.
- Needs D's reader and `import_reference` (by about 00:30; synthetic tests don't wait).

**Integration gate (03:00):** `make preannotate` runs end-to-end with the perception stages stubbed, followed by M2. **Final gate:** validated ZIPs by **Saturday 07:00**.

---

## 8. Risks and open questions

| # | Risk / question | Mitigation |
|---|---|---|
| R1 | Reference canopies use the raw contour convention (integer vertices, max 2047); contract §1.4 prescribes +0.5 plus a 0.5 px outset. That could cost several points of canopy IoU. | `mask_to_polygons(pixel_offset, outset_px)` are parameters; the proposed key `canopy.vector_offset_px` lets `eval-examples` sweep {0/0, 0.5/0.5} so the canopy owner can pick the winner. Add a contract §1.2 erratum in §13. |
| R2 | Example row_ids have 2 digits (`R01`); the contract emits `R001`. | Reference and Marcaj use relaxed regexes; `row_index_of` accepts both. The score depends on grouping, not values. |
| R3 | The part count is 5, not the ~7 in arch; the Marcaj guide says "all five parts". | Derived count is 5 (measured), balanced at ~80 MB each. The publish checklist counts 311 files, not parts. |
| R4 | The actual XML size is unknown until the real AnnSet exists. | Conservative per-tile deflate estimate; post-write `getsize` is authoritative; automatic n+1 repack; validator hard limit. |
| R5 | Marcaj behaviour: `source`, the "No objects" flag, and the name matcher are unverified. | `shape_source` is configurable (default manual); `empty_tiles.csv`; test ZIP uploaded tonight (M1), including a box and an empty frame. |
| R6 | Marcaj export format drift (paths, 2 decimals, masks, duplicates across job exports). | Tolerant lxml reader plus synthetic Marcaj-style tests; rehearsal with a real export Saturday evening. |
| R7 | Python and environment: brew 3.14 `pyexpat` is broken (observed); conda PROJ/GDAL variables; numpy 2.5 vs skimage/opencv. `sknw` may pull `numba`, which caps numpy (unverified). | `uv python install 3.12` (managed interpreter); lxml for XML; `_env` + Makefile `unexport`; numpy left unpinned (≥ 2.1) and resolved by the lock. If sknw/numba conflicts, move it to a `route` extra or vendor the builder. `doctor` runs first. |
| R8 | Spawn pool pitfalls: pickling, torch imported in workers, thread oversubscription. | Top-level tile functions only; lazy `STAGE_MODULES`; `init_worker`; test asserts no torch; workers=1 inline path for debugging. |
| R9 | TargetKind conflict: the contract has row_gap/row_end_short/missing_row; arch adds missing-plant and sparse targets. | Proposal: add `missing_plant` (MIS) and `sparse` (SPA) to the enum and `TARGET_RE` in `CONTRACT_VERSION 1.1`; the post-Marcaj owner confirms before 23:15. |
| R10 | Rasterio and tifffile JPEG decoding differ by ±1 DN, so the veg mask deviates from the prototype. | Parity test with 99.5% agreement instead of bit-exact; re-run eval to confirm no score drop. |
| R11 | Rounding to 0.1 px can make a valid polygon invalid (thin canopy slivers). | Validity loop in the writer (§3.7); dropped objects counted in the manifest and qa. |
| R12 | Paths contain a space and `&` ("data & info"). | All Makefile and Docker paths quoted; config resolves against the project root; a test uses the real path. |
| R13 | Contract §2.1 says `tile_index` has `nodata_frac`, filled by tile_prep, which would give one file two owners. | ingest writes NaN; tile_prep writes `tile_valid`; `read_tile_index` joins them. Documented in the contract history. |
| R14 | Export blocked at 07:00 by noisy qa errors or a single bad tile. | Staged export with a precise report per tile and object; `--allow-qa-errors` escape hatch (logged); per-tile failure leaves an empty `<image>` flagged in `empty_tiles.csv` with reason `failed`. |
| R15 | Everyone depends on the foundation. | Interface freeze of stubs at 23:15 (`tiling`, `enums`, `ids`, `schema_defs`, config models, `StageSpec`/`RunContext` signatures); the others code against the stubs. |

### Critical Files for Implementation
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/geo/tiling.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/contracts/schema_defs.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/pipeline/runner.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/cvat/export.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/eval/metrics.py