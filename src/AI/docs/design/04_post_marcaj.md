# Design: post-Marcaj chain (import → derive → passable → targets → route → measure → web_bundle → publish)

Root used below: `$AI = /Users/maleticimiroslav/Vin Gigahack/src/AI`. Doc citations: **A** = `ARHITECTURA_Siret3.md` (wins on conflict), **C** = `00_contracte.md`, **R** = `CERCETARE_Siret3.md`.

## 1. Files

### Package modules

| Path | Responsibility | ~Lines |
|---|---|---|
| `$AI/vineyard/cvat/reader.py` | Parses CVAT 1.1 `.xml`, `.zip` (every `*.xml` at any depth) or several files into frozen `RawDoc/RawImage/RawShape`. Matches images by basename + NFC name, not by `id`. Reads coordinates as float. Ignores `id/subset/task_id/group_id/source`. Accepts `<tag>`, empty `<image>` and `<points>`. Uses lxml. | 260 |
| `$AI/vineyard/cvat/rle.py` | Decodes a CVAT `<mask rle left top width height>` to a binary box, then to polygons in px (+0.5 per C§1.4). | 90 |
| `$AI/vineyard/cvat/normalize.py` | Enum and ID normalization (C§4.4). Returns a `Normalized` value and status and never mutates its input. | 130 |
| `$AI/vineyard/annset/model.py` | `AnnSet`, `Provenance`, `ImportResult` frozen dataclasses and layer-name constants. | 90 |
| `$AI/vineyard/annset/io.py` | `read_annset` / `write_annset` (C§2.6: parquet + `annset.json`, atomic). Written only if the foundations subsystem does not own it. | 140 |
| `$AI/vineyard/annset/from_cvat.py` | `RawDoc` → AnnSet frames. Steps: px→UTM, `make_valid`, explode multi-geometries, orient CCW, box→polygon. Also IDs (C§2.4), provenance and per-object metrics (`area_m2`, `n_vertices`, `along_m`, `is_clump`, `touches_edge`, `length_m`). | 320 |
| `$AI/vineyard/annset/canopy_rows.py` | Assigns `canopies.row_id` (nearest row piece in the same tile, ≤ 0.5 m, C§2.5.7). Computes `row_pieces.max_gap_m` with ends included (C§2.5.8). | 140 |
| `$AI/vineyard/annset/import_checks.py` | Image-set checks: unknown names, size ≠ 2048, duplicates across files (last wins), missing tiles vs 311. Also `missing_attr` and `id_case_collision`. | 200 |
| `$AI/vineyard/annset/qa.py` | `QaIssue`, deterministic `Q00001…` numbering, `issues_to_gdf`, `review_queue.csv` (messages in Romanian). | 180 |
| `$AI/vineyard/annset/lines.py` | PCA direction, projection ordering, `midline(a, b)`, lateral offset, block normal (C§1.5), UTM `angle_deg`, linear extrapolation. | 180 |
| `$AI/vineyard/annset/gaps.py` | `Interval/RowOccupancy/Gap`, `row_occupancy`, `find_gaps`, `max_gap_m` (vector port of `attrs.row_gap`). | 220 |
| `$AI/vineyard/annset/merge_rows.py` | C§4.5 steps 1–6 → `rows` layer + qa issues. | 280 |
| `$AI/vineyard/annset/row_order.py` | n·c ordering per block, `row_index`, `spacing_prev/next_m`, `angle_deg`. | 170 |
| `$AI/vineyard/annset/blocks.py` | Blocks from any AnnSet (`union(buffer 2.5).buffer(-2.5)`), block stats, `blocks_should_merge`, `block_split`, `waste_block_unknown`. | 260 |
| `$AI/vineyard/annset/link_interrows.py` | C§4.5.7 → `interrow_pieces_linked` + global `interrows`. | 260 |
| `$AI/vineyard/annset/derive.py` | `DeriveResult` + `derive()` orchestration. | 110 |
| `$AI/vineyard/route/targets.py` | `build_targets`: orchestration, dedupe, IDs, priority, `route_role`, preliminary reachability. | 250 |
| `$AI/vineyard/route/target_rules.py` | Target rules: row gaps (including long-gap sampling), missing plants, sparse, `row_end_short`, `missing_row` → `TargetDraft`. | 320 |
| `$AI/vineyard/route/target_waste.py` | Waste targets: merges `W0001a/b` split boxes into one centre, plus links. | 110 |
| `$AI/vineyard/route/domain.py` | `DomainSet(raw, inner, eroded)` and `build_domain` (union with grid 0.001, seam closing, − forbidden, − canopies, `make_valid`, `prepare`). | 190 |
| `$AI/vineyard/route/skeleton.py` | Polygon → raster at 0.25 m (`rasterio.features`) → `skeletonize` → `sknw.build_sknw(multi=False, iso=False)` → UTM polylines, pruning spurs < 2 m. | 230 |
| `$AI/vineyard/route/centerlines.py` | Interrow midlines between consecutive rows, clipped to `domain.eroded`. Uninked interrow pieces fall back to `skeleton.py`. | 200 |
| `$AI/vineyard/route/graph_types.py` | `WalkGraph`, `CandidateSet`, `Reach` frozen types. **Committed first** so the route package can code against it. | 110 |
| `$AI/vineyard/route/graph.py` | Builds `WalkGraph`: KD-tree node snapping, edge splitting, `simplify(0.25)` guarded by `covered_by`, connected components. | 330 |
| `$AI/vineyard/route/connectors.py` | Row-end connectors, seam/snap joins, START link, target spurs; outside length and cost. | 230 |
| `$AI/vineyard/route/graph_io.py` | `WalkGraph` ↔ `walk_nodes` / `walk_edges` layers. | 120 |
| `$AI/vineyard/route/candidates.py` | `augment_with_targets`: projection nodes per side, spurs, final reachability. | 240 |
| `$AI/vineyard/route/distances.py` | CSR graph, chunked multi-source `scipy.sparse.csgraph.dijkstra`, path extraction. | 150 |
| `$AI/vineyard/route/solver.py` | OR-Tools GTSP: `RoutingIndexManager(n,1,0)`, mandatory `AddDisjunction`, integer costs in cm, `RegisterTransitMatrix`. | 200 |
| `$AI/vineyard/route/solver_fallback.py` | Fallback solver: `python-tsp` local search + LK on representatives, then an exact set-choice DP for the fixed order. | 170 |
| `$AI/vineyard/route/unroll.py` | Tour → LineString (START exact, duplicates removed), string pulling that keeps anchors. | 220 |
| `$AI/vineyard/route/planner.py` | `plan_route`: must/optional phases, coverage loop, node-cap ladder, stops, visits. | 300 |
| `$AI/vineyard/route/baseline.py` | Naive baselines: targets in ID order, and every interrow walked once. | 100 |
| `$AI/vineyard/route/validate.py` | `RouteValidation`: fast `covered_by` check, then exact per-segment `difference`. | 220 |
| `$AI/vineyard/route/geojson.py` | Writes and reads `route.geojson` (2 decimals, `crs` member, START exact, `length_m` computed after rounding). Also `check_route_file`. | 170 |
| `$AI/vineyard/measure/measurements.py` | C§5.2 table (total / block / row). | 280 |
| `$AI/vineyard/measure/csv_io.py` | CSV formatting and rounding, header check, `measurements.json`. | 130 |
| `$AI/vineyard/web/geojson4326.py` | pyproj `always_xy` 32635→4326, 7 decimals, RFC 7946 orientation, `id` = PK. | 150 |
| `$AI/vineyard/web/panels.py` | `rows.json`, `blocks.json`, `route.json`, `summary.json` with `bbox:[w,s,e,n]`. | 220 |
| `$AI/vineyard/web/ortho.py` | `gdal2tiles` XYZ when available; otherwise per-tile 512 px JPEG + `ortho_index.json` with 4 lon/lat corners. | 220 |
| `$AI/vineyard/web/bundle.py` | `build_web_bundle`: layers, canopies split per tile, `utm/` copies, `manifest.json`. | 260 |
| `$AI/vineyard/pipeline/stages/import_marcaj.py` | Stage `import_marcaj`. | 120 |
| `$AI/vineyard/pipeline/stages/import_reference.py` | Stage `import_reference`. | 70 |
| `$AI/vineyard/pipeline/stages/derive.py` | Stage `derive`. | 90 |
| `$AI/vineyard/pipeline/stages/passable.py` | Stage `passable`. | 100 |
| `$AI/vineyard/pipeline/stages/targets.py` | Stage `targets`. | 90 |
| `$AI/vineyard/pipeline/stages/route.py` | Stage `route`. | 140 |
| `$AI/vineyard/pipeline/stages/measure.py` | Stage `measure`. | 80 |
| `$AI/vineyard/pipeline/stages/web_bundle.py` | Stage `web_bundle`. | 90 |
| `$AI/vineyard/pipeline/stages/publish.py` | Stage `publish`: checks + atomic copy to the repo root. | 150 |
| `$AI/vineyard/pipeline/stages/qa.py` | Stage `qa`: aggregates issues into `qa/qa_issues.parquet` + `review_queue.csv`. | 80 |
| `$AI/vineyard/cli_post.py` | typer sub-app: `import-marcaj`, `import-reference`, `derive`, `passable`, `targets`, `route`, `measure`, `web build`, `publish`, `post`, `qa`. | 180 |
| `$AI/vineyard/config/post_import.py` | `ImportCfg`, `DeriveCfg` (pydantic `extra="forbid"`, `frozen=True`). | 90 |
| `$AI/vineyard/config/post_route.py` | `TargetsCfg`, `RouteCfg` with `Domain/Graph/Solver/Validate` sub-models. | 170 |
| `$AI/vineyard/config/post_outputs.py` | `MeasureCfg`, `PublishCfg`, `WebCfg`. | 90 |

### Tests (`$AI/tests/post/`)
- `conftest.py`: shared fixtures (paths to the example XML, `02_route`, small solver time limit).
- `factories.py`: synthetic mini-vineyard AnnSet builder `make_block(n_rows, spacing_m, angle_deg, lengths_m, tiles, gaps)`, the Marcaj-variant XML builder and a synthetic `WalkGraph` builder.
- `fixtures/`: `marcaj_export_sample.zip`, captured from the Friday-night Marcaj test export.
- `test_cvat_reader.py`, `test_normalize.py`, `test_annset_from_cvat.py`, `test_import_checks.py`
- `test_gaps.py`, `test_lines.py`, `test_merge_rows.py`, `test_row_order.py`, `test_blocks.py`, `test_link_interrows.py`
- `test_targets.py`
- `test_domain.py`, `test_skeleton.py`, `test_centerlines.py`, `test_graph.py`, `test_candidates.py`
- `test_solver.py`, `test_unroll.py`, `test_route_validate.py`, `test_planner_e2e.py`
- `test_measure_examples.py`, `test_publish.py`, `test_web_bundle.py`, `test_ortho.py`

## 2. Public interfaces

**CONTRACT** means frozen by C (§2.4–2.7, §5, §7). **REQ** marks additive contract changes I request before the 23:00 freeze; each has a fallback.

```python
# vineyard/cvat/reader.py  (shared: export round-trip self-check + eval use it)
ShapeKind = Literal["polygon", "polyline", "box", "mask", "points", "tag", "track"]
@dataclass(frozen=True)
class RawShape:
    kind: ShapeKind; label: str; order: int
    points_px: tuple[tuple[float, float], ...]          # () for box/tag/mask
    box_px: tuple[float, float, float, float] | None     # xtl, ytl, xbr, ybr
    mask: "MaskRLE | None"
    attributes: tuple[tuple[str, str | None], ...]       # XML order; None = empty element
@dataclass(frozen=True)
class RawImage: tile_id: str; name_raw: str; width: int; height: int; shapes: tuple[RawShape, ...]; src_file: str
@dataclass(frozen=True)
class RawDoc: path: str; sha256: str; images: tuple[RawImage, ...]
class CvatReadError(ValueError): ...
def read_cvat(path: str | Path) -> tuple[RawDoc, ...]
def read_cvat_many(paths: Sequence[str | Path]) -> tuple[RawDoc, ...]
def tile_id_from_image_name(name: str) -> str        # basename, strip ext, NFC; case-sensitive

# vineyard/cvat/normalize.py
@dataclass(frozen=True)
class Normalized: value: str | None; raw: str | None; status: Literal["ok", "normalized", "synonym", "bad", "missing"]
def normalize_enum(raw: str | None, allowed: frozenset[str], synonyms: Mapping[str, str], accept_synonyms: bool) -> Normalized
def normalize_id(raw: str | None) -> Normalized

# vineyard/annset/model.py + io.py  (CONTRACT C§2.6)
@dataclass(frozen=True)
class Provenance: source: Source; run_id: str; model_version: str; confidence: float = 1.0
@dataclass(frozen=True)
class AnnSet:
    canopies: gpd.GeoDataFrame; row_pieces: gpd.GeoDataFrame; interrow_pieces: gpd.GeoDataFrame; waste: gpd.GeoDataFrame
    meta: Mapping[str, Any]            # annset.json; meta["tile_ids"] lists ALL imported images incl. empty ones
def read_annset(annset_dir: Path) -> AnnSet
def write_annset(annset: AnnSet, annset_dir: Path) -> None
@dataclass(frozen=True)
class ImportResult: annset: AnnSet; issues: tuple["QaIssue", ...]
def annset_from_cvat(docs: Sequence[RawDoc], tiles: Mapping[str, TileRef], prov: Provenance,
                     cfg: ImportCfg, tile_valid: gpd.GeoDataFrame | None) -> ImportResult

# vineyard/annset/qa.py
@dataclass(frozen=True)
class QaIssue: severity: Severity; code: str; tile_id: str; object_id: str; message: str; x: float; y: float
def issues_to_gdf(issues: Iterable[QaIssue], prov: Provenance) -> gpd.GeoDataFrame      # qa_issues schema C§2.5.14
def review_queue(qa: gpd.GeoDataFrame) -> pd.DataFrame

# vineyard/annset/gaps.py  (also offered to perception row_attrs)
@dataclass(frozen=True) class Interval: start_m: float; end_m: float
@dataclass(frozen=True) class RowOccupancy: line: LineString; length_m: float; occupied: tuple[Interval, ...]; unknown: tuple[Interval, ...]
@dataclass(frozen=True) class Gap: start_m: float; end_m: float; position: Literal["interior", "head", "tail"]; censored: bool
def row_occupancy(line: LineString, canopies: Sequence[Polygon], corridor_half_m: float, unknown: BaseGeometry | None = None) -> RowOccupancy
def find_gaps(occ: RowOccupancy, min_len_m: float, include_ends: bool) -> tuple[Gap, ...]
def max_gap_m(occ: RowOccupancy, include_ends: bool) -> float

# vineyard/annset/derive.py
@dataclass(frozen=True)
class DeriveResult: rows: gpd.GeoDataFrame; blocks: gpd.GeoDataFrame; interrows: gpd.GeoDataFrame
                    interrow_pieces_linked: gpd.GeoDataFrame; issues: tuple[QaIssue, ...]
def derive(annset: AnnSet, coverage: BaseGeometry, passages: BaseGeometry, forbidden: BaseGeometry,
           cfg: DeriveCfg, prov: Provenance) -> DeriveResult
def merge_rows(row_pieces: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame, coverage: BaseGeometry, cfg: DeriveCfg) -> tuple[gpd.GeoDataFrame, tuple[QaIssue, ...]]
def build_blocks(annset: AnnSet, rows: gpd.GeoDataFrame, passages: BaseGeometry, forbidden: BaseGeometry, cfg: DeriveCfg) -> tuple[gpd.GeoDataFrame, tuple[QaIssue, ...]]
def link_interrows(pieces: gpd.GeoDataFrame, row_pieces: gpd.GeoDataFrame, rows: gpd.GeoDataFrame, cfg: DeriveCfg) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, tuple[QaIssue, ...]]

# vineyard/route/domain.py
@dataclass(frozen=True)
class DomainSet: raw: MultiPolygon; inner: MultiPolygon; eroded: MultiPolygon; grid_size_m: float   # inner/eroded shapely.prepare()d
def build_domain(interrow_pieces: gpd.GeoDataFrame, passages: BaseGeometry, forbidden: BaseGeometry,
                 canopies: gpd.GeoDataFrame | None, cfg: DomainCfg) -> DomainSet
def domain_from_raw(raw: BaseGeometry, cfg: DomainCfg) -> DomainSet

# vineyard/route/targets.py
@dataclass(frozen=True)
class TargetsResult: targets: gpd.GeoDataFrame; extents: gpd.GeoDataFrame; issues: tuple[QaIssue, ...]
def build_targets(annset: AnnSet, derived: DeriveResult, coverage: BaseGeometry, domain: DomainSet | None,
                  forbidden: BaseGeometry, cfg: TargetsCfg, route_cfg: RouteCfg, prov: Provenance) -> TargetsResult

# vineyard/route/graph_types.py + graph.py + graph_io.py
@dataclass(frozen=True)
class WalkGraph:
    node_xy: np.ndarray; node_kind: tuple[str, ...]; node_ref: tuple[str, ...]          # kinds: start|row_end|passage|target|junction
    edge_uv: np.ndarray; edge_geom: tuple[LineString, ...]; edge_len_m: np.ndarray
    edge_cost: np.ndarray; edge_inside_frac: np.ndarray; edge_kind: tuple[str, ...]; edge_ref: tuple[str, ...]
    start_node: int; component: np.ndarray                                             # arrays setflags(write=False)
def build_walk_graph(domain: DomainSet, rows: gpd.GeoDataFrame, interrow_pieces_linked: gpd.GeoDataFrame,
                     passages: BaseGeometry, start_xy: tuple[float, float], cfg: RouteCfg) -> tuple[WalkGraph, tuple[QaIssue, ...]]
def graph_to_layers(g: WalkGraph, prov: Provenance) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
def graph_from_layers(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> WalkGraph

# vineyard/route/candidates.py, distances.py, solver*.py, unroll.py, planner.py
@dataclass(frozen=True) class CandidateSet: target_ids: tuple[str, ...]; nodes: tuple[int, ...]; role: Literal["must", "optional"]
@dataclass(frozen=True) class Reach: target_id: str; reachable: bool; note: str; n_candidates: int; snap_dist_m: float
def augment_with_targets(g: WalkGraph, targets: gpd.GeoDataFrame, domain: DomainSet, cfg: RouteCfg) -> tuple[WalkGraph, tuple[CandidateSet, ...], tuple[Reach, ...]]
def distance_matrix(g: WalkGraph, nodes: Sequence[int], chunk: int) -> np.ndarray          # (k,k) metres
@dataclass(frozen=True) class SolverResult: order: tuple[int, ...]; cost_cm: int; solver: str; solve_time_s: float
def solve_gtsp(d_cm: np.ndarray, sets: Sequence[Sequence[int]], cfg: SolverCfg, time_limit_s: int) -> SolverResult
def unroll(g: WalkGraph, stops: Sequence[int], start_xy: tuple[float, float]) -> tuple[LineString, tuple[int, ...]]
def string_pull(line: LineString, anchors: Sequence[int], domain: DomainSet, cfg: GraphCfg) -> tuple[LineString, tuple[int, ...]]
@dataclass(frozen=True)
class RoutePlan: line: LineString; stops: tuple["Stop", ...]; visits: tuple["TargetVisit", ...]; solver: str
                 solve_time_s: float; iterations: int; optional_delta_m: float | None; baseline: "Baseline"
def plan_route(g: WalkGraph, targets: gpd.GeoDataFrame, domain: DomainSet, start_xy: tuple[float, float], cfg: RouteCfg, final: bool) -> RoutePlan

# vineyard/route/validate.py + geojson.py
@dataclass(frozen=True)
class RouteValidation: length_m: float; closure_m: float; outside_len_m: float; outside_frac: float; n_targets: int
    n_reachable: int; n_visited_est: int; coverage_est: float; coverage_must: float; legs_outside: tuple[tuple[int, float], ...]
    fast_covered: bool; single_linestring: bool; zero_length_segments: int; passed: bool; failures: tuple[str, ...]
def validate_route(line: LineString, domain: DomainSet, start_xy: tuple[float, float], must_xy: np.ndarray, all_xy: np.ndarray, cfg: RouteCfg) -> RouteValidation
def outside_length_m(line: LineString, inner: BaseGeometry, grid_size_m: float) -> tuple[float, np.ndarray]
def write_route_geojson(line: LineString, props: Mapping[str, Any], path: Path, decimals: int) -> LineString
def read_route_geojson(path: Path) -> tuple[LineString, dict]
def check_route_file(path: Path, start_xy: tuple[float, float], domain: DomainSet, cfg: RouteCfg) -> tuple["Check", ...]

# vineyard/measure
MEASUREMENT_COLUMNS: Final = ("level", "vineyard_id", "row_id", "n_blocks", "n_rows", "n_row_pieces", "n_canopies",
  "row_length_m", "canopy_area_m2", "canopy_area_ha", "interrow_area_m2", "interrow_area_ha", "n_waste", "n_targets",
  "max_gap_m", "structure_any", "source", "run_id")                                     # CONTRACT C§5.2
def compute_measurements(annset: AnnSet, rows: gpd.GeoDataFrame, targets: gpd.GeoDataFrame | None, prov: Provenance, cfg: MeasureCfg) -> pd.DataFrame
def write_measurements_csv(df: pd.DataFrame, path: Path, cfg: MeasureCfg) -> None
def measurements_to_json(df: pd.DataFrame, meta: Mapping[str, str]) -> dict
def check_measurements_csv(path: Path) -> tuple["Check", ...]

# vineyard/web
def to_wgs84_featurecollection(gdf: gpd.GeoDataFrame, id_col: str, keep_cols: Sequence[str], decimals: int) -> dict
def tile_corners_lonlat(t: TileRef) -> tuple[tuple[float, float], ...]                  # TL, TR, BR, BL
def build_ortho(tile_index: gpd.GeoDataFrame, out_dir: Path, cfg: WebCfg, workers: int) -> "OrthoResult"
def build_web_bundle(run_dir: Path, out_dir: Path, tile_index: gpd.GeoDataFrame, route_inputs: "RouteInputs", cfg: WebCfg) -> "Manifest"

# vineyard/pipeline/stages/*.py: each exposes STAGE: Stage (foundations' API): name, STAGE_VERSION, cfg_keys, inputs, run(ctx)
```

### Layers read and written (all EPSG:32635 parquet under `work/runs/<run_id>/`, validated with `validate_layer`)

| Stage | Reads | Writes |
|---|---|---|
| `import_marcaj` / `import_reference` | CVAT XML/ZIP, `tile_index` (reference: grid fallback), `tile_valid` (optional) | **CONTRACT** `annset/{canopies,row_pieces,interrow_pieces,waste}.parquet` + `annset.json` (C§2.5.7–10, source = marcaj/reference); `qa/issues_import.parquet` |
| `derive` | AnnSet, `tile_valid`, `in_passages`, `in_forbidden` | **CONTRACT** `layers/rows`, `layers/blocks`, `layers/interrows` (C§2.5.5/6/9); **REQ** `layers/interrow_pieces_linked` (piece columns + `interrow_id/row_left_id/row_right_id` filled); `qa/issues_derive.parquet` |
| `passable` | AnnSet (`interrow_pieces`, `canopies`), `rows`, `interrow_pieces_linked`, `in_passages`, `in_forbidden`, `in_start` | **CONTRACT** `passable_parts`, `passable_domain` (stores `raw`, `erosion_m=0`), `walk_nodes`, `walk_edges` (C§2.5.12) |
| `targets` | AnnSet, derive layers, `passable_domain` (**REQ** DAG edge passable→targets), `tile_valid`, `in_forbidden` | **CONTRACT** `targets` (C§2.5.11 + extra columns `reason`, `route_role`, `along_m`), `target_extents` |
| `route` | `walk_*`, `passable_domain`, `targets`, `in_start` | **CONTRACT** `route`, `route_stops` (C§2.5.13), `metrics/route_validation.json` (C§10); **REQ** `layers/target_visits` (`target_id, reachable_final, reach_note, n_candidates, covered, visit_dist_m, route_role`); `metrics/route_baseline.json`; `exports/route.geojson` |
| `measure` | AnnSet, `rows`, `targets` (**REQ** DAG edge targets→measure, for `n_targets`) | `exports/measurements.csv` (**CONTRACT** C§5.2), `exports/measurements.json` |
| `web_bundle` | all of the above, `tile_index`, tiles | `web.out_dir/**` (**CONTRACT** C§7 manifest/layers) |
| `publish` | `exports/route.geojson`, `exports/measurements.csv`, `passable_domain` | repo-root `route.geojson`, `measurements.csv` (**CONTRACT** C§5.1/5.2); `metrics/publish_report.json` |

## 3. Algorithm notes

### 3.0 Verified input facts (read-only checks run tonight)

**`passages.geojson`**
- One feature: MultiPolygon with 2 polygons, 13 holes (all in polygon 1), 1,487 vertices.
- Net area 40,000.4 m² (exterior 274,992.0 − holes 234,991.6).
- Exterior rings are **CW**, so they are not RFC 7946 compliant; the web export must re-orient.
- Polygon 1 has consecutive duplicate vertices (minimum segment 0.0000 m) and two tiny holes (0.3 m² and 2.3 m²).
- Polygon 0 (2,137 m², E 629913.6–630292.7, N 5219225.6–5219539.5, tiles c017–025 × r032–039) is **disconnected** from the main network.
- Half-width (raster estimate) p10/p50/p90/max: 0.25 / 1.0 / 1.95 / 4.05 m, i.e. typical roads about 2 m wide.
- After eroding (passages − forbidden) by 0.05 / 0.30 / 0.50 m there are still exactly 2 components. The 0.30 m erosion leaves 31,987 m² and 1,790 m².

**`forbidden.geojson`**
- MultiPolygon with 19 polygons and no holes, 744,926.8 m².
- Polygon 3 (740,149 m², the village core) lies mostly outside the study area; only about 6,486 m² of forbidden area is inside it.
- passages ∩ forbidden ≈ 19 m² (raster estimate).

**`study_area.geojson`**
- Polygon with 145 vertices, 815,267.8 m² = 311 × 2,621.44 m².

**`start.geojson`**
- Point (629504.70, 5220250.75), in tile `siret3_r018_c010` at (u, v) = (28.0, 2002.0).
- It is **inside passages polygon 1, not in a hole, 0.868 m from the boundary**. So it is inside both `domain.inner` (−0.05 m) and `domain.eroded` (−0.30 m).
- The passage half-width within 10 m of START reaches 3.25 m.

**Tiles**
- The zips hold 74 + 71 + 78 + 76 + 12 = 311 tiles, including both example tiles and the START tile.

**Examples (vector, shoelace)**

| Tile | Σ canopy area (m²) | Σ interrow area (m²) | Σ row length (m) | Signed px area |
|---|---|---|---|---|
| r021 | 237.1194 | 2068.0311 | 910.1036 | all 699 polygons negative, so CCW in UTM |
| r006 | 299.0556 | 1996.3612 | 1031.4509 | same |

- Example `row_id`s have 2 digits (`V01-R01`). The importer must not assume `R\d{3}` (C§2.4: relaxed rule for marcaj/reference).
- UTM row angles: V01 126.8–133.3° (median 127.5°), V02 111.9–113.3° (median 112.8°). Ordering by n·c descending reproduces R01…Rn on both tiles.
- Interior gaps (canopy extents projected on the axis):
  - r006 has 8 gaps ≥ 5 m: R06 5.054; R08 7.09 / 9.25 / 7.745; R09 12.683 / 7.812 / 12.903; R15 5.009 (the reference labels this row `regular`).
  - Gaps in [2, 5) m: r021 has 22, r006 has 23.

### 3.1 Import (C§4.3/4.4, A§4.13.8, R§6.8)

- **Image names.** `tile_id_from_image_name` uses the basename, strips the extension and applies NFC, and stays case-sensitive. The name must exist in `tile_index`.
  - Unknown names, a size other than 2048, and XML parse errors are **structural**. They are collected, then raise `CvatReadError` listing every offender (never silently skipped).
- **Per-object problems** become `QaIssue`s and the object is skipped. Examples: unknown label, wrong shape for a label, degenerate geometry, `track`.
- **Missing tiles.** Fewer than `grid.expected_tiles` unique images is an error when `import.require_all_tiles` is on; `--partial` lowers it to a warning for intermediate job exports.
- **Normalization is necessary but not enough.** Organizers score the **raw** Marcaj values (rules p.3). So every normalization that changes a raw value also emits a qa warning (`enum_normalized` / `id_whitespace`) that points at the tile and object, so annotators fix it in Marcaj before 15:00. This goes beyond C§4.4, where only synonyms warn.
  - `bad_enum` keeps the raw value in the column. `validate_layer` must relax enum checks for source ≠ model (REQ).
- **Geometry conversion.** px→UTM uses the contract `px_to_utm`. Then `make_valid`. A polygonal Multi/GeometryCollection is exploded into separate objects with the same attributes plus the info issue `geom_repaired`. Then `orient(…, 1.0)`.
  - Boxes are normalized (swap with a warning when `xtl > xbr`) and become UTM rectangles.
  - Masks go through RLE → contours → `index_to_px` (+0.5) with a warning.
- **IDs** (C§2.4):
  - canopy `<tile>:C0001` in XML order;
  - row piece `<row_id>@<tile>`, with `#2`, `#3` for duplicates;
  - interrow piece `<tile>:I001`;
  - waste `W0001`…, sorted by `(tile_id, ytl, xtl)`;
  - `model_version = marcaj-export@<sha8 of the concatenated input bytes>` (reference: `reference-examples@<sha8>`).
- **Port of `analyze.py`'s nearest-axis assignment.** `canopy_rows.assign` uses an STRtree over the tile's row pieces and polygon-to-line distance ≤ 0.5 m. Ties go to the smaller centroid perpendicular distance, instead of centroid-only distance.

### 3.2 Occupancy and gaps (port of `attrs.row_gap` and `fit.corridor`)

The prototype rasterized canopies and binned whole pixels along a 2-point axis. The port changes this:

- Each canopy is clipped to `line.buffer(corridor_half_m, cap_style="flat")` (no raster, so no `np.round` error per C§1.4).
- Clipped vertices are projected with `shapely.line_locate_point` onto the (possibly polyline) row, giving `[min, max]` intervals that are then merged.
- Unknown intervals = the row minus `coverage`, where `coverage` = ∪ tile boxes of `annset.meta["tile_ids"]` ∩ `tile_valid`.
- Gaps touching unknown are `censored`. They are kept for targets only if their **known** length is ≥ the threshold. No target is ever placed in unknown or nodata (C§1.7).
- `max_gap_m(include_ends=True)` feeds `row_pieces.max_gap_m`, matching the prototype's "ends to axis ends" rule. `rows.max_gap_m` uses interior gaps only (C§2.5.5).

### 3.3 Merging row pieces (C§4.5)

1. Take the majority `vineyard_id`, and emit `row_multi_block` if the pieces disagree.
2. Get the PCA direction of all vertices and orient each piece along it.
3. Sort pieces by midpoint projection and concatenate. Drop vertices whose projection does not increase by more than `vertex_dedupe_m`; this makes the "all vertices ordered by projection" rule robust to overlapping duplicate pieces.
4. At each junction, the lateral offset (B's start point to A's extended support line) must be ≤ 0.4 m, else `row_piece_misaligned`.
5. `row_missing_in_tile`: the merged line crosses a tile that is in `coverage` for ≥ `row_missing_min_len_m` but has no piece with that row id.
6. `length_m` = Σ piece lengths.

**Port of `attrs.order`.** The prototype sorted in pixel space (y down). The port uses the C§1.5 normal `n = (−sin θ, cos θ)` flipped so `n_y > 0`, in UTM, with R001 = max n·c. θ is the circular median of the row angles, taken mod 180.

### 3.4 Blocks and interrow linking (C§2.5.6, §4.5.7–8)

- **Blocks.** The outline is `union_all(buffer(objects, 2.5), grid_size).buffer(-2.5)` per `vineyard_id`, over all 4 labels.
  - A MultiPolygon outline means `block_split`.
  - `blocks_should_merge` fires when two IDs are < 5 m apart and the shortest connecting segment does not cross `in_passages`.
  - A `vineyard_id` that appears only on waste gives `waste_block_unknown` (C Q3).
  - `is_garden` is informative only: outline within `garden_forbidden_dist_m` of forbidden and `n_rows ≤ garden_max_rows`.
- **Interrow linking.** For each piece, take its centroid's signed offset along the block normal against the row pieces of the same tile and block. Pick the nearest negative and nearest positive offset within `interrow_link_max_m`.
  - When both row IDs match `-R(\d+)$`: `interrow_id = f"{vid}-I{min(k):03d}"`.
  - Otherwise use `row_index` ordering.
  - A piece with only one side is `interrow_unlinked` (REQ qa code).

### 3.5 Targets (A§4.10 merged with C§2.5.11)

| Kind | Rule |
|---|---|
| `row_gap` (GAP) | Interior gap ≥ `gap_min_m` on the global row. One target at the centre if length ≤ `long_gap_m`; otherwise `n = ceil(L / gap_step_m)` targets at sub-segment centres (A). A `target_extents` segment covers the whole gap (C). Extents coverage is reported only: sample every `gap_sample_step_m` and count points within 2 m of the route. |
| `missing_plant` | Gaps in [`missing_min_m`, `gap_min_m`). Only when `include_missing` is true. |
| `sparse` | 20 m windows (step 10 m) with occupancy < 0.10, ≥ 90% of the window known, and no gap ≥ 5 m. Only when `include_sparse` is true. |
| `row_end_short` (END) | (a) The global row line runs ≥ `end_short_min_m` beyond its last canopy; or (b) both neighbouring rows extend ≥ `end_short_min_m` beyond this row's end, target on the linearly extrapolated axis. Both require the end **not** to lie within `edge_margin_m` of the `coverage` boundary; this is why the isolated example tiles give 0. (b) also needs a well-defined comparison: the two neighbours on opposite sides, each ≥ `end_neighbour_min_offset_m` across the row (collinear fragments of one row split at a tile seam, and a spurious row inside an interrow whose neighbours are half a spacing away, are no comparison), and the whole extension imaged (it never crosses unprocessed tiles or nodata). No END on the outermost rows of a block (`end_skip_outer_rows`). |
| `missing_row` (MRW) | Consecutive spacing > `missing_row_spacing_factor` × block median. The virtual axis is `lines.midline(a, b)`, sampled like long gaps. |
| `waste` (WST) | Centre of each waste box; split boxes `Wxxxxa/b` are merged first. |

- `missing_plant` and `sparse` are not in the contract `TargetKind`. REQ adds `missing_plant` (T-MSP) and `sparse` (T-SPR); fallback is `kind=other`, `reason=…`.
- **Priority:** 1 = waste and gaps > 10 m; 2 = other GAP, END, MRW; 3 = MSP, SPR.
- **Route role** (user decision, 26.09): `route_role` = `must` for the kinds in `targets.must_kinds` (default `row_gap`, `missing_row`, `waste`), `optional` for every other kind (`missing_plant`, `row_end_short`, `sparse`). Optional targets are routed only when cheap (§3.8 phase B).
- **Edge artefacts:** row-derived targets (every kind but waste) within `targets.edge_margin_m` (3 m) of the coverage boundary (tile_valid union: unprocessed tiles and nodata) are dropped (`edge_dropped` in `metrics/targets.json`, next to `end_outer_row`, `end_ill_defined`, `end_on_boundary`). 3 m ≈ one row spacing (2.3–3.0 m measured) and is the largest margin that keeps every organizer-annotated row gap of the example tiles (nearest 3.25 m).
- **Dedupe:** targets within `dedupe_m` merge and keep the best priority.
- **Preliminary reachability:** `false` if the target is in forbidden (`in_forbidden`) or farther than `candidate_radius_m` from `domain.inner`. The final graph-based reachability is decided in `route`.

### 3.6 Domain (A§4.11.1 + C§2.5.12)

1. `union_all(interrow_pieces ∪ passages, grid_size=0.001)`.
2. **Seam closing** `buffer(+s, join_style="mitre").buffer(-s)` with `s = seam_close_m` = 0.05 (C `snap_tol_m`). Without it, a 1–2 cm gap between Marcaj-edited pieces at tile edges is widened to about 10 cm by `inner`. That would add about 0.1 m outside per crossing (~2,000 crossings), roughly 0.7%.
3. `− forbidden`, then `− union(canopies)` if `subtract_canopies`.
4. `make_valid`, keep the polygonal parts.
5. `inner = buffer(-0.05)`, `eroded = buffer(-0.3)`, then `shapely.prepare` on both.

### 3.7 Graph (A§4.11.2, R§4.7–4.8)

- **Centerlines.** For consecutive `row_index` pairs per block:
  - sample row A every 2 m and project onto B;
  - drop samples whose projection clamps to B's ends, so only the overlap is kept;
  - take the midpoints, build a line, `intersection(domain.eroded)`, and keep pieces ≥ `centerline_min_len_m`.
  - Interrow pieces not within 0.5 m of any centerline edge fall back to the skeleton of that piece.
- **Deviation from A (explained):** graph **nodes** are placed only at edge ends, junctions and target projections. The "every 2 m / every 5 m" spacing becomes geometry vertex density only. With about 190 km of interrows, one node every 2 m would mean about 95k nodes, and the k × N Dijkstra would need about 1 GB. Projection nodes (§3.8) make dense nodes unnecessary for candidate discovery.
- **Passage skeleton.** Take (passages ∩ domain).buffer(−0.3), rasterize with `rasterio.features.rasterize` on each component's bbox at 0.25 m (≈ 46 Mpx total, bool), run `skimage.morphology.skeletonize`, then `sknw.build_sknw(ske, multi=False, iso=False)`.
  - Edge `pts` go to UTM via pixel centres (`x = minx + (c + 0.5)·res`), with the node `o` prepended and appended.
  - Terminal edges < 2 m are pruned, repeated until stable (at most 3 passes).
  - Each edge is `simplify(0.25)`, kept only if `covered_by(eroded)`; otherwise the original is kept.
- **Connectors** (`kind=connector`, cost = `length + connector_outside_penalty × outside_len`, `outside_len` measured per segment against `inner`):
  - row-end node → nearest point on another edge within `connector_max_m`;
  - consecutive clipped pieces of the same centerline;
  - KD-tree snap joins ≤ `snap_join_m` that are `covered_by(inner)`;
  - **START** → nearest edge point with the segment `covered_by(inner)`. If that fails, the stage stops with an error.
- **Components** come from `scipy.sparse.csgraph.connected_components`. Every component without START gives a `domain_disconnected` qa issue with its edge length.

### 3.8 Candidates and GTSP (A§4.11.3–5, R§4.5–4.6)

- For each reachable target, an STRtree `dwithin(1.9 m)` query over edges gives a projection point per edge.
  - The side is the sign of the cross product with the row direction (for waste: edge identity).
  - Keep ≤ `max_candidates_per_side` per side. Split those edges once, at all sorted parameters, and rebuild the graph immutably.
  - If no edge is within 1.9 m but one is within `max_snap_m`: add a spur edge to the point 1.5 m short of the target (`target_spur`) when `covered_by(inner)`. Otherwise the target is unreachable (`too_far`). Candidates outside START's component make it unreachable (`disconnected`).
- **TSP index space.** START at index 0, plus a **per-target duplicate** of every candidate node, so every node is in exactly one mandatory disjunction (the doc's "targets on the same node are deduplicated" becomes zero-cost duplicates).
- **Distances.** `dijkstra(csr, indices=chunk)` in chunks of 256 unique nodes; keep only the k × k block; costs `int(round(d·100))`.
- **OR-Tools.**
  - `RegisterTransitMatrix(D_cm.tolist())` (C++ API confirmed in `routing.h`; the Python callback path is the fallback).
  - `routing.AddDisjunction(idxs)` with `kNoPenalty` = hard, exactly one index active.
  - PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH, `time_limit` 30 s (60 s with `--final`).
- **Fallback** (on any exception or no solution):
  1. pick a representative per set (the candidate nearest START);
  2. `python_tsp.heuristics.solve_tsp_local_search(D, x0=nn_tour, perturbation_scheme="two_opt", max_processing_time=20)`, then `solve_tsp_lin_kernighan`;
  3. exact DP over the fixed set order that picks the best member per set (O(Σ|Sᵢ||Sᵢ₊₁|)).
- **Phases.**
  - Phase A: `must` targets only.
  - Coverage loop (≤ `cover_iterations`): `dwithin(targets, line, candidate_radius_m)`. Targets covered for free drop out; re-solve; keep the result only if it is shorter **and** still covers 100% of must targets. Otherwise revert. Iteration 0 is 100% by construction, because every must target has an anchor within 1.9 m.
  - Phase B (optional targets): add those still uncovered **whose cheapest insertion into the must tour** (between two consecutive stops, graph cost = metres + outside penalty) is ≤ `route.solver.optional_max_detour_m` (25 m), re-solve, and record `optional_delta_m` (A§4.10: the length cost of `include_missing`). The others get `reach_note=optional_detour`. Without must targets every optional target is routed.
- **Node-cap ladder** when TSP nodes exceed `max_tsp_nodes`:
  1. 1 candidate per side;
  2. merge targets whose projections are within `merge_node_m` on the same edge;
  3. truncate optional targets by priority, then gap length (reported);
  4. if must targets alone still exceed the cap: raise with the counts.

### 3.9 Unroll, string pulling, validation (A§4.11.5–6, R§4.8–4.10)

- **Unroll.** Concatenate the oriented edge geometries of each Dijkstra leg (a single-source run per leg, with predecessors). Set the first and last vertex to START exactly and drop consecutive duplicates.
- **String pulling** only between consecutive **anchors** (stops and START), so coverage is preserved:
  - build a vectorized window of candidate segments `[p_i, p_j]` for j ≤ min(next anchor, i + `pull_lookahead`);
  - run `shapely.covered_by(segs, eroded)` on the prepared geometry and take the largest j that is True;
  - `j = i + 1` is always allowed, because it is an original edge.
- **Validation** runs on the geometry re-read from the written GeoJSON (2 decimals):
  - (a) fast: `covered_by(line, inner)`;
  - (b) exact: explode into **individual segments**, then `shapely.difference(segs, inner, grid_size=0.001)` and sum the lengths. A dead-end interrow walked out and back has collinear self-overlaps, and an overlay on the whole LineString would count the overlapping outside parts only once;
  - (c) `closure_m` = distance of first and last vertex to START (must be ≤ 0.01; the writer sets them exactly);
  - (d) LineString, `is_valid`, 0 zero-length segments;
  - (e) 100% of must targets within `visit_radius_m` (blocking); optional and all-target coverage reported;
  - **no `is_simple`**.
  - **Outside policies** (policies.py, policy_choice.py): every policy is probed in order; a probe is *acceptable* when it validates and its outside share is ≤ `route.plan_outside_frac` = `max_outside_frac_publish` − `plan_outside_margin` (0.015 − 0.001). Probing stops at the first acceptable plan that loses no must target; otherwise the plan with most must targets visited, then optional visited, then interrows reachable, then shortest wins (none acceptable: smallest outside share). The chosen policy is solved again in full; `route_validation.json` reports `policy`, `policy_accepted`, `plan_outside_limit` and one line per probed policy.
- **Baselines.**
  - `baseline_id_order_m` = Σ D over targets in `target_id` order using the first candidate;
  - `baseline_all_interrows_m` = Σ centerline edge lengths.
  - Savings are reported in km, % and minutes at 4 km/h (A§4.11.8).

### 3.10 Measurements, publish, web

- **Measurements.** Canopy area = Σ over tiles of `area(union(canopies in tile))` (block level: per tile per `vineyard_id`). Interrow area = union likewise, which is what the contract specifies; `measurements.json` also carries `interrow_area_sum_m2`, and a qa warning fires if union and sum differ by > 0.1%.
  - `n_blocks` counts distinct non-empty `vineyard_id` over all 4 labels.
  - Values are written with fixed formatting (`f"{x:.2f}"`, ha `:.4f`); NaN becomes an empty cell. Sort is block, then row, by ID.
- **Publish** recomputes everything; it never trusts the file's own properties:
  - `outside_frac` against this run's `passable_domain`, must be ≤ `route.max_outside_frac_publish` (0.015; the official elimination is 0.02);
  - closure ≤ 0.01; single LineString;
  - `length_m` equals the recomputed length within 0.01;
  - `crs` member is `urn:ogc:def:crs:EPSG::32635`; coordinates have ≤ 2 decimals;
  - CSV header is exact, with exactly one `total` row, and Σ block `row_length_m` equals the total within 0.05;
  - optional `require_source`.
  - Refusal raises `PublishRefused` (exit ≠ 0) with the reasons. On success the copy is atomic (`tmp` + `os.replace`).
- **Web.** Transforms use a `pyproj.Transformer(32635 → 4326, always_xy=True)`, then `np.round(…, 7)` via `shapely.transform`, then `orient(…, 1.0)` (fixes the CW source rings). Canopies go to `canopies/<tile_id>.geojson` with `id, vineyard_id, row_id, area_m2`.
  - **Ortho.** If `gdal2tiles.py` and `gdalbuildvrt` are both on PATH, run A§4.14's commands into `ortho/{z}/{x}/{y}.webp`. Otherwise, per tile:
    1. rasterio decimated read at `(3, 512, 512)` with `Resampling.average`;
    2. `cv2.imencode(".jpg", BGR, q80)`;
    3. write `ortho_index.json` with `{tile_id, file, corners: [TL, TR, BR, BL]}`, the order MapLibre's `image` source expects.
  - The manifest's `ortho.mode` tells the frontend which layout exists.

## 4. Config keys I own

These are single-YAML sections with pydantic `extra="forbid"` and `frozen`. `import` is a Python keyword, so the field is `import_: ImportCfg = Field(alias="import")`. Paths are relative to `$AI`.

| YAML path | Default | Source |
|---|---|---|
| `import.accept_enum_synonyms` | true | C§9 |
| `import.duplicate_policy` | last_wins | C§9 |
| `import.require_all_tiles` | true | A§4.13.8 |
| `import.enum_synonyms` | {"bare soil": bare_soil, bare: bare_soil, grass: vegetation, veg: vegetation, unknown: unassessable} | C§4.4 |
| `import.canopy_row_assign_max_m` | 0.5 | C§2.5.7 |
| `import.min_line_length_m` | 0.05 | C§2.7 |
| `import.accept_masks` | true | C§4.3 |
| `derive.block_buffer_m` | 2.5 | C§2.5.6/§4.5.8 |
| `derive.join_lateral_max_m` | 0.4 | C§4.5.4 |
| `derive.row_missing_min_len_m` | 0.5 | C§9 `export.min_row_piece_m` |
| `derive.vertex_dedupe_m` | 0.001 | A§4.11.1 grid |
| `derive.blocks_merge_warn_m` | 5.0 | C§4.5.8 |
| `derive.interrow_link_max_m` | 3.8 | A§3.6 spacing gate |
| `derive.min_rows_per_block_warn` | 3 | A§4.4.3 |
| `derive.garden_forbidden_dist_m` / `garden_max_rows` | 20.0 / 15 | new (informative `is_garden`) |
| `derive.gap_corridor_half_m` | 0.30 | A§4.8 |
| `targets.gap_min_m` | 5.0 | A§3.6 / C§9 |
| `targets.long_gap_m` | 20.0 | A§4.10 |
| `targets.gap_step_m` | 15.0 | A§3.6 |
| `targets.gap_sample_step_m` | 3.0 | C§9 (extents reporting only) |
| `targets.missing_min_m` | 2.0 | A§3.6 |
| `targets.include_missing` | true | A§3.6 (Slack Q3) |
| `targets.include_sparse` | true | A§4.10 |
| `targets.sparse_window_m` / `sparse_step_m` / `sparse_occ_max` / `sparse_known_min` | 20.0 / 10.0 / 0.10 / 0.9 | A§4.10 (+ new step and known fraction) |
| `targets.end_short_min_m` | 5.0 | C§9 |
| `targets.missing_row_spacing_factor` | 1.8 | C§9 |
| `targets.include_waste` | true | C§9 |
| `targets.priority_long_gap_m` | 10.0 | C§2.5.11 |
| `targets.dedupe_m` | 1.0 | new |
| `targets.must_kinds` | [row_gap, missing_row, waste] | user decision 26.09 |
| `targets.edge_margin_m` | 3.0 | user decision 26.09 (replaces the 0.5 m END boundary constant) |
| `targets.end_skip_outer_rows` / `end_neighbour_min_offset_m` | true / 1.5 | user decision 26.09 (1.5 m ≈ 0.6 × row spacing: rejects collinear fragments and spurious rows inside an interrow) |
| `targets.corridor_half_m` | 0.30 | A§4.8 |
| `route.start_file` | 02_route/start.geojson | C§9 |
| `route.start_tolerance_m` | 5.0 | A§3.6 (= C `return_tol_m`) |
| `route.visit_radius_m` | 2.0 | A/C |
| `route.candidate_radius_m` | 1.9 | A§3.6 (replaces C `plan_radius_m` 1.5) |
| `route.max_candidates_per_side` | 3 | A§3.6 |
| `route.max_snap_m` | 5.0 | C§9 |
| `route.max_outside_frac_official` | 0.02 | C§9 |
| `route.max_outside_frac_publish` | 0.015 | user decision 26.09 (was 0.005): the one limit of publish, validator, passable report and (minus the margin) planner |
| `route.plan_outside_margin` | 0.001 | new: the planner accepts ≤ 0.014 |
| `route.solver.include_optional` / `optional_max_detour_m` | true / 25.0 | new (were module constants / optional = visited when cheap) |
| `route.solver.probe_time_limit_s` / `budget_rounds` | 5 / 10 | were module constants |
| `route.walking_speed_kmh` | 4.0 | C§9 |
| `route.domain.grid_size_m` | 0.001 | A§3.6 |
| `route.domain.inner_buffer_m` | 0.05 | A§3.6 |
| `route.domain.eroded_buffer_m` | 0.3 | A§3.6 `string_pull_erode_m` (replaces C `erosion_m` 0.25) |
| `route.domain.seam_close_m` | 0.05 | C§9 `snap_tol_m` |
| `route.domain.subtract_canopies` | true | C§2.5.12 |
| `route.graph.centerline_step_m` | 2.0 | A§4.11.2 |
| `route.graph.centerline_min_len_m` | 1.0 | new |
| `route.graph.skeleton_res_m` | 0.25 | A§3.6 |
| `route.graph.skeleton_erode_m` | 0.3 | A§4.11.2 |
| `route.graph.spur_min_m` | 2.0 | A§3.6 |
| `route.graph.simplify_tol_m` | 0.25 | A§4.11.2 |
| `route.graph.connector_max_m` | 6.0 | new |
| `route.graph.connector_outside_penalty` | 10.0 | new (A: "penalizare suplimentară") |
| `route.graph.snap_join_m` | 0.5 | new |
| `route.graph.pull_lookahead` | 200 | new |
| `route.graph.target_spur_standoff_m` | 1.5 | C§9 `plan_radius_m` |
| `route.solver.method` | auto | C§9 |
| `route.solver.time_limit_s` / `time_limit_final_s` | 30 / 60 | A§3.6 |
| `route.solver.first_solution` / `metaheuristic` | PATH_CHEAPEST_ARC / GUIDED_LOCAL_SEARCH | A§3.6 |
| `route.solver.cost_per_m` | 100 | A§3.6 `cost_unit: cm` |
| `route.solver.cover_iterations` | 3 | A§3.6 |
| `route.solver.fallback_time_s` | 20 | A§4.11.5 |
| `route.solver.max_tsp_nodes` | 1500 | A§4.11.3 |
| `route.solver.merge_node_m` | 1.0 | new |
| `route.solver.dijkstra_chunk` | 256 | new |
| `route.validate.coord_decimals` | 2 | C§1.8 |
| `route.validate.closure_max_m` | 0.01 | C§5.1 |
| `route.validate.length_tol_m` | 0.01 | C§5.1 |
| `route.validate.zero_len_eps_m` | 0.005 | new |
| `measure.round_m` / `round_ha` | 0.01 / 0.0001 | C§9 |
| `measure.total_label` | ALL | C§5.2 |
| `measure.area_union_sum_warn_frac` | 0.001 | new |
| `publish.root_dir` | ../.. | user decision |
| `publish.require_source` | null (Makefile `final` sets marcaj) | new |
| `publish.sum_check_tol_m` | 0.05 | new |
| `web.out_dir` | ../Web/data | user decision |
| `web.survey_id` / `survey_name` | siret3 / Sireț3 (bundle dir `surveys/<survey_id>/pipeline/`; id `^[a-z0-9][a-z0-9-]{1,39}$`, name 2-160 chars, as the web's `public.survey`) | C§6.1 |
| `web.geojson_decimals_4326` | 7 | C§9 |
| `web.utm_decimals` | 3 | C§1.8 |
| `web.ortho_mode` | auto | A§4.14 + C§7 |
| `web.ortho_px` / `ortho_quality` | 512 / 80 | C§9 |
| `web.canopy_minzoom` | 18 | C§9 |
| `web.gdal2tiles` | {zoom: "14-20", resampling: average, tiledriver: WEBP, webp_quality: 75, processes: 8} | A§4.14 |
| `web.canopy_mvt` | auto (tippecanoe if on PATH) | A§4.14 |
| `web.utm_copies` | true | C§7 |

## 5. Tests to write first (TDD)

**`test_cvat_reader.py`**
- Example XML gives 2 images:
  - r021 has 25 polylines (each 2 points) and 423 polygons;
  - r006 has 26 polylines and 276 polygons;
  - 0 boxes, 0 tags.
- The same XML nested at `export/annotations.xml` in a tmp zip parses identically.
- Marcaj variant (`id="57"`, `name="images/siret3_r021_c012.tif"`, `subset`, `task_id`, `source="file"`, `group_id`, `"2048.00,32.10"`) gives the same coordinates (1e-9).
- Name without an extension resolves. An NFD name resolves via NFC. `SIRET3_R021_C012.tif` raises `CvatReadError`.
- Empty `<image/>` is kept with 0 shapes. `<tag>`/`<points>` produce info issues. `<track>` is an error issue and is skipped.
- Mask with rle "2,8,2" in a 4×3 box at (10, 20) gives 8 px² (0.005 m²).
- Duplicate image across 2 files: last wins, with a warning.

**`test_normalize.py`**

| Input | Result |
|---|---|
| "Regular" | regular, status normalized, warning |
| "bare soil", "bare-soil" | bare_soil |
| "grass" | vegetation (synonym) |
| "unknown" | unassessable |
| "foo" | bad, raw kept |
| "" / None | missing |
| " V01 " | "V01" + `id_whitespace` |

- `V03` and `v03` both present: `id_case_collision` error.

**`test_annset_from_cvat.py`** (examples, source = reference)
- Counts:
  - canopies 399 / 251 (650);
  - row_pieces 25 / 26 (51);
  - interrow_pieces 24 / 25 (49);
  - waste 0.
- `vineyard_id`: all 448 r021 objects are V01, all 302 r006 objects are V02.
- `row_structure`: r021 25 regular; r006 21 regular / 5 disrupted.
- `interrow_cover`: r021 24 bare_soil; r006 21 bare_soil / 4 mixed.
- Σ canopy area 237.12 / 299.06 (±0.01 m²); Σ interrow area 2068.03 / 1996.36; Σ row length 910.10 / 1031.45.
- All exteriors are CCW in UTM and valid.
- V01-R01 first point (2048.0, 32.1) px maps to UTM (629657.6, 5220146.3975) within 1e-6.
- IDs: `siret3_r021_c012:C0001…C0399`, `V01-R01@siret3_r021_c012`, `siret3_r021_c012:I001…I024`.
- `confidence` is 1.0.
- Canopies per row: V02-R17, R24, R25, R26 have 1 each; V02-R01 has 2.
- `row_pieces.max_gap_m` (ends included), each ±0.1: V02-R09 12.9, V02-R07 11.33, V02-R15 5.0, V01-R16 3.8.

**`test_import_checks.py`**
- 310 of 311 images with `require_all_tiles=true` is an error; with `--partial` it is a warning listing the missing tile.
- A 1024-px image raises `CvatReadError`.
- A row without `row_id` gives `missing_attr`.

**`test_gaps.py`** (50 m line)
- Canopies [0,10], [15,20], [40,50] give interior gaps 5.0 and 20.0; exactly 5.0 counts.
- Unknown [25,30] splits the 20 m gap into [20,25] (5.0, censored, kept) and [30,40].
- `max_gap_m(include_ends=True)` with canopies [3,10], [15,45] is 5.0.
- A polyline row works.

**`test_merge_rows.py`**
- Collinear pieces in r010_c010 and r010_c011 merge into 1 row with `length_m` = `extent_m` = 102.4 and `tile_ids="siret3_r010_c010,siret3_r010_c011"`.
- Offset 0.6 m gives `row_piece_misaligned`; 0.3 m gives nothing.
- Same `row_id` twice in one tile gives `dup_row_in_tile` and `#2`.
- A missing middle piece gives `row_missing_in_tile`, but nothing if that tile is not in coverage.
- V01/V01/V02 gives `row_multi_block` with V01 chosen.
- Reference gives 51 rows.

**`test_row_order.py`**
- UTM angles: V01 rows in [126.5, 133.5], V02 rows in [111.5, 113.5].
- `row_index` equals the numeric suffix for all 51 rows (R01 = max n·c).
- Median spacing: V02 2.53 ± 0.02, V01 2.78 ± 0.05.

**`test_blocks.py`**
- Reference gives 2 blocks (V01 with 25 rows / 399 canopies, V02 with 26 / 251), each outline ≤ 2621.44 m².
- Same ID 8 m apart gives `block_split`.
- Two IDs 3 m apart give `blocks_should_merge`, but not when a passage strip lies between.
- A waste-only ID gives `waste_block_unknown`.

**`test_link_interrows.py`**
- Reference: all 49 pieces linked.
- r021's first piece is `V01-I001` (R01 | R02); 24 / 25 distinct `interrow_id`s.
- Free-text row IDs fall back to `row_index` ordering (`V01-I003`).
- A one-sided piece gives `interrow_unlinked`.

**`test_targets.py`** (reference, `include_missing=false`)
- r006 gives exactly 8 T-GAP targets: R06 ×1, R08 ×3, R09 ×3, R15 ×1. r021 gives 0.
- Each gap target is ≤ 0.05 m from its axis.
- Priority 1 for exactly 2 targets (R09 12.68 and 12.90).
- 0 END targets.
- IDs match `^T-GAP-\d{4}$`, ordered by (vineyard_id, row_id, along).

Other cases:
- `include_missing=true` gives about 22 (±2) MSP targets on r021 and 23 (±2) on r006.
- Synthetic long gaps: 40 m gives 3 targets at 6.67 / 20 / 33.33; 20 m gives 1; 20.5 m gives 2. Extents length equals gap length.
- Spacing 2.5 with one pair at 5.2 m gives MRW targets; a pair at 4.0 m gives none.
- Split boxes `W0001a/b` give 1 WST target at the union centre.
- Waste in forbidden: `reachable=False, reach_note=in_forbidden`. Waste 3 m outside the domain: `reachable=False`.
- No target lies in a `tile_valid` hole.

**`test_domain.py`** (real `02_route`, passages only)
- 2 polygons; 39,950 < area < 40,001 m².
- START ∈ `inner` and ∈ `eroded`; boundary distance is in [0.85, 0.90] m.
- Adjacent rectangles with a 0.01 m gap close to 1 polygon; with a 0.2 m gap they stay 2.
- Bowtie input is repaired.
- Canopy and forbidden are subtracted.

**`test_skeleton.py`**
- A 3 m-wide L corridor gives a path length equal to the centreline ±0.5 m.
- A 1 m stub is pruned; a 3 m stub is kept.
- `@slow` real passages: exactly 2 components; the small one lies in the bbox E 629913–630293, N 5219226–5219540.

**`test_centerlines.py`**
- Rows 2.6 m apart, lengths 50 and 40 (overlap 40): midline ≈ 39.4 m at 1.3 m from both.
- Fan of 2°: equidistance within 0.01 m.
- A tree hole gives 2 pieces.

**`test_graph.py`**
- Mini-vineyard (4 rows × 60 m, 4 m headland passages, START in a passage) gives 1 component.
- Non-connector edges are `covered_by(eroded)`.
- An interrow end 1 m short of the passage gives a connector with `inside_frac < 1` and cost = len + 10 × outside.
- The START segment is `covered_by(inner)`.
- `graph_io` round-trip is identical.

**`test_candidates.py`**
- A target on the middle axis gets 2 candidates at 1.3 m, or 1 with `max_candidates_per_side=1`.
- A target 2.5 m from edges but inside the domain gets a spur.
- 6 m away: unreachable `too_far`. In another component: `disconnected`.

**`test_solver.py`**
- GTSP toy where the far-side candidate is optimal: OR-Tools equals brute force (≤ 6 sets).
- Fallback + set-DP equals the same optimum.
- 1.004 m becomes 100 cm.
- With ortools import monkeypatched away, `solver == "python_tsp"`.

**`test_unroll.py`**
- Route starts and ends exactly at START, with no consecutive duplicates.
- String pulling on a zigzag shortens it, keeps anchors, and stays `covered_by(eroded)`.

**`test_route_validate.py`**
- 3% outside is rejected (`outside_frac ≈ 0.03`).
- A valid out-and-back (not simple) passes.
- 1 m outside traversed twice gives `outside_len_m == 2.0`.
- START off by 0.02 fails. MultiLineString fails. A zero-length segment fails.
- A must target at 2.05 m fails; at 1.95 m it passes.

**`test_planner_e2e.py`**
- Mini-vineyard with 6 GAP + 1 WST: validation passes, `coverage_must == 1.0`, `outside_frac ≤ 0.005`, length < `baseline_all_interrows_m`.
- A target covered for free is dropped in iteration 2 and final coverage is still 100%.
- Two runs of the tiny instance give the same length.

**`test_measure_examples.py`** (reference)

| Level | Expected |
|---|---|
| total | n_blocks 2, n_rows 51, n_row_pieces 51, n_canopies 650, row_length_m 1941.55, canopy_area_m2 = independent shapely union (in (536.0, 536.2), tolerance ±0.01), interrow_area_m2 4064.39 ±0.01, canopy_area_ha 0.0536, interrow_area_ha 0.4064 |
| V01 | 910.10 m, 25 rows |
| V02 | 1031.45 m, 26 rows |
| row V02-R09 | max_gap_m 12.90, structure_any disrupted |

- The CSV header string is exact; 54 data lines; the total row has `vineyard_id=ALL`; n/a cells are empty.

**`test_publish.py`**
- A valid run is copied byte-identically.
- Refusals: outside 0.006; closure 0.02; MultiLineString; `length_m` off by 0.02; bad header; `require_source=marcaj` on a reference run.
- The copied file has the `crs` member, 2 decimals, and first = last = [629504.7, 5220250.75].

**`test_web_bundle.py` / `test_ortho.py`**
- START (629504.70, 5220250.75) transforms to (28.7073776, 47.1230335) within 2e-7 (the README's values).
- Coordinates have ≤ 7 decimals, exteriors are CCW, `id` = PK.
- `canopies/siret3_r021_c012.geojson` has 399 features; r006 has 251.
- The manifest counts match the files; `summary.json` totals equal the measurements total.
- Fallback ortho gives 512×512 JPEGs and `ortho_index.json` with 4 corners (TL, TR, BR, BL).
- Each `rows.json` bbox contains its row.

## 6. Dependencies

**Needed from other subsystems**
- **contracts:** `enums` (Source, RowStructure, InterrowCover, TargetKind, EdgeKind, Severity, Label), `ids` (formatters and regexes), `schemas.validate_layer(gdf, name, strict)` with relaxed enums for source ≠ model (REQ), `CONTRACT_VERSION`.
- **geo/tiling:** `TileRef`, `tile_ref_from_grid`, `px_to_utm`, `utm_to_px`, `index_to_px`, `tiles_for_bounds`, `GSD_M`, `TILE_M`.
- **geo/vector_io:** `read_layer`, `write_layer`, `write_geojson(crs member, decimals)`; `geo/ops`: `make_valid` / `orient` / `explode`.
- **pipeline:** `Stage` registry, `RunContext`, cache keys, `atomic_write`, `runs/LATEST_*`, `parallel_map` (for ortho). **config:** root loader that mounts my 3 section modules.
- **ingest:** `tile_index`, `in_passages`, `in_forbidden`, `in_study_area`, `in_start`. I fall back to reading `02_route` directly when ingest has not run.
- **tile_prep:** `tile_valid`. If absent: full tiles are assumed, with a warning.
- **assemble:** AnnSet(model) with the same schema, plus `tile_status`. **export_cvat:** `id_registry.json` (optional QA of new IDs).
- **Contract REQs (before 23:00):**
  - TargetKind += `missing_plant` (MSP), `sparse` (SPR);
  - layers += `interrow_pieces_linked`, `target_visits`;
  - DAG edges passable→targets and targets→measure;
  - qa codes += `interrow_unlinked`, `enum_normalized`, `id_whitespace`, `domain_disconnected`, `target_unreachable`, `connector_outside`, `block_too_few_rows`, `images_missing`, `geom_repaired`.

**Provided to others**
- `cvat.reader.read_cvat`: for export_cvat's round-trip self-check (C§4.1) and eval. This is **early and on the critical path**.
- `import_reference` → AnnSet(reference): for eval's canopy ≥ 0.80 / axes ≥ 0.95 regression gate before Publish.
- `annset.gaps.row_occupancy` / `find_gaps` (optional reuse in `row_attrs`).
- `annset.qa.QaIssue` / `issues_to_gdf` (other stages can emit the same issue type).
- Web data bundle and `manifest.json` for the frontend in `src/Web`.
- `review_queue.csv` for the Marcaj correction team.
- Repo-root `route.geojson` and `measurements.csv`.

## 7. Work packages

**WP1: CVAT reader and import** (`cvat/reader`, `rle`, `normalize`; `annset/model`, `io`, `from_cvat`, `canopy_rows`, `import_checks`, `qa`; stages `import_marcaj`, `import_reference`; `config/post_import.ImportCfg`)
- **In:** example XML, synthetic Marcaj variants, contracts/tiling.
- **Out:** `read_cvat`, AnnSet(reference) on disk.
- **Accept:** `test_cvat_reader`, `test_normalize`, `test_annset_from_cvat`, `test_import_checks` green, ≥ 80% coverage.
- **Estimate:** 3.5 h.
- **Order:** start now; deliver `read_cvat` by about Sat 00:30 (export round-trip and eval need it before 07:00). After the Friday-night Marcaj test, save the real export as a fixture.

**WP2: derive** (`annset/lines`, `gaps`, `merge_rows`, `row_order`, `blocks`, `link_interrows`, `derive`; stage `derive`; `DeriveCfg`; `tests/post/factories.py` **within its first hour**)
- **In:** AnnSet schema (a synthetic factory until WP1 lands).
- **Out:** `rows`, `blocks`, `interrows`, `interrow_pieces_linked`, qa issues.
- **Accept:** the gaps, lines, merge_rows, row_order, blocks and link_interrows tests; reference gives 51 rows / 2 blocks / 49 linked.
- **Estimate:** 4 h.
- **Order:** parallel with WP1 (only the reference-based cases wait for WP1).

**WP3: targets, measurements, publish** (`route/targets`, `target_rules`, `target_waste`; `measure/*`; stages `targets`, `measure`, `publish`, `qa`; `TargetsCfg`, `MeasureCfg`, `PublishCfg`)
- **In:** WP2 layers (synthetic factory first), `DomainSet` (a stub from `domain_from_raw` is enough).
- **Out:** `targets`, `target_extents`, `measurements.csv/json`, publish.
- **Accept:** `test_targets` (8 GAP / 0 END on reference), `test_measure_examples` (1941.55 m, 51 rows), `test_publish`.
- **Estimate:** 4 h.
- **Order:** after WP2's `lines`/`gaps` interfaces (about +1.5 h).

**WP4: passable** (`route/domain`, `skeleton`, `centerlines`, `graph_types` (**first commit**), `graph`, `connectors`, `graph_io`; stage `passable`; `RouteCfg.domain/graph`)
- **In:** `02_route`, WP2 `rows` + `interrow_pieces_linked` (synthetic first).
- **Out:** `passable_*`, `walk_*`, `WalkGraph`.
- **Accept:** `test_domain` (real START facts), `test_skeleton` (2 components), `test_centerlines`, `test_graph`.
- **Estimate:** 4.5 h.
- **Order:** parallel; `graph_types.py` is committed within 30 min for WP5.

**WP5: route** (`route/candidates`, `distances`, `solver`, `solver_fallback`, `unroll`, `planner`, `baseline`, `validate`, `geojson`; stage `route`; `RouteCfg.solver/validate`)
- **In:** `WalkGraph` (synthetic builder), targets schema.
- **Out:** `route`, `route_stops`, `target_visits`, `exports/route.geojson`, `route_validation.json`, `route_baseline.json`.
- **Accept:** `test_candidates`, `test_solver`, `test_unroll`, `test_route_validate`, `test_planner_e2e`.
- **Estimate:** 5 h.
- **Order:** starts on synthetic graphs; integrates with WP4 at the end.

**WP6: web bundle** (`web/geojson4326`, `panels`, `ortho`, `bundle`; stage `web_bundle`; `WebCfg`; `cli_post.py`)
- **In:** any run directory (AnnSet(reference) + stub route).
- **Out:** `src/Web/data/**`.
- **Accept:** `test_web_bundle`, `test_ortho`; the frontend team confirms that `manifest.json` loads.
- **Estimate:** 3 h.
- **Order:** parallel; full data after WP3 and WP5.

**Integration gates**
1. `vineyard post --annset <model run>` end-to-end by Sat about 16:00. This does not block the Sat 07:00 / 12:00 gates; WP1's `read_cvat` is the only critical-path item.
2. Run on the intermediate Marcaj export Sat about 20:00; qa list goes to the annotators.
3. Final: Sun 11:00–13:00 with `route.solver.time_limit_final_s`, then `publish --require-source marcaj`, then commit and push.

## 8. Risks and open questions

| # | Risk | Mitigation |
|---|---|---|
| 1 | The real Marcaj export deviates from the example (paths in names, 2 decimals, masks, extra attributes, job-level splits). | Tolerant reader. Capture the Friday-night Marcaj test export as a fixture. Structural errors fail loudly; object errors become qa issues. |
| 2 | Normalizing changes the values we measure, but organizers score raw Marcaj values. | Every normalization emits a warning into `review_queue.csv` so it is fixed in Marcaj before 15:00. |
| 3 | Interrow strips don't touch passages (headland outside `passages`), so the domain has disconnected components or long connectors eat the 0.5% budget. | Connectors record their outside length and carry a penalty. Each non-START component gives `domain_disconnected`. Outside budget and per-leg outside lengths go in `route_validation.json`. Re-solve with penalty × 10. Passage polygon 0 is disconnected by construction (verified). |
| 4 | TSP explodes with `include_missing` (about 20 MSP per vineyard tile, so thousands of targets). | Must/optional split, coverage loop, node-cap ladder, `RegisterTransitMatrix`. `optional_delta_m` is reported to support the Slack Q3 decision. |
| 5 | Self-overlapping out-and-back routes make overlay undercount the outside length. | Exact per-segment `difference`; a dedicated test. |
| 6 | 1–2 cm tile-seam gaps in Marcaj-edited interrows make the outside share look larger (~0.7%). | `seam_close_m` closing on the union; validation against the closed domain. |
| 7 | GLS with a time limit is not deterministic (contract §3.3 idempotency). | Cache the route by key and rerun only with `--force`. Record solver, time and cost. Tests use tiny instances that reach the optimum. |
| 8 | `sknw` pulls in `numba` / `llvmlite`. `numba 0.67.0` requires `numpy<2.6,>=1.22`, compatible with numpy 2.5 (verified on PyPI tonight), but it adds about 100 MB to Docker and JIT warm-up. | Accept. The `skeleton.py` interface hides it; a pure-numpy tracer is a fallback only if `uv lock` fails. |
| 9 | Contract gaps: TargetKind lacks MSP/SPR; layers `interrow_pieces_linked`/`target_visits`; DAG edges; new qa codes. | REQs sent before 23:00 freeze. Fallbacks: `kind=other` + `reason`; targets computes the domain itself; `n_targets` left empty. |
| 10 | START is only 0.868 m from the passage edge; a larger erosion would push it out of `eroded`. | START is linked via `inner`, not `eroded`. A config validator asserts `eroded_buffer_m` < the START boundary distance at load. |
| 11 | Row IDs inconsistent across tiles after correction produce wrong midlines. | Midlines are clipped to `eroded`. Skeleton fallback for unlinked pieces. `row_piece_misaligned` / `row_missing_in_tile` go to annotators Saturday evening. |
| 12 | Interrow area: union vs sum is ambiguous (brief says union only for canopy; C says union). | CSV uses union; JSON carries both; warning if they differ by > 0.1%. |
| 13 | The hidden target definition is unknown (Slack Q3). | Flags `include_missing` / `include_sparse`; extents coverage reported. |
| 14 | Post chain runtime on Sunday. Estimate: import 30 s, derive 30 s, passable ≈ 60 s (skeleton ≈ 46 Mpx), targets 20 s, route ≤ 5 min (3 iterations + phase B), measure 10 s, web 2–3 min. | `--fast` preset (10 s solver); ortho cached by key. |
| 15 | OR-Tools Python binding of `RegisterTransitMatrix` unverified in 9.15 (the C++ API is confirmed). | Fallback to a transit callback over a numpy lookup. |
| 16 | gdal and tippecanoe are not installed. | The pure-Python ortho path is the default when absent; MVT is optional. |
| 17 | Publishing to the repo root from Docker. | `publish.root_dir` is configurable; README documents the volume mount. |

### Critical Files for Implementation
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/cvat/reader.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/annset/merge_rows.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/route/graph.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/route/planner.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/route/validate.py