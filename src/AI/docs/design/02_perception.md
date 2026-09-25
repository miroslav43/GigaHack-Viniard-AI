# Subsystem P: classical perception → AnnSet(model). Design v1

Scope: `vineyard/perception/*` (pure numpy/cv2/shapely, no I/O), `vineyard/annset/assemble.py`, `vineyard/qa/{render,review,overview}.py`, and the stage wrappers `rows_detect, rows_link, blocks, canopy, interrow, row_attrs, assemble, qa_previews`. All paths below are relative to `/Users/maleticimiroslav/Vin Gigahack/src/AI/`.

**Facts measured while designing** (read-only, from the reference XML). Tests and decisions below depend on them.

| Fact | r021_c012 (V01) | r006_c004 (V02) |
|---|---|---|
| UTM row angle, median (range) | 127.5° (126.8–133.3, a fan) | 112.8° (111.9–113.3) |
| Rows sorted by `n·c` descending equal R01..Rnn | yes | yes |
| Spacing between consecutive rows, measured at the overlap midpoint | 2.58, 2.89, 2.71, 3.04, 2.82, 2.85, 3.04, 2.78, **3.74, 3.61, 2.49**, 2.68 … 2.52; median-of-3 steps up to **0.93 m** | 2.34–2.78; median-of-3 steps ≤ 0.15 m |
| Max gap from reference canopies, ends included | all < 5 m | R06 5.054 (disrupted), R07 11.329 (end gap), R08 9.25, R09 12.903, R23 6.773 (end gap); **R15 5.009 but labelled regular** |
| Gaps shared across adjacent rows | — | R07 [12.6, 23.9], R08 [14.3, 22.1] and R09 [17.4, 30.3] share a **4.7 m band [17.4, 22.1]**, while R10–R12 are continuous there |
| Total row length (reference) | 910.1 m | 1031.5 m |

What follows from these facts:
- **Arch §4.4.2 spacing/phase cut:** it would split V01 into several blocks, so it ships disabled by default.
- **Transverse-band cut:** a rule of "≥ 3 rows" would split V02. The band must instead run across the full width of the block.
- **Gap threshold:** the 5.0 m threshold yields 50/51 (R15 is the exception), not the 51/51 that arch §1 claims.

---

## 1. Files

### Package `vineyard/`

| Path | Responsibility | ≈ lines |
|---|---|---|
| `perception/__init__.py` | Re-exports the public API. | 30 |
| `perception/types.py` | Frozen dataclasses and enums: `TileMasks`, `Profile`, `SpacingEstimate`, `PeakOffset`, `BandFit`, `RowFeatures`, `RowCandidate`, `OrientationResult`, `TileDetection`, `LinkResult`, `BlockResult`, `CanopyResult`, `CanopyStats`, `CoverStats`, `Gap`, `OccupancyProfile`, `StructureResult`, `FusionVariant`. Includes `readonly(arr)`, which sets `writeable=False`. | 240 |
| `perception/settings.py` | Pydantic section models (`frozen=True`, `extra="forbid"`): `VegSettings`, `RowsSettings{detect,link,filter}`, `OrchardSettings`, `BlocksSettings`, `CanopySettings`, `InterrowSettings`, `RowStructureSettings`, `QaSettings`, plus the `Overrides` file model. `config.py` imports and composes them. | 320 |
| `perception/vegmask.py` | Lab −a\*, blur, threshold, Otsu-in-corridor helper, vis codes (shadow/overexposed/nodata), texture points (std-V), NN probability upsample and fusion. | 230 |
| `perception/profile.py` | Point sampling (+0.5), coarse-to-fine dominant angle, offset histogram, autocorrelation spacing, spectral SNR, 2s-harmonic flag, peaks at 0.6·s, lattice deviation. | 270 |
| `perception/linefit.py` | Slab pre-selection, iterated Huber `fitLine`, residual p95, percentile ends, `extend_to_clip` (ray cast to the clip polygon), curved-row tracking (P1), DP simplify. | 250 |
| `perception/row_features.py` | Along-row occupancy (0.05 m stations, ±0.30 m), gap list, perpendicular run widths (p50/p80), along-row FFT period and duty, hard/soft reject reasons. | 270 |
| `perception/rows_detect.py` | Per-tile orchestration: orientation loop (up to 3, removing explained pixels), candidates plus features, `detection_to_candidates` to UTM. | 280 |
| `perception/rows_link.py` | Global support-line linking: candidate link costs, greedy union-find with a same-tile constraint, rescue of soft rejects, chain refit (straight or polyline), end extension to the clip edge, duplicate removal. | 330 |
| `perception/blocks_graph.py` | Neighbour pairs (parallel and collinear), passage cuts, full-width transverse bands, splitting chains at bands, optional spacing-step cut, connected components. | 330 |
| `perception/blocks.py` | `build_blocks`: minimum rows, orchard rejection (P1), IDs, `rows`/`blocks`/`row_pairs` frames, block outline, `rows` length fields. | 300 |
| `perception/ordering.py` | Canonical normal (n_y > 0, else n_x > 0), block order `(−round(cy), round(cx))`, row order by `n·c` descending, canopy sort key. | 140 |
| `perception/corridor.py` | `clip_rows_to_tile` (one polyline per row per tile, first to last valid vertex), flat-cap corridor polygon, corridor label raster using `fillPoly((uv−0.5)*16, shift=4)`. | 210 |
| `perception/canopy.py` | Veg (or fused) ∩ corridor labels → CC8 → area ≥ min → contour, `approxPolyDP`, +0.5, UTM, mitre outset, ∩ own corridor, ∩ clip → IDs. Also `vine_mask` for NN pseudo-labels. | 330 |
| `perception/trees.py` (P1) | Tree-in-row mask: wide perpendicular runs > 1.5 m, compact round blob (> 2 m², axis ratio < 1.5); tree polygons for QA and interrow holes. | 200 |
| `perception/interrow.py` | Band polygon from offset curves (±0.30 m) trimmed to the shorter row, forbidden holes, per-tile cut, `cover_stats`, `classify_cover`. | 320 |
| `perception/attrs.py` | **Shared with the targets subsystem.** Occupancy profile from vector canopies, `find_gaps` (ends included, unknown spans excluded), visible fraction, `classify_row_structure`, row-piece attributes. | 290 |
| `perception/overrides.py` | Apply `exclude_areas`, `force_empty_tiles`, `delete_rows`, `extend_rows`, `add_rows` deterministically in UTM; digest for cache keys; unmatched entries produce a QA issue. | 230 |
| `perception/topology.py` | Contract §2.7 invariants returning `QaIssue`s; removes canopy overlap from interrow pieces. | 190 |
| `annset/assemble.py` | Builds `AnnSet(model)`, `tile_status` rows and `qa_issues` from stage outputs; applies forced-empty tiles. | 300 |
| `qa/render.py` | Per-tile JPEG overlay (colours as in the official previews, `row_id` labels, header bar). | 280 |
| `qa/review.py` | Review priority rules, `review_queue.csv` and `empty_tiles.csv` frames. | 180 |
| `qa/overview.py` | Study-area mosaic from thumbnails, with block colours and `V` labels. | 130 |
| `pipeline/stages/_raster_source.py` | `TileRasters`: lazy `valid()`, `veg()`, `vis()`, `rgb()`, `canopy_prob()` loaders; `load_rows_for_tile`. | 160 |
| `pipeline/stages/rows_detect.py` | T stage → `cache/rows_detect/<t>.parquet` + `.json`. | 160 |
| `pipeline/stages/rows_link.py` | G stage → `layers/rows_raw.parquet`, `layers/row_candidates.parquet` (concatenated, for QA). | 130 |
| `pipeline/stages/blocks.py` | G stage → `layers/{rows,blocks,row_pairs,rows_rejected}.parquet`. | 170 |
| `pipeline/stages/canopy.py` | T stage → `cache/canopy/<t>.parquet` + `.json` (+ `cache/trees/<t>.parquet` when P1 is on). | 170 |
| `pipeline/stages/interrow.py` | B→T: global pre-step writes `layers/interrows.parquet`; tile step writes `cache/interrow/<t>.parquet`. | 200 |
| `pipeline/stages/row_attrs.py` | T stage → `cache/row_attrs/<t>.parquet` + `<t>.gaps.parquet`. | 140 |
| `pipeline/stages/assemble.py` | G stage → `annset/*`, `annset.json`, `layers/tile_status.parquet`, `qa/{qa_issues.parquet, tile_status.csv, review_queue.csv, empty_tiles.csv}`, `exports/geojson/*`. | 230 |
| `pipeline/stages/qa_previews.py` | T stage → `qa/previews/<t>.jpg`; global tail writes `qa/overview.jpg`. | 140 |

### Tests `tests/`

| Path | Responsibility | ≈ lines |
|---|---|---|
| `conftest.py` | Fixtures: settings defaults, example tiles, synthetic generators, tmp run dir. | 150 |
| `helpers/synth.py` | Striped masks (angle, spacing, width), arcs, orchard blobs, noise, multi-tile candidate frames. | 220 |
| `helpers/examples.py` | Loads the 2 example tifs and the reference via `cvat.reader`, with a 40-line lxml fallback; returns axes/canopies/interrows in px and UTM. | 160 |
| `helpers/proto_metrics.py` | Test-only port of `fit.score` and `axes.row_f1` (raster), until `eval.metrics` lands. | 120 |
| `perception/test_*.py` (one per module: settings, vegmask, profile, linefit, row_features, rows_detect, rows_link, blocks, ordering, corridor, canopy, trees, interrow, attrs, overrides, topology) | Unit tests. | 1,900 total |
| `annset/test_assemble.py`, `qa/test_review.py`, `qa/test_render.py` | Unit tests. | 350 |
| `stages/test_perception_examples_e2e.py` | `@pytest.mark.slow`: stage chain on the 2 example tiles. | 150 |

---

## 2. Public interfaces

### 2.1 Layers read and written

**CONTRACT** marks schemas frozen in contract §2.5. Common columns (`source, run_id, model_version, confidence, qa_flags`) are always written.

| Stage | Reads | Writes (columns) |
|---|---|---|
| `rows_detect` (T) | `cache/{valid,veg,vis,stats}/<t>`, `tile_valid`, `in_forbidden`, optional `cache/nn/<mv>/canopy_prob/<t>.png`, lazily `work/tiles/<t>.tif` (texture fallback) | **CONTRACT** `row_candidates`: `cand_id, tile_id, angle_deg (UTM), offset_m, length_m, support_frac, width_med_m, local_spacing_m, vine_score, rejected_reason`. Additive internal columns: `orientation, method, snr, spacing_tile_m, width_p80_m, lattice_dev_frac, residual_p95_m, is_curved, along_period_m, along_duty, harmonic_flag, soft_flags, gaps_json, near_edge_start, near_edge_end`. Also a `.json` per tile: angles, spacing, snr, status_hint, counts. |
| `rows_link` (G) | all `row_candidates`, `in_passages`, `tile_valid`, `overrides` | `rows_raw` (internal): `chain_id (L00001), member_cand_ids, tile_ids, n_tiles, angle_deg, extent_m, support_frac, width_p80_m, along_duty, harmonic_frac, rescued_frac, is_curved, gaps_json, confidence`, LineString |
| `blocks` (G) | `rows_raw`, `in_passages`, `in_forbidden`, `tile_index`, `tile_valid`, `overrides` | **CONTRACT** `rows`: `row_id, vineyard_id, row_index, length_m (Σ pieces), extent_m, n_pieces, tile_ids, angle_deg, spacing_prev_m, spacing_next_m`, with `max_gap_m`, `n_gaps_ge5`, `structure_any` left NaN/"" (derive fills them). **CONTRACT** `blocks`: `vineyard_id, n_rows, n_row_pieces, n_tiles, row_length_m, outline_area_m2, angle_deg, spacing_med_m, is_garden`; canopy and interrow areas NaN. Internal `row_pairs` (additive): `vineyard_id, row_a, row_b, k_a, spacing_m, overlap_from_m, overlap_to_m, angle_diff_deg`, connector LineString. Internal `rows_rejected`: `chain_id, reason`. |
| `canopy` (T) | `veg`, `valid`, `rows ∩ tile ⊕ canopy.rows_margin_m`, nn prob, lazily tif (Otsu) | **CONTRACT** `canopies`: `canopy_id, tile_id, vineyard_id, row_id, area_m2, n_vertices, along_m, is_clump, touches_edge`. `.json`: `corridor_veg_frac, otsu_used, n_components, n_small_dropped, vineyard_score_p95`. P1: `cache/trees/<t>.parquet`. |
| `interrow` (B→T) | `rows`, `row_pairs`, `in_forbidden`, `veg`, `vis`, (trees when P1) | **CONTRACT** `interrows`: `interrow_id, vineyard_id, row_left_id, row_right_id, area_m2, width_mean_m, length_m`. **CONTRACT** `interrow_pieces`: `piece_id, interrow_id, vineyard_id, tile_id, row_left_id, row_right_id, interrow_cover, veg_frac, shadow_frac, area_m2, width_mean_m, n_notches`. |
| `row_attrs` (T) | `rows ∩ tile`, `cache/canopy/<t>`, `vis`, `valid` | **CONTRACT** `row_pieces`: `piece_id (<row_id>@<tile_id>), row_id, vineyard_id, tile_id, row_structure, length_m, max_gap_m, n_vertices`. Internal `gaps`: `piece_id, start_m, end_m, length_m, kind`, LineString. |
| `assemble` (G) | caches from the 4 stages above, `layers/waste.parquet` (optional, from the waste subsystem), `tile_index`, `tile_valid`, `cache/stats`, `overrides` | **CONTRACT** AnnSet: `annset/{canopies,row_pieces,interrow_pieces,waste}.parquet` + `annset.json`. **CONTRACT** `tile_status` (`upload_zip=null`, `image_id=-1`; export fills them). **CONTRACT** `qa_issues`. CSVs under `qa/`. |
| `qa_previews` (T) | tif, AnnSet slice, `row_candidates` (rejected), `rows_rejected`, trees | `qa/previews/<t>.jpg`, `qa/overview.jpg` |

New QA codes (additive to contract §2.5.14): `row_offlattice`, `row_rescued_low_snr`, `orchard_rejected`, `tree_removed`, `tree_hole`, `otsu_fallback`, `texture_fallback`, `transverse_band_cut`, `curved_row`, `override_applied`, `override_unmatched`.

### 2.2 Python signatures

Aliases: `BoolMask = NDArray[np.bool_]`, `U8 = NDArray[np.uint8]`, `F32 = NDArray[np.float32]`, `I32 = NDArray[np.int32]`. GeoDataFrames are never mutated in place: functions return new frames (`.assign`, `.copy()`). Arrays inside frozen dataclasses are read-only.

Frozen interfaces that I only consume: `geo.tiling.{TileRef, px_to_utm, utm_to_px, index_to_px, tiles_for_bounds}`, the layer schemas in §2.1, the ID formats (contract §2.4), and the NN outputs (contract §6).

Interfaces marked **PROPOSED-CONTRACT** below are ones other subsystems call. They should be frozen at 23:00.

```python
# vegmask.py
def compute_tile_masks(rgb: U8, valid_eroded: BoolMask, cfg: VegSettings) -> TileMasks      # PROPOSED-CONTRACT (tile_prep)
#   TileMasks(veg: BoolMask, vis: U8 {0 ok,1 shadow,2 overexp,3 nodata}, method: str, threshold: float, veg_frac: float)
def neg_a(rgb: U8, blur_sigma_px: float) -> F32
def veg_mask(rgb: U8, valid: BoolMask, cfg: VegSettings, threshold: float | None = None) -> BoolMask
def vis_codes(rgb: U8, valid: BoolMask, cfg: VegSettings) -> U8
def otsu_threshold(values: F32) -> float
def texture_points(rgb: U8, valid: BoolMask, cfg: TextureFallbackSettings) -> BoolMask
def upsample_prob(prob_1024: U8 | None, size_px: int) -> F32 | None                        # INTER_LINEAR, /255
def fuse_masks(veg: BoolMask, prob: F32 | None, variant: FusionVariant, thr: float) -> BoolMask  # PROPOSED-CONTRACT (nn ablation A–E)

# profile.py
def sample_points(mask: BoolMask, max_points: int, seed: int) -> F32        # (N,2) CVAT px = index + 0.5
def dominant_angle_px(points: F32, cfg: RowsDetectSettings) -> float          # [0,180)
def offset_profile(points: F32, angle_px_deg: float, bin_px: float, weights: F32 | None = None) -> Profile
def estimate_spacing(profile: Profile, cfg: RowsDetectSettings) -> SpacingEstimate   # spacing_m, snr, harmonic_flag, ok
def find_row_offsets(profile: Profile, sp: SpacingEstimate, cfg: RowsDetectSettings) -> tuple[PeakOffset, ...]
def px_angle_to_utm(angle_px_deg: float) -> float                              # (−a) mod 180

# linefit.py
def fit_band_line(points: F32, offsets: F32, peak_px: float, angle_px_deg: float, cfg: RowsDetectSettings) -> BandFit | None
def line_extent(fit: BandFit, cfg: RowsDetectSettings) -> LineString          # percentile ends, px
def extend_to_clip(line: LineString, clip: Polygon, max_dist: float) -> LineString
def track_row(points: F32, fit: BandFit, cfg: RowsDetectSettings) -> LineString   # P1

# row_features.py
def along_occupancy(veg: BoolMask, line_px: LineString, half_m: float, bin_m: float) -> F32
def occupancy_gaps(occ: F32, bin_m: float, min_gap_m: float) -> tuple[tuple[float, float], ...]
def perpendicular_widths(veg: BoolMask, line_px: LineString, cfg: RowsFilterSettings, max_half_m: float) -> F32
def along_periodicity(occ: F32, bin_m: float, cfg: OrchardSettings) -> tuple[float, float]   # (period_m, duty)
def compute_row_features(veg: BoolMask, line_px: LineString, sp: SpacingEstimate, local_spacing_m: float,
                         prob: F32 | None, cfg: RowsSettings, orchard: OrchardSettings) -> RowFeatures
def hard_reject_reason(f: RowFeatures, cfg: RowsSettings, prob_present: bool) -> str | None
def soft_flags(f: RowFeatures, sp: SpacingEstimate, cfg: RowsSettings) -> tuple[str, ...]

# rows_detect.py
def detect_tile_rows(veg: BoolMask, clip_px: Polygon, tile_id: str, cfg: RowsSettings, orchard: OrchardSettings,
                     seed: int, prob: F32 | None = None, fallback_points: BoolMask | None = None) -> TileDetection
def detection_to_candidates(det: TileDetection, tile: TileRef) -> gpd.GeoDataFrame          # row_candidates, UTM

# rows_link.py
def link_candidates(cands: gpd.GeoDataFrame, passages: BaseGeometry | None, clips: Mapping[str, BaseGeometry],
                    cfg: RowsLinkSettings, detect: RowsDetectSettings) -> LinkResult   # (rows_raw, decisions df, issues)
def refit_chain(pieces: Sequence[LineString], cfg: RowsDetectSettings) -> LineString
def extend_chain_ends(line: LineString, clips: Mapping[str, BaseGeometry], max_m: float) -> LineString

# blocks_graph.py / blocks.py / ordering.py
def neighbour_pairs(rows: gpd.GeoDataFrame, cfg: BlocksSettings, detect: RowsDetectSettings,
                    passages: BaseGeometry | None = None) -> pd.DataFrame                   # PROPOSED-CONTRACT (derive/QA reuse)
def transverse_bands(rows: gpd.GeoDataFrame, pairs: pd.DataFrame, cfg: BlocksSettings) -> tuple[Band, ...]
def split_rows_at_bands(rows: gpd.GeoDataFrame, bands: Sequence[Band]) -> gpd.GeoDataFrame
def build_blocks(rows_raw: gpd.GeoDataFrame, passages: BaseGeometry | None, forbidden: BaseGeometry | None,
                 tiles: gpd.GeoDataFrame, clips: Mapping[str, BaseGeometry], cfg: BlocksSettings,
                 orchard: OrchardSettings, detect: RowsDetectSettings, corridor_half_m: float) -> BlockResult
def canonical_normal(angle_utm_deg: float) -> tuple[float, float]
def order_blocks(reps: Sequence[Point]) -> tuple[int, ...]
def order_rows(lines: Sequence[LineString], angle_utm_deg: float) -> tuple[int, ...]

# corridor.py
def clip_rows_to_tile(rows: gpd.GeoDataFrame, tile: TileRef, clip: BaseGeometry,
                      margin_m: float = 0.0) -> gpd.GeoDataFrame                            # PROPOSED-CONTRACT (waste, derive)
def corridor_polygon(axis: LineString, half_m: float) -> Polygon                            # flat caps
def corridor_labels(pieces: gpd.GeoDataFrame, tile: TileRef, half_m: float) -> I32          # PROPOSED-CONTRACT (nn pseudolabels)

# canopy.py / trees.py
def vine_mask(veg_or_fused: BoolMask, labels: I32, tree: BoolMask | None) -> I32            # PROPOSED-CONTRACT (nn pseudolabels)
def extract_canopies(veg: BoolMask, pieces: gpd.GeoDataFrame, tile: TileRef, clip: BaseGeometry,
                     cfg: CanopySettings, veg_cfg: VegSettings, prob: F32 | None = None,
                     variant: FusionVariant = FusionVariant.NONE, tree: BoolMask | None = None,
                     otsu_veg: Callable[[I32], BoolMask] | None = None) -> CanopyResult
def tree_mask(veg: BoolMask, pieces: gpd.GeoDataFrame, tile: TileRef, cfg: CanopySettings,
              orchard: OrchardSettings, prob: F32 | None = None) -> BoolMask                # P1

# interrow.py
def build_interrow_bands(rows: gpd.GeoDataFrame, pairs: pd.DataFrame, forbidden: BaseGeometry | None,
                         cfg: InterrowSettings) -> gpd.GeoDataFrame                          # contract `interrows`
def band_polygon(left: LineString, right: LineString, offset_m: float) -> Polygon | None
def cut_interrows_to_tile(interrows: gpd.GeoDataFrame, tile: TileRef, clip: BaseGeometry,
                          holes: BaseGeometry | None, min_piece_m2: float) -> gpd.GeoDataFrame
def cover_stats(veg: BoolMask, vis: U8, piece: Polygon, tile: TileRef) -> CoverStats
def classify_cover(s: CoverStats, width_mean_m: float, cfg: InterrowSettings) -> tuple[str, tuple[str, ...]]

# attrs.py — PROPOSED-CONTRACT, shared with targets (same code on AnnSet(model|marcaj))
def occupancy_profile(axis: LineString, canopies: Sequence[Polygon], cfg: RowStructureSettings,
                      half_m: float, unknown: BaseGeometry | None = None) -> OccupancyProfile
def find_gaps(p: OccupancyProfile, min_length_m: float, include_ends: bool) -> tuple[Gap, ...]  # Gap(start_m,end_m,kind)
def gap_centre(axis: LineString, gap: Gap) -> Point
def visible_fraction(vis: U8, axis_px: LineString, half_px: float, cfg: RowStructureSettings) -> float
def classify_row_structure(p: OccupancyProfile, visible_frac: float, cfg: RowStructureSettings) -> StructureResult
def row_pieces_attributes(pieces: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame, vis: U8 | None, tile: TileRef,
                          clip: BaseGeometry, cfg: RowStructureSettings, half_m: float) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]

# overrides.py
class Overrides(BaseModel): ...   # version, force_empty_tiles, exclude_areas, delete_rows, extend_rows, add_rows (UTM, each with id)
def apply_candidate_overrides(cands: gpd.GeoDataFrame, ov: Overrides) -> tuple[gpd.GeoDataFrame, tuple[QaIssue, ...]]
def apply_row_overrides(rows_raw: gpd.GeoDataFrame, ov: Overrides) -> tuple[gpd.GeoDataFrame, tuple[QaIssue, ...]]
def overrides_digest(ov: Overrides) -> str

# topology.py / annset/assemble.py / qa
def check_invariants(annset: AnnSet, clips: Mapping[str, BaseGeometry], rows: gpd.GeoDataFrame) -> tuple[QaIssue, ...]
def remove_canopy_overlap(pieces: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame, min_piece_m2: float) -> tuple[gpd.GeoDataFrame, int]
def assemble_annset(parts: AnnSetParts, tile_index: gpd.GeoDataFrame, tile_valid: gpd.GeoDataFrame,
                    stats: Mapping[str, TileStats], ov: Overrides, meta: AnnSetMeta, min_interrow_m2: float) -> AssembleResult
def review_queue(tile_status: gpd.GeoDataFrame, issues: gpd.GeoDataFrame, cfg: QaSettings) -> pd.DataFrame
def empty_tiles(tile_status: gpd.GeoDataFrame) -> pd.DataFrame
def render_preview(rgb: U8, tile: TileRef, objs: TileObjects, header: PreviewHeader, cfg: QaSettings) -> U8
def build_overview(thumbs: Mapping[str, U8], blocks: gpd.GeoDataFrame, cfg: QaSettings) -> U8
```

**Stage modules** follow the protocol assumed from `pipeline/dag.py`. Each exposes `STAGE_NAME`, `STAGE_VERSION`, `SCOPE`, `CFG_KEYS`, `item_key(ctx, tile_id) -> str`, and either a top-level `run_tile(ctx: RunContext, tile_id: str) -> TileOutcome` (picklable for spawn) or `run(ctx) -> StageOutcome`.

**`interrow` has two steps:** `run_global` (bands) and then `run_tile`. If the DAG runner cannot run a pre-step, register a separate `interrow_bands` (G) stage; this is additive.

### 2.3 NN `canopy_prob` plug-in (contract §6)

`TileRasters.canopy_prob()` returns `None` in any of these cases: `nn.enabled=false`, the file is missing, or the weights sha256 does not match. In that case it logs one warning and `model_version` records `nn=none`. Otherwise it returns `upsample_prob(png_1024, 2048)`. The cache key includes `weights_sha256`, so a new model invalidates exactly the tiles that consume it.

| Consumer | Probability present | `None` (classical) |
|---|---|---|
| `rows_detect` | If `rows.detect.use_nn_weight`, profile points are `veg ∧ prob ≥ nn.prob_threshold`. `vine_score` = mean prob in the corridor. `rows.filter.min_vine_score` applies. | Points = veg; `vine_score = NaN`; filter skipped |
| `canopy` | `fuse_masks(veg, prob, canopy.nn_fusion)`: `none`=A, `nn`=B, `and`=C, `or`=D. E (no corridor) is ablation-only via `FusionVariant.NN_NO_CORRIDOR`. | Variant A always |
| `tile_status.vineyard_score` | p95 of prob over the corridors | NaN |
| `trees` / interrow holes (P1) | Tree blob must also have prob < thr | Geometric test only |
| NN pseudo-labels (nn subsystem) | — | Uses `corridor_labels` + `vine_mask(veg, labels, tree)`; vegetation outside corridors = 0 |

The default is `canopy.nn_fusion: none`. The NN owner flips it only if B, C or D wins on both example tiles (arch §4.6).

---

## 3. Algorithm notes

These cover only choices the docs leave open and places where porting needs care.

**1. Pixel conventions (contract §1.2/§1.4).** The prototype mixes index and continuous coordinates: `np.nonzero` without +0.5, `round(uv*16)`, and references rasterized with `np.round(p)`.
- The port uses `index_to_px` (+0.5) for every point set and contour.
- Corridor and piece rasterization use `fillPoly(round((uv−0.5)*16), shift=4)`.
- Expect about ±0.005 score drift against the RAPORT numbers. The gates in §5 absorb it.

**2. `veg_mask` (port of `axes.veg_mask`).** Same maths: `na = −(a*−128)`, `GaussianBlur` with σ 2.5, `na > 4`.
- Extra steps: `∧ valid_eroded` (contract §1.7), and `∧ ¬forbidden` when `rows.detect.mask_forbidden` is on (arch §4.2.1).
- Vis codes use HSV V: shadow is **V < 50**, taken from arch §4.7 over the contract's 35. Overexposed is V > 245.
- Otsu fallback (P1, flag): the `canopy` stage computes `corridor_veg_frac`. Only when it falls outside [0.10, 0.90] does it lazily read the tif and threshold `neg_a` by Otsu on corridor pixels. It logs `otsu_fallback`.

**3. `dominant_angle` (port of `axes.dominant_angle`).**
- Coarse-to-fine search: 2° steps over [0, 180), then 0.5° steps within ±2° (about 100 evaluations instead of 360).
- 200k-point sample drawn with `default_rng(runtime.seed)`; histogram bins of 0.1 m.
- UTM angle = (−angle_px) mod 180.

**4. `detect_rows` (port of `axes.detect_rows`).** Replaced: the fixed `distance=0.9 m` and the prominence of 5%. New sequence:
- (a) Offset histogram in 0.05 m bins, smoothed with σ = 3 bins.
- (b) Spacing `s` = first autocorrelation maximum with lag in [1.8, 3.8] m.
- (c) SNR = power at 1/s divided by the median power over periods [0.5, 10] m. `harmonic_flag` = power at 1/(2s) > power at 1/s.
- (d) `find_peaks(distance=0.6·s)`, no height criterion. Noise peaks die on occupancy.
- (e) Lattice: `o0` = circular mean of the peak phases; `lattice_dev = |off − (o0 + k·s)|/s`. Above 0.25 the candidate gets the **soft** flag `row_offlattice` and must still pass the normal tests. On r021 the fan gives deviations around 0.35·s, so this can never be a hard reject.
- (f) Performance: pre-select a slab with `argsort(offsets)` + `searchsorted` on ±`band_max_drift_m` (0.8 m) once, then iterate the Huber fit on the slab only. The prototype recomputed residuals over all veg points on every iteration.
- (g) Iteration: initial band `band_init_m` 0.45, then 3 × {`fitLine` Huber, keep r < 0.35 ∧ |off − c| < 0.8}. Drop if slab area < 0.2 m² or the angle deviates more than 3°.
- (h) Ends at the p0.5 / p99.5 of the along-projection. `extend_to_clip` ray-casts against `clip_px = tile_box ∩ tile_valid`, not `[0, 2048]` as the prototype's `clip_to_tile` did (contract §1.6).
- (i) Multi-orientation: remove pixels within ±0.35 m of accepted lines and repeat while the residual veg is ≥ 15% and the new angle differs from previous ones by ≥ 10°, up to 3 orientations.
- (j) The whole-tile SNR gate is **soft** (`rejected_reason=low_snr`): candidates are still emitted. Hard rejects are `too_wide` (width p80 ≥ 1.2 m), `low_occupancy` (< 0.15), `angle_gate`, `short` (< 1 m) and `min_band_area`.
- (k) Candidate IDs are `K01..` in profile order across orientations. The contract regex `K\d{2}` can overflow beyond 99; the proposal is to relax it to `K\d{2,3}`.
- (l) Each candidate stores `gaps_json`: occupancy gaps ≥ 3 m in along-metres. Blocks need this to see turning zones that fall inside a single per-tile piece.

**5. `rows_link` (arch §4.3 global, replaces endpoint linking).** Candidate pairs come from tiles at Chebyshev distance ≤ 1 + `link_gap_max_tiles`, plus same-tile pairs for multi-orientation duplicates.
- Reference line: the **longer** piece L. If both pieces are ≥ 5 m, require |Δθ| ≤ 3°; short stubs have unreliable angles, so they skip this test. The shorter piece's endpoints must lie within ≤ 0.3 m of L's support line, or its end tangent if L is a polyline.
- Along-gap ≤ 1 tile (51.2 m). The connecting segment must not intersect `passages.buffer(−0.5)`.
- **Greedy union** in ascending (offset, gap) order. Refuse a union that would put two pieces of the same tile overlapping by more than 0.5 m along into one chain. This stops transitive drift across parallel rows.
- Chains made only of soft-rejected pieces are dropped. Soft-rejected pieces that join an accepted chain are rescued and flagged `row_rescued_low_snr`. This recovers corner stubs and blocks at the edge of grassy tiles without any raster access in a global stage.
- Refit uses vertices ordered by PCA projection. If the max deviation from TLS is ≤ 0.15 m, output a 2-point line; otherwise a polyline DP-simplified at 0.1 m.
- Global ends get the 3 m snap to the boundary of the end tile's clip.
- **Deferred to P2:** grid indexing `round((offset − o0)/s)` (arch §4.4.4). Measured r021 spacing makes a block-level grid invalid, and chain linking already covers the edge-gap case it was meant to fix.

**6. `blocks` (arch §4.4), with two evidence-driven deviations.**
- Edges:
  - Parallel pairs: ≤ 5°, overlap ≥ 20% of the shorter row, spacing 1.8–4.0 m measured at the overlap midpoint; the edge is cut if the connector crosses a passage.
  - Collinear pairs: lateral ≤ 0.3 m, gap ≤ 5 m, no passage crossing.
- **Transverse band = full width only.** An interval I of width ≥ `transverse_band_min_m` (4.0) is a band if **every** row of the component that spans I has a gap covering I, and at least 3 such rows exist. Gaps come from chain holes plus mapped `gaps_json`. Matching rows are split at I. The r006 partial band (R07–R09, 4.7 m) does not qualify.
- **Spacing/phase cut:** behind `spacing_cut_enabled: false`. When enabled it requires a step between the median of 3 rows before and 3 after, with ≥ 3 rows on each side. The r021 steps of up to 0.93 m show that the literal 0.2 m / 0.3·s rule breaks V01.
- Components are then built; blocks with fewer than 3 rows are dropped.
- IDs: blocks sorted by `(−round(cy), round(cx))` → `V01…`; rows by `n·c` descending → `R001…`.
- Row order does not affect the score. Interrows use `row_pairs` adjacency, not index arithmetic, so a fan mis-ordering is harmless.
- Outline: `union(buffer(corridor, 2.5)).buffer(−2.5)`.
- Orchard rejection (P1) runs per block. A block is rejected if any of these holds:
  - median `width_p80 / spacing` > 0.45;
  - more than 50% of its rows have an along-row period in [3, 6] m with duty < 0.6;
  - more than 50% of its members carry `harmonic_flag`.
- Overrides are applied first: `apply_row_overrides` (delete, then extend, then add).

**7. `canopy` (port of `axes.canopies` + `fit.corridor`).**
- Corridors are built from rows clipped with a **1 m margin**. The prototype cut the corridor with a flat cap exactly at the tile edge, which lost the oblique triangle at the edge. The reference's apparent "overhang" of up to 0.71 m in r006 is exactly this artifact: 0.3·tan 67° ≈ 0.7.
- Labels raster with value = piece index; `mask ∧ labels>0`; CC8 via `connectedComponentsWithStats`; keep ≥ ⌈0.19 / 0.025²⌉ = 304 px.
- Row assignment: the label with maximum overlap (`bincount` over (component, label)).
- Per component, on its bbox crop: `findContours(RETR_EXTERNAL, NONE)` → `approxPolyDP(2.5 px)` → +0.5 → `px_to_utm` → `make_valid` → `buffer(0.5 px · GSD, join_style="mitre")` → ∩ own `corridor_polygon` → ∩ clip → explode → keep parts ≥ 0.19 m² → `orient(+1)` (CCW).
- `canopy_id` sorted by (vineyard_id, row_index, along_m).
- Grass in the corridor stays in the canopy (r006 R16–R18 reference clumps); `split_clumps` stays false.
- Tree filter (P1): a local perpendicular run > 1.5 m sustained over ≥ 0.75 m of the row, **and** a round blob (> 2 m², axis ratio < 1.5). Elongated grass never qualifies.

**8. `interrow` (port of `attrs.band` / `ir_mask`, raster → vector).**
- For each pair in `row_pairs`: take `offset_curve(left, +0.30)` and `offset_curve(right, −0.30)`, oriented alike, cut both with `substring` to the common along-interval (this is the shorter-row rule), and build `Polygon(left_off + right_off[::-1])`. Subtract forbidden holes ≥ 1 m².
- Tree holes (P1) only when `canopy.tree_filter_enabled`. A fully grassy interrow would otherwise be carved out.
- Holes become notches here via `geo.ops.notch_holes`, so the AnnSet stays 1:1 with the CVAT content.
- Per tile: ∩ clip → explode → pieces ≥ 0.25 m².
- Cover: rasterize the piece; `veg_frac` = mean(veg); `shadow_frac` = mean(vis ∈ {1, 2}). The result is `unassessable` if `shadow_frac` > 0.5 or width < 0.4 m, otherwise the 0.25 / 0.75 thresholds apply. `cover_borderline` is raised within ±0.05 of a threshold.
- Overlap with canopy is zero by construction (both stop at exactly ±0.30 m of the same axis). `assemble` still subtracts the canopy union as a guarantee, so the contract DAG (canopy ∥ interrow) is unchanged.

**9. `row_attrs` (port of `attrs.row_gap`, raster → vector, arch §4.8).**
- Canopy polygons are affine-transformed into the axis frame (along, across) and rasterized at 0.05 m within ±0.30 m.
- Per-bin cover fraction → mean over a 0.5 m window ≥ 0.20 → occupied → 1D closing 0.4 m.
- Bins over nodata are `unknown`; gaps split around them rather than counting them.
- Gaps include the ends. `disrupted` if max ≥ 5.0 m. `unassessable` if the visible fraction (bins with ≤ 50% shadow/overexposed/nodata) is < 0.5. `structure_borderline` if the max gap is in [4.5, 7].
- The arch rule "fill only if < 1.2 m wide and not grass" is enforced through the corridor plus the tree filter. The reference counts grass in the corridor as occupancy (r006 R16–R18 are regular).
- The same functions give the targets subsystem its 2–5 m "missing" gaps on any AnnSet.

**10. `assemble`.**
- Concatenate the tile caches; drop objects of `force_empty_tiles`; apply `remove_canopy_overlap`; run `validate_layer` on every layer.
- Invariants produce `qa_issues` with severity `error`; export refuses to run on errors.
- `tile_status`:
  - `status` = `empty_nodata` (from stats), `no_vineyard` (0 objects or forced empty), `failed`, or `ok`;
  - `has_vineyard` = any canopy or row piece;
  - `review_priority`: 1 if the tile has an error, failed, an empty-tile confirmation, SNR < `qa.snr_review_max` with rows, an orchard rejection, tree removal, a band cut or an override; 2 if it only has warnings; 3 otherwise.
- `issue_id` = Q00001… after a stable sort on (severity, code, tile_id, object_id).
- Waste: consume `layers/waste.parquet` if present. If absent, write an empty waste layer and a warning.
- `source` enum has no `override` value, so added rows use `source=model` with `qa_flags=override:A001`.

**11. Determinism.** No iteration over `set`; stable sorts by ID; seed from config; override application order fixed as exclude → force_empty → link → delete → extend → add.

---

## 4. Config keys I own

Sources: **A** = ARHITECTURA, **C** = contract, **new** = introduced here. Every key lives in `configs/default.yaml` and is validated with `extra=forbid`. Where A and C both have a key, A wins (arch §10) and the A name is used inside the C structure.

### `veg`

| YAML path | Default | Source |
|---|---|---|
| `veg.method` | `lab_a` | C§9 (A `vegmask.index`) |
| `veg.blur_sigma_px` | 2.5 | A§3.6 |
| `veg.threshold` | 4.0 | A§3.6 |
| `veg.morph_close_px` | 0 | C§9 |
| `veg.otsu_fallback_enabled` | false (P1) | new |
| `veg.otsu_fallback_veg_frac` | [0.10, 0.90] | A§3.6 / §4.5 |
| `veg.shadow_v_max` | 50 | A§4.7 (supersedes C `interrow.shadow_v_max: 35`) |
| `veg.overexposed_v_min` | 245 | A§4.8 |
| `veg.texture_fallback.{enabled, window_m, top_frac, trigger_min_veg_frac}` | false (P1), 0.25, 0.30, 0.50 | A§3.6 `v_std_0.25m` + new |

### `rows.detect`

| YAML path | Default | Source |
|---|---|---|
| `rows.detect.angle_step_deg` / `angle_coarse_step_deg` | 0.5 / 2.0 | C§9 / new (performance) |
| `rows.detect.angle_bin_m` | 0.1 | C§9 |
| `rows.detect.max_sample_points` | 200000 | A§4.3.1 |
| `rows.detect.profile_bin_m` / `smooth_sigma_bins` | 0.05 / 3 | C§9 |
| `rows.detect.spacing_estimate` | `autocorr` | A§3.6 |
| `rows.detect.spacing_min_m` / `spacing_max_m` | 1.8 / 3.8 | A§3.6 (user decision; C `spacing_range_m` [2.2, 3.8] superseded) |
| `rows.detect.peak_min_dist_factor` | 0.6 | A§3.6 (C `peak_min_distance_m` 0.9 superseded) |
| `rows.detect.periodicity_min_snr` | 3.0 | A§3.6 (C `peak_prominence_frac` superseded) |
| `rows.detect.snr_band_m` | [0.5, 10.0] | new |
| `rows.detect.onlattice_tol_factor` | 0.25 | A§3.6 |
| `rows.detect.band_init_m` / `band_m` | 0.45 / 0.35 | C§9 / A§3.6 |
| `rows.detect.band_max_drift_m` / `min_band_area_m2` | 0.8 / 0.2 | new (prototype literals) |
| `rows.detect.fit_iterations` / `angle_gate_deg` | 3 / 3.0 | C§9 / A |
| `rows.detect.end_percentiles` | [0.5, 99.5] | C§9 |
| `rows.detect.snap_to_edge_m` / `min_row_len_m` | 3.0 / 1.0 | A§3.6 |
| `rows.detect.residual_split_m` / `track_window_px` | 0.15 / 256 | A§3.6 |
| `rows.detect.track_vertex_m` / `dp_tolerance_m` / `tracking_enabled` | 7.5 / 0.1 / false (P1) | A§4.3.4 |
| `rows.detect.max_orientations` | 3 | A§4.3.6 |
| `rows.detect.multi_min_residual_frac` / `multi_min_angle_sep_deg` | 0.15 / 10.0 | new |
| `rows.detect.gap_record_min_m` | 3.0 | new |
| `rows.detect.mask_forbidden` | true | A§4.2.1 |
| `rows.detect.use_nn_weight` | false | C§6 |

### `rows.link`, `rows.filter`, `orchard`

| YAML path | Default | Source |
|---|---|---|
| `rows.link.link_angle_max_deg` | 3.0 | A§3.6 (C `angle_tol_deg`) |
| `rows.link.link_offset_max_m` | 0.3 | A§3.6 (C `perp_tol_m` 0.5 superseded) |
| `rows.link.link_gap_max_tiles` | 1 | A§3.6 (C `max_join_gap_m` superseded) |
| `rows.link.angle_min_len_m` / `passage_erode_m` / `dup_overlap_m` | 5.0 / 0.5 / 0.5 | new |
| `rows.link.rescue_soft_rejects` | true | new |
| `rows.filter.width_p80_max_m` | 1.2 | A§4.2.2 (C `max_canopy_width_m` 1.0 superseded) |
| `rows.filter.width_station_m` | 0.25 | new |
| `rows.filter.occupancy_min` | 0.15 | A§4.2.2 |
| `rows.filter.min_vine_score` | 0.35 | C§9 (only when NN is present) |
| `orchard.width_spacing_ratio_max` | 0.45 | A§3.6 |
| `orchard.along_period_m` / `along_duty_max` | [3, 6] / 0.60 | A§3.6 |
| `orchard.tree_blob_area_m2` / `tree_blob_axis_ratio_max` | 2.0 / 1.5 | A§3.6 |
| `orchard.block_reject_enabled` / `block_majority_frac` | false (P1) / 0.5 | new |

### `blocks`

| YAML path | Default | Source |
|---|---|---|
| `blocks.neighbour_max_m` | 4.0 | A§3.6 |
| `blocks.parallel_max_deg` / `min_overlap_frac` | 5.0 / 0.20 | A§4.4.1 |
| `blocks.collinear_gap_max_m` | 5.0 | A§3.6 (= C `gap_close_m`) |
| `blocks.spacing_cut_enabled` | **false** | new (r021 evidence) |
| `blocks.spacing_jump_max_m` / `phase_tol_factor` | 0.2 / 0.3 | A§3.6 |
| `blocks.headland_min_rows` | 3 | A§3.6 |
| `blocks.transverse_band_min_m` | 4.0 | A§4.4.2, research §2.7 |
| `blocks.transverse_full_width` | true | new (r006 evidence) |
| `blocks.min_rows_per_block` | 3 | A§3.6 (= C `min_rows`) |
| `blocks.cut_by_passages` | true | C§9 |
| `blocks.outline_buffer_m` | 2.5 | C§2.5.6 |
| `blocks.id_prefix` / `id_width` / `row_id_width` | V / 2 / 3 | C§9 |
| `blocks.garden_max_row_length_m` | 300.0 | new (`is_garden`, informational) |

### `canopy`, `interrow`, `row_structure`

| YAML path | Default | Source |
|---|---|---|
| `canopy.corridor_half_m` / `min_area_m2` | 0.30 / 0.19 | A / C |
| `canopy.connectivity` | 8 | C§9 |
| `canopy.simplify_px` | 2.5 | A§3.6 (= C `approx_eps_px`) |
| `canopy.vector_outset_px` | 0.5 | C§9 |
| `canopy.clump_area_m2` / `split_clumps` | 2.0 / false | C§9 |
| `canopy.max_perp_width_tree_m` / `tree_filter_enabled` | 1.5 / false (P1) | A§3.6 / new |
| `canopy.rows_margin_m` | 1.0 | C§3.2 stage 7 |
| `canopy.nn_fusion` | `none` | A§4.6 |
| `interrow.offset_m` | 0.30 | A§3.6 (= C `offset_from_axis_m`) |
| `interrow.cover_bare_max` / `cover_veg_min` | 0.25 / 0.75 | A§3.6 |
| `interrow.shadow_unassessable` / `min_width_assessable_m` | 0.5 / 0.4 | A§3.6 / A§4.7.5 |
| `interrow.cover_borderline_margin` | 0.05 | new |
| `interrow.hole_min_area_m2` / `hole_sources` | 1.0 / [forbidden] | C§9 (`tree_mask` added when P1 is on) |
| `row_structure.gap_disrupted_m` / `include_end_gaps` / `borderline_m` | 5.0 / true / [4.5, 7.0] | A / C |
| `row_structure.unassessable_visible_min` | 0.5 | A§3.6 |
| `row_structure.occ_bin_m` | 0.05 | new |
| `row_structure.occ_close_m` / `occ_window_m` / `occ_min` | 0.4 / 0.5 / 0.20 | A§3.6 |
| `row_structure.gap_fill_max_width_m` | 1.2 | A§3.6 |
| `row_structure.visible_bin_max_hidden` | 0.5 | new |

### `qa`, `paths`

| YAML path | Default | Source |
|---|---|---|
| `qa.preview_px` / `preview_jpeg_quality` / `overview_px_per_tile` | 1024 / 85 / 128 | new |
| `qa.colors_bgr` | row `[0,0,255]`, disrupted `[255,0,255]`, unassessable `[0,165,255]`, canopy `[0,255,0]`, interrow `[255,255,0]`, waste `[0,255,255]`, rejected `[128,128,128]`, tree `[255,0,0]` | README preview legend |
| `qa.canopy_fill_alpha` / `snr_review_max` | 0.35 / 6.0 | new |
| `paths.overrides` | `configs/overrides.yaml` | A§2.4 |

Read-only keys owned elsewhere: `grid.*`, `nodata.veg_erode_px`, `nn.{enabled, prob_threshold, in_gsd_m, weights_sha256}`, `runtime.{seed, n_workers}`, `export.{min_row_piece_m 0.5, min_interrow_piece_m2 0.25}`.

---

## 5. Tests to write first (TDD)

Example tile origins: r021_c012 at x0 = 629606.4, y0 = 5220147.2; r006_c004 at x0 = 629196.8, y0 = 5220915.2.

**`test_settings.py`**
- Defaults load.
- An unknown key raises `ValidationError`.
- Models are frozen.
- `spacing_min_m == 1.8`.

**`test_vegmask.py`**
- Pure green (0, 255, 0) gives `na > 4`; soil (150, 100, 70) gives `na < 4`.
- Veg is False outside `valid_eroded`.
- Vis codes: V = 40 → 1, V = 250 → 2, invalid → 3.
- `TileMasks` arrays are not writeable.
- `fuse_masks`: `None` prob returns veg for every variant; truth tables for B, C, D; C with prob ≡ 1 equals A.
- Otsu fallback is **not** triggered on either example (corridor veg fraction about 0.47 and 0.52 with reference axes).

**`test_profile.py`**
- Synthetic stripes at 30° px, 2.5 m spacing, 0.4 m wide: angle 30 ± 0.5°, spacing 2.50 ± 0.05 m, SNR ≥ 10, peak count equals the number of stripes.
- Random 20% noise: SNR < 3.
- Orchard rows at 5.0 m: `harmonic_flag` True.
- A spurious peak at o0 + 0.5·s gets `lattice_dev_frac ≥ 0.25`.
- Examples: UTM angle 127.5 ± 1.0° (r021) and 112.8 ± 1.0° (r006); spacing 2.78 ± 0.25 m and 2.53 ± 0.10 m; SNR ≥ 3 on both.

**`test_linefit.py`**
- Straight band: angle error < 0.1°, `residual_p95` < 0.05 m.
- Arc with R = 300 m: residual > 0.15 m; `track_row` max deviation ≤ 0.1 m (P1).
- End 2 m from the edge snaps exactly to u or v ∈ {0, 2048}; 4 m from the edge stays unchanged.
- A nodata corner stops the extension at the valid boundary.

**`test_row_features.py`**
- Continuous row 0.5 m wide: `width_p80` ≈ 0.5 ± 0.05, duty > 0.6, no reject.
- Round crowns of 3 m every 5 m: `too_wide`; period 5.0 ± 0.3 m; duty < 0.6.
- Occupancy 0.1 → `low_occupancy`.
- Gaps recorded: canopies [0, 2], [3, 4], [10, 12] give gap (4, 10).

**`test_rows_detect.py`**
- Examples, single tile, reference ends on the edge: F1 ≥ 0.95 per tile (prototype: 24/25 and 25/26 matched, F1 0.98).
- **No reference row matched by a hard-rejected candidate** (51/51 survive the filters).
- Empty mask gives `status_hint="no_periodicity"` and 0 candidates.
- Two-orientation synthetic (30° and 80°): both found.
- IDs match `^siret3_r021_c012:K\d{2}$`.
- `@slow`: ≤ 4 s per tile.

**`test_rows_link.py`**
- Collinear pieces in (r, c) and (r, c+1) with 0.2 m offset and 1° difference: 1 chain.
- 0.5 m offset: 2 chains.
- Gap over one empty tile: linked; over two tiles: not linked.
- Connector crossing a passage: not linked.
- Transitive drift (B and C in one tile, 2.5 m apart): never in one chain.
- 0.8 m stub with a 10° angle error: links on offset alone.
- Three collinear pieces refit to 2 vertices; pieces on an R = 300 m arc give a polyline with deviation ≤ 0.1 m.
- A soft-only chain is dropped; a soft piece joining an accepted chain is rescued and flagged.
- Shuffled input gives identical `rows_raw` (IDs and WKB).

**`test_blocks.py`**
- 10 synthetic rows at 2.5 m: 1 block `V01`, `R001..R010`, R001 has the maximum `n·c`.
- Passage between two plantings: 2 blocks; the northern one is V01.
- 2-row block dropped with `block_too_few_rows`.
- Full-width 6 m band across all rows: split into 2 blocks.
- **r006-like partial band** (R07–R09 common gap [17.4, 22.1], R10–R12 continuous): not split.
- **r021 spacing sequence** with `spacing_cut_enabled` false: 1 block. With it true and the 0.2 m rule, the test documents that it splits.
- Reference axes as `rows_raw`: 1 block per example with 25 and 26 rows; order equals reference R01..Rnn; regexes `^V\d{2,3}$` and `^V\d{2,3}-R\d{3}$`.
- `length_m` = Σ pieces; totals 910.1 ± 1 m and 1031.5 ± 1 m.

**`test_ordering.py`**
- Angle 0°: n = (0, 1). Angle 90°: n = (1, 0).
- Block order uses `(−round(cy), round(cx))`.

**`test_corridor.py`**
- A row across 3 tiles gives 3 pieces with the same `row_id`; Σ length equals the row length (1e-6).
- Nodata cut keeps one polyline from the first to the last valid vertex.
- Horizontal axis at v = 100.0 with half-width 0.30 m fills rows 88..111 (24 px).
- Flat caps: the polygon length equals the axis length.

**`test_canopy.py`**
- A 3×3 px block at indices (10..12, 20..22) with outset 0.5 gives polygon [10, 13] × [20, 23], area 9 px² = 0.005625 m², CCW in UTM.
- Component of 0.18 m² dropped; 0.20 m² kept.
- Every vertex ≤ 0.30 m + 1e-6 from its axis.
- IDs `siret3_r021_c012:C0001`…, and C0001 is on R001 at the smallest along position.
- Reference axes: proto raster score ≥ 0.84 mean (RAPORT 0.855), each tile ≥ 0.80; counts within ±10% of 399 and 251; union within ±10% of 256.9 and 319.4 m²; `touches_edge` counts 15 ± 5 and 16 ± 5.
- `prob=None` is identical to variant A.

**`test_trees.py` (P1)**
- Round 3 m crown on the axis: masked, canopies removed there.
- Grass strip 20 × 1.8 m: not a tree.
- Both examples: removed area = 0.

**`test_interrow.py`**
- Rows 2.5 m apart, 40 m and 30 m long, overlapping 30 m: width 1.9 m, length 30 m, 4 vertices, area 57.0 m².
- N rows give N−1 interrows; none across blocks.
- Forbidden square inside: 1 polygon with `n_notches = 1`.
- Band across a tile edge: 2 pieces; Σ area equals the band area.
- Cover: 0.10 → `bare_soil`; 0.5 → `mixed`; 0.8 → `vegetation`; shadow 0.6 → `unassessable`; width 0.35 m → `unassessable`; 0.27 → `cover_borderline`.
- Reference axes: 24 and 25 pieces; IoU ≥ 0.98 with the reference; cover 49/49 (24 `bare_soil`; 21 `bare_soil` + 4 `mixed`); `bare_soil` veg_frac ≤ 0.16, `mixed` in [0.51, 0.68]; overlap with canopies ≤ 0.01 m² per tile.

**`test_attrs.py`**
- Canopies [0, 2], [3, 4], [10, 12] on a 15 m axis: gaps 1.0, 6.0 and end gap 3.0; max 6.0 → `disrupted`.
- Canopy [7, 15]: start gap 7.0 → `disrupted`.
- Holes < 0.4 m closed.
- Nodata span is not a gap.
- Visible fraction 0.4 → `unassessable`.
- Max gap 5.5 → `structure_borderline`.
- Reference axes and canopies: max gaps R06 5.05, R07 11.33, R08 9.25, R09 12.90, R23 6.77 (±0.10); disrupted ⊇ {R06, R07, R08, R09, R23}; accuracy ≥ 50/51, and the only allowed miss is **V02-R15** (5.009 m), which must be flagged borderline.
- All 51 reference rows have visible fraction ≥ 0.5.
- Gaps in [2, 5) m are returned with centre points (for targets).

**`test_overrides.py`**
- Unknown key raises an error.
- `delete_rows` near (x, y) with tol 1 m removes exactly one chain.
- `add_rows` gives `source=model` and `qa_flags` containing `override:A001`.
- `extend_rows` moves the end to the projection of `to`.
- Forced-empty tile has 0 objects, `no_vineyard`, and appears in `empty_tiles.csv` with reason `forced_empty`.
- START (629504.70, 5220250.75) resolves to `siret3_r018_c010` at (28.0, 2002.0), used to find the affected tiles.
- An unmatched override raises `override_unmatched` and is never silent.

**`test_topology.py` + `annset/test_assemble.py`**
- Canopy ∩ interrow of 0.02 m² in a tile: resolved to < 0.01 by `remove_canopy_overlap`, otherwise `canopy_interrow_overlap` error.
- A `row_id` with two `vineyard_id`s: `row_multi_block`.
- Object 2 mm outside its clip: error.
- AnnSet built from the reference: 0 errors; counts 399/251 canopies, 25/26 row pieces, 24/25 interrow pieces.
- `annset.json` has every contract field.
- `issue_id` order is stable.

**`qa/test_review.py`, `qa/test_render.py`**
- Priority rules (empty tile → 1, warnings only → 2, clean → 3); queue sorted by (priority, tile_id); exact `empty_tiles.csv` columns.
- Preview is 1024×1024; the pixel at a regular axis midpoint is red and at a disrupted one magenta.

**`stages/test_perception_examples_e2e.py` (`@slow`)**
- Full stage chain on the 2 tiles: row F1 ≥ 0.95 each.
- Canopy score ≥ 0.80 mean and ≥ 0.76 per tile (prototype 0.782 / 0.86).
- Interrow IoU ≥ 0.94 each (prototype 0.955).
- Attributes ≥ 0.9; 1 block per tile.

Coverage gate: ≥ 80% on `perception/`, `annset/assemble.py` and `qa/review.py`.

---

## 6. Dependencies

**I need:**

| Interface | Owner | Fallback if late |
|---|---|---|
| `geo.tiling` (**CONTRACT**) | geo | none (blocking; about 60 lines) |
| `geo.raster.{read_tile, rasterize_px, resize_aligned}` | geo | rasterio or cv2 directly inside `_raster_source.py` |
| `geo.ops.{orient, make_valid, notch_holes, split_multi}` | geo | notches: drop holes and log `tree_hole` |
| `geo.vector_io.{read_layer, write_layer, write_geojson}` | geo | `geopandas.to_parquet` directly |
| `contracts.{enums, ids, schemas.validate_layer}`, `contracts.qa.QaIssue` + `issues_to_gdf` | contracts | I define `QaIssue` in `perception/types.py`; contracts re-exports it |
| `pipeline.{dag, context}`: `RunContext`, stage registration, cache keys, `parallel_map` (spawn) | pipeline | e2e test calls the stage functions sequentially |
| `config.load_config` composing my settings models | config | tests build the settings directly |
| `tile_prep` outputs: `cache/{valid,veg,stats}`; requested `cache/vis/<t>.png`; layers `tile_index`, `tile_valid`, `in_passages`, `in_forbidden` | ingest | `vis` computed lazily from the tif |
| `cache/nn/<mv>/canopy_prob/<t>.png` + weights sha | nn | `None` (classical) |
| `layers/waste.parquet` (AnnSet schema, W-ids) | waste | empty layer + warning |
| `annset.{AnnSet, AnnSetMeta, write_annset, read_annset}` | annset/post-Marcaj | I write it if absent by 01:00 (about 60 lines) |
| `cvat.reader` (examples → reference) | cvat | lxml fallback in `tests/helpers/examples.py` |
| `eval.metrics` | eval | `tests/helpers/proto_metrics.py` |

**I provide:**

| To | What |
|---|---|
| tile_prep | `vegmask.compute_tile_masks` |
| nn | `corridor_labels`, `vine_mask`, `tree_mask`, `fuse_masks`, plus the `rows` and `canopies` layers for pseudo-labels |
| waste | `rows`, `blocks`, `clip_rows_to_tile`, and `cache/canopy` (plant centroids for the tube-phase filter) |
| export_cvat | AnnSet(model), `tile_status`, `qa_issues` (errors block export), `qa/empty_tiles.csv`, `qa/review_queue.csv` |
| targets | `attrs.occupancy_profile`, `find_gaps`, `gap_centre` (same code on AnnSet(model) and AnnSet(marcaj)) |
| derive / QA | `blocks_graph.neighbour_pairs`, `ordering.*` |
| web | `rows`, `blocks`, `interrows` layers |

---

## 7. Work packages

Timeline: now Friday 22:05; contract freeze 23:00. Commit and push after each package passes its acceptance tests.

**WP0 Foundations** (one agent, 23:00–23:45, 0.75 h)
- Builds: `perception/types.py`, `perception/settings.py`, `tests/conftest.py`, `tests/helpers/*`.
- Acceptance: settings tests pass; fixtures load both tifs (2048×2048×3), 51 rows, 650 canopies and 49 interrows.
- WP1–4 can start writing tests against §2 signatures in parallel.

**WP1 Per-tile detection** (Agent A, 23:00–02:30, 3.5 h)
- Builds: `vegmask`, `profile`, `linefit` (no tracking), `row_features`, `rows_detect` (multi-orientation included), `stages/rows_detect`.
- Input: example tifs and veg masks.
- Output: `row_candidates` per tile.
- Acceptance: `test_vegmask`, `test_profile`, `test_linefit`, `test_row_features`, `test_rows_detect` (F1 ≥ 0.95 each; 51/51 reference rows survive the filters; ≤ 4 s per tile).

**WP2 Linking + blocks + overrides** (Agent B, 23:00–02:30, 3.5 h)
- Builds: `rows_link`, `blocks_graph`, `blocks`, `ordering`, `overrides`, `stages/{rows_link, blocks}`.
- Input: synthetic candidate frames and reference axes turned into candidates (no WP1 code needed).
- Output: `rows_raw`, `rows`, `blocks`, `row_pairs`, `rows_rejected`.
- Acceptance: `test_rows_link`, `test_blocks`, `test_ordering`, `test_overrides` (one block per example; r006 not split; r021 not split; deterministic).

**WP3 Canopy + interrow** (Agent C, 23:00–02:30, 3.5 h)
- Builds: `corridor`, `canopy`, `interrow`, `stages/{canopy, interrow}`.
- Input: `rows` built from reference axes; veg and vis rasters.
- Output: canopies, `interrows`, `interrow_pieces`.
- Acceptance: `test_corridor`, `test_canopy` (≥ 0.84 with reference axes), `test_interrow` (IoU ≥ 0.98, 24/25 pieces, cover 49/49).

**WP4 Attributes + assemble + QA** (Agent D, 23:00–02:30, 3.5 h)
- Builds: `attrs`, `topology`, `annset/assemble`, `qa/{render, review, overview}`, `stages/{row_attrs, assemble, qa_previews}`.
- Input: reference rows and canopies.
- Output: AnnSet(model), `tile_status`, `qa_issues`, CSVs, previews.
- Acceptance: `test_attrs` (≥ 50/51; only R15 allowed, flagged), `test_topology`, `test_assemble` (0 errors on the reference AnnSet), `qa` tests; one preview renders in < 1 s.

**M1 integration** (02:30–03:00): e2e test on the 2 examples (§5 gates).

**M2 full run** (03:00–03:30): all 311 tiles. Emit `rows_detect` SNR per tile into `review_queue` and look at its distribution for calibration. Record timings.

**WP5 Robustness** (Agent A, or B after M2; 03:30–06:00, 2.5 h)
- Builds: `trees` + canopy/interrow integration, orchard block rejection, texture fallback, Otsu fallback, curved-row tracking.
- Every item stays behind a flag that defaults to off.
- Acceptance per item: no-op on both examples (score ±0.005, rows unchanged), its synthetic positive test passes, and the list of affected tiles is surfaced in `review_queue` (priority 1).
- A flag is switched on only after the team QA-reviews its effect.

**M3 QA** (03:30–06:00, team plus WP5): review previews by priority, fill `overrides.yaml`, re-run. The cache recomputes only the tiles those rows touch.

**M4 final run** (06:00–06:45): final assemble with overrides, hand off to `export_cvat`, validated ZIPs by 07:00.

Ordering constraints:
- WP0 types first; WP1–4 are otherwise independent because they are coupled only through the §2.1 schemas.
- M1 needs WP1–4 plus ingest/tile_prep.
- WP5 needs WP1 and WP3 merged.
- The NN pseudo-label run needs M2 output.

---

## 8. Risks and open questions

| # | Risk / question | Mitigation |
|---|---|---|
| 1 | Arch §4.4.2 spacing/phase cut would split r021's V01 (median-of-3 spacing steps up to 0.93 m inside one reference block). | `spacing_cut_enabled: false`; a test pins this behaviour. QA flags `blocks_should_merge` / `block_split` for humans. |
| 2 | A "≥ 3 rows" transverse-band rule would split r006's V02 (4.7 m shared gap on R07–R09). | Full-width requirement plus test. **Open:** 4.0 m (arch intent: turning zones) vs 5.0 m (rules §6: < 5 m of non-vineyard ground means the same block). Default 4.0, logged as `transverse_band_cut` for QA. |
| 3 | The 5.0 m threshold misclassifies R15 (5.009 m, labelled regular) vs R06 (5.054 m, disrupted). | Keep 5.0 (no tuning on 2 tiles); accept 50/51; every 4.5–7 m row goes to the review queue. |
| 4 | Width p80 / occupancy / lattice filters could reject grassy true rows (r006 R16–R18; fan rows on r021 are about 0.35·s off-lattice). | Lattice is a soft flag only; acceptance "51/51 reference rows survive"; orchard filters are block-level and behind P1 flags. |
| 5 | Tree filter or tree holes delete grass clumps or grassy interrows. | Require round + wide + compact blobs; no-op test on the examples; default off; QA overlay shows the removed polygons. |
| 6 | SNR threshold 3.0 is not calibrated on non-vineyard tiles. | Soft gate + rescue in linking; SNR in `review_queue`; set the threshold after M2 from the distribution (tiles with SNR < 6 and rows get priority 1). |
| 7 | Rows aligned across roads that are missing from `passages.geojson` get linked into one block. | Full-width band cut; `qa/overview.jpg` coloured by block; `exclude_areas` / `delete_rows` overrides. |
| 8 | Corner blocks in grassy tiles are missed by per-tile detection. | Multi-orientation (up to 3) + rescue of soft-rejected pieces. P2: guided end extension. |
| 9 | Half-pixel convention changes shift metrics against the prototype. | 3×3 px and corridor-raster unit tests; gates set 0.015–0.02 below measured values. |
| 10 | Otsu fallback misfires on sparse young vines. | Behind a flag; fires only outside [0.10, 0.90]; test that it does not fire on the examples; logged `otsu_fallback`. |
| 11 | Performance (360-angle scan, per-peak fits over all veg points). | Coarse-to-fine angle search; slab pre-selection; target ≤ 4 s per tile, about 3 min for 311 tiles on 8 spawn workers. |
| 12 | Late dependencies (DAG, AnnSet I/O, cvat reader, eval). | Thin stage wrappers; local shims in tests; pure modules testable without the DAG. |
| 13 | Contract nits: `K\d{2}` can overflow; `Source` has no `override`; `row_candidates` gets extra columns; `row_pairs` and `rows_rejected` layers are new; `cache/vis`; new QA codes. | All additive. Propose them before the 23:00 freeze; overrides use `source=model` + `qa_flags=override:<id>`. |
| 14 | Canopies in one tile coming from two blocks with equal `row_index` make ordering ambiguous. | Sort key starts with `vineyard_id`. |
| 15 | Garden vineyards inside `in_forbidden` get masked (arch §4.2.1). | `rows.detect.mask_forbidden` flag; QA can add them back with `add_rows`. |
| 16 | P1 features (tracking, texture, orchard, trees) not finished by 06:00. | The MVP path alone meets the example gates; unfinished flags stay off (arch §6 cut list order). |

### Critical Files for Implementation
- /Users/maleticimiroslav/Vin Gigahack/analiza_exemple/scripts/axes.py
- /Users/maleticimiroslav/Vin Gigahack/analiza_exemple/scripts/attrs.py
- /Users/maleticimiroslav/Vin Gigahack/arhitectura/00_contracte.md
- /Users/maleticimiroslav/Vin Gigahack/arhitectura/ARHITECTURA_Siret3.md
- /Users/maleticimiroslav/Vin Gigahack/data & info/05_examples/siret3_examples_cvat/annotations.xml