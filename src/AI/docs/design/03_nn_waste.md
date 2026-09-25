# Design: Neural network and waste subsystem (`vineyard/nn`, `vineyard/perception/waste`)

This design is for the self-contained project at `/Users/maleticimiroslav/Vin Gigahack/src/AI/`. All paths below are relative to that folder. Section references use A§ for `ARHITECTURA_Siret3.md` v1.1, C§ for `00_contracte.md` v1.0 and R§ for `CERCETARE_Siret3.md`. Where A and C disagree, A wins.

## 0. Facts checked during exploration that affect this design

- **The two reference tiles are in the 311 upload set.** `siret3_r006_c004.tif` is byte-identical in `01_tiles/…part1of5/` and `05_examples/…/images/` (checked with `cmp`), and both names appear in the upload ZIPs. Holding them out of training therefore has to be done by `tile_id`. A test must also assert that neither ever enters the patch store.
- **SAM 3.1 cannot be loaded through transformers.**
  - `/Users/maleticimiroslav/Vin Gigahack/models/sam3.1/README.md` says: "This repository hosts only the SAM 3.1 model checkpoints — there is no Hugging Face Transformers integration."
  - `config.json` declares `architectures: ["Sam3VideoModel"]` and `model_type: sam3_video`.
  - The only weights file is `sam3.1_multiplex.pt`, a native checkpoint with no `model.safetensors`. It is still downloading (a 3.5 GB `.incomplete` file at 22:04).
  - So `Sam3Model.from_pretrained("facebook/sam3.1")` will fail. The adapter checks the repo's file list before loading and skips any id without transformers weights.
  - The path that works needs access to the separate gated repo `facebook/sam3`. Request it now.
- **Nothing ML is installed yet.** No `transformers`, `open_clip`, `sklearn`, `torch`, `smp` or `timm` in either python3 or python3.12. All API claims below about transformers SAM3, open_clip 3.3 and smp 0.5 still need an import test right after `uv lock`.
- **scikit-learn is missing from the A§4.1 pin list.** The probe needs it at training time only. Scoring is pure numpy.
- **What the repo docs say about UAVVaste** (A§4.9, R§1.6, R§1.11):
  - Apache-2.0 licence; 772 low-altitude UAV images; 3,718 COCO-like annotations with bbox and segmentation; a single class, `rubbish`.
  - Scenes are urban and natural: streets, parks, lawns.
  - Download is either `python3 main.py` in `github.com/PUTvision/UAVVaste` or Zenodo record 8214061. The paper is Sci Data s41597-023-01976-9.
  - **GSD and altitude are not published.** The docs don't give archive size, image resolution, or whether the Zenodo record holds the images or only the annotations. These are checked at fetch time (see §3.W7).
- **Contract conflicts resolved here** (A wins):
  - `waste.min_area_m2` becomes 0.015 (C said 0.05).
  - C's `waste.export_min_confidence` and `waste.detector` are replaced by A§4.9's two-model decision rule (`waste.decide.*`, `waste.sam3.enabled`).
  - C's `nn` block is kept as the key structure and filled with A§3.6 values (`batch_size` = `batch`, `prob_threshold` = `threshold`, `in_gsd_m` = `gsd_m`).
  - The probability threshold comparison is `>=` (C§6). Both readings agree after uint8 quantisation (0.5 → 128 → 0.502).

---

## 1. Files

`src/AI/vineyard/nn/`:

| Path | Responsibility | ~lines |
|---|---|---|
| `nn/__init__.py` | torch-free re-exports (`probs`, `fusion`, `weights` version helpers) | 20 |
| `nn/config.py` | pydantic `NnConfig` (+ `PseudoLabelCfg`, `TrainCfg`, `AblationCfg`), `extra="forbid"`, frozen; imported by root `config.py` | 130 |
| `nn/errors.py` | `NnError`, `WeightsIntegrityError`, `PseudoLabelError`, `NnTrainingError(epoch, batch)`, `ProbRasterError` | 40 |
| `nn/env.py` | sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and `PYTORCH_MPS_HIGH_WATERMARK_RATIO` before torch is imported; `select_device(pref, fallback)`; `sync(device)` for timing | 70 |
| `nn/probs.py` | **torch-free, contract C§6**: probability PNG paths; write/read uint8 1024²; INTER_LINEAR upsample to 2048; corridor statistics | 150 |
| `nn/fusion.py` | **torch-free**: mask variants A–E, returns `FusedMask` | 90 |
| `nn/pseudolabels.py` | **torch-free**: per-tile label {0,1,255} at 0.05 m; ignore bands; training-tile selection | 230 |
| `nn/patch_store.py` | **torch-free**: extracts 512² patches; memmap `.npy` shards + `manifest.json`; asserts holdout tiles are absent | 180 |
| `nn/dataset.py` | torch `PatchDataset` over the store; flips, rot90, ±10 % colour jitter; deterministic per (seed, epoch, idx) | 150 |
| `nn/model.py` | `smp.Unet(resnet18, classes=1)`; ImageNet normalisation wrapper; uint8 batch → probabilities | 110 |
| `nn/losses.py` | masked BCE with label smoothing plus soft Dice, both ignoring 255 | 90 |
| `nn/schedule.py` | `EarlyStopper` (immutable state transitions); cosine LR; `drop_noisiest` index selection | 110 |
| `nn/validate.py` | per-epoch validation on the 2 reference tiles: variant B with reference-axis corridor, then a raster canopy score (ported from `fit.score`) | 170 |
| `nn/train.py` | orchestration: spawn DataLoader, epoch loop, smoke mode, optional drop-noisiest, best checkpoint, history, writes weights + card | 320 |
| `nn/infer.py` | main-process batched full-tile inference → `canopy_prob` PNG; nodata set to zero; MPS-synced timing | 220 |
| `nn/weights.py` | `ModelCard`; save/load `state_dict`; sha256; fetch + verify; version strings | 200 |
| `nn/ablation.py` | variants A–F on the reference tiles through the canopy/eval interfaces; per-tile + mean; promotion rule; JSON + MD | 260 |
| `nn/panels.py` | RGB \| a\* \| NN \| diff JPEG panels; automatic pick of hard tiles | 150 |
| `nn/cli.py` | typer sub-app `nn`: `pseudolabels`, `train [--smoke] [--variant F]`, `infer`, `ablate`, `fetch` | 180 |
| `pipeline/stages/nn_infer.py` | stage wrapper: `STAGE_VERSION`, cfg keys, per-tile key = sha256(tile) + weights sha; degrades when disabled or weights are missing | 130 |

`src/AI/vineyard/perception/waste/`:

These files sit where the user's spec puts them. Modules that load models (`clip_embed`, `probe` load/save, `sam3_adapter`, `uavvaste`) keep file and network access behind `load_*`/`fetch_*` functions, so the scoring logic stays pure. None of them are imported eagerly by `__init__`.

| Path | Responsibility | ~lines |
|---|---|---|
| `__init__.py` | torch-free exports only | 15 |
| `config.py` | pydantic `WasteConfig` (+ `ColourCfg`, `DecideCfg`, `ProbeCfg`, `Sam3Cfg`, `ReviewCfg`) | 170 |
| `types.py` | frozen `BoxPx`, `Candidate`, `CandidateScores`, `Decision`, `Confirmation`, `BlockAssignment`; StrEnums `ColourClass`, `RejectReason`, `Category`, `Detector` | 170 |
| `candidates.py` | HSV "vivid" and "bright" masks; 8-connected components; per-component stats; box in CVAT px (corner convention); padding and clipping | 200 |
| `lattice.py` | plant lattice (pitch, phase) per row piece from near-axis white blobs; periodicity test | 130 |
| `filters.py` | near-axis (by mode), periodic, too-small-white, tube shape, hose, forbidden, area range / vehicle; `apply_filters` | 260 |
| `nms.py` | box IoU matrix; greedy NMS | 70 |
| `crops.py` | probe crop (box + 50 %, min 64, resized to 224); 512 px SAM window placement; `BoxPx` crop↔tile transforms | 120 |
| `clip_embed.py` | `ClipEmbedder` Protocol; OpenCLIP loader (fallback: transformers `CLIPModel`); batch embed; zero-shot margin | 190 |
| `probe.py` | train LR (sklearn); threshold calibration on out-of-fold scores grouped by tile; `.npz` save/load with sha (no pickle); numpy sigmoid scoring | 230 |
| `positives.py` | COCO parse; cut out objects via segmentation; rescale to target px; paste onto Sireț3 backgrounds; JPEG round-trip | 220 |
| `negatives.py` | hard negatives = unfiltered candidates on the 2 example tiles; random soil/grass/canopy boxes | 130 |
| `uavvaste.py` | fetch via Zenodo API (git fallback); md5 verify; `SOURCE.json`; licence copy | 170 |
| `sam3_adapter.py` | tries the model-id list with a file-list preflight; CPU load; text prompt + negative boxes; overlap rule; budgeted loop; `Sam3Status` | 300 |
| `verify.py` | main-process scoring: crops → CLIP/probe → SAM (budgeted) → `CandidateScores` | 200 |
| `decide.py` | auto/review rule; rank score; degradation levels L0–L3 | 110 |
| `assign.py` | `vineyard_id` (block containing the box, or nearest ≤ 10 m); W-id assignment; a/b pairing across tile edges | 180 |
| `confirm.py` | confirmations CSV schema and validation; IoU matching; `merge_confirmed` → AnnSet `waste` (called by `assemble`) | 200 |
| `review.py` | `qa/waste_candidates.csv`, `qa/waste_crops/*.jpg`, `qa/waste_review.html` contact sheet | 170 |
| `cli.py` | typer sub-app `waste`: `fetch-positives`, `train-probe`, `sam3-check`, `run`, `review`, `confirm` | 170 |
| `pipeline/stages/waste.py` | stage: per-tile pool (candidates + filters, no torch) → verify (main process) → finalize (NMS, ids, `vineyard_id`) → layers + review | 260 |

Other files:
- `configs/default.yaml`: the `nn` and `waste` sections.
- `configs/waste_confirmed.csv`: header only, committed.
- `THIRD_PARTY/UAVVaste/LICENSE`: copied at fetch time.

`src/AI/tests/`:

| Path | ~lines |
|---|---|
| `tests/nn/conftest.py`: synthetic tiles, labels, a tiny CPU model | 100 |
| `tests/nn/test_probs.py`, `test_fusion.py`, `test_pseudolabels.py`, `test_patch_store.py`, `test_dataset.py`, `test_losses.py`, `test_model.py`, `test_schedule.py`, `test_validate.py`, `test_weights.py`, `test_infer.py`, `test_ablation.py`, `test_nn_infer_stage.py`, `test_import_hygiene.py` | 60–150 each |
| `tests/nn/test_train_smoke.py` (`@pytest.mark.slow`) | 80 |
| `tests/waste/conftest.py`, `test_candidates.py`, `test_lattice.py`, `test_filters.py`, `test_nms.py`, `test_crops.py`, `test_probe.py`, `test_positives.py`, `test_uavvaste.py` (fake HTTP), `test_sam3_adapter.py`, `test_decide.py`, `test_assign.py`, `test_confirm.py`, `test_review.py`, `test_waste_stage.py` | 60–150 each |
| `tests/integration/test_examples_waste_zero_fp.py`, `test_examples_nn_baseline.py` (`@pytest.mark.examples`, skipped if data is missing) | 80 each |

Markers: `slow`, `examples`, `mps`, `network`. The last two never run in CI.

---

## 2. Public interfaces

Items marked **CONTRACT** are fixed by C§6 or C§2.5.10. Changing them requires bumping `CONTRACT_VERSION`.

```python
# vineyard/nn/probs.py  (torch-free)  -- CONTRACT C§6
PROB_PX: Final = 1024
CHANNELS: Final = ("canopy_prob", "axis_prob")
@dataclass(frozen=True)
class ProbStats: mean: float; p95: float; n_px: int
def prob_png_path(cache_root: Path, nn_version: str, channel: str, tile_id: str) -> Path   # cache/nn/<mv>/<channel>/<tile_id>.png
def write_prob_png(path: Path, prob: np.ndarray) -> None          # float32 (1024,1024) in [0,1] -> uint8 rint(p*255); atomic tmp+replace
def read_prob_png(path: Path) -> np.ndarray                       # uint8 (1024,1024); ProbRasterError on shape/dtype
def upsample_prob(prob_u8: np.ndarray, out_px: int = 2048) -> np.ndarray   # float32, cv2.INTER_LINEAR
def load_canopy_prob(cache_root: Path, nn_version: str | None, tile_id: str, out_px: int = 2048) -> np.ndarray | None  # None if nn disabled/no weights
def corridor_prob_stats(prob: np.ndarray, corridor: np.ndarray) -> ProbStats   # rows_detect.vine_score=mean, tile_status.vineyard_score=p95

# vineyard/nn/fusion.py  (torch-free)
class FusionVariant(StrEnum): A = "A"; B = "B"; C = "C"; D = "D"; E = "E"
@dataclass(frozen=True)
class FusedMask: mask: np.ndarray; variant_used: FusionVariant; fell_back: bool
def fuse_masks(veg: np.ndarray, prob: np.ndarray | None, corridor: np.ndarray | None,
               variant: FusionVariant, threshold: float) -> FusedMask
# A=veg∩corr; B=(p>=t)∩corr; C=veg∧(p>=t)∩corr; D=(veg∨p>=t)∩corr; E=(p>=t) (ablation only).
# prob None and variant!=A -> A with fell_back=True; caller logs "nn.fallback_classic" with tile_id.

# vineyard/nn/weights.py
@dataclass(frozen=True)
class DataSource: name: str; url: str; licence: str
@dataclass(frozen=True)
class ModelCard:   # CONTRACT C§6 fields
    name: str; version: str; arch: str; encoder: str; in_gsd_m: float; classes: tuple[str, ...]
    train_data: tuple[DataSource, ...]; metrics: Mapping[str, float]; sha256: str; created_at: str; train_cmd: str
@dataclass(frozen=True)
class LoadedWeights: card: ModelCard; weights_path: Path
def save_weights(state_dict: Mapping[str, Any], card: ModelCard, models_dir: Path) -> ModelCard   # models/<name>/<version>/{weights.pt,model_card.json}
def load_weights(models_dir: Path, name: str, version: str, expected_sha256: str | None) -> LoadedWeights  # WeightsIntegrityError
def fetch_weights(url: str, models_dir: Path, name: str, version: str, expected_sha256: str) -> LoadedWeights
def nn_version_string(card: ModelCard | None) -> str      # "vine-unet@v1+1a2b3c4d" | "none"   CONTRACT C§2.2/C§6
def resolve_nn_version(cfg: NnConfig, models_dir: Path) -> str   # for RunContext.model_version and consumers' cache keys

# vineyard/nn/pseudolabels.py  (torch-free)
@dataclass(frozen=True)
class LabelParams: factor: int; ignore_band_px: int; ignore_small_in_corridor: bool; ignore_clumps: bool
@dataclass(frozen=True)
class TileSelection: positives: tuple[str, ...]; empties: tuple[str, ...]; source: Literal["qa_file", "auto"]
def downsample_rgb(rgb: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray     # invalid->0, INTER_AREA (C§6)
def build_pseudolabel(canopy: np.ndarray, clumps: np.ndarray, corridor: np.ndarray, veg: np.ndarray,
                      valid: np.ndarray, p: LabelParams) -> np.ndarray                  # 2048² inputs -> uint8 1024² {0,1,255}
def select_training_tiles(status: pd.DataFrame, qa_approved: Sequence[str] | None,
                          holdout: Sequence[str], p: PseudoLabelCfg) -> TileSelection

# vineyard/nn/train.py / infer.py / ablation.py
@dataclass(frozen=True)
class EpochStats: epoch: int; train_loss: float; val_score: float; val_iou: float; val_f1: float; lr: float; img_per_s: float; wall_s: float
@dataclass(frozen=True)
class TrainResult: card: ModelCard; best_epoch: int; history: tuple[EpochStats, ...]; stopped_early: bool; device: str
def train_model(cfg: NnConfig, seed: int, store_dir: Path, val: Sequence["ValTile"], models_dir: Path, *,
                smoke: bool = False, variant: Literal["main", "F"] = "main") -> TrainResult
def infer_tiles(jobs: Sequence["InferJob"], weights: LoadedWeights, cfg: NnConfig, cache_root: Path) -> tuple["InferOutcome", ...]
@dataclass(frozen=True)
class VariantScore: variant: str; tile_id: str; iou: float; f1: float; score: float; axes: Literal["model", "reference"]
@dataclass(frozen=True)
class AblationReport: scores: tuple[VariantScore, ...]; means: Mapping[str, float]; promoted: FusionVariant; reason: str
def decide_promotion(scores: Sequence[VariantScore], baseline: str, candidates: Sequence[str],
                     tiles: Sequence[str], min_gain: float) -> tuple[FusionVariant, str]

# vineyard/pipeline/stages/nn_infer.py  (shape follows pipeline.dag's Stage protocol, owned by the pipeline subsystem)
STAGE_NAME = "nn_infer"; STAGE_VERSION = "1"; CFG_KEYS = ("nn", "nodata", "grid")
def run(ctx: RunContext, tile_ids: Sequence[str]) -> StageReport
```

```python
# vineyard/perception/waste/types.py
@dataclass(frozen=True)
class BoxPx: xtl: float; ytl: float; xbr: float; ybr: float        # CVAT continuous px, validated xtl<xbr, ytl<ybr, in [0,2048]
@dataclass(frozen=True)
class Candidate:
    tile_id: str; cand_key: str; box: BoxPx; centroid_px: tuple[float, float]; area_px: int; area_m2: float
    colour_class: ColourClass; is_white: bool; aspect: float; length_m: float; width_m: float
    mean_hsv: tuple[float, float, float]; reject_reason: RejectReason | None = None
@dataclass(frozen=True)
class CandidateScores: cand_key: str; probe_p: float | None; clip_pos_p: float | None; clip_margin: float | None
                       sam_score: float | None; category: Category
@dataclass(frozen=True)
class Decision: cand_key: str; auto: bool; review: bool; rank_score: float; detector: Detector
@dataclass(frozen=True)
class Confirmation: line_no: int; tile_id: str; box: BoxPx; decision: Literal["accept", "reject", "add"]
                    category: Category; reviewer: str; note: str
@dataclass(frozen=True)
class BlockAssignment: vineyard_id: str; dist_block_m: float

# candidates.py / filters.py / decide.py  (pure)
def find_candidates(rgb: np.ndarray, valid: np.ndarray, tile_id: str, p: CandidateParams) -> tuple[Candidate, ...]
@dataclass(frozen=True)
class TileWasteContext: axes_px: tuple[np.ndarray, ...]; forbidden_px: BaseGeometry | None; gsd_m: float
def apply_filters(cands: Sequence[Candidate], ctx: TileWasteContext, p: FilterParams) -> tuple[Candidate, ...]  # same order; rejected get reason
def nms(cands: Sequence[Candidate], rank: Mapping[str, float], iou_max: float) -> tuple[Candidate, ...]
def decide(s: CandidateScores, probe_auto_min: float | None, p: DecideParams) -> Decision

# assign.py / confirm.py  -- called by `assemble` and reused by import QA
def assign_vineyard_id(box_utm: Polygon, blocks: gpd.GeoDataFrame, max_dist_m: float) -> BlockAssignment
def assign_waste_ids(waste: gpd.GeoDataFrame, edge_tol_px: float) -> gpd.GeoDataFrame          # W0001.., a/b edge pairs
def read_confirmations(path: Path, known_tiles: frozenset[str]) -> tuple[Confirmation, ...]    # ConfirmationFileError(line_no, field, value)
def merge_confirmed(candidates: gpd.GeoDataFrame, confirmations: Sequence[Confirmation], blocks: gpd.GeoDataFrame,
                    tiles: Mapping[str, TileRef], p: MergeParams) -> gpd.GeoDataFrame           # -> AnnSet `waste` layer (CONTRACT C§2.5.10 columns)

# sam3_adapter.py / probe.py / clip_embed.py
@dataclass(frozen=True)
class Sam3Status: available: bool; model_id: str | None; reason: str; tried: tuple[tuple[str, str], ...]
class Sam3Verifier(Protocol):
    def verify(self, crop_rgb: np.ndarray, cand_box: BoxPx, negatives: Sequence[BoxPx]) -> "SamResult": ...
def load_sam3(p: Sam3Params, loader: Sam3Loader | None = None, lister: FileLister | None = None) -> tuple[Sam3Verifier | None, Sam3Status]
def verify_with_budget(v: Sam3Verifier, jobs: Sequence["SamJob"], budget_s: float,
                       clock: Callable[[], float] = time.monotonic) -> tuple["SamResult | None", ...]
@dataclass(frozen=True)
class ProbeModel: coef: np.ndarray; intercept: float; auto_threshold: float; embed_model: str; version: str; sha256: str; recall_at_threshold: float
def train_probe(pos: np.ndarray, neg: np.ndarray, neg_groups: np.ndarray, neg_survives_filters: np.ndarray, p: ProbeParams) -> ProbeModel
def score_probe(m: ProbeModel, emb: np.ndarray) -> np.ndarray
class ClipEmbedder(Protocol):
    def embed(self, crops_u8: np.ndarray) -> np.ndarray: ...                    # (N,224,224,3) -> (N,D) L2-normalised
    def zero_shot(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # pos_p, margin, best_prompt_idx
def waste_version_string(probe: ProbeModel | None, clip: str | None, sam: Sam3Status) -> str   # e.g. "probe@v1+ab12cd34,sam3=none"

# vineyard/pipeline/stages/waste.py
STAGE_NAME = "waste"; STAGE_VERSION = "1"; CFG_KEYS = ("waste", "grid")
def waste_tile_worker(job: WasteTileJob) -> WasteTileResult      # top-level, picklable, no torch import
def run(ctx: RunContext, tile_ids: Sequence[str]) -> StageReport
```

**Layers and files this subsystem reads:**
- `tile_index` (`path`, `sha256`, `tile_id`)
- `cache/valid/<t>.png` and `cache/veg/<t>.png` (from `tile_prep`)
- `layers/rows.parquet` (`row_id`, `vineyard_id`, LineString) and `layers/blocks.parquet` (`vineyard_id`, MultiPolygon)
- `in_forbidden`
- `annset/canopies.parquet` from the model AnnSet (`canopy_id`, `tile_id`, `is_clump`)
- `tile_status` (`status`, `has_vineyard`, `n_row_pieces`) and `qa_issues`
- AnnSet(reference): `canopies` and `row_pieces` of the 2 example tiles
- the QA list file `nn.pseudolabels.train_tiles_file`
- `configs/waste_confirmed.csv`

**Files and layers this subsystem writes:**

NN:
- `cache/nn/<mv>/canopy_prob/<t>.png` + `.key` (**CONTRACT**)
- `work/nn/stores/<hash>/{img_000.npy, lbl_000.npy, manifest.json}`
- `models/vine-unet/<ver>/{weights.pt, model_card.json}`
- `runs/<id>/metrics/{nn_train.json, nn_ablation.json}`
- `reports/nn_ablation.md` and `qa/nn_panels/*.jpg`

Waste:
- `cache/waste/<t>.parquet` + `.key`: all candidates of the tile, including rejected ones with their reason.
- `layers/waste_candidates.parquet`: C§2.5.10 columns plus extra columns that `contracts.schemas` must allow: `cand_key`, `colour_class`, `probe_p`, `clip_pos_p`, `clip_margin`, `sam_score`, `rank_score`, `auto`, `review`.
- `qa/waste_candidates.csv`, `qa/waste_crops/<cand_key>.jpg`, `qa/waste_review.html`.
- The AnnSet `waste` layer is written by `assemble` using the output of `merge_confirmed`. Its columns are:
  - `waste_id`, `tile_id`, `vineyard_id`, `dist_block_m`
  - `px_xtl`, `px_ytl`, `px_xbr`, `px_ybr`
  - `area_m2`, `category`, `detector`, `exported=True`
  - provenance columns (C§2.2)

---

## 3. Algorithm notes

These cover only the places where the docs leave a choice open or where porting needs care.

**N1. Where NN inputs are consumed (C§6, C§3.2).**
- Consumers include the nn cache key in their own keys only if they actually use the probabilities:
  - canopy stage: when `nn.fusion != A`;
  - rows_detect: when `nn.use_in_rows`;
  - interrow: when `nn.use_in_interrow`.
- The two `use_in_*` flags default to false. Promoting the NN therefore re-runs only canopy → row_attrs → interrow → assemble → export, and not row detection. This keeps a late re-export cheap and safe.
- The first run (Friday night) has no weights. `nn_infer` logs `nn.weights_missing` once and returns "skipped". Every consumer receives `None` (C§6 degradation rule).

**N2. Pseudo-labels (A§4.6, R§3.4, R§3.6).**
- Label 1 is the rasterised model AnnSet `canopies` of the tile. Those polygons are already "a\* ∩ corridor, after the tree filter, ≥ 0.19 m²".
- Rasterisation follows C§1.4: `fillPoly((uv−0.5)*16, shift=4)` at 2048, then INTER_AREA down to 1024 and a ≥ 0.5 threshold. The same applies to the corridor, built from `rows` clipped to the tile with ±0.30 m.
- Pixels become 255 (ignore) in these cases:
  - nodata;
  - a 3 px band (7×7 kernel, dilate XOR erode) around the edges of both the label and the corridor;
  - canopies with `is_clump` (area > 2 m²). **This is a choice.** Grass inside the corridor is labelled canopy by the reference (the 55 m polygons in r006, RAPORT §2). Grass outside the corridor is labelled 0. Without the ignore, the network would see contradictory targets.
  - vegetation inside the corridor that belongs to components under 0.19 m² (`ignore_small_in_corridor`). A per-pixel network cannot enforce a component-area rule.
- Everything else is 0, which includes all vegetation outside corridors.
- Empty tiles confirmed by QA (`no_vineyard`) are added as all-zero labels, capped at 20 % of tiles. This teaches orchard, garden and grass = 0, which matters because false canopies on empty tiles are penalised.
- Tile selection:
  - If the QA list file exists, it is used.
  - Otherwise (logged `nn.train_tiles.auto`): `status == ok` and `has_vineyard` and `n_row_pieces ≥ 3` and no error-severity `qa_issues`.
  - Holdout tiles are always removed, with a warning if they appear in the list.
  - Training aborts if fewer than `min_train_tiles` remain.

**N3. Patches and augmentation.**
- Each tile is 1024² at 0.05 m; with stride 512 that gives 4 patches of 512² per tile (~400–600 per epoch).
- Stored as uint8 memmaps so DataLoader workers (spawn, `persistent_workers=True`, top-level picklable dataset) open them lazily.
- Geometric augmentations (flips, rot90) are applied to image and label together. Colour jitter (brightness, contrast, saturation factor in [0.9, 1.1]) is applied to the image only. No hue jitter, because hue is the signal.
- RNG = `np.random.default_rng((seed, epoch, idx))`, so a run is reproducible.

**N4. Loss (A§4.6).**
- Mask m = (label ≠ 255).
- Soft target y_s = y·(1−ε) + ε/2 with ε = 0.05, so 1 → 0.975 and 0 → 0.025.
- BCE is taken as the mean over m. Soft Dice uses the hard y over m, with smooth = 1.
- Total = w_bce·BCE + w_dice·Dice. A batch with m empty gives loss 0 (no NaN).
- A NaN loss raises `NnTrainingError(epoch, batch)`. Suggested remedy: `--set nn.device=cpu`.

**N5. Validation and early stopping.**
- Port `fit.score` (analiza_exemple/scripts/fit.py) into `nn/validate.py` as `raster_canopy_score`, clearly marked as an approximation (C§1.4 allows this).
- Fix carried over from the prototype: reference polygons are rasterised with C§1.4 `fillPoly(shift=4)` instead of `np.round(p)`.
- Prediction path: full-tile 1024² inference → INTER_LINEAR up to 2048 → `>= 0.5` → ∩ corridor from the **reference** axes (±0.30 m) → components ≥ 304 px (0.19 m²). The components step needs the canopy subsystem's `label_components`, a port of `axes.canopies`.
- The early-stopping signal is the mean score over the 2 tiles, patience 3; the best checkpoint is kept.
- Optional drop-noisiest: after epoch 5, compute per-patch loss in eval mode, drop the top 10 %, and continue with the new index tuple.

**N6. Inference (C§6).**
- Invalid pixels are set to 0 before INTER_AREA downsampling, in both training and inference.
- The full 1024² tile goes through the network in one pass (divisible by 32), so there are no seams.
- `infer_batch_size` = 4 on MPS, with `torch.inference_mode()` and `model.eval()`.
- Output = rint(p·255) multiplied by valid₁₀₂₄. Pixel k covers u ∈ [2k, 2k+2).
- `encoder_weights=None` at inference, so it runs offline. Device order: MPS, then CPU. If the first batch raises a "not implemented" RuntimeError, the error is logged with its context and the run retries on CPU.

**N7. Ablation (A§4.6, R§3.7).**
- Variants A–E come from `fuse_masks` on the 2 reference tiles. F uses a second set of weights (`version=v1f`) trained with `ignore_band_px=0` and `label_smoothing=0`.
- Masks become polygons through the canopy subsystem's `mask_to_canopies`, which is the same code as the export path. Scoring uses eval's **vector** `canopy_score` (0.6·IoU + 0.4·F1@0.5).
- Two axis sources are reported:
  - model rows: the primary result, since it matches what is exported;
  - reference axes: isolates the quality of the mask itself.
- Promotion: pick among {B, C, D} only if the variant beats A on **both** tiles by at least `min_gain` (default 0.0, per A§4.6). If several qualify, take the highest mean; otherwise A.
- The result is **not** applied automatically. The CLI prints `--set nn.fusion=X`, and the change is committed to `configs/default.yaml` together with `reports/nn_ablation.md`.
- A baseline regression check comes for free: the RAPORT end-to-end values are r021 0.774/0.795 → 0.782 and r006 0.841/0.890 → 0.861.

**W1. Candidates (A§4.9 step 1).**
- OpenCV HSV (H 0–179).
- vivid = S ≥ `vivid_s_min` ∧ V ≥ `vivid_v_min` ∧ H ∉ `veg_hue_range`.
- bright = V > 200 ∧ S < 45 (A§4.9).
- Both masks are multiplied by `valid`, then an optional 1 px opening, then 8-connected components.
- Area kept: 24–9600 px (0.015–6 m²).
- Box uses the corner convention (C§1.2): pixels i0..i1 map to [i0, i1+1]. **No +0.5**, because the box encloses whole pixels.
- Padding: `box_pad` = 10 % of each dimension in total (5 % per side), then clipped to [0, 2048]. **This is a choice.** "Padding de 10 %" in A§4.9 is ambiguous.
- `is_white` = the component comes from the bright mask.
- `cand_key` = `<tile_id>@<round(cx):04d>_<round(cy):04d>`. It is used for crop file names and display only.
- The default vivid thresholds are engineering estimates. They are calibrated on the 2 examples plus 20 random tiles, targeting a median of about 20 or fewer candidates per tile before filters. The calibration table goes to `reports/waste_calibration.md`.

**W2. Filters (A§4.9 step 2).**
- Distance to the axis is measured from the **centroid** to the row lines (tile rows ⊕ 1 m, in UTM, via `px_to_utm`). Centroid is the choice because tubes are point-like; the box-edge distance would also exclude real litter lying beside the row.
- `axis_exclusion_mode`:
  - `all` (default): reject anything < 0.6 m from an axis. This follows A§4.9 and makes rule 2a a subset.
  - `periodic`: near-axis objects are rejected only if white and on the plant lattice, so rule 2a becomes the operative rule. This mode is for the case where Slack Q7 says on-row waste counts.
- Lattice:
  - Take the projections t of white blobs within 0.6 m of an axis onto that axis (at least 4 of them).
  - pitch p = the mode of the pairwise differences within [0.8, 2.0] m (0.05 m histogram, fundamental only); phase φ = circular mean of t mod p.
  - A candidate is periodic if |((t − φ + p/2) mod p) − p/2| ≤ 0.25·p.
- Tube: aspect ratio (from `minAreaRect`) > 2.5 and area < 192 px.
- Hose: long side > 120 px (3 m) and width = area_px / long_side_px < 2 px (5 cm).
- Too-small-white: white and < 48 px.
- Forbidden: box intersects `in_forbidden`. Vehicle: area > 9600 px.
- NMS at IoU 0.3 is **not** a filter here. It runs after scoring, ordered by rank score (A§4.9 2d), in the finalize step.

**W3. Probe crops and CLIP.**
- Crop = box + 50 % context, side = max(64, 1.5·max(w, h)), square, shifted to stay inside the tile, resized to 224 with INTER_CUBIC.
- Zero-shot: softmax over the positive and negative prompt lists (`logit_scale` taken from the model); margin = max_pos − max_neg (A§4.9, R§1.7).
- `ClipEmbedder` is a Protocol so that `transformers.CLIPModel("openai/clip-vit-base-patch32")` can replace open_clip if open_clip fails to resolve in the lock (also ViT-B/32).

**W4. Positives and domain gap (the main risk of the probe).**
- UAVVaste has an unknown GSD and a different sensor. A probe trained on plain UAVVaste crops against Sireț3 negatives can learn "which dataset" instead of "trash or not".
- **Primary method: paste augmentation.**
  - Cut each UAVVaste object out with its COCO segmentation mask.
  - Resize so its long side falls in U[8, 40] px, which randomises scale over 0.2–1.0 m objects at 2.5 cm/px.
  - Paste it onto random Sireț3 background windows from tiles that are not holdout (Gaussian blur σ 0.5 px, JPEG q 90 round-trip), then crop with the same W3 policy.
- Secondary: plain rescaled UAVVaste crops with the same size policy (`positives.mode: both`).
- Negatives:
  - every candidate `find_candidates` returns on the 2 example tiles (examples have 0 waste; tubes and stakes are explicitly NOT WASTE);
  - 200 random candidate-sized boxes per example tile (soil, grass, canopy).
- Human `reject` decisions become extra negatives and human `accept`/`add` become in-domain positives when the probe is retrained (seconds, because embeddings are cached).

**W5. Probe calibration (A§4.9: "0 FP on examples").**
- LR (C = 1, `class_weight="balanced"`) on L2-normalised embeddings.
- Out-of-fold negative scores come from 2 folds grouped by tile (train on the r021 negatives plus all positives, score the r006 negatives, and the reverse).
- τ* = max(OOF scores of example negatives that **survive the geometric filters**; if none survive, all example negatives) + 0.02.
- Effective auto threshold = max(0.9, τ*).
- `recall_at_threshold` is measured on held-out pasted positives (5-fold). If it is below `min_recall` (0.2), auto-accept is disabled and the probe is used for ranking only; this is flagged in the report.
- Saved as `.npz` (coef, intercept, thresholds) plus `model_card.json`. No pickle.

**W6. SAM3 adapter (A§4.9, R§1.1–1.4).**
- For each id in `model_ids`, which may also be local directories:
  - preflight with `huggingface_hub.list_repo_files` (or listing the directory): the repo must contain `model.safetensors`, `model.safetensors.index.json` or `pytorch_model.bin`, otherwise skip with reason `no_transformers_weights`. **`facebook/sam3.1` is skipped this way.**
  - then lazy-import `transformers.Sam3Processor`/`Sam3Model`; `from_pretrained(id)`, fp32, CPU, eval.
  - catch `GatedRepoError`, `RepositoryNotFoundError`, `OSError`, `ImportError` and `ValueError` per id, and record them in `Sam3Status.tried`.
  - no id loads → `(None, status)`, logged as `waste.sam3.unavailable` and written to `run.json`.
  - the HF token is never logged.
- Per candidate:
  - 512 px window centred on the centroid, clamped to [0, 1536].
  - Up to 3 negative boxes: the nearest rejected white candidates (`near_axis`, `tube_shape`, `too_small_white`) inside the window, with label 0.
  - Each of the 5 text prompts is run separately. If the transformers API allows reusing vision features across prompts, the adapter does that (unverified; fallback is 5 full passes).
  - `post_process_instance_segmentation(threshold=0.4, mask_threshold=0.5)`.
  - An instance counts only if its mask covers ≥ 20 % of the candidate box.
  - sam_score = the maximum over prompts; category comes from the prompt map.
- Budget:
  - candidates ordered by probe score descending, only those with probe ≥ 0.3;
  - keep going while elapsed + last per-candidate cost ≤ 1800 s;
  - candidates left over get `sam_score=None`, so the CLIP-margin branch still applies to them.
- The MLX port is documented in the README as a fallback only. It needs Python 3.13+ and has an ambiguous licence (R§1.3), so it cannot share our 3.12 environment.

**W7. Fetching UAVVaste at implementation time (`vineyard waste fetch-positives`).**
1. `GET https://zenodo.org/api/records/8214061` and read `files[].{key, size, checksum="md5:…", links.self}`.
   - Abort if the total exceeds `max_download_gb` (8).
   - Log the file list, then stream each file to `work/external/uavvaste/<key>.part`, verify md5 and rename.
   - Write `SOURCE.json` with record id, DOI 10.5281/zenodo.8214061, files, md5 values, `retrieved_at` and `licence: Apache-2.0`.
2. If the record does not contain the images and COCO JSON, fall back to `git clone --depth 1 https://github.com/PUTvision/UAVVaste work/external/uavvaste/repo`:
   - record the commit sha;
   - read its README;
   - run its `main.py` downloader inside `work/external/` only, with the user's explicit OK, because it is third-party code.
3. Licence handling:
   - Apache-2.0 allows use and derivative models.
   - We **do not redistribute** images or crops (`work/` is gitignored). We ship only the probe coefficients.
   - Copy the upstream `LICENSE` (and `NOTICE`, if present) to `THIRD_PARTY/UAVVaste/`.
   - List it in the probe's `model_card.train_data` and in the README section "Third-party data and models". That section also covers: SAM License (used, not redistributed), OpenCLIP/LAION weights (licence taken from the model card and recorded at fetch time), and smp/timm (Apache-2.0).
   - The brief permits "open datasets with verified licences" and "any open pretrained models (e.g. SAM)".

**W8. Decision rule and ids (A§4.9 steps 4–6, C§2.4, C§1.6).**
- `auto` = probe ≥ max(0.9, τ*) ∧ (sam ≥ 0.5 ∨ margin > 0.2).
- `rank_score` = probe_p, else clip_pos_p, else a rule-only saliency score (S·log area).
- `review` = ¬auto ∧ rank_score ≥ 0.3. In rule-only mode (L3) the list is capped at the top 300.
- `detector`: `sam3` if the SAM branch fired; `nn` if the CLIP branch fired; `rule` for review-only candidates; `manual` for human `add`.
- Confirmation matching: same tile and IoU ≥ 0.5. `reject` overrides `auto`. An unmatched `accept` is kept with the confirmation's own box and `qa_flags=confirm_unmatched`.
- Final ids are assigned inside `merge_confirmed` on the exported set only:
  - sorted by (tile_id, ytl, xtl) → `W0001…`;
  - boxes touching a shared tile edge (≤ 1 px) whose extents along that edge overlap by ≥ 50 % of the smaller one share an id with suffixes `a`/`b` (a = the lexically smaller tile_id).
- **Contract clarification needed:** `waste_candidates` gets PK `waste_id` = `<tile_id>:W0001` (tile-local, like `:C0001`), so it never collides with exported `W…` ids.

---

## 4. Config keys owned by this subsystem

Source codes: A = `ARHITECTURA_Siret3.md`, C = `00_contracte.md`, R = `CERCETARE_Siret3.md`, "new" = design choice, marked calibrate where it must be tuned on examples.

**`nn`:**

| YAML path | Default | Source |
|---|---|---|
| `enabled` / `name` / `version` / `arch` / `encoder` | true / vine-unet / v1 / unet / resnet18 | C§9, A§3.6 |
| `in_gsd_m` | 0.05 | C§9 (= A `gsd_m`) |
| `out_channels` / `axis_sigma_m` | 1 / 0.15 | C§6 (axis head reserved, off) |
| `device` / `fallback_device` | mps / cpu | C§9, A§4.1 |
| `batch_size` / `infer_batch_size` | 8 / 4 | A§3.6 `batch`; A§4.1 "4–8" |
| `prob_threshold` | 0.5 | C§9 / A `threshold` |
| `weights_sha256` / `weights_url` | null / null | C§9 / C§6 |
| `fusion` | A | A§4.6 (promotion result) |
| `use_in_rows` / `use_in_interrow` | false / false | new (C§6 consumers) |
| `holdout_tiles` | [siret3_r006_c004, siret3_r021_c012] | A§4.6 |
| `mps_high_watermark_ratio` | 0.8 | A§4.1 |
| `pseudolabels.ignore_band_px` | 3 | A§3.6 |
| `pseudolabels.ignore_small_in_corridor` / `.ignore_clumps` | true / true | new (§3 N2) |
| `pseudolabels.train_tiles_file` | null | A§4.6 (QA list) |
| `pseudolabels.include_empty_tiles` / `.max_empty_tile_frac` / `.min_train_tiles` | true / 0.2 / 30 | new |
| `pseudolabels.patch_px` / `.patch_stride_px` | 512 / 512 | A§3.6 / new |
| `train.epochs` / `.lr` / `.weight_decay` | 20 / 3.0e-4 / 0.01 | A§3.6 / torch AdamW default |
| `train.label_smoothing` / `.patience` | 0.05 / 3 | A§3.6 |
| `train.bce_weight` / `.dice_weight` | 1.0 / 1.0 | A§4.6 |
| `train.num_workers` / `.colour_jitter` / `.encoder_weights` | 4 / 0.10 / imagenet | A§4.6 |
| `train.smoke_max_batches` | 50 | new |
| `train.drop_noisy.{enabled, after_epoch, frac}` | false / 5 / 0.10 | A§4.6 |
| `ablation.variants` / `.promotable` | [A,B,C,D,E,F] / [B,C,D] | A§4.6 |
| `ablation.min_gain` / `.axes_sources` | 0.0 / [model, reference] | new |
| `ablation.panel_tiles` / `.n_auto_panels` | [] / 2 | A§4.6 / new |

**`waste`:**

| YAML path | Default | Source |
|---|---|---|
| `enabled` | true | C§9 |
| `min_area_m2` / `max_area_m2` / `white_min_area_m2` | 0.015 / 6.0 / 0.03 | A§3.6 (min area overrides C) |
| `axis_exclusion_m` / `axis_exclusion_mode` | 0.6 / all | C name + A `min_dist_axis_m` / new (Slack Q7) |
| `plant_phase_tol` | 0.25 | A§3.6 |
| `lattice.pitch_range_m` / `lattice.min_points` | [0.8, 2.0] / 4 | new (RAPORT pitch 1.0–1.5 m) |
| `tube_max_aspect` / `tube_max_area_m2` | 2.5 / 0.12 | C§9, A§4.9 |
| `hose_min_len_m` / `hose_max_width_m` | 3.0 / 0.05 | A§3.6 |
| `nms_iou` / `box_pad` | 0.3 / 0.10 | A§3.6 |
| `block_assign_max_m` | 10.0 | C§9 (= A `block_radius_m`) |
| `max_candidates_total` | 9999 | new (4-digit id width) |
| `colour.vivid_s_min` / `.vivid_v_min` / `.veg_hue_range` | 150 / 70 / [25, 95] | new, calibrate |
| `colour.bright_v_min` / `.bright_s_max` / `.open_px` | 200 / 45 / 1 | A§4.9 / new |
| `decide.candidate_min` | 0.3 | A§3.6 |
| `decide.auto_probe_min` / `.auto_sam3_min` / `.auto_clip_margin_min` | 0.9 / 0.5 / 0.2 | A§3.6 |
| `decide.rule_only_max_review` | 300 | new |
| `probe.enabled` / `.name` / `.version` | true / waste-probe / v1 | new |
| `probe.clip_model` / `.clip_pretrained` / `.device` / `.embed_batch` | ViT-B-32 / laion2b_s34b_b79k / mps / 64 | A§4.9 / new |
| `probe.crop_context` / `.crop_min_px` / `.crop_resize_px` | 0.5 / 64 / 224 | A§4.9 |
| `probe.lr_c` / `.calibration_margin` / `.min_recall` | 1.0 / 0.02 / 0.2 | new |
| `probe.positives.mode` / `.long_side_px` / `.max_n` / `.jpeg_quality` / `.paste_blur_sigma_px` | both / [8, 40] / 1500 / 90 / 0.5 | new (§3 W4) |
| `probe.positives.zenodo_record` / `.repo_url` / `.licence` / `.max_download_gb` | 8214061 / https://github.com/PUTvision/UAVVaste / Apache-2.0 / 8 | A§4.9 / new |
| `probe.negatives.random_per_tile` | 200 | new |
| `probe.clip_pos_prompts` | ["trash", "a plastic bag", "a plastic bottle", "a tire", "a heap of rubbish", "litter on the ground"] | A§4.9 |
| `probe.clip_neg_prompts` | ["a white plastic vine tube", "a wooden stake", "a stone", "bare soil", "grass", "a grapevine", "an irrigation hose", "a car"] | A§4.9 NOT-WASTE list |
| `sam3.enabled` / `.model_ids` / `.device` | true / [facebook/sam3.1, facebook/sam3] / cpu | user decision (the preflight skips 3.1), A§4.9 |
| `sam3.crop_px` / `.text_prompts` | 512 / [trash, plastic bag, plastic bottle, tire, rubbish heap] | A§3.6, A§4.9 |
| `sam3.prompt_categories` | {trash: debris, plastic bag: bag, plastic bottle: bottle, tire: tyre, rubbish heap: heap} | C§2.5.10 enum |
| `sam3.max_negative_boxes` / `.threshold` / `.mask_threshold` | 3 / 0.4 / 0.5 | A§4.9 / R§1.1 |
| `sam3.min_overlap_frac` / `.time_budget_s` / `.min_probe_to_verify` | 0.2 / 1800 / 0.3 | new / A§4.9 / new |
| `review.crop_px` / `.confirmed_file` / `.confirm_match_iou` / `.edge_tol_px` | 256 / configs/waste_confirmed.csv / 0.5 / 1.0 | new |

Removed from C§9: `waste.detector`, `waste.export_min_confidence` (A§4.9 wins).

---

## 5. Tests to write first (TDD)

Values are from the docs or derived at GSD 0.025 m, i.e. 1 px = 0.000625 m².

**`tests/nn/test_probs.py`**
- `write_prob_png`: p = 0.5 → stored 128; p = 0.498 → 127; round-trips to (1024,1024) uint8.
- Wrong shape or dtype on read → `ProbRasterError`.
- Alignment (C§6): a 2048² image with ones only at u∈[10,12), v∈[20,22), downsampled with INTER_AREA, has exactly one nonzero pixel at (row 10, col 5) with value 1.0.
- `upsample_prob` of a constant stays constant; a linear ramp stays monotonic.
- `load_canopy_prob(None, …)` returns None.
- `corridor_prob_stats`: prob = 0.8 everywhere, corridor of 100 px → mean 0.8, p95 0.8, n_px 100.

**`tests/nn/test_fusion.py`**
- Truth tables for A–E on 2×2 masks. Example: veg = [1,1,0,0], p = [.9,.1,.9,.1], corr = all 1 → B = [1,0,1,0], C = [1,0,0,0], D = [1,1,1,0].
- prob None with variant C → mask = A, `fell_back=True`.
- Threshold uses `>=`: p exactly 0.5 counts as positive.

**`tests/nn/test_pseudolabels.py`**
- Isolated 20×20 canopy square at 1024 with corridor covering everything → exactly 480 pixels = 255 (26² − 14²), interior 14² = 196 pixels = 1.
- Nodata → 255.
- Vegetation outside the corridor → 0.
- A clump canopy → 255.
- A small in-corridor component → 255 when the flag is on, 0 when off.
- `select_training_tiles` never returns `siret3_r006_c004` or `siret3_r021_c012`, even if they appear in the QA list (warning logged).
- Raises `PseudoLabelError` below `min_train_tiles`.
- Empties are capped at 20 %.

**`tests/nn/test_patch_store.py`**
- A 1024² tile with stride 512 gives 4 patches with correct offsets.
- Manifest round-trip.
- Holdout id inside a store → `AssertionError` with the tile id in the message.

**`tests/nn/test_dataset.py`**
- Same seed/epoch/idx → identical sample.
- rot90 and flips are applied to label and image together (checked with an asymmetric marker).
- Jitter never changes the label.
- Output dtypes are float32 and int64.

**`tests/nn/test_losses.py`**
- Smoothing targets are 0.975 and 0.025.
- All-255 label → loss 0.0, finite.
- A perfect logit prediction has lower loss than an inverted one.
- Gradient is zero at ignored pixels.

**`tests/nn/test_model.py`** (CPU)
- `build_unet("resnet18", 1, None)`: (2,3,512,512) → (2,1,512,512); (1,3,1024,1024) → (1,1,1024,1024).
- Parameter count is ≈ 14.3 M (assert between 13 and 16 M).
- `encoder_weights=None` makes no network call (socket monkeypatched to raise).

**`tests/nn/test_schedule.py`**
- Validation scores [0.70, 0.74, 0.73, 0.735, 0.72] with patience 3 → stop after epoch 5, best epoch 2.
- Cosine LR: 3e-4 at epoch 0, about 0 at the last epoch.
- `drop_noisiest(losses=[…10 values], frac=0.1)` removes exactly the single worst index.

**`tests/nn/test_validate.py`**
- `raster_canopy_score(ref, ref)` = (1.0, 1.0).
- Disjoint prediction → (0, 0).
- One canopy shifted by half its width → IoU < 0.5, so it is not a TP.

**`tests/nn/test_weights.py`**
- The card's sha256 equals `sha256_file`.
- Flipping one byte → `WeightsIntegrityError`.
- `nn_version_string` → `vine-unet@v1+<sha[:8]>`, and `"none"` for None.
- `fetch_weights` with a fake URL payload verifies the sha.

**`tests/nn/test_infer.py`**
- A fake model returning constant 0.7 → PNG pixels all 179 (rint 178.5 = 178? no: rint(0.7·255 = 178.5) = 178, half-to-even), and 0 wherever valid = 0.
- Batch of 3 tiles gives 3 files plus `.key`.

**`tests/nn/test_nn_infer_stage.py`**
- `enabled=false` or no weights → stage returns "skipped", no files written, consumers receive None.
- The key changes when the weights sha changes.

**`tests/nn/test_ablation.py`**
- `decide_promotion` cases:
  - A = {r021: 0.782, r006: 0.861}; B = {0.80, 0.85} → **A** ("B loses on r006").
  - C = {0.79, 0.87} → **C**.
  - B = {0.79, 0.865} and C = {0.80, 0.87} → **C** (higher mean).
  - E and F are never promoted.
- Report JSON contains per-tile values and means for both axis sources.

**`tests/nn/test_import_hygiene.py`**
- In a subprocess, importing `vineyard.nn.probs`, `fusion`, `pseudolabels`, `patch_store` and `vineyard.perception.waste.{candidates,filters,lattice,nms,crops,decide,assign,confirm}` leaves `"torch" not in sys.modules`.

**`tests/nn/test_train_smoke.py`** (slow, CPU)
- 1 epoch on 8 synthetic patches: loss is finite, `weights.pt` and the card are written, the history has 1 entry.

**`tests/integration/test_examples_nn_baseline.py`** (examples marker)
- Variant A with model rows: r021 ≈ 0.782 ± 0.02, r006 ≈ 0.861 ± 0.02, mean ≥ 0.80 (RAPORT §3, A§5).
- With reference axes: mean ≈ 0.855 ± 0.02.

**`tests/waste/test_candidates.py`**
- A 5×5 px blue blob (25 px = 0.0156 m²) → 1 candidate. A 4×5 px blob (20 px) → none (min 24 px).
- A pure white 10×10 blob → `is_white=True`, class `bright`.
- A green blob (H = 60) → no candidate.
- Box of component pixels i = 100..119 is [100.0, 120.0]; with 10 % padding it becomes [99.0, 121.0].
- Nodata region → no candidates.
- A 100×100 vivid blob (10,000 px > 9,600) is rejected as a vehicle (see filters).

**`tests/waste/test_lattice.py`**
- White blobs at t = 0.3 + 1.2·k m (k = 0..9) → pitch 1.2 ± 0.05, phase 0.3 ± 0.05.
- Candidate at t = 3.9 → periodic (3.9 = 0.3 + 3·1.2). Candidate at t = 4.5 → not periodic (0.6 off, larger than 0.25·1.2 = 0.3).
- Fewer than 4 points → None.

**`tests/waste/test_filters.py`**
- Centroid at 0.59 m from the axis → `near_axis`; at 0.61 m → kept.
- In `periodic` mode, a vivid red candidate at 0.3 m from the axis is kept; a white one on the lattice is rejected with `periodic_tube`.
- White candidate of 47 px → `too_small_white`; 48 px → kept.
- Aspect 3.0 and 150 px → `tube_shape`; aspect 3.0 and 200 px → kept.
- Line 130 px long, 2 px wide → `hose`.
- Box intersecting forbidden → `forbidden`.
- 9,601 px → `vehicle`.
- The output tuple has the same length and order as the input, and the inputs are not mutated (frozen).

**`tests/waste/test_nms.py`**
- Two boxes with IoU 0.35 → only the higher-ranked one survives.
- IoU 0.29 → both survive.

**`tests/waste/test_crops.py`**
- A 20×10 box → a 64² crop centred on it.
- A 60×40 box → a 90² crop.
- Crops near tile edges are shifted inside [0, 2048].
- The SAM window for a centroid at (10, 2040) is [0, 1536]–[512, 2048].
- Box transforms tile↔crop round-trip exactly.

**`tests/waste/test_probe.py`**
- Separable synthetic embeddings (±μ): training succeeds; τ* equals the maximum OOF negative score + 0.02 (checked with fixed data); auto threshold = max(0.9, τ*).
- `.npz` round-trip gives identical scores, and the file contains no pickle (`allow_pickle=False` works).
- Inseparable data → `recall_at_threshold` < 0.2 → the report flags auto as disabled.

**`tests/waste/test_positives.py`**
- A tiny COCO fixture (2 images, 3 annotations with segmentation) parses into 3 objects.
- Pasting with long side 16 px gives a pasted-region bbox long side of 16 ± 1.
- Output is uint8 224².

**`tests/waste/test_uavvaste.py`**
- Fake Zenodo JSON: md5 mismatch → error naming the file and both hashes; oversize total → abort before downloading; success writes `SOURCE.json` with licence `Apache-2.0`.

**`tests/waste/test_sam3_adapter.py`** (fake lister and loader, no network)
- The file list `[config.json, processor_config.json, sam3.1_multiplex.pt]` → skip with reason `no_transformers_weights`.
- A loader raising `GatedRepoError` → next id is tried; all fail → `(None, status)` with 2 `tried` entries.
- Budget with a fake clock (10 s per verify, budget 30 s) → exactly 3 of 5 processed, the rest `None`.
- Negative boxes are passed with label 0 and capped at 3.
- A mask with 15 % overlap → ignored; 25 % → counted.

**`tests/waste/test_decide.py`** (probe, sam, margin → decision)
- (0.95, 0.6, 0.0) → auto; (0.95, 0.4, 0.25) → auto; (0.95, 0.4, 0.1) → review.
- (0.89, 0.9, 0.9) → review; (0.95, None, 0.21) → auto; (0.95, None, None) → review.
- rank 0.30 → in the review list; 0.29 → dropped.
- τ* = 0.93 and probe 0.92 → not auto.
- Detector is `sam3` versus `nn` depending on which branch fired.

**`tests/waste/test_assign.py`**
- A box inside V02 → ("V02", 0.0).
- A box 9.99 m from V01 → ("V01", 9.99). At 10.01 m → ("", 10.01).
- A box overlapping two blocks → the one with the larger intersection.
- W-ids for (r010_c005, ytl 50, xtl 900), (r010_c005, 50, 100), (r009_c007, 1000, 10) → r009 = W0001, xtl 100 = W0002, xtl 900 = W0003.
- A box with xbr = 2048.0 in c005 and one with xtl = 0.0 in c006 that overlap along v → W0004a (c005) and W0004b (c006).
- All ids match `^W\d{4}[ab]?$`.

**`tests/waste/test_confirm.py`**
- Bad decision value → `ConfirmationFileError(line_no=3, field="decision")`.
- `reject` at IoU 0.6 against an auto candidate → not exported.
- `accept` of a review candidate → exported.
- `add` → `detector=manual`, confidence 1.0, `vineyard_id` assigned.
- Unmatched `accept` → exported with flag `confirm_unmatched`.
- Output passes `validate_layer(gdf, "waste")`.

**`tests/waste/test_waste_stage.py`**
- Per-tile worker on a synthetic tile (read functions monkeypatched): cache parquet + key written; a second call is a cache hit.
- With the probe and SAM unavailable, the stage completes at level L3 and the degradation level is written to `run.json`.

**`tests/integration/test_examples_waste_zero_fp.py`** (examples marker)
- On both example tiles: `auto` count = 0 at every degradation level. With the trained probe, using out-of-fold scores, exported = 0.
- Logged, not asserted: the number of white blobs ≥ 0.01 m², for comparison with RAPORT §2 (327 and 648, of which 68 and 308 are within 0.6 m of an axis; the "white" definition used there was not recorded).

---

## 6. Dependencies

**What this subsystem needs from others:**
- **config**: the root `Config` composes `NnConfig` and `WasteConfig` (`extra="forbid"`); `cfg_hash(stage)`; `runtime.seed`.
- **contracts**: `enums` (Source, Label); `ids.format_waste_id` and `WASTE_ID_RE`; `schemas.validate_layer`. `LAYER_SCHEMAS["waste_candidates"]` must accept the extra columns listed in §2 and the tile-local PK.
- **geo**:
  - `tiling`: `TileRef`, `px_to_utm`, `utm_to_px`, `tiles_for_bounds`;
  - `raster`: `read_tile(path) -> uint8 (2048,2048,3)`, `read_mask_png`, `rasterize_px` (C§1.4 `fillPoly shift=4`);
  - `vector_io`: parquet read/write; `ops.clip_to_tile`.
- **pipeline**: `Stage` protocol; `RunContext` (paths, cfg, logger, run_id, `model_version`); `parallel_map` (spawn); atomic-write and cache-key helpers.
- **tile_prep**: `cache/valid` and `cache/veg` PNGs.
- **blocks**: `layers/rows.parquet`, `layers/blocks.parquet`.
- **canopy**:
  - `corridor_mask(lines_px, half_m, shape) -> uint8`;
  - `label_components(mask, min_area_px) -> int32 labels` (port of `axes.canopies`);
  - `mask_to_canopies(mask, rows_in_tile, tile, cfg) -> GeoDataFrame` (the export path);
  - AnnSet(model) `canopies` with `is_clump`.
- **eval**: `canopy_score(pred_gdf, ref_gdf) -> (iou, f1, score)`, vector version (C§1.4).
- **import_reference**: AnnSet(reference) for `siret3_r006_c004` and `siret3_r021_c012`.
- **ingest**: `tile_index`, `in_forbidden`.
- **QA**: the training-tile list file and the tiles confirmed as `no_vineyard`; a person working through `waste_review.html` and editing `configs/waste_confirmed.csv`.
- **assemble**: calls `merge_confirmed` and `corridor_prob_stats` (for `tile_status.vineyard_score`); composes `model_version` using `resolve_nn_version` and `waste_version_string`.
- **CLI / Makefile / Docker**:
  - `app.add_typer(nn_app, name="nn")` and the same for `waste_app`;
  - `nn.env.configure()` runs before any torch import;
  - Makefile targets `nn-overnight` (pseudolabels → train main → train F → infer 311 → ablate, run sequentially and never at the same time as the CPU pool) and `waste`;
  - Docker runs with the waste ML disabled when CLIP/SAM weights are absent (declared in the README).

**What this subsystem provides:**
- `nn.probs.load_canopy_prob` and `corridor_prob_stats` → rows_detect, canopy, interrow, assemble (**CONTRACT C§6**).
- `nn.fusion.fuse_masks` → canopy.
- `nn.weights.resolve_nn_version` → RunContext and consumers' cache keys.
- Stages `nn_infer` and `waste`; layer `waste_candidates`.
- `perception.waste.confirm.merge_confirmed` → AnnSet `waste` (**CONTRACT C§2.5.10 columns**).
- `perception.waste.assign.assign_vineyard_id` → import-QA check `waste_block_unknown`.
- Metrics and reports: `nn_ablation.json`/`.md`, panels, `nn_train.json` (timing and img/s), model cards, release assets (weights ~57 MB as a GitHub Release asset, because `*.pt` is gitignored).
- README text for the licences listed in §3 W7.

---

## 7. Work packages

Times are Chișinău time. Nothing in this subsystem is on the 07:00 ZIP critical path except the waste stage running at level L3 or better. The NN is promoted only by re-export, and only if the ablation finishes by Saturday 09:30. Otherwise export ships with `nn.fusion=A`. Each package ends with its tests passing, then commit and push.

**WP-N1: NN core, torch-free parts plus model/loss/dataset.** ~3 h, Fri 22:30–01:30.
- Inputs: C§1.4, C§6, the stubs in §2.
- Outputs: `nn/{config, errors, env, probs, fusion, pseudolabels, patch_store, dataset, model, losses, schedule}.py` and their tests.
- Acceptance: the tests for these modules pass; coverage ≥ 80 %; the import-hygiene test passes.
- Order: none. Other packages can start against the frozen signatures.

**WP-N2: NN train, validate, infer, weights, stage and CLI.** ~3.5 h, Fri 23:00–02:00, in parallel with N1 once signatures are agreed.
- Outputs: `nn/{validate, train, infer, weights, cli}.py` and `pipeline/stages/nn_infer.py`.
- Acceptance:
  - `vineyard nn train --smoke` runs 1 epoch on MPS on a store built from the reference labels of the 2 examples. This store is for timing only: flagged `smoke`, version `v0-smoke`, never promoted. No NaN; img/s logged. **This is the A§5 Friday gate.**
  - Inference on the 2 examples writes aligned PNGs.
  - The stage degrades correctly with no weights; the tamper test passes.
- Order: after N1's model/losses/dataset interfaces exist; the real run needs the classical pipeline outputs.

**WP-N3: ablation, panels, overnight chain.** ~2 h, Sat 00:30–02:30.
- Inputs: `eval.canopy_score` and `canopy.mask_to_canopies` (stubbed until they land), N2 outputs.
- Outputs: `nn/{ablation, panels}.py`, `make nn-overnight`.
- Acceptance: with fake probabilities equal to the a\* mask, B ≈ A and the promotion is A; the examples baseline test passes (0.782 / 0.861 ± 0.02).
- Order: after N2. The overnight chain starts at ~02:00, after classical run 1 and the first QA pass. Estimated duration ~1.5 h, including training F.

**WP-W1: waste geometry and bookkeeping (pure).** ~4 h, Fri 22:30–02:30.
- Outputs: `perception/waste/{config, types, candidates, lattice, filters, nms, crops, decide, assign, confirm, review}.py` and their tests.
- Acceptance:
  - All tests listed for these modules pass; coverage ≥ 80 %.
  - Candidate counts per tile on the 2 examples plus 20 random tiles, and the calibrated `colour.*` values, are recorded in `reports/waste_calibration.md`.
- Order: none.

**WP-W2: probe data and model.** ~3.5 h plus download time, Fri 22:30–02:00. Start the Zenodo fetch first, in the background.
- Outputs: `perception/waste/{uavvaste, positives, negatives, clip_embed, probe}.py`, `THIRD_PARTY/UAVVaste/LICENSE`, `models/waste-probe/v1/{probe.npz, model_card.json}`.
- Acceptance: τ* is reported; 0 out-of-fold FP on the examples at the auto threshold; recall on held-out pasted positives is reported.
- Time-box: if the positives are not ready by Saturday 03:00, fall back to level L2 (CLIP ranking only, no auto-accept).
- Order: only `types` and `crops` from W1, both of which are small.

**WP-W3: waste stage, then the SAM adapter.**
- (a) Stage orchestration, ~1.5 h, Sat 02:30–04:00.
  - Outputs: `pipeline/stages/waste.py`, `verify.py`, the CLI.
  - Acceptance: `vineyard run --until waste` on 311 tiles in < 10 min without SAM; candidates CSV, crops and HTML written; `auto` = 0 on the examples.
  - The result goes into the ~07:00 export at level L1 or L2.
- (b) `sam3_adapter.py`, strictly 1.5 h, and only if `facebook/sam3` access has been granted and `hf auth login` is done. A§6 lists it as the first thing to cut.
  - Acceptance: the adapter tests pass; `vineyard waste sam3-check` measures seconds per crop on CPU; the budget holds within ±5 %.
- Order: (a) after W1 and the W2 interfaces; (b) after (a). Never run at the same time as NN training (memory).

After 07:00 (outside these packages): the reviewer (role E) goes through `waste_review.html` from 07:00 to 10:00 → `configs/waste_confirmed.csv` → re-export at ~10:00, with `nn.fusion` changed if it was promoted → validator → upload. The candidate list is still useful after Publish for adding boxes by hand in Marcaj.

---

## 8. Risks and open questions

1. **`facebook/sam3.1` has no transformers weights**; this is confirmed from its README, config and file list. The 3.5 GB download will not help the transformers path.
   - Mitigation: request `facebook/sam3` access now; the preflight skips 3.1 in under a second.
   - Converting the multiplex checkpoint to transformers format is not attempted: it is unverified and there is no time.
   - Without SAM, auto-accept can still happen through the CLIP-margin branch (level L1).
2. **UAVVaste availability, size and GSD are unknown.** Mitigation: the Zenodo API preflight with size cap and md5; the git fallback; paste augmentation with scale randomised over [8, 40] px; time-box to 03:00, then level L2.
3. **Domain gap: the probe may learn "UAVVaste versus Sireț3".** Mitigation: paste positives onto Sireț3 backgrounds; `recall_at_threshold` must be at least 0.2 or auto-accept is switched off; human rejects and accepts feed a retrain in seconds (cached embeddings).
   - This fails safely: nothing gets auto-accepted, so there are no false positives.
4. **The same 2 tiles are used for early stopping and for the ablation**, which inflates the NN variants' scores.
   - Mitigation: also report the last-epoch checkpoint; F and E act as controls; `min_gain` is configurable.
   - Open question: raise `min_gain` above 0.0? Recommended 0.005, but that is the user's decision.
5. **Grass inside corridors is labelled canopy by the reference but ambiguous for the network** (r006 clumps). Mitigation: `ignore_clumps`. The ablation with reference axes shows whether B/C lose clump area. D (a\* OR NN) keeps it.
6. **The first QA pass may not have produced a training-tile list by 02:00.** Mitigation: automatic selection (logged, with holdout enforced); retrain in ~20 min once the list arrives, if time allows.
7. **MPS instability or NaN, or torch 2.14 regressions.** Mitigation: fp32 without autocast; fallback env vars; CPU retry; NaN raises with context. Fallback pin torch 2.13.x. CPU training (~1.5 h for 20 epochs) still fits overnight with `--set nn.train.epochs=10`.
8. **Memory contention on 18 GB of unified memory.** SAM in fp32 on CPU takes ~3.5 GB, NN training uses MPS and the CPU pool runs 8 processes. Mitigation: the Makefile runs them strictly one after another, and SAM/CLIP only execute in the main process after the pool has closed.
9. **Lock conflicts between transformers 5.17, open-clip 3.3, smp 0.5 and timm** (unverified). Mitigation: put `transformers` and `open-clip-torch` in the optional extra `waste-ml`, so the core lock is never blocked. `ClipEmbedder` can fall back to transformers `CLIPModel`. Add `scikit-learn` as a pin and run the import test right after `uv lock`.
10. **The axis exclusion removes real waste lying on the row** (A§8 Q7). Mitigation: `axis_exclusion_mode: periodic` exists as a switch, to use once Slack answers.
11. **Colour thresholds may produce too many candidates** (soil, flowers, roofs, cars). Mitigation: calibrate on the examples plus 20 tiles before running on 311; the forbidden filter removes the village; `max_candidates_total` raises an error instead of truncating silently; rule-only review lists are capped at 300.
12. **Contract clarifications to confirm with the contract owner:**
    - `waste_candidates` PK `<tile_id>:W0001`, plus the extra columns;
    - removal of `waste.detector` and `export_min_confidence`;
    - the new `nn.*` keys, including the consumer gating flags `use_in_rows`/`use_in_interrow`;
    - `>=` for the probability threshold.

    These are additive changes and would need `CONTRACT_VERSION` 1.1.
13. **Docker runs offline without CLIP or SAM weights**, so waste ML is disabled there. The deliverables (`route.geojson`, `measurements.csv`) come from the Marcaj export and are not affected. This is declared in the README. NN inference in Docker uses the CPU wheel (~8 min for 311 tiles) after `nn fetch`, or with `models/` mounted.

### Critical Files for Implementation
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/nn/pseudolabels.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/nn/train.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/nn/probs.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/perception/waste/filters.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/pipeline/stages/waste.py