// The AI bundle contract (src/Web/CLAUDE.md §6.2–6.4) and the survey facts (§6.6) the converter relies on.
export const BUNDLE_CRS = "EPSG:32635";
export const CRS_URN = "urn:ogc:def:crs:EPSG::32635";
export const STAGES = ["model", "marcaj_corrected"];
export const TILES_TOTAL = 311;
export const TILE_M = 51.2;

export const ROW_STRUCTURES = ["regular", "disrupted", "unassessable"];
export const INTERROW_COVERS = ["bare_soil", "vegetation", "mixed", "unassessable"];
export const TARGET_TYPES = ["gap", "missing", "waste"];

export const MANIFEST_FILE = "manifest.json";
export const CSV_FILE = "measurements.csv";
// canopies may be GeoJSONSeq (one Feature per line, §6.2) or a plain FeatureCollection
export const CANOPY_FILES = ["canopies.geojsonl", "canopies.geojson"];
export const MANIFEST_REQUIRED = ["survey_id", "name", "captured_at", "gsd_m", "crs", "source", "license", "stage", "generated_at"];
export const CSV_HEADER =
  "level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,interrow_area_m2,interrow_area_ha,plant_count,row_structure";

const POLYGONAL = ["Polygon", "MultiPolygon"];
/**
 * Per layer: file, allowed geometry types, properties that must be set (`required`), properties that must be
 * present but may be null (`nullable`), the unique id property and the lowercase enums (§6.2, §6.3).
 */
export const LAYERS = {
  blocks: { file: "blocks.geojson", geometry: POLYGONAL, required: ["vineyard_id"], unique: "vineyard_id" },
  rows: {
    file: "rows.geojson",
    geometry: ["LineString", "MultiLineString"],
    required: ["row_id", "vineyard_id", "row_structure"],
    unique: "row_id",
    enums: { row_structure: ROW_STRUCTURES },
  },
  canopies: { file: CANOPY_FILES[0], geometry: POLYGONAL, required: ["canopy_id", "vineyard_id", "tile"], unique: "canopy_id" },
  interrows: {
    file: "interrows.geojson",
    geometry: POLYGONAL,
    required: ["interrow_id", "vineyard_id", "interrow_cover", "tile"],
    unique: "piece_id",
    enums: { interrow_cover: INTERROW_COVERS },
  },
  waste: { file: "waste.geojson", geometry: POLYGONAL, required: ["waste_id", "tile"], nullable: ["vineyard_id"], unique: "waste_id" },
  // vineyard_id is null on a waste target more than 10 m from every block, like its waste (still set on gap / missing)
  targets: {
    file: "targets.geojson",
    geometry: ["Point"],
    required: ["target_id", "type"],
    nullable: ["vineyard_id", "row_id", "route_order"],
    unique: "target_id",
    enums: { type: TARGET_TYPES },
  },
  route: { file: "route.geojson", geometry: ["LineString"], required: ["length_m"] },
};

// map-performance guard: past this point canopies need a per-tile split or PMTiles (docs/DECISIONS.md ADR-006)
export const CANOPY_GUARD = { features: 40_000, bytes: 25 * 1024 * 1024 };

// ---------- optional: survey tiles and their vegetation masks (web bundle v3; older bundles have neither) ----------
export const TILE_STATUSES = ["vineyard", "no_vineyard"];
// src/AI/configs/tile_review.csv: a tile listed there needs completion in Marcaj
export const REVIEW_STATUSES = ["missed", "partial", "verify"];
export const TILE_ID_RE = /^siret3_r(\d{3})_c(\d{3})$/;
// tile origin (north-west corner) in EPSG:32635 (§6.6): x = 628992 + c·51.2, y = 5220966.4 − (r−5)·51.2
export const TILE_GRID = { x0: 628992, y0: 5220966.4, r0: 5 };
// footprint vs grid: the bundle writes coordinates at 3 decimals
export const FOOTPRINT_TOL_M = 0.002;
export const TILE_COUNTS = ["n_rows", "n_canopies", "n_interrows", "n_waste"];
export const TILE_FRACTIONS = ["veg_frac", "nodata_frac"];
export const TILES = {
  file: "tiles.geojson",
  geometry: ["Polygon"],
  required: ["tile", "status"],
  nullable: [...TILE_FRACTIONS, "review_priority", "review_note", "review_status"],
  unique: "tile",
  enums: { status: TILE_STATUSES, review_status: REVIEW_STATUSES },
};
export const MASKS_DIR = "masks";
export const MASK_PX = 1024;

// ---------- optional: farms and road classes (older bundles have neither) ----------
// a farm = vineyard blocks at most 10 m apart and not separated by a public road; its vineyard_ids are the source of
// truth for block membership (blocks.geojson farm_id is filled from them when missing)
export const FARMS = { file: "farms.geojson", geometry: POLYGONAL, required: ["farm_id"], unique: "farm_id" };
export const ROAD_CLASSES = ["public", "field", "internal"];
// "cadastre" = a public road found only in the cadastre (styled like an OSM road)
export const ROAD_SOURCES = ["osm", "detected", "cadastre"];
// farms and blocks may carry a cadastre snapshot: n_parcels, cadastral_codes, landuse_counts (each may be null)
// name, surface, farm_id and cadastral (bool) may be null or absent (read as null); length_m is measured when absent
export const ROADS = {
  file: "roads.geojson",
  geometry: ["LineString", "MultiLineString"],
  required: ["road_id", "road_class"],
  unique: "road_id",
  enums: { road_class: ROAD_CLASSES, source: ROAD_SOURCES },
};
