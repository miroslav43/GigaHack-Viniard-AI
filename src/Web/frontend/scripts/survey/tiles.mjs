// The optional survey tile layer (tiles.geojson, web bundle v3): one footprint per supplied tile with its status,
// object counts, vegetation / nodata fractions and the review flag from src/AI/configs/tile_review.csv.
// Checks (validateTiles), the 4326 map layer (reprojected like every layer) and the summary.json "tiles" block.
// validate.mjs never imports this module (read-bundle.mjs runs both), so there is no import cycle.
import { FOOTPRINT_TOL_M, TILE_COUNTS, TILE_FRACTIONS, TILE_GRID, TILE_ID_RE, TILE_M, TILES, TILES_TOTAL } from "./contract.mjs";
import { bbox, isNum, ll, positions } from "./geo.mjs";
import { layerProblems } from "./validate.mjs";

const SAMPLE = 3;
const isNil = (v) => v === undefined || v === null;
const mm = (v) => Math.round(v * 1000) / 1000;
const idOf = (f, i) => f.properties?.tile ?? `feature #${i + 1}`;

/** One problem for every feature matching `bad`, like validate.mjs does for the other layers. */
const aggregate = (features, bad, what) => {
  const hits = features.flatMap((f, i) => (bad(f) ? [idOf(f, i)] : []));
  if (!hits.length) return [];
  const names = `${hits.slice(0, SAMPLE).join(", ")}${hits.length > SAMPLE ? ", …" : ""}`;
  return [{ file: TILES.file, message: `${what}: ${hits.length} feature${hits.length === 1 ? "" : "s"} (${names})` }];
};

/** Grid square of a tile id in EPSG:32635 as [minX, minY, maxX, maxY] (§6.6), or null for a malformed id. */
export const gridSquare = (tile) => {
  const m = TILE_ID_RE.exec(tile ?? "");
  if (!m) return null;
  const [r, c] = [Number(m[1]), Number(m[2])];
  const x0 = TILE_GRID.x0 + c * TILE_M, y1 = TILE_GRID.y0 - (r - TILE_GRID.r0) * TILE_M;
  return [x0, y1 - TILE_M, x0 + TILE_M, y1].map(mm);
};

const near = (a, b) => Math.abs(a - b) <= FOOTPRINT_TOL_M;
/** true when the exterior ring is the tile's grid square: every vertex on a corner, the bbox equal to the square. */
const onGrid = (f) => {
  const square = gridSquare(f.properties.tile);
  if (!square) return true; // reported as a malformed id
  const [x0, y0, x1, y1] = square;
  const ring = f.geometry.coordinates[0] ?? [];
  const isCorner = ([x, y]) => (near(x, x0) || near(x, x1)) && (near(y, y0) || near(y, y1));
  return ring.length >= 5 && ring.every(isCorner) && bbox(ring).every((v, i) => near(v, square[i]));
};

const isCount = (v) => Number.isInteger(v) && v >= 0;
const isFraction = (v) => isNil(v) || (isNum(v) && v >= 0 && v <= 1);
const hasObjects = (p) => p.n_rows + p.n_canopies > 0;

/** Property checks beyond the generic layer checks (run only when those passed). */
const propertyProblems = (features) => [
  ...aggregate(features, (f) => !TILE_ID_RE.test(f.properties.tile), "tile id not siret3_rNNN_cNNN"),
  ...TILE_COUNTS.flatMap((k) => aggregate(features, (f) => !isCount(f.properties[k]), `"${k}" not a non-negative integer`)),
  ...TILE_FRACTIONS.flatMap((k) => aggregate(features, (f) => !isFraction(f.properties[k]), `"${k}" not a fraction 0..1 or null`)),
  ...aggregate(features, (f) => !isNil(f.properties.review_priority) && !Number.isInteger(f.properties.review_priority),
    "\"review_priority\" not an integer or null"),
  ...aggregate(features, (f) => !isNil(f.properties.review_note) && typeof f.properties.review_note !== "string",
    "\"review_note\" not a string or null"),
  ...aggregate(features, (f) => typeof f.properties.has_mask !== "boolean", "missing boolean property \"has_mask\""),
  ...aggregate(features, (f) => !onGrid(f), `footprint is not the tile's ${TILE_M} m grid square (§6.6)`),
];

/**
 * tiles.geojson checks → { errors, warnings }. expected = manifest.tiles (311): a different feature count and a
 * status that disagrees with the object counts are warnings.
 */
export const validateTiles = (fc, expected = TILES_TOTAL) => {
  const generic = layerProblems(TILES, fc);
  if (generic.length) return { errors: generic, warnings: [] };
  const errors = propertyProblems(fc.features);
  if (errors.length) return { errors, warnings: [] };
  const n = fc.features.length;
  return {
    errors: [],
    warnings: [
      ...(n === expected ? [] : [{ file: TILES.file, message: `${n} tiles, expected one per supplied tile (${expected})` }]),
      ...aggregate(fc.features, (f) => hasObjects(f.properties) !== (f.properties.status === "vineyard"),
        "status disagrees with n_rows + n_canopies (vineyard ⇔ any row piece or canopy)"),
    ],
  };
};

/** The tile needs completion in Marcaj (listed in tile_review.csv). */
export const toComplete = (p) => !isNil(p.review_status);

/** summary.json "tiles" block. */
export const tilesSummary = (fc) => {
  const props = fc.features.map((f) => f.properties);
  return {
    total: props.length,
    vineyard: props.filter((p) => p.status === "vineyard").length,
    no_vineyard: props.filter((p) => p.status === "no_vineyard").length,
    to_complete: props.filter(toComplete).length,
  };
};

/** MapLibre image-source corners of a footprint (EPSG:32635 Polygon): [lon, lat] TL, TR, BR, BL (north up). */
export const imageCorners = (geometry) => {
  const [x0, y0, x1, y1] = bbox(positions(geometry));
  return [ll([x0, y1]), ll([x1, y1]), ll([x1, y0]), ll([x0, y0])];
};
