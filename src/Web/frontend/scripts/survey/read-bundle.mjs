// Loads an AI survey bundle (src/Web/data/surveys/<id>/pipeline, src/Web/CLAUDE.md §6.2–6.4, EPSG:32635)
// and validates it. Every problem names the bundle file it comes from. tiles.geojson is optional (web bundle v3);
// its masks/ PNGs are checked by masks.mjs (async, headers only).
import fs from "node:fs";
import path from "node:path";
import { CANOPY_FILES, CSV_FILE, CSV_HEADER, LAYERS, MANIFEST_FILE, TILES, TILES_TOTAL } from "./contract.mjs";
import { validateTiles } from "./tiles.mjs";
import { validateBundle } from "./validate.mjs";

const LAYER_FILES = Object.fromEntries(
  Object.entries(LAYERS).filter(([key]) => key !== "canopies").map(([key, spec]) => [key, spec.file]),
);
const CSV_COLUMNS = CSV_HEADER.split(",");
const CSV_TEXT_COLUMNS = new Set(["level", "vineyard_id", "row_id", "row_structure"]);

export class BundleError extends Error {
  constructor(dir, problems) {
    super(`bundle ${dir} is invalid (${problems.length} problem${problems.length === 1 ? "" : "s"}):\n` +
      problems.map((p) => `  - ${p.file}: ${p.message}`).join("\n"));
    this.name = "BundleError";
    this.problems = problems;
  }
}

const readText = (dir, file) => {
  const full = path.join(dir, file);
  if (!fs.existsSync(full)) return { problem: { file, message: "missing file" } };
  return { text: fs.readFileSync(full, "utf8") };
};
const parseJson = (file, text) => {
  try {
    return { value: JSON.parse(text) };
  } catch (err) {
    return { problem: { file, message: `invalid JSON (${err.message})` } };
  }
};
const readJson = (dir, file) => {
  const { text, problem } = readText(dir, file);
  return problem ? { problem } : parseJson(file, text);
};

/** GeoJSONSeq: one Feature per non-empty line (an RFC 8142 record separator is tolerated). */
const RECORD_SEPARATOR = "\u001e";
const parseGeojsonl = (file, text) => {
  const lines = text.split("\n").map((l) => (l.startsWith(RECORD_SEPARATOR) ? l.slice(1) : l).trim());
  const features = [];
  for (const [i, line] of lines.entries()) {
    if (!line) continue;
    const { value, problem } = parseJson(`${file} line ${i + 1}`, line);
    if (problem) return { problem };
    features.push(value);
  }
  return { value: { type: "FeatureCollection", features, geojsonl: true } };
};

const readCanopies = (dir) => {
  const file = CANOPY_FILES.find((f) => fs.existsSync(path.join(dir, f)));
  if (!file) return { problem: { file: CANOPY_FILES[0], message: `missing file (nor ${CANOPY_FILES[1]})` } };
  const { text } = readText(dir, file);
  const parsed = file.endsWith(".geojsonl") ? parseGeojsonl(file, text) : parseJson(file, text);
  return parsed.problem ? parsed : { value: parsed.value, file };
};

/** measurements.csv → { header, lines: [{ line, level, vineyard_id, …, row_structure }] } (numbers parsed; "" → null). */
export const parseMeasurements = (text) => {
  const [header = "", ...rest] = text.replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n");
  const lines = rest.map((raw, i) => {
    const cells = raw.split(",");
    const record = Object.fromEntries(
      CSV_COLUMNS.map((col, c) => {
        const cell = cells[c] ?? "";
        return [col, cell === "" ? null : CSV_TEXT_COLUMNS.has(col) ? cell : Number(cell)];
      }),
    );
    return { line: i + 2, cellCount: cells.length, ...record };
  });
  return { header, lines };
};

// a bare route Feature (§6.3) becomes a one-feature FeatureCollection that keeps its crs member
const asCollection = (geojson) =>
  geojson?.type === "Feature"
    ? { type: "FeatureCollection", crs: geojson.crs, features: [Object.fromEntries(Object.entries(geojson).filter(([k]) => k !== "crs"))] }
    : geojson;

// optional files (web bundle v3): absent in older bundles, so absence is not a problem
const readOptionalJson = (dir, file) => (fs.existsSync(path.join(dir, file)) ? readJson(dir, file) : { value: null });

/** Reads every bundle file; returns { bundle, problems } (problems = missing or unparseable files). */
export const loadBundle = (dir) => {
  const reads = {
    manifest: readJson(dir, MANIFEST_FILE),
    ...Object.fromEntries(Object.entries(LAYER_FILES).map(([key, file]) => [key, readJson(dir, file)])),
    canopies: readCanopies(dir),
    csv: readText(dir, CSV_FILE),
    tiles: readOptionalJson(dir, TILES.file),
  };
  const problems = Object.values(reads).flatMap((r) => (r.problem ? [r.problem] : []));
  const layers = Object.fromEntries(Object.keys(LAYER_FILES).map((key) => [key, reads[key].value ?? null]));
  const bundle = {
    dir,
    manifest: reads.manifest.value ?? null,
    ...layers,
    route: asCollection(layers.route),
    canopies: reads.canopies.value ?? null,
    canopyFile: reads.canopies.file ?? CANOPY_FILES[0],
    csv: reads.csv.text === undefined ? null : parseMeasurements(reads.csv.text),
    // null = an older bundle without the tile layer
    tiles: reads.tiles.value ?? null,
  };
  return { bundle, problems };
};

/** tiles.geojson checks, when the bundle ships one (one feature per supplied tile: manifest.tiles, else 311). */
const tileChecks = (bundle) => {
  if (!bundle.tiles) return { errors: [], warnings: [] };
  const expected = Number.isInteger(bundle.manifest?.tiles) ? bundle.manifest.tiles : TILES_TOTAL;
  return validateTiles(bundle.tiles, expected);
};

/**
 * Loads and validates a bundle. Throws BundleError listing every problem; returns { bundle, warnings }.
 * opts: { surveyId, geofence (4326 Polygon|MultiPolygon, targets must lie inside) }
 */
export const readBundle = (dir, opts) => {
  if (!fs.existsSync(dir)) throw new BundleError(dir, [{ file: ".", message: "bundle directory does not exist" }]);
  const { bundle, problems } = loadBundle(dir);
  if (problems.length) throw new BundleError(dir, problems);
  const layers = validateBundle(bundle, opts);
  const tiles = tileChecks(bundle);
  const errors = [...layers.errors, ...tiles.errors];
  if (errors.length) throw new BundleError(dir, errors);
  return { bundle, warnings: [...layers.warnings, ...tiles.warnings] };
};
