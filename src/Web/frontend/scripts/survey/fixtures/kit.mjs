// Test support: temporary copies of the mini-bundle fixture, edits, and the NEW bundle schema
// (canopy area_m2, int route_order, speed_kmh, priority / route_role / skip_reason, interrow_total_m2,
// manifest model_version / tiles_with_objects / bbox_32635).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readBundle } from "../read-bundle.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const MINI_BUNDLE = path.join(HERE, "mini-bundle");
export const FRONTEND = path.resolve(HERE, "../../..");
export const SIRETI = JSON.parse(fs.readFileSync(path.join(FRONTEND, "data/osm/sireti_19100171.geojson"), "utf8")).features[0].geometry;

/** Fresh temp dir; removed by the test's `after` hook. */
export const tempDir = (t, prefix = "build-survey-") => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
};

/**
 * Copy of the fixture bundle, optionally in the new schema. tiles: false strips the web bundle v3 extras
 * (tiles.geojson, masks/, manifest counts.tiles and masks), giving a bundle as older pipelines wrote it.
 */
export const copyBundle = (t, { schema = "old", tiles = true } = {}) => {
  const dir = path.join(tempDir(t), "pipeline");
  fs.cpSync(MINI_BUNDLE, dir, { recursive: true });
  if (schema === "new") enrich(dir);
  if (!tiles) stripTiles(dir);
  return dir;
};

export const readJson = (dir, file) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
export const editJson = (dir, file, fn) => fs.writeFileSync(path.join(dir, file), JSON.stringify(fn(readJson(dir, file))));

// the fixture tiles (make-mini-bundle.mjs): A and B hold the objects and have masks, B and C are to complete in Marcaj
export const TILE_A = "siret3_r018_c010", TILE_B = "siret3_r018_c011", TILE_C = "siret3_r018_c012";

/** Removes tiles.geojson, masks/ and their manifest entries. */
export const stripTiles = (dir) => {
  fs.rmSync(path.join(dir, "tiles.geojson"));
  fs.rmSync(path.join(dir, "masks"), { recursive: true });
  const without = (o, key) => Object.fromEntries(Object.entries(o).filter(([k]) => k !== key));
  editJson(dir, "manifest.json", (m) => ({ ...without(m, "masks"), counts: without(m.counts, "tiles") }));
};

/** Rewrites the properties of the tiles.geojson feature of `tile`. */
export const editTile = (dir, tile, fn) => editJson(dir, "tiles.geojson", (fc) => ({
  ...fc, features: fc.features.map((f) => (f.properties.tile === tile ? { ...f, properties: fn(f.properties) } : f)),
}));
export const editText = (dir, file, fn) => fs.writeFileSync(path.join(dir, file), fn(fs.readFileSync(path.join(dir, file), "utf8")));
const mapFeatures = (fc, fn) => ({ ...fc, features: fc.features.map((f, i) => ({ ...f, properties: fn(f.properties, i) })) });

// values chosen to differ from what the converter would compute, so tests can tell "kept" from "derived"
export const NEW = {
  canopyAreas: [0.2512, 0.3612, 0.4912, 1.0012],
  interrowTotals: { "V01-I001": 398.5, "V02-I001": 173.5 },
  speedKmh: 4.5,
  modelVersion: "pipe@0.1.0+abc1234;nn=vine-unet/v1;waste=clip",
  tilesWithObjects: 7,
  targets: {
    T001: { priority: 1, route_role: "must", skip_reason: null },
    T002: { priority: 2, route_role: "optional", skip_reason: null },
    T003: { priority: 2, route_role: "optional", skip_reason: "optional_detour" },
    T004: { priority: 1, route_role: "must", skip_reason: "disconnected" },
    T005: { priority: 1, route_role: "must", skip_reason: "too_far" },
  },
};

/** Rewrites a bundle directory in the new (enriched) schema. */
export const enrich = (dir) => {
  const lines = fs.readFileSync(path.join(dir, "canopies.geojsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  fs.writeFileSync(path.join(dir, "canopies.geojsonl"),
    lines.map((f, i) => JSON.stringify({ ...f, properties: { ...f.properties, area_m2: NEW.canopyAreas[i] } })).join("\n") + "\n");
  editJson(dir, "interrows.geojson", (fc) => mapFeatures(fc, (p) => ({ ...p, interrow_total_m2: NEW.interrowTotals[p.interrow_id] })));
  editJson(dir, "targets.geojson", (fc) => mapFeatures(fc, (p) => ({ ...p, ...NEW.targets[p.target_id] })));
  editJson(dir, "route.geojson", (fc) => mapFeatures(fc, (p) => ({
    ...p, speed_kmh: NEW.speedKmh, duration_min: Math.round((p.length_m / 1000 / NEW.speedKmh) * 60 * 1000) / 1000,
  })));
  editJson(dir, "manifest.json", (m) => ({
    ...m, model_version: NEW.modelVersion, tiles_with_objects: NEW.tilesWithObjects, bbox_32635: [629504, 5220249.6, 629606.4, 5220300.8],
  }));
};

/** Validated bundle (throws BundleError), as the converter reads it. */
export const load = (dir, surveyId = "siret3") => readBundle(dir, { surveyId, geofence: SIRETI });
