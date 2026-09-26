import assert from "node:assert/strict";
import test from "node:test";
import { copyBundle, editJson, editTile, load, MINI_BUNDLE, TILE_A, TILE_B, TILE_C } from "./fixtures/kit.mjs";
import { ll } from "./geo.mjs";
import { BundleError } from "./read-bundle.mjs";
import { gridSquare, imageCorners, tilesSummary, toComplete, validateTiles } from "./tiles.mjs";

const tilesOf = (dir) => load(dir).bundle.tiles;
const tileProps = (fc, tile) => fc.features.find((f) => f.properties.tile === tile).properties;

/** Asserts that loading `dir` fails with a tiles.geojson problem matching `pattern`. */
const rejects = (dir, pattern) =>
  assert.throws(() => load(dir), (err) => {
    assert.ok(err instanceof BundleError, err.message);
    assert.ok(err.problems.some((p) => p.file === "tiles.geojson" && pattern.test(p.message)), `no tiles.geojson problem ${pattern} in:\n${err.message}`);
    return true;
  });

test("the fixture tile layer: 311 footprints, A and B vineyard with their objects, B and C to complete", () => {
  const fc = tilesOf(MINI_BUNDLE);
  assert.equal(fc.features.length, 311);
  assert.deepEqual(tileProps(fc, TILE_A), {
    tile: TILE_A, status: "vineyard", n_rows: 2, n_canopies: 2, n_interrows: 1, n_waste: 0, veg_frac: 0.01, nodata_frac: 0,
    review_priority: null, review_note: null, review_status: null, has_mask: true,
  });
  assert.equal(tileProps(fc, TILE_B).review_status, "partial");
  assert.equal(tileProps(fc, TILE_C).status, "no_vineyard");
  assert.deepEqual(tilesSummary(fc), { total: 311, vineyard: 2, no_vineyard: 309, to_complete: 2 });
});

test("gridSquare follows §6.6 (r018_c010 holds START); imageCorners are TL, TR, BR, BL", () => {
  assert.deepEqual(gridSquare(TILE_A), [629504, 5220249.6, 629555.2, 5220300.8]);
  assert.equal(gridSquare("siret3_r18_c10"), null);
  const geometry = tilesOf(MINI_BUNDLE).features.find((f) => f.properties.tile === TILE_A).geometry;
  assert.deepEqual(imageCorners(geometry), [ll([629504, 5220300.8]), ll([629555.2, 5220300.8]), ll([629555.2, 5220249.6]), ll([629504, 5220249.6])]);
});

test("toComplete: any review_status (missed | partial | verify); null is not", () => {
  assert.equal(toComplete({ review_status: "verify" }), true);
  assert.equal(toComplete({ review_status: null }), false);
  assert.equal(toComplete({}), false);
});

test("rejects bad statuses, review statuses, counts, fractions and a missing has_mask", (t) => {
  const dir = copyBundle(t);
  editTile(dir, TILE_A, (p) => ({ ...p, status: "Vineyard" }));
  rejects(dir, /"status" outside vineyard\|no_vineyard \(lowercase\): 1 feature \(siret3_r018_c010\)/);
  editTile(dir, TILE_A, (p) => ({ ...p, status: "vineyard", review_status: "todo" }));
  rejects(dir, /"review_status" outside missed\|partial\|verify/);
  editTile(dir, TILE_A, (p) => ({ ...p, review_status: null, n_rows: -1, veg_frac: 1.5 }));
  rejects(dir, /"n_rows" not a non-negative integer: 1 feature/);
  rejects(dir, /"veg_frac" not a fraction 0\.\.1 or null/);
  editTile(dir, TILE_A, (p) => ({ ...p, n_rows: 2, veg_frac: null, has_mask: "yes" }));
  rejects(dir, /missing boolean property "has_mask"/);
});

test("rejects duplicate tiles, malformed ids, a missing nullable key and another crs", (t) => {
  const dir = copyBundle(t);
  editTile(dir, TILE_B, (p) => ({ ...p, tile: TILE_A }));
  rejects(dir, /duplicate tile: siret3_r018_c010/);
  const fresh = copyBundle(t);
  editTile(fresh, TILE_A, (p) => ({ ...p, tile: "r018_c010" }));
  rejects(fresh, /tile id not siret3_rNNN_cNNN: 1 feature \(r018_c010\)/);
  editTile(fresh, "r018_c010", (p) => ({ ...p, tile: TILE_A }));
  editTile(fresh, TILE_C, (p) => Object.fromEntries(Object.entries(p).filter(([k]) => k !== "review_note")));
  rejects(fresh, /missing property "review_note" \(null allowed\): 1 feature \(siret3_r018_c012\)/);
  editJson(fresh, "tiles.geojson", (fc) => ({ ...fc, crs: { type: "name", properties: { name: "urn:ogc:def:crs:EPSG::4326" } } }));
  rejects(fresh, /crs is "urn:ogc:def:crs:EPSG::4326"/);
});

test("rejects a footprint that is not the tile's grid square", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "tiles.geojson", (fc) => ({
    ...fc,
    features: fc.features.map((f) => (f.properties.tile === TILE_A
      ? { ...f, geometry: { ...f.geometry, coordinates: [f.geometry.coordinates[0].map(([x, y]) => [x + 0.5, y])] } }
      : f)),
  }));
  rejects(dir, /footprint is not the tile's 51\.2 m grid square \(§6\.6\): 1 feature \(siret3_r018_c010\)/);
});

test("warns on a feature count other than manifest.tiles and on a status that disagrees with the counts", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "tiles.geojson", (fc) => ({ ...fc, features: fc.features.slice(0, 310) }));
  editJson(dir, "manifest.json", (m) => ({ ...m, counts: { ...m.counts, tiles: 310 } }));
  editTile(dir, TILE_A, (p) => ({ ...p, status: "no_vineyard" }));
  assert.deepEqual(load(dir).warnings.map((w) => `${w.file}: ${w.message}`), [
    "tiles.geojson: 310 tiles, expected one per supplied tile (311)",
    "tiles.geojson: status disagrees with n_rows + n_canopies (vineyard ⇔ any row piece or canopy): 1 feature (siret3_r018_c010)",
  ]);
});

test("validateTiles returns generic problems first and does not mutate its input", () => {
  const fc = tilesOf(MINI_BUNDLE);
  const before = JSON.stringify(fc);
  assert.deepEqual(validateTiles(fc), { errors: [], warnings: [] });
  assert.deepEqual(validateTiles({ type: "Feature" }).errors, [{ file: "tiles.geojson", message: "not a GeoJSON FeatureCollection" }]);
  assert.equal(JSON.stringify(fc), before);
});
