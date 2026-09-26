import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { copyBundle, editJson, editText, load, MINI_BUNDLE, SIRETI } from "./fixtures/kit.mjs";
import { BundleError, loadBundle, readBundle } from "./read-bundle.mjs";

/** Asserts that loading `dir` fails with a problem on `file` whose message matches `pattern`. */
const rejects = (dir, file, pattern, surveyId = "siret3") => {
  assert.throws(() => load(dir, surveyId), (err) => {
    assert.ok(err instanceof BundleError, err.message);
    assert.ok(err.problems.some((p) => p.file === file && pattern.test(p.message)), `no ${file} problem matching ${pattern} in:\n${err.message}`);
    assert.match(err.message, new RegExp(file.replace(".", "\\.")));
    return true;
  });
};

/** Rewrites the properties of one target of the bundle in `dir`. */
const editTarget = (dir, targetId, fn) => editJson(dir, "targets.geojson", (fc) => ({
  ...fc, features: fc.features.map((f) => (f.properties.target_id === targetId ? { ...f, properties: fn(f.properties) } : f)),
}));

test("the old-schema fixture is a valid bundle without warnings", () => {
  const { bundle, warnings } = load(MINI_BUNDLE);
  assert.deepEqual(warnings, []);
  assert.equal(bundle.canopies.features.length, 4);
  assert.equal(bundle.interrows.features.length, 4);
  assert.equal(bundle.targets.features.length, 5);
  assert.equal(bundle.csv.lines.length, 6);
  assert.equal(bundle.csv.lines[0].level, "survey");
  const t005 = bundle.targets.features.find((f) => f.properties.target_id === "T005").properties;
  assert.deepEqual([t005.type, t005.waste_id, t005.vineyard_id], ["waste", "W1", null], "like W1, > 10 m from every block");
  assert.equal(bundle.waste.features[0].properties.vineyard_id, null);
});

test("targets: vineyard_id may be null on waste targets only, and must always be present", (t) => {
  const dir = copyBundle(t);
  editTarget(dir, "T002", (p) => ({ ...p, vineyard_id: null }));
  rejects(dir, "targets.geojson", /missing property "vineyard_id" \(null only on waste targets\): 1 feature \(T002\)/);
  editTarget(dir, "T002", (p) => ({ ...p, vineyard_id: "V02" }));
  editTarget(dir, "T005", (p) => Object.fromEntries(Object.entries(p).filter(([k]) => k !== "vineyard_id")));
  rejects(dir, "targets.geojson", /missing property "vineyard_id" \(null allowed\): 1 feature \(T005\)/);
  editTarget(dir, "T005", (p) => ({ ...p, vineyard_id: "V02" }));
  assert.deepEqual(load(dir).warnings, [], "a waste target near a block keeps its block");
});

test("the new-schema variant (enriched fields) is valid too", (t) => {
  const { bundle, warnings } = load(copyBundle(t, { schema: "new" }));
  assert.deepEqual(warnings, []);
  assert.equal(bundle.manifest.tiles_with_objects, 7);
  assert.equal(bundle.targets.features.find((f) => f.properties.target_id === "T001").properties.route_role, "must");
});

test("rejects a manifest in another CRS", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "manifest.json", (m) => ({ ...m, crs: "EPSG:4326" }));
  rejects(dir, "manifest.json", /crs is "EPSG:4326", expected "EPSG:32635"/);
});

test("rejects a layer whose crs member is not EPSG:32635", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "rows.geojson", (fc) => ({ ...fc, crs: { type: "name", properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" } } }));
  rejects(dir, "rows.geojson", /crs is "urn:ogc:def:crs:OGC:1\.3:CRS84"/);
});

test("rejects degrees labelled as EPSG:32635", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "blocks.geojson", (fc) => ({
    ...fc, features: fc.features.map((f, i) => (i === 0 ? { ...f, geometry: { type: "Polygon", coordinates: [[[28.7, 47.1], [28.8, 47.1], [28.8, 47.2], [28.7, 47.1]]] } } : f)),
  }));
  rejects(dir, "blocks.geojson", /coordinates not in EPSG:32635 metres: 1 feature \(V01\)/);
});

test("rejects another survey_id", (t) => {
  rejects(copyBundle(t), "manifest.json", /survey_id is "siret3", expected "siret3-nn1"/, "siret3-nn1");
});

test("rejects a missing file, naming it", (t) => {
  const dir = copyBundle(t);
  fs.rmSync(path.join(dir, "targets.geojson"));
  rejects(dir, "targets.geojson", /missing file/);
  fs.rmSync(path.join(dir, "canopies.geojsonl"));
  rejects(dir, "canopies.geojsonl", /missing file/);
});

test("rejects a missing bundle directory", (t) => {
  assert.throws(() => readBundle(path.join(copyBundle(t), "nope"), { surveyId: "siret3" }), /bundle directory does not exist/);
});

test("rejects invalid JSON and GeoJSONSeq lines with their location", (t) => {
  const dir = copyBundle(t);
  editText(dir, "canopies.geojsonl", (s) => s.replace(/\n$/, "\n{broken\n"));
  rejects(dir, "canopies.geojsonl line 5", /invalid JSON/);
});

test("rejects uppercase enums, missing required properties and duplicate ids", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "rows.geojson", (fc) => ({ ...fc, features: fc.features.map((f, i) => (i === 0 ? { ...f, properties: { ...f.properties, row_structure: "Regular" } } : f)) }));
  rejects(dir, "rows.geojson", /row_structure" outside regular\|disrupted\|unassessable/);
  editJson(dir, "interrows.geojson", (fc) => ({ ...fc, features: fc.features.map((f) => ({ ...f, properties: { ...f.properties, tile: null, piece_id: "same" } })) }));
  rejects(dir, "interrows.geojson", /missing property "tile": 4 features/);
  rejects(dir, "interrows.geojson", /duplicate piece_id: same/);
});

test("route_order: 1.0 is accepted, 1.5 and repeated values are not", (t) => {
  const dir = copyBundle(t);
  assert.match(fs.readFileSync(path.join(dir, "targets.geojson"), "utf8"), /"route_order":1\.0/);
  editTarget(dir, "T002", (p) => ({ ...p, route_order: 1.5 }));
  rejects(dir, "targets.geojson", /route_order not a positive integer or null: 1 feature \(T002\)/);
  editTarget(dir, "T002", (p) => ({ ...p, route_order: 1 }));
  rejects(dir, "targets.geojson", /duplicate route_order: 1/);
});

test("rejects targets outside the Sireți geofence", (t) => {
  const dir = copyBundle(t);
  // ~20 km west, in Cojușna's direction
  editJson(dir, "targets.geojson", (fc) => ({ ...fc, features: fc.features.map((f) => (f.properties.target_id === "T004" ? { ...f, geometry: { type: "Point", coordinates: [609000, 5220000] } } : f)) }));
  rejects(dir, "targets.geojson", /outside the UAT geofence .*: 1 feature \(T004\)/);
  assert.doesNotThrow(() => readBundle(dir, { surveyId: "siret3", geofence: null }), "no geofence → no check");
  assert.ok(SIRETI.type === "Polygon" || SIRETI.type === "MultiPolygon");
});

test("measurements.csv: exact header, counts and ids must match the layers", (t) => {
  const dir = copyBundle(t);
  editText(dir, "measurements.csv", (s) => s.replace("level,vineyard_id", "Level,vineyard_id"));
  rejects(dir, "measurements.csv", /header differs from §6\.4/);
  fs.cpSync(path.join(MINI_BUNDLE, "measurements.csv"), path.join(dir, "measurements.csv"));
  editText(dir, "measurements.csv", (s) => s.replace("survey,,,2,3,", "survey,,,2,4,").replace("row,V02,V02-R001", "row,V02,V02-R009"));
  rejects(dir, "measurements.csv", /survey row_count 4 ≠ 3 features in rows\.geojson/);
  rejects(dir, "measurements.csv", /row ids in rows\.geojson but not in the CSV: V02-R001/);
});

test("loadBundle reports every missing file at once and accepts a bare route Feature", (t) => {
  const dir = copyBundle(t);
  editJson(dir, "route.geojson", (fc) => ({ ...fc.features[0], crs: fc.crs }));
  assert.doesNotThrow(() => load(dir));
  fs.rmSync(path.join(dir, "waste.geojson"));
  fs.rmSync(path.join(dir, "measurements.csv"));
  const { problems } = loadBundle(dir);
  assert.deepEqual(problems.map((p) => p.file).sort(), ["measurements.csv", "waste.geojson"]);
});

test("warns (does not fail) on manifest tiles and counts that differ", (t) => {
  const dir = copyBundle(t, { tiles: false });
  editJson(dir, "manifest.json", (m) => ({ ...m, tiles: 145, counts: { ...m.counts, canopies: 5 } }));
  const { warnings } = load(dir);
  assert.deepEqual(warnings.map((w) => w.message), ["tiles is 145, expected 311", "counts.canopies is 5 but the layer has 4 features"]);
});

test("tiles.geojson is optional: an older bundle without it (nor masks/) loads without warnings", (t) => {
  const { bundle, warnings } = load(copyBundle(t, { tiles: false }));
  assert.equal(bundle.tiles, null);
  assert.deepEqual(warnings, []);
  assert.equal(load(MINI_BUNDLE).bundle.tiles.features.length, 311);
});

test("optional manifest fields may be null (a bundle without geometries has no bbox, a manual AnnSet no model)", (t) => {
  const dir = copyBundle(t, { schema: "new" });
  editJson(dir, "manifest.json", (m) => ({ ...m, bbox_32635: null, model_version: null }));
  assert.deepEqual(load(dir).warnings, []);
  editJson(dir, "manifest.json", (m) => ({ ...m, bbox_32635: [1, 2], tiles_with_objects: -1 }));
  rejects(dir, "manifest.json", /tiles_with_objects must be a non-negative integer/);
  editJson(dir, "manifest.json", (m) => ({ ...m, tiles_with_objects: 3 }));
  assert.deepEqual(load(dir).warnings.map((w) => w.message), ["bbox_32635 is not [minx, miny, maxx, maxy]; ignored"]);
});

test("rejects a stage outside model|marcaj_corrected; warns when block lines do not add up to the survey line", (t) => {
  const dir = copyBundle(t);
  editText(dir, "measurements.csv", (s) => s.replace("block,V02,,,1,20.00,1.00,0.0001,174.00", "block,V02,,,1,20.00,1.00,0.0001,184.00"));
  assert.deepEqual(load(dir).warnings.map((w) => `${w.file}: ${w.message}`),
    ["measurements.csv: block interrow_area_m2 lines sum to 583.00, survey line says 573.00 (+1.75%)"]);
  editJson(dir, "manifest.json", (m) => ({ ...m, stage: "mock" }));
  rejects(dir, "manifest.json", /stage is "mock", expected model\|marcaj_corrected/);
});
