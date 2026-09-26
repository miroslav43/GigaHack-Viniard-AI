import assert from "node:assert/strict";
import test from "node:test";
import { copyBundle, editJson, F01_OUTLINE, load, MINI_BUNDLE } from "./fixtures/kit.mjs";
import { farmsSummary, labelPoint, lineLength, roadsSummary } from "./farms.mjs";
import { ll } from "./geo.mjs";
import { buildLayers } from "./layers.mjs";
import { BundleError } from "./read-bundle.mjs";
import { buildSummary } from "./summary.mjs";

const build = (dir) => {
  const { bundle, warnings } = load(dir);
  const { files } = buildLayers(bundle, { interrowTol: 0.0125 });
  const targets = files["targets.geojson"];
  return { bundle, warnings, files, summary: buildSummary({ bundle, targetCount: targets.features.length, uatAreaHa: 1, targets }) };
};
const props = (fc) => fc.features.map((f) => f.properties);

/** Asserts that loading `dir` fails with a problem of `file` matching `pattern`. */
const rejects = (dir, file, pattern) =>
  assert.throws(() => load(dir), (err) => {
    assert.ok(err instanceof BundleError, err.message);
    assert.ok(err.problems.some((p) => p.file === file && pattern.test(p.message)), `no ${file} problem ${pattern} in:\n${err.message}`);
    return true;
  });

test("a bundle without farms / roads writes neither layer nor summary block, and blocks keep their properties", () => {
  const { files, summary, bundle } = build(MINI_BUNDLE);
  assert.equal(bundle.farms, null);
  assert.equal(bundle.roads, null);
  assert.equal(files["farms.geojson"], undefined);
  assert.equal(files["roads.geojson"], undefined);
  assert.equal(summary.farms, undefined);
  assert.equal(summary.roads, undefined);
  assert.equal("farm_count" in summary.totals, false);
  assert.deepEqual(props(files["blocks.geojson"]), [{ vineyard_id: "V01", area_m2: 1200 }, { vineyard_id: "V02", area_m2: 510 }]);
});

test("farms: 4326 layer with n_blocks, area (kept or measured) and an inside label point; blocks gain farm_id", (t) => {
  const { files, warnings } = build(copyBundle(t, { farms: true }));
  assert.deepEqual(warnings, []);
  const farms = files["farms.geojson"];
  assert.deepEqual(farms.features.map((f) => f.id), [1, 2]);
  assert.deepEqual(props(farms), [
    { farm_id: "F01", vineyard_ids: ["V01"], n_blocks: 1, area_m2: 1100, label_point: ll([629515, 5220285]) },
    { farm_id: "F02", vineyard_ids: ["V02"], n_blocks: 1, area_m2: 510, label_point: ll([629590, 5220263.5]) },
  ]);
  assert.deepEqual(farms.features[0].geometry.coordinates[0][0], ll([629510, 5220270]), "EPSG:4326, 7 decimals");
  assert.deepEqual(props(files["blocks.geojson"]).map((p) => p.farm_id), ["F01", "F02"]);
});

test("roads: 4326 layer with every optional key, lengths rounded or measured", (t) => {
  const roads = props(build(copyBundle(t, { farms: true })).files["roads.geojson"]);
  assert.deepEqual(roads.map((p) => [p.road_id, p.road_class, p.length_m, p.farm_id]), [
    ["D0001", "public", 100, null], ["D0002", "field", 30, null], ["D0003", "internal", 20, "F01"],
  ]);
  assert.deepEqual(roads[1], { road_id: "D0002", road_class: "field", highway: "track", source: "osm", name: null, surface: null, farm_id: null, length_m: 30 });
});

test("summary: farms summed from the block lines add up to the survey; road lengths per class; farm_count", (t) => {
  const { summary } = build(copyBundle(t, { farms: true }));
  assert.deepEqual(summary.farms, [
    { farm_id: "F01", vineyard_ids: ["V01"], n_blocks: 1, area_m2: 1100, row_count: 2, row_length_m: 46.44, canopy_area_m2: 1.1,
      interrow_area_m2: 399, plant_count: 3, target_count: 3 },
    { farm_id: "F02", vineyard_ids: ["V02"], n_blocks: 1, area_m2: 510, row_count: 1, row_length_m: 20, canopy_area_m2: 1,
      interrow_area_m2: 174, plant_count: 1, target_count: 1 },
  ]);
  for (const k of ["row_count", "row_length_m", "canopy_area_m2", "interrow_area_m2", "plant_count"])
    assert.equal(Math.round(summary.farms.reduce((s, f) => s + f[k] * 100, 0)) / 100, summary.totals[k], k);
  assert.equal(summary.totals.farm_count, 2);
  assert.deepEqual(summary.roads, { public_m: 100, field_m: 30, internal_m: 20 });
  assert.deepEqual(Object.keys(summary).slice(-3), ["tiles", "farms", "roads"]);
});

test("farmsSummary hands the rounding residual of the block lines to the largest farms, only when every block is in a farm", () => {
  const blocks = [
    { vineyard_id: "V01", row_count: 1, row_length_m: 10.01, canopy_area_m2: 1, interrow_area_m2: 2, plant_count: 1 },
    { vineyard_id: "V02", row_count: 1, row_length_m: 20.01, canopy_area_m2: 1, interrow_area_m2: 2, plant_count: 1 },
    { vineyard_id: "V03", row_count: 1, row_length_m: 5.01, canopy_area_m2: 1, interrow_area_m2: 2, plant_count: 1 },
  ];
  const farm = (farm_id, vineyard_ids) => ({ type: "Feature", properties: { farm_id, vineyard_ids, area_m2: 1 }, geometry: null });
  const farms = { features: [farm("F01", ["V01", "V03"]), farm("F02", ["V02"])] };
  const survey = { row_length_m: 35.02, canopy_area_m2: 3, interrow_area_m2: 6 }; // Σ blocks = 35.03: one cent too many
  const noTargets = { features: [] };
  const out = farmsSummary({ farms, blocks, survey, targets: noTargets });
  assert.deepEqual(out.map((f) => [f.farm_id, f.row_length_m]), [["F01", 15.02], ["F02", 20.0]]);
  const partial = farmsSummary({ farms: { features: [farms.features[1]] }, blocks, survey, targets: noTargets });
  assert.deepEqual(partial.map((f) => f.row_length_m), [20.01], "a block outside every farm: no balancing");
});

test("roadsSummary sums the rounded road lengths per class; lineLength is planar", () => {
  const road = (road_class, length_m) => ({ properties: { road_class, length_m }, geometry: { type: "LineString", coordinates: [[0, 0], [3, 4]] } });
  assert.deepEqual(roadsSummary({ features: [road("public", 0.104), road("public", 0.104), road("internal", undefined)] }),
    { public_m: 0.2, field_m: 0, internal_m: 5 });
  assert.equal(lineLength({ type: "MultiLineString", coordinates: [[[0, 0], [3, 4]], [[0, 0], [0, 2], [2, 2]]] }), 9);
});

test("labelPoint: the centroid when inside, else the middle of the widest scanline stretch of the largest polygon", () => {
  assert.deepEqual(labelPoint({ type: "Polygon", coordinates: [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]] }), [2, 1]);
  assert.deepEqual(labelPoint({ type: "Polygon", coordinates: F01_OUTLINE }), [629515, 5220285]);
  const small = [[[100, 100], [101, 100], [101, 101], [100, 101], [100, 100]]];
  assert.deepEqual(labelPoint({ type: "MultiPolygon", coordinates: [small, [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]]] }), [2, 1]);
});

test("rejects unknown and doubled blocks, bad vineyard_ids, bad road classes and lengths; warns on the rest", (t) => {
  const dir = copyBundle(t, { farms: true });
  const editFarm = (id, fn) => editJson(dir, "farms.geojson", (fc) => ({
    ...fc, features: fc.features.map((f) => (f.properties.farm_id === id ? { ...f, properties: fn(f.properties) } : f)),
  }));
  editFarm("F02", (p) => ({ ...p, vineyard_ids: ["V02", "V09"] }));
  rejects(dir, "farms.geojson", /vineyard_ids name blocks missing from blocks\.geojson: V09/);
  editFarm("F02", (p) => ({ ...p, vineyard_ids: ["V02", "V01"] }));
  rejects(dir, "farms.geojson", /blocks listed in more than one farm: V01/);
  editFarm("F02", (p) => ({ ...p, vineyard_ids: "V02" }));
  rejects(dir, "farms.geojson", /"vineyard_ids" not a non-empty list of block ids: 1 feature \(F02\)/);
  editFarm("F02", (p) => ({ ...p, vineyard_ids: ["V02"] }));
  editJson(dir, "roads.geojson", (fc) => ({ ...fc, features: fc.features.map((f, i) => (i === 0 ? { ...f, properties: { ...f.properties, road_class: "Public", length_m: -1 } } : f)) }));
  rejects(dir, "roads.geojson", /"road_class" outside public\|field\|internal \(lowercase\): 1 feature \(D0001\)/);
  editJson(dir, "roads.geojson", (fc) => ({ ...fc, features: fc.features.map((f, i) => (i === 0 ? { ...f, properties: { ...f.properties, road_class: "public" } } : f)) }));
  rejects(dir, "roads.geojson", /"length_m" not a non-negative number: 1 feature \(D0001\)/);

  const warn = copyBundle(t, { farms: true });
  editJson(warn, "farms.geojson", (fc) => ({ ...fc, features: [{ ...fc.features[0], properties: { ...fc.features[0].properties, n_blocks: 2 } }] }));
  editJson(warn, "blocks.geojson", (fc) => ({ ...fc, features: fc.features.map((f) => ({ ...f, properties: { ...f.properties, farm_id: "F09" } })) }));
  editJson(warn, "manifest.json", (m) => ({ ...m, counts: { ...m.counts, farms: 1 } }));
  const { warnings, files } = build(warn);
  assert.deepEqual(warnings.map((w) => `${w.file}: ${w.message}`), [
    "farms.geojson: 1 block (V02) in no farm: the farms do not add up to the survey",
    "farms.geojson: n_blocks differs from the vineyard_ids count (the list wins): 1 feature (F01)",
    "blocks.geojson: farm_id disagrees with farms.geojson vineyard_ids (farms win): 1 block (V01)",
  ]);
  assert.deepEqual(props(files["blocks.geojson"]).map((p) => p.farm_id), ["F01", "F09"], "farms win; a block outside every farm keeps its own");
});

test("cadastre snapshot properties pass through to the layers; the farm summary gains n_parcels; bad types are rejected", (t) => {
  const dir = copyBundle(t, { farms: true });
  const snapshot = { n_parcels: 3, cadastral_codes: ["5537213.001", "5537213.002", "5537213.003"], landuse_counts: { "vii": 2, "arabil": 1 } };
  editJson(dir, "farms.geojson", (fc) => ({ ...fc, features: fc.features.map((f, i) => ({ ...f, properties: { ...f.properties, ...(i === 0 ? snapshot : { n_parcels: null }) } })) }));
  editJson(dir, "blocks.geojson", (fc) => ({ ...fc, features: fc.features.map((f) => ({ ...f, properties: { ...f.properties, ...snapshot } })) }));
  editJson(dir, "roads.geojson", (fc) => ({ ...fc, features: fc.features.map((f, i) => ({ ...f, properties: { ...f.properties, cadastral: i === 0, ...(i === 1 ? { source: "cadastre" } : {}) } })) }));
  const { files, summary } = build(dir);
  assert.deepEqual(props(files["farms.geojson"])[0].landuse_counts, snapshot.landuse_counts);
  assert.deepEqual(props(files["blocks.geojson"])[0].cadastral_codes, snapshot.cadastral_codes);
  assert.deepEqual(props(files["roads.geojson"]).map((p) => [p.cadastral, p.source]), [[true, "osm"], [false, "cadastre"], [false, "detected"]]);
  assert.deepEqual(summary.farms.map((f) => f.n_parcels), [3, null]);
  assert.equal("n_parcels" in build(copyBundle(t, { farms: true })).summary.farms[0], false, "absent without a snapshot");
  editJson(dir, "blocks.geojson", (fc) => ({ ...fc, features: fc.features.map((f) => ({ ...f, properties: { ...f.properties, n_parcels: -1 } })) }));
  rejects(dir, "blocks.geojson", /"n_parcels" not a non-negative integer or null: 2 features \(V01, V02\)/);
});

test("a road naming an unknown farm and a crs other than EPSG:32635", (t) => {
  const dir = copyBundle(t, { farms: true });
  editJson(dir, "roads.geojson", (fc) => ({ ...fc, features: fc.features.map((f) => ({ ...f, properties: { ...f.properties, farm_id: f.properties.farm_id && "F77" } })) }));
  assert.deepEqual(load(dir).warnings.map((w) => w.message), ["farm_id missing from farms.geojson: 1 road (D0003)"]);
  editJson(dir, "farms.geojson", (fc) => ({ ...fc, crs: { type: "name", properties: { name: "urn:ogc:def:crs:EPSG::4326" } } }));
  rejects(dir, "farms.geojson", /crs is "urn:ogc:def:crs:EPSG::4326"/);
});
