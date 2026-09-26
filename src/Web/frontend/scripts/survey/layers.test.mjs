import assert from "node:assert/strict";
import test from "node:test";
import { copyBundle, load, MINI_BUNDLE, NEW } from "./fixtures/kit.mjs";
import { ll, polygonArea, positions, r4 } from "./geo.mjs";
import { buildLayers, interrowsLayer, routeOrder, targetsLayer } from "./layers.mjs";

const OPTS = { interrowTol: 0.0125, targets: "all" };
const build = (dir, opts = OPTS) => {
  const { bundle } = load(dir);
  return { bundle, ...buildLayers(bundle, opts) };
};
const ids = (fc, key) => fc.features.map((f) => f.properties[key]);
const isRounded7 = (v) => Math.abs(v * 1e7 - Math.round(v * 1e7)) < 1e-6;

test("every layer is a 4326 FeatureCollection with numeric ids 1..n, 7-decimal coordinates and no crs member", () => {
  const { files } = build(MINI_BUNDLE);
  assert.deepEqual(Object.keys(files).sort(),
    ["blocks.geojson", "canopies.geojson", "interrows.geojson", "route.geojson", "rows.geojson", "targets.geojson", "waste.geojson"]);
  for (const [name, fc] of Object.entries(files)) {
    assert.equal(fc.type, "FeatureCollection", name);
    assert.equal(fc.crs, undefined, name);
    assert.deepEqual(fc.features.map((f) => f.id), fc.features.map((_, i) => i + 1), name);
    for (const [lon, lat] of fc.features.flatMap((f) => positions(f.geometry))) {
      assert.ok(lon > 28.70 && lon < 28.72 && lat > 47.12 && lat < 47.13, `${name}: ${lon},${lat}`);
      assert.ok(isRounded7(lon) && isRounded7(lat), `${name}: ${lon},${lat}`);
    }
  }
  assert.deepEqual(files["route.geojson"].features[0].geometry.coordinates[0], ll([629504.7, 5220250.75]));
});

test("rows keep their bundle properties; length_m is the CSV row figure", () => {
  const { bundle, files } = build(MINI_BUNDLE);
  const rows = files["rows.geojson"];
  assert.deepEqual(ids(rows, "row_id"), ids(bundle.rows, "row_id"), "bundle order kept");
  const r010 = rows.features.find((f) => f.properties.row_id === "V01-R010").properties;
  assert.equal(r010.length_m, 16.44, "CSV 16.44, not r2(16.445) = 16.45");
  assert.deepEqual(r010.tile_structures, { siret3_r018_c010: "regular" });
  assert.equal(rows.features.find((f) => f.properties.row_id === "V01-R002").geometry.type, "MultiLineString");
});

test("old schema: canopy area_m2 is the UTM shoelace area; blocks, waste and route keep their properties", () => {
  const { bundle, files } = build(MINI_BUNDLE);
  assert.deepEqual(ids(files["canopies.geojson"], "area_m2"), [0.25, 0.36, 0.49, 1]);
  assert.deepEqual(files["canopies.geojson"].features[0].properties,
    { ...bundle.canopies.features[0].properties, area_m2: 0.25 });
  assert.deepEqual(files["blocks.geojson"].features.map((f) => f.properties), bundle.blocks.features.map((f) => f.properties));
  assert.equal(files["blocks.geojson"].features[1].geometry.type, "MultiPolygon");
  assert.deepEqual(files["waste.geojson"].features[0].properties, { waste_id: "W1", vineyard_id: null, tile: "siret3_r018_c011", confidence: 0.87 });
  assert.deepEqual(files["route.geojson"].features[0].properties, { ...bundle.route.features[0].properties, mock: false });
});

test("new schema: canopy area_m2 and interrow_total_m2 come from the bundle (same-tile pieces too)", (t) => {
  const { files } = build(copyBundle(t, { schema: "new" }));
  assert.deepEqual(ids(files["canopies.geojson"], "area_m2"), NEW.canopyAreas);
  assert.deepEqual(ids(files["interrows.geojson"], "interrow_total_m2"), [398.5, 398.5, 173.5, 173.5]);
});

test("old schema: interrow_total_m2 is the sum of the pieces only when they are in different tiles; piece area_m2 kept", () => {
  const { bundle, files } = build(MINI_BUNDLE);
  const pieces = bundle.interrows.features.map((f) => f.properties.area_m2);
  const out = files["interrows.geojson"].features.map((f) => f.properties);
  const csvInterrow = (vid) => bundle.csv.lines.find((l) => l.level === "block" && l.vineyard_id === vid).interrow_area_m2;
  assert.deepEqual(out.map((p) => p.area_m2), pieces);
  assert.deepEqual(out[0].row_ids, ["V01-R002", "V01-R010"]);
  // V01-I001: one piece in each tile, so the sum is the union (the block's only inter-row: its CSV area)
  assert.deepEqual(out.slice(0, 2).map((p) => p.interrow_total_m2), [csvInterrow("V01"), csvInterrow("V01")]);
  // V02-I001: two overlapping pieces in one tile ("#2"), whose sum would overstate the union: no total
  assert.deepEqual(out.slice(2).map((p) => p.piece_id), ["V02-I001@siret3_r018_c011", "V02-I001@siret3_r018_c011#2"]);
  assert.ok(pieces[2] + pieces[3] > 1.5 * csvInterrow("V02"), `${pieces[2]} + ${pieces[3]} vs union ${csvInterrow("V02")}`);
  assert.ok(out.slice(2).every((p) => !("interrow_total_m2" in p)));
});

test("interrowsLayer: a summed total only for pieces in distinct tiles; a bundle total is always kept; input untouched", () => {
  const square = (x, y) => ({ type: "Polygon", coordinates: [[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10], [x, y]]] });
  const piece = (id, tile, [x, y], extra = {}) =>
    ({ type: "Feature", properties: { interrow_id: id, tile, interrow_cover: "bare_soil", ...extra }, geometry: square(x, y) });
  const input = { type: "FeatureCollection", features: [
    piece("A", "t1", [629600, 5220200]), piece("A", "t1", [629601, 5220200]), // 90 m² overlap: union 110, sum 200
    piece("B", "t1", [629600, 5220220]), piece("B", "t2", [629610, 5220220]),
    piece("C", "t1", [629600, 5220240], { interrow_total_m2: 110 }), piece("C", "t1", [629601, 5220240], { interrow_total_m2: 110 }),
  ] };
  const snapshot = structuredClone(input);
  const out = interrowsLayer(input, 0).layer.features.map((f) => f.properties);
  assert.deepEqual(out.map((p) => p.area_m2), [100, 100, 100, 100, 100, 100]);
  assert.deepEqual(out.map((p) => p.interrow_total_m2), [undefined, undefined, 200, 200, 110, 110]);
  assert.ok(!("interrow_total_m2" in out[0]) && !("interrow_total_m2" in out[1]));
  assert.deepEqual(input, snapshot);
});

test("inter-rows are simplified in UTM with the area kept within 0.1%; --interrow-tol 0 keeps every vertex", () => {
  const { stats } = build(MINI_BUNDLE);
  const s = stats.interrows;
  assert.ok(s.coordsAfter < s.coordsBefore / 10, `${s.coordsBefore} → ${s.coordsAfter}`);
  assert.ok(Math.abs(s.areaAfter - s.areaBefore) / s.areaBefore < 0.001);
  assert.ok(s.maxPieceChange < 0.001);
  const exact = build(MINI_BUNDLE, { ...OPTS, interrowTol: 0 }).stats.interrows;
  assert.equal(exact.coordsAfter, exact.coordsBefore);
  assert.equal(r4(exact.areaAfter), r4(exact.areaBefore));
});

test("targets: route_order int|null, sorted with nulls last (bundle order kept among them)", () => {
  const { files } = build(MINI_BUNDLE);
  const targets = files["targets.geojson"];
  assert.deepEqual(ids(targets, "target_id"), ["T001", "T002", "T003", "T004", "T005"]);
  assert.deepEqual(ids(targets, "route_order"), [1, 2, null, null, null]);
  assert.ok(targets.features.every((f) => f.properties.route_order === null || Number.isInteger(f.properties.route_order)));
  assert.equal(targets.features[3].properties.row_id, null);
  assert.deepEqual([targets.features[4].properties.type, targets.features[4].properties.vineyard_id], ["waste", null]);
  assert.deepEqual(targets.features[0].geometry.coordinates, ll([629560, 5220280]));
});

test("--targets route keeps only the routed targets", () => {
  const { files } = build(MINI_BUNDLE, { ...OPTS, targets: "route" });
  assert.deepEqual(ids(files["targets.geojson"], "target_id"), ["T001", "T002"]);
  assert.deepEqual(files["targets.geojson"].features.map((f) => f.id), [1, 2]);
});

test("new schema: priority / route_role / skip_reason pass through", (t) => {
  const { files } = build(copyBundle(t, { schema: "new" }));
  const t003 = files["targets.geojson"].features.find((f) => f.properties.target_id === "T003").properties;
  assert.deepEqual([t003.priority, t003.route_role, t003.skip_reason, t003.route_order], [2, "optional", "optional_detour", null]);
});

test("routeOrder normalises floats and missing values; targetsLayer does not mutate its input", () => {
  assert.deepEqual([routeOrder({ route_order: 3.0 }), routeOrder({ route_order: null }), routeOrder({})], [3, null, null]);
  const input = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { target_id: "B", route_order: null }, geometry: { type: "Point", coordinates: [629530, 5220290] } },
    { type: "Feature", properties: { target_id: "A", route_order: 1.0 }, geometry: { type: "Point", coordinates: [629560, 5220280] } },
  ] };
  const snapshot = structuredClone(input);
  assert.deepEqual(ids(targetsLayer(input), "target_id"), ["A", "B"]);
  assert.deepEqual(input, snapshot);
});

test("canopy area in UTM equals the fixture squares", () => {
  const { bundle } = load(MINI_BUNDLE);
  assert.deepEqual(bundle.canopies.features.map((f) => r4(polygonArea(f.geometry))), [0.25, 0.36, 0.49, 1]);
});
