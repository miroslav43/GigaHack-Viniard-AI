import assert from "node:assert/strict";
import test from "node:test";
import proj4 from "proj4";
import { MINI_BUNDLE, readJson } from "./fixtures/kit.mjs";
import {
  bbox, ll, naturalCompare, pointInPolygonal, polygonArea, positions, ringArea, simplifyLine, simplifyPolygonal, simplifyRing, toWgs84, UTM,
} from "./geo.mjs";

test("START control point (src/Web/CLAUDE.md §6.6) reprojects to 28.7073776 E, 47.1230335 N at 7 decimals", () => {
  assert.deepEqual(ll([629504.7, 5220250.75]), [28.7073776, 47.1230335]);
  // unrounded proj4 value, to make sure the rounding is the only difference
  const [lon, lat] = proj4(UTM, "EPSG:4326").forward([629504.7, 5220250.75]);
  assert.ok(Math.abs(lon - 28.7073776) < 5e-8 && Math.abs(lat - 47.1230335) < 5e-8);
});

test("toWgs84 keeps the geometry type and nesting", () => {
  const multi = { type: "MultiPolygon", coordinates: [[[[629504, 5220250], [629505, 5220250], [629505, 5220251], [629504, 5220250]]]] };
  const out = toWgs84(multi);
  assert.equal(out.type, "MultiPolygon");
  assert.equal(out.coordinates[0][0].length, 4);
  assert.deepEqual(out.coordinates[0][0][0], ll([629504, 5220250]));
  assert.deepEqual(toWgs84({ type: "Point", coordinates: [629504.7, 5220250.75] }).coordinates, [28.7073776, 47.1230335]);
});

test("polygonArea subtracts holes and sums MultiPolygon parts", () => {
  const outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const hole = [[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]];
  assert.equal(ringArea(outer), 100);
  assert.equal(polygonArea({ type: "Polygon", coordinates: [outer, hole] }), 96);
  assert.equal(polygonArea({ type: "MultiPolygon", coordinates: [[outer], [hole]] }), 104);
  assert.equal(polygonArea({ type: "LineString", coordinates: outer }), 0);
});

test("Douglas-Peucker drops collinear and sub-tolerance vertices, keeps corners", () => {
  const line = [[0, 0], [1, 0.001], [2, 0], [3, 5], [4, 0]];
  assert.deepEqual(simplifyLine(line, 0.01), [[0, 0], [2, 0], [3, 5], [4, 0]]);
  assert.equal(simplifyLine(line, 0), line);
});

test("a simplified dense inter-row keeps its area within 0.1% and loses most vertices", () => {
  const piece = readJson(MINI_BUNDLE, "interrows.geojson").features[0].geometry; // 5 cm vertices, ±3 mm jitter, one hole
  const simplified = simplifyPolygonal(piece, 0.0125);
  const [a0, a1] = [polygonArea(piece), polygonArea(simplified)];
  assert.ok(Math.abs(a1 - a0) / a0 < 0.001, `area changed ${a0} → ${a1}`);
  assert.ok(positions(simplified).length < positions(piece).length / 10);
  assert.equal(simplified.coordinates.length, 2, "the hole is kept");
});

test("a ring that would collapse is kept as it is", () => {
  const sliver = [[0, 0], [5, 0.001], [10, 0], [5, -0.001], [0, 0]];
  assert.equal(simplifyRing(sliver, 0.01), sliver);
  const triangle = [[0, 0], [10, 0], [5, 5], [0, 0]];
  assert.deepEqual(simplifyRing(triangle, 0.01), triangle);
});

test("pointInPolygonal honours holes; bbox and natural sort", () => {
  const poly = { type: "Polygon", coordinates: [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]] };
  assert.equal(pointInPolygonal([5, 5], poly), true);
  assert.equal(pointInPolygonal([3, 3], poly), false);
  assert.equal(pointInPolygonal([11, 5], poly), false);
  assert.deepEqual(bbox([[1, 5], [3, 2], [2, 9]]), [1, 2, 3, 9]);
  assert.deepEqual(["V01-R10", "V01-R2", "V01-R1"].sort(naturalCompare), ["V01-R1", "V01-R2", "V01-R10"]);
});
