// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import type { LineString, Position } from "geojson";
import { cumulative, cutAt, routeSlicer, type RouteLines } from "./sliceLine.ts";

// at the equator a degree of longitude and of latitude have the same length, so the numbers stay simple
const L: Position[] = [
  [0, 0],
  [2, 0],
  [2, 2],
];
const fc = (...lines: Position[][]): RouteLines => ({
  type: "FeatureCollection",
  features: lines.map((coordinates, i) => ({ type: "Feature", properties: { i }, geometry: { type: "LineString", coordinates } })),
});

test("cumulative lengths along a line", () => {
  assert.deepEqual(cumulative(L), [0, 2, 4]);
});

test("cutAt interpolates inside the segment that holds the distance", () => {
  const cum = cumulative(L);
  assert.equal(cutAt(L, cum, 0), null);
  assert.deepEqual(cutAt(L, cum, 1), [
    [0, 0],
    [1, 0],
  ]);
  assert.deepEqual(cutAt(L, cum, 3), [
    [0, 0],
    [2, 0],
    [2, 1],
  ]);
  assert.equal(cutAt(L, cum, 9), L);
});

test("routeSlicer draws the features in order and returns the input at the end", () => {
  // a second line on the equator too, 4 units long (8 in all)
  const input = fc(L, [
    [10, 0],
    [14, 0],
  ]);
  const slice = routeSlicer(input);
  assert.equal(slice(1), input);
  assert.equal(slice(0).features.length, 0);
  const half = slice(0.5); // 4 of 8 units: the whole first line, nothing of the second
  assert.equal(half.features.length, 1);
  assert.deepEqual((half.features[0].geometry as LineString).coordinates, L);
  const most = slice(0.75); // 6 of 8: the first line and half of the second
  assert.equal(most.features.length, 2);
  assert.deepEqual((most.features[1].geometry as LineString).coordinates, [
    [10, 0],
    [12, 0],
  ]);
  assert.deepEqual(most.features[1].properties, { i: 1 });
});

test("a MultiLineString is cut part by part", () => {
  const input: RouteLines = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: [L, L] } }],
  };
  const g = routeSlicer(input)(0.75).features[0].geometry;
  assert.equal(g.type, "MultiLineString");
  assert.equal(g.coordinates.length, 2);
  assert.deepEqual(g.coordinates[1], [
    [0, 0],
    [2, 0],
  ]);
});
