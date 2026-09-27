// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWalkGraph, type WalkLine } from "./graph.ts";
import { interRowLanes, rowSegments, type RowAxis } from "./lanes.ts";
import { pathTo, shortestPaths } from "./dijkstra.ts";
import { solveTour, tourLength, TOUR_LIMITS } from "./tour.ts";
import { FARM_MARGIN_M, planFarmRoute, START_ID } from "./plan.ts";
import { withinFarm } from "./area.ts";
import { projectUtm, unprojectUtm } from "../utm.ts";

// two parallel 100 m rows 3 m apart; their lanes run at y = -1.5 (outer), 1.5 (between) and 4.5 (outer)
const rowAxes: RowAxis[] = [
  { rowId: "R1", blockId: "V1", parts: [[[0, 0], [100, 0]]] },
  { rowId: "R2", blockId: "V1", parts: [[[0, 3], [100, 3]]] },
];
const laneLinesOf = (axes: RowAxis[]): WalkLine[] => interRowLanes(axes).map((l) => ({ kind: "lane" as const, rowIds: l.rowIds, points: l.points }));
const obstacles = rowSegments(rowAxes);
const road: WalkLine = { kind: "road", rowIds: [], points: [[-10, -50], [-10, 50]] };

test("lanes run between the rows and outside the edge rows, past the row ends, never on a row", () => {
  const lanes = interRowLanes(rowAxes);
  const ys = lanes.map((l) => l.points[0][1]).sort((a, b) => a - b);
  assert.deepEqual(ys.map((y) => Math.round(y * 10) / 10), [-1.5, 1.5, 4.5]);
  for (const l of lanes) {
    const xs = l.points.map((p) => p[0]);
    assert.ok(Math.min(...xs) < 0 && Math.max(...xs) > 100, "lanes run past the row ends");
  }
  assert.deepEqual(lanes.find((l) => Math.abs(l.points[0][1] - 1.5) < 0.01)!.rowIds.sort(), ["R1", "R2"]);
});

test("rows one after another on the same line are never paired, and no join crosses a longer row", () => {
  // A (0..40) and B (50..100) on y = 0, C (0..100) on y = 3: A and B each pair with C, never with each other
  const axes: RowAxis[] = [
    { rowId: "A", blockId: "V1", parts: [[[0, 0], [40, 0]]] },
    { rowId: "B", blockId: "V1", parts: [[[50, 0], [100, 0]]] },
    { rowId: "C", blockId: "V1", parts: [[[0, 3], [100, 3]]] },
  ];
  for (const l of interRowLanes(axes)) assert.ok(!(l.rowIds.includes("A") && l.rowIds.includes("B")), `lane pairs A and B: ${l.rowIds}`);
  const g = buildWalkGraph(interRowLanes(axes).map((l) => ({ kind: "lane" as const, rowIds: l.rowIds, points: l.points })), [
    { id: "a", point: [20, 2.8], rowId: "C", seenFromLine: true }, // between A and C
    { id: "b", point: [20, 3.2], rowId: "C", seenFromLine: true }, // outside C
  ], rowSegments(axes));
  const sp = shortestPaths(g, g.terminalNode.get("a")!);
  for (const n of pathTo(sp, g.terminalNode.get("b")!)) {
    const [x, y] = g.coords[n];
    assert.ok(x < 0 || x > 100 || Math.abs(y - 3) >= 0.5, `on row C at ${x},${y}`);
  }
});

test("targets beside the same lane are chained along it; a target is seen from the lane, not stepped onto", () => {
  const g = buildWalkGraph(laneLinesOf(rowAxes), [
    { id: "a", point: [30, 0.2], rowId: "R1", seenFromLine: true },
    { id: "b", point: [60, 0.2], rowId: "R1", seenFromLine: true },
  ], obstacles);
  const sp = shortestPaths(g, g.terminalNode.get("a")!);
  // ± the 0.5 m node grid (a snap point merges into a lane vertex that close)
  assert.ok(Math.abs(sp.dist[g.terminalNode.get("b")!] - 30) < 0.5, `got ${sp.dist[g.terminalNode.get("b")!]}`);
  assert.ok(Math.abs(g.coords[g.terminalNode.get("a")!][1] - 1.5) < 1e-9); // on the lane between the rows
});

test("moving to the next lane goes round the row end over the headland", () => {
  const g = buildWalkGraph(laneLinesOf(rowAxes), [
    { id: "a", point: [90, 2.8], rowId: "R2", seenFromLine: true }, // lane 1.5
    { id: "b", point: [90, 3.2], rowId: "R2", seenFromLine: true }, // lane 4.5, across row R2
  ], obstacles);
  const sp = shortestPaths(g, g.terminalNode.get("a")!);
  // 11.5 m to the lane end past x = 100, 3 m across, 11.5 m back
  assert.ok(Math.abs(sp.dist[g.terminalNode.get("b")!] - 26) < 1e-6, `got ${sp.dist[g.terminalNode.get("b")!]}`);
  for (const n of pathTo(sp, g.terminalNode.get("b")!)) {
    const [x, y] = g.coords[n];
    assert.ok(x > 100 || Math.abs(y - 3) >= 1, `on row R2 at ${x},${y}`);
  }
});

test("a start far from every line is bridged in, and the bridge is marked off-network", () => {
  const g = buildWalkGraph(laneLinesOf(rowAxes), [
    { id: "s", point: [0, -400] },
    { id: "a", point: [50, 0.2], rowId: "R1", seenFromLine: true },
  ], obstacles);
  const sp = shortestPaths(g, g.terminalNode.get("s")!);
  assert.ok(Number.isFinite(sp.dist[g.terminalNode.get("a")!]));
  assert.ok(g.offNetwork.size >= 1);
  assert.ok(pathTo(sp, g.terminalNode.get("a")!).length > 2);
  void road;
});

const randomMatrix = (n: number, seed: number) => {
  let x = seed;
  const rnd = () => ((x = (x * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
  const pts = Array.from({ length: n }, () => [rnd() * 100, rnd() * 100]);
  return pts.map((p) => pts.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
};
const bruteForce = (d: number[][]) => {
  const perms = (xs: number[]): number[][] => (xs.length <= 1 ? [xs] : xs.flatMap((x, i) => perms([...xs.slice(0, i), ...xs.slice(i + 1)]).map((p) => [x, ...p])));
  return Math.min(...perms(d.map((_, i) => i).slice(1)).map((p) => tourLength(d, [0, ...p])));
};

test("small tours are optimal and start at the start", () => {
  for (const seed of [1, 2, 3, 4]) {
    const d = randomMatrix(7, seed);
    const order = solveTour(d);
    assert.equal(order[0], 0);
    assert.equal(new Set(order).size, 7);
    assert.ok(Math.abs(tourLength(d, order) - bruteForce(d)) < 1e-9);
  }
});

test("large tours visit every stop once and beat nearest neighbour order", () => {
  const n = TOUR_LIMITS.exactMaxStops + 40;
  const d = randomMatrix(n, 7);
  const order = solveTour(d);
  assert.equal(order[0], 0);
  assert.deepEqual([...order].sort((a, b) => a - b), d.map((_, i) => i));
  assert.ok(tourLength(d, order) < tourLength(d, d.map((_, i) => i)));
});

const ll = (x: number, y: number) => unprojectUtm([629500 + x, 5220250 + y]);
const local = (p: readonly [number, number]) => {
  const [x, y] = projectUtm([p[0], p[1]]);
  return [x - 629500, y - 5220250] as const;
};
const line = (pts: [number, number][], row_id: string | null) => ({
  type: "Feature" as const,
  properties: { row_id },
  geometry: { type: "LineString" as const, coordinates: pts.map(([x, y]) => ll(x, y)) },
});
const box = (x0: number, y0: number, x1: number, y1: number) => ({
  type: "Polygon" as const,
  coordinates: [[ll(x0, y0), ll(x1, y0), ll(x1, y1), ll(x0, y1), ll(x0, y0)]],
});

test("planFarmRoute closes the loop at the start, orders every target and never walks on a row", () => {
  const rowsGeo = [line([[0, 0], [100, 0]], "R1"), line([[0, 3], [100, 3]], "R2")].map((f) => ({ ...f, properties: { ...f.properties, vineyard_id: "V1" } }));
  const r = planFarmRoute({
    start: ll(-10, 0),
    farm: box(-12, -2, 102, 5),
    roads: [line([[-10, -50], [-10, 50]], null)],
    rows: rowsGeo,
    targets: [
      { type: "Feature", properties: { target_id: "T1", row_id: "R1" }, geometry: { type: "Point", coordinates: ll(80, 0.2) } },
      { type: "Feature", properties: { target_id: "T2", row_id: "R2" }, geometry: { type: "Point", coordinates: ll(80, 2.8) } },
    ],
  });
  assert.deepEqual([...r.order].sort(), ["T1", "T2"]);
  assert.ok(!r.order.includes(START_ID));
  assert.deepEqual(r.line[0].map((v) => v.toFixed(6)), r.line[r.line.length - 1].map((v) => v.toFixed(6)));
  // both targets are seen from the lane between the rows: 1.5 up the road, 8.5 to the lane, 81.5 in, and back = 183
  assert.ok(Math.abs(r.lengthM - 183) < 0.5, `got ${r.lengthM}`);
  for (const p of r.line) {
    const [x, y] = local(p);
    if (x > 0.5 && x < 99.5) assert.ok(Math.abs(y) >= 1 && Math.abs(y - 3) >= 1, `on a row at ${x.toFixed(1)},${y.toFixed(1)}`);
  }
});

test("the walk stays inside the farm and a start clicked outside moves onto its edge", () => {
  const rows = [line([[0, 0], [100, 0]], "R1"), line([[0, 3], [100, 3]], "R2")].map((f) => ({ ...f, properties: { ...f.properties, vineyard_id: "V1" } }));
  const r = planFarmRoute({
    start: ll(50, 60),
    farm: box(-2, -2, 102, 5),
    // a road 40 m north would be a shortcut between the row ends, but it lies outside the farm
    roads: [line([[-5, 5], [-5, 40], [105, 40], [105, 5]], null)],
    rows,
    targets: [
      { type: "Feature", properties: { target_id: "T1", row_id: "R1" }, geometry: { type: "Point", coordinates: ll(20, 0.2) } },
      { type: "Feature", properties: { target_id: "T2", row_id: "R2" }, geometry: { type: "Point", coordinates: ll(80, 2.8) } },
    ],
  });
  assert.ok(Math.abs(r.startMovedM - 55) < 0.5, `moved ${r.startMovedM}`);
  const [sx, sy] = local(r.start);
  assert.ok(Math.abs(sx - 50) < 0.5 && Math.abs(sy - 5) < 0.5, `start ${sx},${sy}`);
  const rings = [[[-2, -2], [102, -2], [102, 5], [-2, 5], [-2, -2]] as [number, number][]];
  for (const p of r.line) assert.ok(withinFarm(local(p), rings, FARM_MARGIN_M + 0.5), `off the farm: ${local(p)}`);
});
