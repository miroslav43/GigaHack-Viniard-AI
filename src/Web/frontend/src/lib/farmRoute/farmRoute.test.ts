// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWalkGraph, type WalkLine } from "./graph.ts";
import { pathTo, shortestPaths } from "./dijkstra.ts";
import { solveTour, tourLength, TOUR_LIMITS } from "./tour.ts";
import { planFarmRoute, START_ID } from "./plan.ts";
import { unprojectUtm } from "../utm.ts";

// two parallel 100 m rows 3 m apart (headland at both ends) and a road 10 m below their west ends
const rows: WalkLine[] = [
  { kind: "row", rowId: "R1", points: [[0, 0], [100, 0]] },
  { kind: "row", rowId: "R2", points: [[0, 3], [100, 3]] },
];
const road: WalkLine = { kind: "road", rowId: null, points: [[-10, -50], [-10, 50]] };

test("targets on the same row are chained along it, not via the row ends", () => {
  const g = buildWalkGraph([...rows, road], [
    { id: "a", point: [30, 0], rowId: "R1" },
    { id: "b", point: [60, 0], rowId: "R1" },
  ]);
  const sp = shortestPaths(g, g.terminalNode.get("a")!);
  assert.ok(Math.abs(sp.dist[g.terminalNode.get("b")!] - 30) < 1e-6);
});

test("the headland joins neighbouring rows and row ends reach the road", () => {
  const g = buildWalkGraph([...rows, road], [
    { id: "a", point: [90, 0], rowId: "R1" },
    { id: "b", point: [90, 3], rowId: "R2" },
    { id: "s", point: [-10, 0] },
  ]);
  const sp = shortestPaths(g, g.terminalNode.get("a")!);
  // 10 m to the east end, 3 m across, 10 m back
  assert.ok(Math.abs(sp.dist[g.terminalNode.get("b")!] - 23) < 1e-6);
  const toStart = sp.dist[g.terminalNode.get("s")!];
  assert.ok(Math.abs(toStart - 100) < 1e-6, `got ${toStart}`);
  assert.ok(pathTo(sp, g.terminalNode.get("s")!).length > 2);
});

test("a start far from every line is bridged in, and the bridge is marked off-network", () => {
  const g = buildWalkGraph([...rows], [
    { id: "s", point: [0, -400] },
    { id: "a", point: [50, 0], rowId: "R1" },
  ]);
  const sp = shortestPaths(g, g.terminalNode.get("s")!);
  assert.ok(Number.isFinite(sp.dist[g.terminalNode.get("a")!]));
  assert.ok(g.offNetwork.size >= 1);
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

test("planFarmRoute closes the loop at the start and orders every target", () => {
  const ll = (x: number, y: number) => unprojectUtm([629500 + x, 5220250 + y]);
  const line = (pts: [number, number][], row_id: string | null) => ({
    type: "Feature" as const,
    properties: { row_id },
    geometry: { type: "LineString" as const, coordinates: pts.map(([x, y]) => ll(x, y)) },
  });
  const r = planFarmRoute({
    start: ll(-10, 0),
    roads: [line([[-10, -50], [-10, 50]], null)],
    rows: [line([[0, 0], [100, 0]], "R1"), line([[0, 3], [100, 3]], "R2")],
    targets: [
      { type: "Feature", properties: { target_id: "T1", row_id: "R1" }, geometry: { type: "Point", coordinates: ll(80, 0) } },
      { type: "Feature", properties: { target_id: "T2", row_id: "R2" }, geometry: { type: "Point", coordinates: ll(80, 3) } },
    ],
  });
  assert.deepEqual([...r.order].sort(), ["T1", "T2"]);
  assert.ok(!r.order.includes(START_ID));
  assert.deepEqual(r.line[0].map((v) => v.toFixed(6)), r.line[r.line.length - 1].map((v) => v.toFixed(6)));
  // rows are crossed only at their ends: 10 to R1, 80 + 20 to its east end, 3 across, 20 + 80 back along R2,
  // 10 to the road and 3 down to the start = 226 (±0.5 m of projection error)
  assert.ok(Math.abs(r.lengthM - 226) < 0.5, `got ${r.lengthM}`);
});
