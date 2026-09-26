// Farm route: from a start point chosen on the map, through every target of one farm, back to the start (ADR-028).
// Input is the survey GeoJSON (EPSG:4326); the maths runs planar in UTM 35N like the measuring tool (ADR-010).
import type { Feature, LineString, MultiLineString, Point, Position } from "geojson";
import { projectUtm, unprojectUtm, type LonLat } from "../utm.ts";
import { buildWalkGraph, edgeKey, type Terminal, type WalkLine } from "./graph.ts";
import { pathTo, shortestPaths, type ShortestPaths } from "./dijkstra.ts";
import { solveTour } from "./tour.ts";
import { dist, type XY } from "./geometry.ts";

/** walking speed of the official route (route.geojson duration_min) */
export const WALK_KMH = 4;
/** roads farther than this from the farm and the start are left out of the graph (m) */
export const ROAD_MARGIN_M = 500;
export const START_ID = "__start__";

export interface FarmRouteInput {
  start: LonLat;
  roads: Feature<LineString | MultiLineString>[];
  /** the farm's rows (rows.geojson features with row_id) */
  rows: Feature<LineString | MultiLineString, { row_id?: string | null }>[];
  /** the farm's targets */
  targets: Feature<Point, { target_id: string; row_id?: string | null }>[];
}

export interface FarmRouteResult {
  /** closed line start → … → start (EPSG:4326) */
  line: LonLat[];
  lengthM: number;
  durationMin: number;
  /** target ids in visiting order */
  order: string[];
  /** metres walked straight across the terrain, off mapped roads and rows (to and from the start, between pieces) */
  offNetworkM: number;
}

const partsOf = (g: LineString | MultiLineString): Position[][] => (g.type === "LineString" ? [g.coordinates] : g.coordinates);
const toXY = (p: Position): XY => projectUtm([p[0], p[1]]);

function bboxAround(points: readonly XY[], margin: number) {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  return [Math.min(...xs) - margin, Math.min(...ys) - margin, Math.max(...xs) + margin, Math.max(...ys) + margin] as const;
}

export function planFarmRoute(input: FarmRouteInput): FarmRouteResult {
  if (input.targets.length === 0) throw new Error("no targets");
  const start = toXY(input.start);
  const rowLines: WalkLine[] = input.rows.flatMap((f) =>
    partsOf(f.geometry).map((part) => ({ kind: "row" as const, rowId: f.properties?.row_id ?? null, points: part.map(toXY) })),
  );
  const box = bboxAround([start, ...rowLines.flatMap((l) => l.points)], ROAD_MARGIN_M);
  const inBox = (p: XY) => p[0] >= box[0] && p[0] <= box[2] && p[1] >= box[1] && p[1] <= box[3];
  const roadLines: WalkLine[] = input.roads
    .flatMap((f) => partsOf(f.geometry).map((part) => part.map(toXY)))
    .filter((pts) => pts.some(inBox))
    .map((points) => ({ kind: "road" as const, rowId: null, points }));

  const terminals: Terminal[] = [
    { id: START_ID, point: start },
    ...input.targets.map((f) => ({ id: f.properties.target_id, point: toXY(f.geometry.coordinates), rowId: f.properties.row_id ?? null })),
  ];
  const graph = buildWalkGraph([...roadLines, ...rowLines], terminals);
  const nodes = terminals.map((t) => graph.terminalNode.get(t.id)!);
  const trees: ShortestPaths[] = nodes.map((n) => shortestPaths(graph, n));
  const matrix = trees.map((sp) => nodes.map((n) => sp.dist[n]));
  const order = solveTour(matrix);

  const cycle = [...order, order[0]];
  const pathNodes = cycle.slice(1).flatMap((b, i) => {
    const leg = pathTo(trees[cycle[i]], nodes[b]);
    return i === 0 ? leg : leg.slice(1);
  });
  const xy = pathNodes.map((n) => graph.coords[n]);
  const lengthM = xy.reduce((s, p, i) => (i === 0 ? 0 : s + dist(xy[i - 1], p)), 0);
  const offNetworkM = pathNodes.reduce(
    (s, n, i) => (i > 0 && graph.offNetwork.has(edgeKey(pathNodes[i - 1], n)) ? s + dist(xy[i - 1], xy[i]) : s),
    0,
  );
  return {
    line: xy.map(unprojectUtm),
    lengthM,
    durationMin: lengthM / ((WALK_KMH * 1000) / 60),
    order: order.slice(1).map((i) => terminals[i].id),
    offNetworkM,
  };
}
