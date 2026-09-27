// Farm route: from a start point chosen on the map, through every target of one farm, back to the start (ADR-028).
// Input is the survey GeoJSON (EPSG:4326); the maths runs planar in UTM 35N like the measuring tool (ADR-010).
import type { Feature, LineString, MultiLineString, MultiPolygon, Point, Polygon, Position } from "geojson";
import { projectUtm, unprojectUtm, type LonLat } from "../utm.ts";
import { buildWalkGraph, edgeKey, type Terminal, type WalkLine } from "./graph.ts";
import { pathTo, shortestPaths, type ShortestPaths } from "./dijkstra.ts";
import { solveTour } from "./tour.ts";
import { dist, type XY } from "./geometry.ts";
import { clipToFarm, nearestEdge, withinFarm, type FarmRings } from "./area.ts";

/** walking speed of the official route (route.geojson duration_min) */
export const WALK_KMH = 4;
/** the walk stays inside the farm: roads count only where they run inside it or this close to its outline (m) */
export const FARM_MARGIN_M = 10;
export const START_ID = "__start__";

export interface FarmRouteInput {
  /** the point clicked; moved onto the farm outline when it is outside the farm */
  start: LonLat;
  /** the farm outline (farms.geojson) */
  farm: Polygon | MultiPolygon;
  roads: Feature<LineString | MultiLineString>[];
  /** the farm's rows (rows.geojson features with row_id) */
  rows: Feature<LineString | MultiLineString, { row_id?: string | null }>[];
  /** the farm's targets */
  targets: Feature<Point, { target_id: string; row_id?: string | null }>[];
}

export interface FarmRouteResult {
  /** where the walk starts and ends: the clicked point, or the nearest point of the farm outline */
  start: LonLat;
  /** how far the clicked point was moved to reach the farm (0 when it was inside) */
  startMovedM: number;
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

const ringsOf = (g: Polygon | MultiPolygon): FarmRings =>
  (g.type === "Polygon" ? [g.coordinates] : g.coordinates).flatMap((poly) => poly.map((ring) => ring.map(toXY)));

function bboxOf(points: readonly XY[], margin: number) {
  const b = points.reduce(
    (acc, [x, y]) => [Math.min(acc[0], x), Math.min(acc[1], y), Math.max(acc[2], x), Math.max(acc[3], y)] as const,
    [Infinity, Infinity, -Infinity, -Infinity] as const,
  );
  return [b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin] as const;
}

export function planFarmRoute(input: FarmRouteInput): FarmRouteResult {
  if (input.targets.length === 0) throw new Error("no targets");
  const rings = ringsOf(input.farm);
  const clicked = toXY(input.start);
  const edge = withinFarm(clicked, rings, FARM_MARGIN_M) ? null : nearestEdge(clicked, rings);
  const start = edge?.point ?? clicked;
  const rowLines: WalkLine[] = input.rows.flatMap((f) =>
    partsOf(f.geometry).map((part) => ({ kind: "row" as const, rowId: f.properties?.row_id ?? null, points: part.map(toXY) })),
  );
  // roads only where they run through the farm or along its edge: the inspector walks inside the farm
  const box = bboxOf(rings.flat(), FARM_MARGIN_M);
  const touchesBox = (pts: readonly XY[]) => {
    const b = bboxOf(pts, 0);
    return b[0] <= box[2] && b[2] >= box[0] && b[1] <= box[3] && b[3] >= box[1];
  };
  const roadLines: WalkLine[] = input.roads
    .flatMap((f) => partsOf(f.geometry).map((part) => part.map(toXY)))
    .filter(touchesBox)
    .flatMap((pts) => clipToFarm(pts, rings, FARM_MARGIN_M))
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
    start: unprojectUtm(start),
    startMovedM: edge?.d ?? 0,
    line: xy.map(unprojectUtm),
    lengthM,
    durationMin: lengthM / ((WALK_KMH * 1000) / 60),
    order: order.slice(1).map((i) => terminals[i].id),
    offNetworkM,
  };
}
