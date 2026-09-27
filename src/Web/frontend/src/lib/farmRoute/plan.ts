// Farm route: from a start point chosen on the map, through every target of one farm, back to the start (ADR-028).
// Input is the survey GeoJSON (EPSG:4326); the maths runs planar in UTM 35N like the measuring tool (ADR-010).
import type { Feature, LineString, MultiLineString, MultiPolygon, Point, Polygon, Position } from "geojson";
import { projectUtm, unprojectUtm, type LonLat } from "../utm.ts";
import { buildWalkGraph, SECOND_SIDE, type Terminal, type WalkLine } from "./graph.ts";
import { groupsOf, solveOnGraph } from "./solve.ts";
import type { XY } from "./geometry.ts";
import { clipToFarm, nearestEdge, withinFarm, type FarmRings } from "./area.ts";
import { interRowLanes, rowSegments, type RowAxis } from "./lanes.ts";
import { sweepBaselineM } from "./baseline.ts";

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
  /** the farm's rows (rows.geojson): the lanes are built between them, the rows themselves are never walked */
  rows: RowFeature[];
  /** the farm's targets */
  targets: TargetFeature[];
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
  /** the normal walk it is compared with: every inter-row swept, from the same start (baseline.ts) */
  baselineM: number;
  /** farms in visiting order (the route through all farms) */
  farmOrder?: string[];
}

export type RowFeature = Feature<LineString | MultiLineString, { row_id?: string | null; vineyard_id?: string | null }>;
export type TargetFeature = Feature<Point, { target_id: string; row_id?: string | null }>;

export const partsOf = (g: LineString | MultiLineString): Position[][] => (g.type === "LineString" ? [g.coordinates] : g.coordinates);
export const toXY = (p: Position): XY => projectUtm([p[0], p[1]]);
export const minutesAt4Kmh = (m: number) => m / ((WALK_KMH * 1000) / 60);

export const rowAxesOf = (rows: readonly RowFeature[]): RowAxis[] =>
  rows.map((f, i) => ({
    rowId: f.properties?.row_id ?? `row-${i}`,
    blockId: f.properties?.vineyard_id ?? "",
    parts: partsOf(f.geometry).map((part) => part.map(toXY)),
  }));

export const ringsOf = (g: Polygon | MultiPolygon): FarmRings =>
  (g.type === "Polygon" ? [g.coordinates] : g.coordinates).flatMap((poly) => poly.map((ring) => ring.map(toXY)));

export function bboxOf(points: readonly XY[], margin: number) {
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
  const rowAxes = rowAxesOf(input.rows);
  const lanes = interRowLanes(rowAxes);
  const laneLines: WalkLine[] = lanes.map((l) => ({ kind: "lane" as const, rowIds: l.rowIds, points: l.points }));
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
    .map((points) => ({ kind: "road" as const, rowIds: [], points }));

  const terminals: Terminal[] = [
    { id: START_ID, point: start },
    ...input.targets.map((f) => ({
      id: f.properties.target_id,
      point: toXY(f.geometry.coordinates),
      rowId: f.properties.row_id ?? null,
      seenFromLine: true,
      bothSides: true,
    })),
  ];
  const graph = buildWalkGraph([...roadLines, ...laneLines], terminals, rowSegments(rowAxes));
  const tour = solveOnGraph(graph, groupsOf(graph, terminals.map((t) => t.id), SECOND_SIDE));
  return {
    start: unprojectUtm(start),
    startMovedM: edge?.d ?? 0,
    line: tour.points.map(unprojectUtm),
    lengthM: tour.lengthM,
    durationMin: minutesAt4Kmh(tour.lengthM),
    order: tour.order,
    offNetworkM: tour.offNetworkM,
    baselineM: sweepBaselineM(lanes, start),
  };
}
