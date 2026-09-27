// Route through every farm (ADR-028): one closed tour from the start through every target of every farm and back.
// The walk uses the inter-row lanes of all farms and the roads around them (public ones included, to go from farm
// to farm), never across a row. The chain-compressed graph (compress.ts) keeps 1,200+ targets to a few seconds.
import type { Feature, LineString, MultiLineString, MultiPolygon, Polygon } from "geojson";
import { unprojectUtm, type LonLat } from "../utm.ts";
import { buildWalkGraph, SECOND_SIDE, type Terminal, type WalkLine } from "./graph.ts";
import type { XY } from "./geometry.ts";
import { interRowLanes, rowSegments } from "./lanes.ts";
import { sweepBaselineM } from "./baseline.ts";
import { groupsOf, solveOnGraph } from "./solve.ts";
import {
  bboxOf,
  minutesAt4Kmh,
  partsOf,
  ringsOf,
  rowAxesOf,
  START_ID,
  toXY,
  type FarmRouteResult,
  type RowFeature,
  type TargetFeature,
} from "./plan.ts";

/** roads farther than this from every farm and from the start are left out (m) */
export const BETWEEN_FARMS_MARGIN_M = 300;
/** local search budget for the big tour (ms) */
const ALL_FARMS_BUDGET_MS = 4000;

export interface FarmInput {
  farmId: string;
  geometry: Polygon | MultiPolygon;
  rows: RowFeature[];
  targets: TargetFeature[];
}

export interface AllFarmsInput {
  start: LonLat;
  roads: Feature<LineString | MultiLineString>[];
  farms: FarmInput[];
}

export function planAllFarms(input: AllFarmsInput): FarmRouteResult {
  const farms = input.farms.filter((f) => f.targets.length > 0);
  if (farms.length === 0) throw new Error("no targets");
  const start = toXY(input.start);
  const rowAxes = rowAxesOf(farms.flatMap((f) => f.rows));
  const lanes = interRowLanes(rowAxes);
  const box = bboxOf([start, ...farms.flatMap((f) => ringsOf(f.geometry).flat())], BETWEEN_FARMS_MARGIN_M);
  const inBox = (p: XY) => p[0] >= box[0] && p[0] <= box[2] && p[1] >= box[1] && p[1] <= box[3];
  const lines: WalkLine[] = [
    ...input.roads
      .flatMap((f) => partsOf(f.geometry).map((part) => part.map(toXY)))
      .filter((pts) => pts.some(inBox))
      .map((points) => ({ kind: "road" as const, rowIds: [], points })),
    ...lanes.map((l) => ({ kind: "lane" as const, rowIds: l.rowIds, points: l.points })),
  ];
  const terminals: Terminal[] = [
    { id: START_ID, point: start },
    ...farms.flatMap((f) =>
      f.targets.map((t) => ({ id: t.properties.target_id, point: toXY(t.geometry.coordinates), rowId: t.properties.row_id ?? null, seenFromLine: true, bothSides: true })),
    ),
  ];
  const graph = buildWalkGraph(lines, terminals, rowSegments(rowAxes));
  const tour = solveOnGraph(graph, groupsOf(graph, terminals.map((t) => t.id), SECOND_SIDE), ALL_FARMS_BUDGET_MS);

  // farms in the order the tour first reaches them
  const farmOf = new Map(farms.flatMap((f) => f.targets.map((t) => [t.properties.target_id, f.farmId] as const)));
  const farmOrder = [...new Set(tour.order.map((id) => farmOf.get(id)!))];
  return {
    start: input.start,
    startMovedM: 0,
    line: tour.points.map(unprojectUtm),
    lengthM: tour.lengthM,
    durationMin: minutesAt4Kmh(tour.lengthM),
    order: tour.order,
    offNetworkM: tour.offNetworkM,
    baselineM: sweepBaselineM(lanes, start),
    farmOrder,
  };
}
