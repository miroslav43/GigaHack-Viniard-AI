// The walking graph of the farm route: roads + the inter-row lanes of the farm (lanes.ts; never the row axes, which
// run through the canopies), noded where they touch, with short connectors from lane ends to nearby lane ends and to
// the road network, and every terminal (start, targets) spliced in.
// Everything is in UTM 35N metres. Build once per request; the result is never mutated afterwards.
import { closestOnSegment, dist, SegmentGrid, segmentIntersection, type XY } from "./geometry.ts";

/** road = roads.geojson clipped to the farm; lane = an inter-row lane (lanes.ts) */
export type LineKind = "road" | "lane";

export interface WalkLine {
  kind: LineKind;
  /** rows beside a lane (targets on them are inspected from it); empty for roads */
  rowIds: readonly string[];
  points: readonly XY[];
}

export interface Terminal {
  id: string;
  point: XY;
  /** preferred row: the terminal snaps to a lane beside it (a target's row_id) */
  rowId?: string | null;
  /** seen from the line (a target): its node is the snap point itself, the walk does not step off the lane */
  seenFromLine?: boolean;
  /** also a second candidate on the lane across its row (terminal id + SECOND_SIDE): a row gap can be inspected
   *  from either side, the tour picks one (the pipeline's candidate sets, solve_gtsp) */
  bothSides?: boolean;
}

export const SECOND_SIDE = "#2";

export const GRAPH_LIMITS = {
  /** coordinates closer than this share a node (m) */
  nodeGrid: 0.5,
  /** a lane end connects to other lanes within this distance: the headland turn round a row end, or another block (m) */
  headlandM: 8,
  /** a lane end connects to the nearest road within this distance (m) */
  laneToRoadM: 60,
  /** a dangling road end connects to another road within this distance: T-junctions not noded in OSM (m) */
  roadGapM: 5,
  /** a target snaps to a lane beside its own row when one is closer than this, otherwise to the nearest line (m) */
  ownLaneM: 4,
} as const;

export interface WalkGraph {
  coords: readonly XY[];
  /** CSR adjacency: neighbours of node i are adjTo[adjStart[i] .. adjStart[i+1]) */
  adjStart: Int32Array;
  adjTo: Int32Array;
  adjW: Float64Array;
  /** terminal id → node */
  terminalNode: ReadonlyMap<string, number>;
  /** straight walks off the mapped lines (terminal → its line, bridges between pieces), as edgeKey(a, b) */
  offNetwork: ReadonlySet<string>;
}

export const edgeKey = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);

interface Segment {
  a: XY;
  b: XY;
  line: number;
}
interface Split {
  t: number;
  point: XY;
  key: string;
}

const gridKey = (p: XY) => `${Math.round(p[0] / GRAPH_LIMITS.nodeGrid)}:${Math.round(p[1] / GRAPH_LIMITS.nodeGrid)}`;

function segmentsOf(lines: readonly WalkLine[]): Segment[] {
  return lines.flatMap((l, line) => l.points.slice(1).map((b, i) => ({ a: l.points[i], b, line })));
}

function nearestSegment(p: XY, segs: readonly Segment[], accept: (s: Segment) => boolean) {
  let best: { seg: number; hit: ReturnType<typeof closestOnSegment> } | null = null;
  segs.forEach((s, i) => {
    if (!accept(s)) return;
    const hit = closestOnSegment(p, s.a, s.b);
    if (!best || hit.d < best.hit.d) best = { seg: i, hit };
  });
  return best as { seg: number; hit: ReturnType<typeof closestOnSegment> } | null;
}

/** Collects nodes and undirected edges; freezes them into a CSR graph at the end. */
class EdgeList {
  readonly coords: XY[] = [];
  private readonly index = new Map<string, number>();
  readonly from: number[] = [];
  readonly to: number[] = [];
  readonly w: number[] = [];
  readonly offNetwork = new Set<string>();

  node(p: XY, key = gridKey(p)): number {
    const found = this.index.get(key);
    if (found !== undefined) return found;
    this.coords.push(p);
    this.index.set(key, this.coords.length - 1);
    return this.coords.length - 1;
  }

  edge(a: number, b: number, w = dist(this.coords[a], this.coords[b]), offNetwork = false) {
    if (a === b) return;
    if (offNetwork) this.offNetwork.add(edgeKey(a, b));
    this.from.push(a);
    this.to.push(b);
    this.w.push(w);
  }
}

function toCsr(n: number, e: EdgeList) {
  const deg = new Int32Array(n + 1);
  e.from.forEach((a, i) => {
    deg[a + 1]++;
    deg[e.to[i] + 1]++;
  });
  for (let i = 0; i < n; i++) deg[i + 1] += deg[i];
  const fill = deg.slice(0, n);
  const adjTo = new Int32Array(deg[n]);
  const adjW = new Float64Array(deg[n]);
  e.from.forEach((a, i) => {
    const b = e.to[i];
    adjTo[fill[a]] = b;
    adjW[fill[a]++] = e.w[i];
    adjTo[fill[b]] = a;
    adjW[fill[b]++] = e.w[i];
  });
  return { adjStart: deg, adjTo, adjW };
}

/** Connected component label of every node (iterative DFS). */
export function components(n: number, from: readonly number[], to: readonly number[]): Int32Array {
  const nb: number[][] = Array.from({ length: n }, () => []);
  from.forEach((a, i) => {
    nb[a].push(to[i]);
    nb[to[i]].push(a);
  });
  const comp = new Int32Array(n).fill(-1);
  let c = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] !== -1) continue;
    const stack = [s];
    comp[s] = c;
    while (stack.length) {
      const v = stack.pop()!;
      for (const u of nb[v])
        if (comp[u] === -1) {
          comp[u] = c;
          stack.push(u);
        }
    }
    c++;
  }
  return comp;
}

const BRIDGE_CELL_M = 25;

/** Nearest node among `nodes` to p, by expanding rings of a uniform grid. */
function nearestInGrid(coords: readonly XY[], grid: Map<string, number[]>, p: XY): { node: number; d: number } {
  const cx = Math.floor(p[0] / BRIDGE_CELL_M), cy = Math.floor(p[1] / BRIDGE_CELL_M);
  let best = { node: -1, d: Infinity };
  for (let r = 0; r < 4000; r++) {
    // every node in ring r is at least (r - 1) cells away: stop once that cannot beat the best
    if ((r - 1) * BRIDGE_CELL_M > best.d) break;
    for (let x = cx - r; x <= cx + r; x++)
      for (let y = cy - r; y <= cy + r; y++) {
        if (Math.max(Math.abs(x - cx), Math.abs(y - cy)) !== r) continue;
        for (const n of grid.get(`${x}:${y}`) ?? []) {
          const d = dist(p, coords[n]);
          if (d < best.d) best = { node: n, d };
        }
      }
  }
  return best;
}

/**
 * Joins every component holding a terminal to the one holding `anchor` (the start) with the shortest straight
 * connector, so each target stays reachable (e.g. a start clicked in a field).
 */
function bridgeComponents(e: EdgeList, anchor: number, terminalNodes: readonly number[]): void {
  for (let guard = 0; guard < terminalNodes.length + 1; guard++) {
    const comp = components(e.coords.length, e.from, e.to);
    const main = comp[anchor];
    const stray = terminalNodes.find((t) => comp[t] !== main);
    if (stray === undefined) return;
    const grid = new Map<string, number[]>();
    comp.forEach((c, i) => {
      if (c !== main) return;
      const k = `${Math.floor(e.coords[i][0] / BRIDGE_CELL_M)}:${Math.floor(e.coords[i][1] / BRIDGE_CELL_M)}`;
      grid.set(k, [...(grid.get(k) ?? []), i]);
    });
    // the closest pair between the stray terminal's component and the main one
    let best = { d: Infinity, a: -1, b: -1 };
    comp.forEach((c, i) => {
      if (c !== comp[stray]) return;
      const hit = nearestInGrid(e.coords, grid, e.coords[i]);
      if (hit.d < best.d) best = { d: hit.d, a: i, b: hit.node };
    });
    if (best.b < 0) return;
    e.edge(best.a, best.b, best.d, true);
  }
}

/** Hits on the segments near p within `radius`, nearest first (each segment once). */
function hitsNear(p: XY, grid: SegmentGrid<Segment>, radius: number, accept: (s: Segment) => boolean) {
  return grid
    .near(p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius)
    .filter((i) => accept(grid.items[i]))
    .map((seg) => ({ seg, hit: closestOnSegment(p, grid.items[seg].a, grid.items[seg].b) }))
    .filter((h) => h.hit.d <= radius)
    .sort((x, y) => x.hit.d - y.hit.d);
}

/**
 * @param obstacles the row axes: no connector may cross one (the canopies); the lanes themselves run between them
 */
export function buildWalkGraph(lines: readonly WalkLine[], terminals: readonly Terminal[], obstacles: readonly (readonly [XY, XY])[] = []): WalkGraph {
  const segs = segmentsOf(lines);
  const grid = new SegmentGrid(segs);
  const rows = new SegmentGrid(obstacles.map(([a, b]) => ({ a, b })));
  const clear = (p: XY, q: XY) => !rows.crossedBy(p, q);
  const splits: Split[][] = segs.map(() => []);
  const e = new EdgeList();
  const addSplit = (seg: number, t: number, point: XY) => {
    const key = gridKey(point);
    splits[seg].push({ t, point, key });
    return key;
  };
  const connectors: [XY, string][] = [];
  const kindOf = (s: Segment) => lines[s.line].kind;
  const endsOf = (l: WalkLine) => (l.points.length < 2 ? [] : [l.points[0], l.points[l.points.length - 1]]);

  // lane ends: the nearest point of each other lane close by (the headland turn, round the row end), and the
  // nearest road; never across a row
  lines.forEach((l, li) => {
    if (l.kind !== "lane") return;
    for (const end of endsOf(l)) {
      const perLane = new Map<number, { seg: number; hit: ReturnType<typeof closestOnSegment> }>();
      for (const h of hitsNear(end, grid, GRAPH_LIMITS.headlandM, (s) => s.line !== li && kindOf(s) === "lane"))
        if (!perLane.has(segs[h.seg].line) && clear(end, h.hit.point)) perLane.set(segs[h.seg].line, h);
      for (const h of perLane.values()) connectors.push([end, addSplit(h.seg, h.hit.t, h.hit.point)]);
      const road = hitsNear(end, grid, GRAPH_LIMITS.laneToRoadM, (s) => kindOf(s) === "road").find((h) => clear(end, h.hit.point));
      if (road) connectors.push([end, addSplit(road.seg, road.hit.t, road.hit.point)]);
    }
  });
  // dangling road ends onto a road passing close by
  lines.forEach((l, li) => {
    if (l.kind !== "road") return;
    for (const end of endsOf(l)) {
      const hit = hitsNear(end, grid, GRAPH_LIMITS.roadGapM, (s) => kindOf(s) === "road" && s.line !== li)[0];
      if (hit && hit.hit.d > 0) connectors.push([end, addSplit(hit.seg, hit.hit.t, hit.hit.point)]);
    }
  });
  // a road (a passage) crossing a lane: a junction, so the walk can switch lanes there
  segs.forEach((r, ri) => {
    if (kindOf(r) !== "road") return;
    for (const li of grid.near(Math.min(r.a[0], r.b[0]), Math.min(r.a[1], r.b[1]), Math.max(r.a[0], r.b[0]), Math.max(r.a[1], r.b[1]))) {
      if (kindOf(segs[li]) !== "lane") continue;
      const x = segmentIntersection(r.a, r.b, segs[li].a, segs[li].b);
      if (!x) continue;
      const point: XY = [r.a[0] + x.t * (r.b[0] - r.a[0]), r.a[1] + x.t * (r.b[1] - r.a[1])];
      addSplit(ri, x.t, point);
      addSplit(li, x.u, point);
    }
  });
  // terminals: a lane beside their own row first, then the nearest line (for the start: one it reaches without
  // crossing a row, when there is one)
  const terminalKey = new Map<string, { at: XY; key: string; seenFromLine: boolean }>();
  for (const term of terminals) {
    const rowId = term.rowId;
    if (term.bothSides && rowId) {
      // the nearest lane beside the row on each side: two hits whose directions from the target are opposite
      const hits = hitsNear(term.point, grid, GRAPH_LIMITS.ownLaneM, (s) => lines[s.line].rowIds.includes(rowId));
      const first = hits[0];
      if (first) {
        const v1: XY = [first.hit.point[0] - term.point[0], first.hit.point[1] - term.point[1]];
        const second = hits.find((h) => (h.hit.point[0] - term.point[0]) * v1[0] + (h.hit.point[1] - term.point[1]) * v1[1] < 0);
        terminalKey.set(term.id, { at: term.point, key: addSplit(first.seg, first.hit.t, first.hit.point), seenFromLine: true });
        if (second) terminalKey.set(term.id + SECOND_SIDE, { at: term.point, key: addSplit(second.seg, second.hit.t, second.hit.point), seenFromLine: true });
        continue;
      }
    }
    const own = rowId ? nearestSegment(term.point, segs, (s) => lines[s.line].rowIds.includes(rowId)) : null;
    const reachable = term.seenFromLine ? null : nearestSegment(term.point, segs, (s) => clear(term.point, closestOnSegment(term.point, s.a, s.b).point));
    const hit = own && own.hit.d <= GRAPH_LIMITS.ownLaneM ? own : (reachable ?? nearestSegment(term.point, segs, () => true));
    const key = hit ? addSplit(hit.seg, hit.hit.t, hit.hit.point) : gridKey(term.point);
    terminalKey.set(term.id, { at: term.point, key, seenFromLine: Boolean(term.seenFromLine && hit) });
  }

  // nodes + edges along every segment, split at the sorted cut points
  segs.forEach((s, i) => {
    const cuts = [...splits[i]].sort((x, y) => x.t - y.t);
    const chain = [e.node(s.a), ...cuts.map((c) => e.node(c.point, c.key)), e.node(s.b)];
    chain.slice(1).forEach((b, k) => e.edge(chain[k], b));
  });
  for (const [p, key] of connectors) {
    const a = e.node(p);
    const b = e.node(p, key); // key exists: it was created by a split or a line vertex
    e.edge(a, b);
  }
  // a target is inspected from its snap point on the lane; the start is a short straight walk away from its own
  const terminalNode = new Map<string, number>();
  for (const [id, { at, key, seenFromLine }] of terminalKey) {
    const onLine = e.node(at, key);
    if (seenFromLine) {
      terminalNode.set(id, onLine);
      continue;
    }
    const own = e.node(at, `t:${id}`);
    e.edge(own, onLine, undefined, true);
    terminalNode.set(id, own);
  }

  const anchor = terminals.length ? terminalNode.get(terminals[0].id)! : 0;
  if (e.coords.length) bridgeComponents(e, anchor, [...terminalNode.values()]);
  return { coords: e.coords, ...toCsr(e.coords.length, e), terminalNode, offNetwork: e.offNetwork };
}
