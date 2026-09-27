// A closed tour from the start through every target and back, on the walking graph: distances on the
// chain-compressed graph (compress.ts), the tour over candidate groups (tour.ts: each target seen from one of its
// candidates, e.g. either lane beside its row), then each leg expanded back to the full line.
import { edgeKey, type WalkGraph } from "./graph.ts";
import { compress } from "./compress.ts";
import { edgesTo, shortestPaths } from "./dijkstra.ts";
import { solveGroupTour, TOUR_LIMITS } from "./tour.ts";
import { dist, type XY } from "./geometry.ts";

export interface SolvedTour {
  /** the walk start → … → start */
  points: XY[];
  lengthM: number;
  /** group ids (all but the start's) in visiting order */
  order: string[];
  /** metres on straight off-network edges (graph.offNetwork) */
  offNetworkM: number;
}

export interface TourGroup {
  /** the target id (or the start's) */
  id: string;
  /** terminal ids of its candidates (graph.terminalNode keys) */
  candidates: readonly string[];
}

/** @param groups the start first (one candidate), then one group per target */
export function solveOnGraph(graph: WalkGraph, groups: readonly TourGroup[], budgetMs: number = TOUR_LIMITS.budgetMs): SolvedTour {
  const candNodes = groups.flatMap((g) => g.candidates.map((id) => graph.terminalNode.get(id)!));
  const candGroups = groups.reduce<number[][]>((acc, g) => {
    const first = acc.reduce((n, x) => n + x.length, 0);
    return [...acc, g.candidates.map((_, k) => first + k)];
  }, []);
  const c = compress(graph, candNodes);
  const at = candNodes.map((v) => c.index[v]);
  // one Dijkstra per distinct node; float32 keeps the 2,400² matrix of the whole survey small
  const rowOf = new Map<number, Float32Array>();
  for (const s of new Set(at)) {
    const sp = shortestPaths(c, s);
    rowOf.set(s, Float32Array.from(at, (t) => sp.dist[t]));
  }
  const matrix = at.map((s) => rowOf.get(s)!);
  const chosen = solveGroupTour(matrix, candGroups, budgetMs);

  const cycle = [...chosen, chosen[0]];
  const path = cycle.slice(1).flatMap((b, i) => {
    const from = at[cycle[i]], to = at[b];
    if (from === to) return [];
    const sp = shortestPaths(c, from, to);
    return edgesTo(sp, to).flatMap((k) => c.adjChain[k].slice(1));
  });
  const full = [candNodes[0], ...path];
  const points = full.map((v) => graph.coords[v]);
  const lengthM = points.reduce((s, p, i) => (i === 0 ? 0 : s + dist(points[i - 1], p)), 0);
  const offNetworkM = full.reduce((s, v, i) => (i > 0 && graph.offNetwork.has(edgeKey(full[i - 1], v)) ? s + dist(points[i - 1], points[i]) : s), 0);

  const groupOfCand = candGroups.flatMap((g, gi) => g.map(() => gi));
  return { points, lengthM, order: chosen.slice(1).map((k) => groups[groupOfCand[k]].id), offNetworkM };
}

/** Tour groups of the terminals: the start alone, then each target with its candidates (both sides when found). */
export const groupsOf = (graph: WalkGraph, terminalIds: readonly string[], secondSide: string): TourGroup[] =>
  terminalIds.map((id, i) => ({
    id,
    candidates: i > 0 && graph.terminalNode.has(id + secondSide) ? [id, id + secondSide] : [id],
  }));
