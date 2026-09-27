// Walkable lanes of a vineyard block: the inter-row axes between neighbouring vine rows, plus lanes outside the edge
// rows. The challenge asks to walk "passable inter-row areas and authorised passages, not paths through canopies":
// a row axis runs through the canopies, so it is never walked. Same idea as the pipeline's interrow centerlines
// (src/AI/vineyard/route/centerlines.py: the mean of two neighbouring row axes), without its row_index: the neighbour
// is found point by point, among the rows that run alongside at that position (rows one after another on the same
// line, or of another sub-block, are never paired). Lane ends are joined later by the graph, only where the join
// crosses no row (graph.ts).
import type { XY } from "./geometry.ts";

export const LANE_LIMITS = {
  /** rows closer than this laterally are one line (duplicate pieces), never neighbours (m) */
  minSpacingM: 0.8,
  /** a row farther than this laterally is not a neighbour: past it the lane runs at half the spacing (m) */
  maxSpacingM: 4.5,
  /** row spacing of a block with a single row or no measurable spacing (m) */
  defaultSpacingM: 2.5,
  /** lanes run this far past the row ends, so the turn into the next lane goes round the row end (m) */
  headlandClearanceM: 1.5,
  /** lane vertex spacing (m) */
  stepM: 2,
} as const;

export interface RowAxis {
  rowId: string;
  blockId: string;
  /** the row's pieces (a MultiLineString keeps its gaps) */
  parts: readonly (readonly XY[])[];
}

export interface Lane {
  points: XY[];
  /** the rows beside the lane: targets on them are inspected from it */
  rowIds: string[];
  blockId: string;
  /** lateral position across the block's rows (m, in the block's frame): orders the lanes of a sweep */
  offset: number;
}

interface Profile {
  rowId: string;
  /** position along the rows (sorted) and lateral offset of the row's points */
  t: number[];
  s: number[];
  t0: number;
  t1: number;
}

const median = (xs: readonly number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

/** Mean axial direction of the block's segments (angles doubled, so opposite digitising orders agree). */
function blockDirection(rows: readonly RowAxis[]): XY {
  let cx = 0, cy = 0;
  for (const r of rows)
    for (const part of r.parts)
      for (let i = 1; i < part.length; i++) {
        const dx = part[i][0] - part[i - 1][0], dy = part[i][1] - part[i - 1][1];
        const a = Math.atan2(dy, dx), len = Math.hypot(dx, dy);
        cx += len * Math.cos(2 * a);
        cy += len * Math.sin(2 * a);
      }
  const a = Math.atan2(cy, cx) / 2;
  return [Math.cos(a), Math.sin(a)];
}

function profileOf(r: RowAxis, d: XY, n: XY): Profile {
  const pts = r.parts
    .flat()
    .map((p) => ({ t: p[0] * d[0] + p[1] * d[1], s: p[0] * n[0] + p[1] * n[1] }))
    .sort((a, b) => a.t - b.t);
  const t = pts.map((p) => p.t);
  return { rowId: r.rowId, t, s: pts.map((p) => p.s), t0: t[0], t1: t[t.length - 1] };
}

/** The row's lateral offset at position t (linear between its points, held constant past its ends). */
function offsetAt(p: Profile, t: number): number {
  if (t <= p.t0) return p.s[0];
  const last = p.t.length - 1;
  if (t >= p.t1) return p.s[last];
  let i = 1;
  while (p.t[i] < t) i++;
  const k = p.t[i] === p.t[i - 1] ? 0 : (t - p.t[i - 1]) / (p.t[i] - p.t[i - 1]);
  return p.s[i - 1] + k * (p.s[i] - p.s[i - 1]);
}

/** The nearest row on `side` (+1 / -1) of row a at position t, among rows that run alongside there. */
function neighbourAt(a: Profile, t: number, side: number, rows: readonly Profile[]): { row: Profile; gap: number } | null {
  const sa = offsetAt(a, t);
  let best: { row: Profile; gap: number } | null = null;
  for (const b of rows) {
    if (b === a || t < b.t0 || t > b.t1) continue;
    const gap = side * (offsetAt(b, t) - sa);
    if (gap >= LANE_LIMITS.minSpacingM && gap <= LANE_LIMITS.maxSpacingM && (!best || gap < best.gap)) best = { row: b, gap };
  }
  return best;
}

/** Positions from t0 to t1, every stepM. */
function samples(t0: number, t1: number): number[] {
  const steps = Math.max(1, Math.ceil((t1 - t0) / LANE_LIMITS.stepM));
  return Array.from({ length: steps + 1 }, (_, i) => t0 + ((t1 - t0) * i) / steps);
}

/** Typical spacing of the block: the median gap between neighbouring rows at their middles. */
function blockSpacing(rows: readonly Profile[]): number {
  const gaps = rows.flatMap((a) => {
    const nb = neighbourAt(a, (a.t0 + a.t1) / 2, 1, rows);
    return nb ? [nb.gap] : [];
  });
  return gaps.length ? median(gaps) : LANE_LIMITS.defaultSpacingM;
}

/**
 * Lanes of one row: on its + side, the lane runs over the row and its + neighbours (midway between them, or half a
 * spacing out where there is none); on its − side, only where it has no neighbour (an edge), half a spacing out.
 */
function rowLanes(a: Profile, rows: readonly Profile[], half: number, d: XY, n: XY, blockId: string): Lane[] {
  const c = LANE_LIMITS.headlandClearanceM;
  const at = (t: number, s: number): XY => [t * d[0] + s * n[0], t * d[1] + s * n[1]];
  const plusNeighbours = new Set<Profile>();
  for (const t of samples(a.t0, a.t1)) {
    const nb = neighbourAt(a, t, 1, rows);
    if (nb) plusNeighbours.add(nb.row);
  }
  const span = [a, ...plusNeighbours];
  const plus = samples(Math.min(...span.map((p) => p.t0)) - c, Math.max(...span.map((p) => p.t1)) + c).map((t) => {
    const nb = neighbourAt(a, t, 1, rows);
    const sa = offsetAt(a, t);
    return at(t, nb ? sa + nb.gap / 2 : sa + half);
  });
  // − side: runs of positions with no neighbour there
  const runs: XY[][] = [];
  let run: XY[] = [];
  for (const t of samples(a.t0 - c, a.t1 + c)) {
    if (neighbourAt(a, Math.min(Math.max(t, a.t0), a.t1), -1, rows)) {
      if (run.length >= 2) runs.push(run);
      run = [];
    } else run = [...run, at(t, offsetAt(a, t) - half)];
  }
  if (run.length >= 2) runs.push(run);
  const offsetOf = (pts: readonly XY[]) => median(pts.map((p) => p[0] * n[0] + p[1] * n[1]));
  return [
    { points: plus, rowIds: [a.rowId, ...[...plusNeighbours].map((p) => p.rowId)], blockId, offset: offsetOf(plus) },
    ...runs.map((points) => ({ points, rowIds: [a.rowId], blockId, offset: offsetOf(points) })),
  ];
}

/** Every walkable lane of the given rows (each block in its own frame). */
export function interRowLanes(rows: readonly RowAxis[]): Lane[] {
  const blocks = new Map<string, RowAxis[]>();
  for (const r of rows) if (r.parts.some((p) => p.length >= 2)) blocks.set(r.blockId, [...(blocks.get(r.blockId) ?? []), r]);
  return [...blocks.entries()].flatMap(([blockId, blockRows]) => {
    const d = blockDirection(blockRows);
    const n: XY = [-d[1], d[0]];
    const profiles = blockRows.map((r) => profileOf(r, d, n));
    const half = blockSpacing(profiles) / 2;
    return profiles.flatMap((a) => rowLanes(a, profiles, half, d, n, blockId));
  });
}

/** Every segment of the rows: the canopies a walk must not cross. */
export const rowSegments = (rows: readonly RowAxis[]): [XY, XY][] =>
  rows.flatMap((r) => r.parts.flatMap((p) => p.slice(1).map((b, i): [XY, XY] => [p[i], b])));
