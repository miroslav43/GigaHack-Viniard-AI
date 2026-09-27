// The farm outline in UTM metres: inside tests, the nearest edge point, and clipping roads to the farm (ADR-028).
import { closestOnSegment, dist, type XY } from "./geometry.ts";

/** Every ring of the farm (outer rings and holes of all its polygons). */
export type FarmRings = readonly (readonly XY[])[];

/** Even–odd rule over all rings: right for polygons with holes and for disjoint multipolygons. */
export function insideRings(p: XY, rings: FarmRings): boolean {
  let inside = false;
  for (const ring of rings)
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
    }
  return inside;
}

/** Closest point of the farm outline to p. */
export function nearestEdge(p: XY, rings: FarmRings): { point: XY; d: number } {
  let best = { point: p, d: Infinity };
  for (const ring of rings)
    for (let i = 1; i < ring.length; i++) {
      const hit = closestOnSegment(p, ring[i - 1], ring[i]);
      if (hit.d < best.d) best = { point: hit.point, d: hit.d };
    }
  return best;
}

/** Inside the farm or at most `margin` metres from its outline. */
export const withinFarm = (p: XY, rings: FarmRings, margin: number) => insideRings(p, rings) || nearestEdge(p, rings).d <= margin;

/** Points every `step` metres along the line (the vertices kept), so clipping follows the outline closely. */
function densify(points: readonly XY[], step: number): XY[] {
  const out: XY[] = points.length ? [points[0]] : [];
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    const n = Math.ceil(dist(a, b) / step);
    for (let k = 1; k <= n; k++) out.push([a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n]);
  }
  return out;
}

/** The pieces of a line that run inside the farm (or within `margin` of it), each with at least two points. */
export function clipToFarm(points: readonly XY[], rings: FarmRings, margin: number, step = 2): XY[][] {
  // local accumulators: a run grows point by point until the line leaves the farm
  const pieces: XY[][] = [];
  let run: XY[] = [];
  for (const p of densify(points, step)) {
    if (withinFarm(p, rings, margin)) run.push(p);
    else {
      if (run.length >= 2) pieces.push(run);
      run = [];
    }
  }
  if (run.length >= 2) pieces.push(run);
  return pieces;
}
