// Planar helpers for the farm route (UTM 35N metres, ADR-028). Pure: no projection, no DOM.

export type XY = readonly [number, number];

export const dist = (a: XY, b: XY) => Math.hypot(b[0] - a[0], b[1] - a[1]);

export interface SegmentHit {
  /** 0..1 along the segment a→b */
  t: number;
  point: XY;
  d: number;
}

/** Closest point of the segment a→b to p. */
export function closestOnSegment(p: XY, a: XY, b: XY): SegmentHit {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2));
  const point: XY = [a[0] + t * dx, a[1] + t * dy];
  return { t, point, d: dist(p, point) };
}

export const polylineLength = (pts: readonly XY[]) => pts.reduce((s, p, i) => (i === 0 ? 0 : s + dist(pts[i - 1], p)), 0);
