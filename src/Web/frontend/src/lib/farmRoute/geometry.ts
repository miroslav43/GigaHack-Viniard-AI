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

const cross = (o: XY, a: XY, b: XY) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** Where p→q properly crosses a→b: the fractions along each, or null (parallel, apart, or only touching an end). */
export function segmentIntersection(p: XY, q: XY, a: XY, b: XY): { t: number; u: number } | null {
  const r: XY = [q[0] - p[0], q[1] - p[1]], s: XY = [b[0] - a[0], b[1] - a[1]];
  const den = r[0] * s[1] - r[1] * s[0];
  if (Math.abs(den) < 1e-12) return null;
  const t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / den;
  const u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den;
  const eps = 1e-9;
  return t > eps && t < 1 - eps && u > eps && u < 1 - eps ? { t, u } : null;
}

/** p→q crosses a→b (proper crossing; used to keep connectors off the rows). */
export const crosses = (p: XY, q: XY, a: XY, b: XY) =>
  cross(p, q, a) * cross(p, q, b) < 0 && cross(a, b, p) * cross(a, b, q) < 0;

/** Uniform grid over segments: the ones near a box, without scanning them all. */
export class SegmentGrid<T extends { a: XY; b: XY }> {
  private readonly cells = new Map<string, number[]>();
  readonly items: readonly T[];
  private readonly cell: number;

  constructor(items: readonly T[], cell = 10) {
    this.items = items;
    this.cell = cell;
    items.forEach((s, i) => {
      for (const k of this.keys(Math.min(s.a[0], s.b[0]), Math.min(s.a[1], s.b[1]), Math.max(s.a[0], s.b[0]), Math.max(s.a[1], s.b[1])))
        this.cells.set(k, [...(this.cells.get(k) ?? []), i]);
    });
  }

  private keys(x0: number, y0: number, x1: number, y1: number): string[] {
    const out: string[] = [];
    for (let x = Math.floor(x0 / this.cell); x <= Math.floor(x1 / this.cell); x++)
      for (let y = Math.floor(y0 / this.cell); y <= Math.floor(y1 / this.cell); y++) out.push(`${x}:${y}`);
    return out;
  }

  /** Indices of the segments whose cells touch the box (each once). */
  near(x0: number, y0: number, x1: number, y1: number): number[] {
    const seen = new Set<number>();
    for (const k of this.keys(x0, y0, x1, y1)) for (const i of this.cells.get(k) ?? []) seen.add(i);
    return [...seen];
  }

  /** p→q crosses one of the segments. */
  crossedBy(p: XY, q: XY): boolean {
    return this.near(Math.min(p[0], q[0]), Math.min(p[1], q[1]), Math.max(p[0], q[0]), Math.max(p[1], q[1])).some((i) =>
      crosses(p, q, this.items[i].a, this.items[i].b),
    );
  }
}
