// The first `fraction` of a set of lines, by length: the official route drawn progressively on the map
// (src/components/map/animation/RouteDraw.tsx). Lengths are planar in lon/lat with the longitude scaled by cos(lat),
// which is plenty for the share of a few-kilometre route drawn at each frame.
import type { Feature, FeatureCollection, LineString, MultiLineString, Position } from "geojson";

export type RouteLines = FeatureCollection<LineString | MultiLineString>;

const DEG = Math.PI / 180;

const segLength = (a: Position, b: Position) => {
  const kx = Math.cos(((a[1] + b[1]) / 2) * DEG);
  return Math.hypot((b[0] - a[0]) * kx, b[1] - a[1]);
};

/** Cumulative length at each vertex of a line (first 0). */
export function cumulative(line: Position[]): number[] {
  const out = [0];
  for (let i = 1; i < line.length; i++) out.push(out[i - 1] + segLength(line[i - 1], line[i]));
  return out;
}

const partsOf = (f: Feature<LineString | MultiLineString>): Position[][] =>
  f.geometry.type === "LineString" ? [f.geometry.coordinates] : f.geometry.coordinates;

/** A line cut at `dist` along it (the whole line when longer); null when there is nothing to draw yet. */
export function cutAt(line: Position[], cum: number[], dist: number): Position[] | null {
  if (dist <= 0 || line.length < 2) return null;
  const end = cum[cum.length - 1];
  if (dist >= end) return line;
  let i = 1;
  while (cum[i] < dist) i++;
  const t = (dist - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
  const [a, b] = [line[i - 1], line[i]];
  return [...line.slice(0, i), [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]];
}

/** Precomputes the lengths once; the returned function gives the collection drawn up to `fraction` ∈ [0, 1]. */
export function routeSlicer(fc: RouteLines): (fraction: number) => RouteLines {
  const parts = fc.features.map((f) => partsOf(f).map((line) => ({ line, cum: cumulative(line) })));
  const total = parts.flat().reduce((sum, p) => sum + p.cum[p.cum.length - 1], 0);

  return (fraction) => {
    if (fraction >= 1 || total === 0) return fc;
    let left = Math.max(0, fraction) * total;
    const features: RouteLines["features"] = [];
    fc.features.forEach((f, k) => {
      const lines: Position[][] = [];
      for (const { line, cum } of parts[k]) {
        const cut = cutAt(line, cum, left);
        left -= cum[cum.length - 1];
        if (cut) lines.push(cut);
        if (left <= 0) break;
      }
      if (lines.length)
        features.push({
          ...f,
          geometry: lines.length === 1 ? { type: "LineString", coordinates: lines[0] } : { type: "MultiLineString", coordinates: lines },
        });
    });
    return { ...fc, features };
  };
}
