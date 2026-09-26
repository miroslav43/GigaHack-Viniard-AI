// Planar measurement in EPSG:32635 (UTM 35N) — the same method the pipeline and API use (ADR-010).
// Only for ad-hoc measurements drawn on the map; official figures come precomputed from the data build.
import proj4 from "proj4";

const UTM35N = "+proj=utm +zone=35 +datum=WGS84 +units=m +no_defs";
const toUtm = proj4("EPSG:4326", UTM35N);

export type LonLat = [number, number];

export const projectUtm = (p: LonLat) => toUtm.forward(p) as [number, number];
export const unprojectUtm = (p: readonly [number, number]) => toUtm.inverse([p[0], p[1]]) as LonLat;

export function lengthM(points: LonLat[]): number {
  const u = points.map(projectUtm);
  let s = 0;
  for (let i = 1; i < u.length; i++) s += Math.hypot(u[i][0] - u[i - 1][0], u[i][1] - u[i - 1][1]);
  return s;
}

/** Area of the polygon closed by the points (shoelace, m²); 0 for fewer than three points. */
export function areaM2(points: LonLat[]): number {
  if (points.length < 3) return 0;
  const u = points.map(projectUtm);
  let s = 0;
  for (let i = 0; i < u.length; i++) {
    const [x1, y1] = u[i];
    const [x2, y2] = u[(i + 1) % u.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}
