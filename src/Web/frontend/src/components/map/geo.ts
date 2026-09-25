import type { Feature, FeatureCollection, Geometry, Position } from "geojson";

export type BBox = [number, number, number, number];

const walk = (coords: unknown, out: Position[]) => {
  if (typeof (coords as number[])[0] === "number") out.push(coords as Position);
  else for (const c of coords as unknown[]) walk(c, out);
};

export function bboxOf(geoms: (Geometry | Feature)[]): BBox | null {
  const pts: Position[] = [];
  for (const g of geoms) {
    const geom = "geometry" in g ? g.geometry : g;
    if (geom && "coordinates" in geom) walk(geom.coordinates, pts);
  }
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

export const bboxOfCollection = (fc: FeatureCollection) => bboxOf(fc.features);

export const intersects = (a: BBox, b: BBox) => a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1];

/** World polygon with the given rings as holes — used to grey out everything outside the geofence. */
export function maskOutside(fc: FeatureCollection): FeatureCollection {
  const holes: Position[][] = [];
  for (const f of fc.features) {
    if (f.geometry.type === "Polygon") holes.push(f.geometry.coordinates[0]);
    if (f.geometry.type === "MultiPolygon") for (const p of f.geometry.coordinates) holes.push(p[0]);
  }
  const world: Position[] = [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]];
  return {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [world, ...holes] } }],
  };
}
