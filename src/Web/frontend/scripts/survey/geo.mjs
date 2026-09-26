// Planar geometry in EPSG:32635 metres and reprojection to EPSG:4326 — same conventions as build-data.mjs
// (proj4 UTM 35N, 7 decimals). Areas are measured in UTM before reprojection.
import proj4 from "proj4";

export const UTM = "+proj=utm +zone=35 +datum=WGS84 +units=m +no_defs";
const toLonLat = proj4(UTM, "EPSG:4326");
export const ll = ([x, y]) => toLonLat.forward([x, y]).map((v) => Math.round(v * 1e7) / 1e7);

export const r2 = (v) => Math.round(v * 100) / 100;
export const r4 = (v) => Math.round(v * 1e4) / 1e4;
export const isNum = (v) => typeof v === "number" && Number.isFinite(v);
export const naturalCompare = (a, b) => String(a).localeCompare(String(b), "en", { numeric: true });

// coordinate nesting depth per geometry type
export const DEPTH = { Point: 0, LineString: 1, MultiLineString: 2, Polygon: 2, MultiPolygon: 3 };
export const mapCoords = (coords, depth, fn = ll) =>
  depth === 0 ? fn(coords) : coords.map((c) => mapCoords(c, depth - 1, fn));
export const toWgs84 = (geometry) => ({
  type: geometry.type,
  coordinates: mapCoords(geometry.coordinates, DEPTH[geometry.type]),
});

/** Every position of a geometry, flattened. */
export const positions = (geometry) => {
  const walk = (coords, depth) => (depth === 0 ? [coords] : coords.flatMap((c) => walk(c, depth - 1)));
  return walk(geometry.coordinates, DEPTH[geometry.type]);
};

/** [minX, minY, maxX, maxY] of a list of positions. */
export const bbox = (points) =>
  points.reduce(
    ([x0, y0, x1, y1], [x, y]) => [Math.min(x0, x), Math.min(y0, y), Math.max(x1, x), Math.max(y1, y)],
    [Infinity, Infinity, -Infinity, -Infinity],
  );

// ---------- areas ----------
// shoelace relative to the first vertex: raw UTM products (~3e12) would cost ~1e-3 m² of float precision per ring
export const ringArea = (ring) => {
  if (ring.length < 3) return 0;
  const [x0, y0] = ring[0];
  let s = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    s += (ring[i][0] - x0) * (ring[i + 1][1] - y0) - (ring[i + 1][0] - x0) * (ring[i][1] - y0);
  }
  return Math.abs(s) / 2;
};
/** Polygon rings of a Polygon / MultiPolygon (empty for any other type). */
export const polygonsOf = (geometry) =>
  geometry.type === "Polygon" ? [geometry.coordinates] : geometry.type === "MultiPolygon" ? geometry.coordinates : [];
/** Planar area: outer rings minus holes. */
export const polygonArea = (geometry) =>
  polygonsOf(geometry).reduce(
    (sum, rings) => sum + rings.reduce((s, ring, i) => s + (i === 0 ? 1 : -1) * ringArea(ring), 0),
    0,
  );

// ---------- Douglas-Peucker ----------
const segmentDistance = (p, a, b) => {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const px = p[0] - a[0], py = p[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(px, py);
  const t = Math.max(0, Math.min(1, (px * dx + py * dy) / len2));
  return Math.hypot(px - t * dx, py - t * dy);
};

/** Douglas-Peucker on a polyline (iterative); tolerance in metres, <= 0 keeps every vertex. */
export const simplifyLine = (pts, tol) => {
  if (!(tol > 0) || pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [i, j] = stack.pop();
    let maxD = 0, k = -1;
    for (let m = i + 1; m < j; m++) {
      const d = segmentDistance(pts[m], pts[i], pts[j]);
      if (d > maxD) [maxD, k] = [d, m];
    }
    if (maxD > tol) {
      keep[k] = 1;
      stack.push([i, k], [k, j]);
    }
  }
  return pts.filter((_, m) => keep[m] === 1);
};

/** Closed ring simplification: keeps >= 4 points (closed triangle); keeps the original if the ring collapses. */
export const simplifyRing = (ring, tol) => {
  const out = simplifyLine(ring, tol);
  return out.length >= 4 && ringArea(out) > 0 ? out : ring;
};

/** Polygon / MultiPolygon with every ring simplified (other geometry types unchanged). */
export const simplifyPolygonal = (geometry, tol) => {
  if (geometry.type === "Polygon") return { ...geometry, coordinates: geometry.coordinates.map((r) => simplifyRing(r, tol)) };
  if (geometry.type === "MultiPolygon")
    return { ...geometry, coordinates: geometry.coordinates.map((p) => p.map((r) => simplifyRing(r, tol))) };
  return geometry;
};

// ---------- point in polygon (ray casting; works in any planar or lon/lat frame) ----------
export const pointInRing = ([x, y], ring) => {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
};
export const pointInPolygonal = (pt, geometry) =>
  polygonsOf(geometry).some(([outer, ...holes]) => pointInRing(pt, outer) && !holes.some((h) => pointInRing(pt, h)));
