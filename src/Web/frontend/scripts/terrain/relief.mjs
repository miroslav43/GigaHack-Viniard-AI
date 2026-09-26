// Synthetic canopy relief: canopy polygons (EPSG:4326) → height tiles over every tile MapLibre can request.
// The orthophoto tiles are RGB only (no altitude), so the relief is made up from the annotations: ground = 0 m,
// each canopy a smooth ridge of `heightM`. Leaves (maxZoom) are rasterised; lower zooms average their children.
import { inRange, metresPerPixel, rangeSize, tileCentreLat, tileRange, toPixel } from "./mercator.mjs";
import { ENCODING_STEP_M } from "./terrarium.mjs";
import { coverageGrid, crop, downsample, gaussianBlur, maxOf, reliefProfile } from "./raster.mjs";

export const RELIEF_DEFAULTS = {
  heightM: 1.1, // canopy height of a trellised vine row (the web UI rescales it, 0–2 m)
  blurM: 0.45, // gaussian sigma of the smoothing, metres: rounds each canopy into a hedge-like ridge
  knee: 0.6, // blurred coverage from which the canopy top is flat (raster.mjs reliefProfile)
  minZoom: 14,
  maxZoom: 19, // 256 px tiles: ~0.2 m/px at 47° N, finer than the narrowest canopies (~0.5 m)
  tileSize: 256,
  marginM: 2, // bounds margin around the canopies, so the blur is never cut at the edge
};
// a survey much larger than Sireț3 (~1 600 tiles) needs a smarter tiling than "every tile in the bbox"
export const MAX_TILES = 60_000;

const M_PER_DEG_LAT = 110_574;
const M_PER_DEG_LON_EQUATOR = 111_320;
const CHILDREN = [[0, 0], [1, 0], [0, 1], [1, 1]]; // NW, NE, SW, SE (raster.mjs downsample order)

/** Outer rings + holes of every Polygon / MultiPolygon feature; `skipped` counts the other features. */
export const canopyPolygons = (fc) => {
  const polygons = [];
  let skipped = 0;
  for (const f of fc?.features ?? []) {
    const g = f?.geometry;
    const parts = g?.type === "Polygon" ? [g.coordinates] : g?.type === "MultiPolygon" ? g.coordinates : null;
    const valid = parts?.filter((rings) => Array.isArray(rings) && rings.length > 0 && rings.every((r) => Array.isArray(r) && r.length >= 3));
    if (!valid?.length) {
      skipped++;
      continue;
    }
    polygons.push(...valid);
  }
  return { polygons, skipped };
};

const roundOut = (v, dir) => (dir < 0 ? Math.floor(v * 1e7) : Math.ceil(v * 1e7)) / 1e7;

/** [w, s, e, n] of the polygons grown by `marginM` metres, rounded outward to 7 decimals. */
export const reliefBounds = (polygons, marginM) => {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const rings of polygons) for (const [lon, lat] of rings[0]) {
    if (lon < w) w = lon;
    if (lon > e) e = lon;
    if (lat < s) s = lat;
    if (lat > n) n = lat;
  }
  const dLat = marginM / M_PER_DEG_LAT;
  const dLon = marginM / (M_PER_DEG_LON_EQUATOR * Math.cos((((s + n) / 2) * Math.PI) / 180));
  return [roundOut(w - dLon, -1), roundOut(s - dLat, -1), roundOut(e + dLon, 1), roundOut(n + dLat, 1)];
};

/** Tile ranges per zoom and the total number of tiles MapLibre can request inside `bounds`. */
export const reliefTileRanges = (bounds, minZoom, maxZoom) => {
  const ranges = {};
  let total = 0;
  for (let z = minZoom; z <= maxZoom; z++) {
    ranges[z] = tileRange(bounds, z);
    total += rangeSize(ranges[z]);
  }
  return { ranges, total };
};

// polygons in global pixels at maxZoom, bucketed by every leaf tile their bbox (+ pad) touches
const indexLeaves = (polygons, { maxZoom, tileSize }, pad) => {
  const buckets = new Map();
  for (const rings of polygons) {
    const px = rings.map((ring) => ring.map((p) => toPixel(p, maxZoom, tileSize)));
    const xs = px[0].map((p) => p[0]), ys = px[0].map((p) => p[1]);
    const [tx0, tx1] = [Math.floor((Math.min(...xs) - pad) / tileSize), Math.floor((Math.max(...xs) + pad) / tileSize)];
    const [ty0, ty1] = [Math.floor((Math.min(...ys) - pad) / tileSize), Math.floor((Math.max(...ys) + pad) / tileSize)];
    for (let tx = tx0; tx <= tx1; tx++) for (let ty = ty0; ty <= ty1; ty++) {
      const key = `${tx}/${ty}`;
      // the index is local to this build: appending in place keeps it linear in the number of polygons
      if (buckets.has(key)) buckets.get(key).push(px);
      else buckets.set(key, [px]);
    }
  }
  return buckets;
};

/**
 * Writes every tile MapLibre can request for `bounds` between minZoom and maxZoom:
 * `writeTile(z, x, y, heights)` is awaited per tile, heights a Float32Array (metres, tileSize², row-major) or null
 * for flat ground. Returns { bounds, tiles, reliefTiles, perZoom }.
 */
export const buildRelief = async (polygons, options, writeTile) => {
  const o = { ...RELIEF_DEFAULTS, ...options };
  if (!polygons.length) throw new Error("no canopy polygons: nothing to raise");
  const bounds = reliefBounds(polygons, o.marginM);
  const { ranges, total } = reliefTileRanges(bounds, o.minZoom, o.maxZoom);
  if (total > MAX_TILES) throw new Error(`${total} relief tiles for bounds ${bounds.join(", ")} (> ${MAX_TILES}): survey area too large`);

  const lat = (bounds[1] + bounds[3]) / 2;
  const sigmaPx = o.blurM / metresPerPixel(lat, o.maxZoom, o.tileSize);
  const pad = Math.ceil(3 * sigmaPx) + 2;
  const leaves = indexLeaves(polygons, o, pad);
  const perZoom = Object.fromEntries(Object.keys(ranges).map((z) => [z, { tiles: 0, relief: 0 }]));

  const leaf = (x, y) => {
    const polys = leaves.get(`${x}/${y}`);
    if (!polys) return null;
    const [ox, oy] = [x * o.tileSize - pad, y * o.tileSize - pad];
    const size = o.tileSize + 2 * pad;
    const local = polys.map((rings) => rings.map((ring) => ring.map(([px, py]) => [px - ox, py - oy])));
    const sigma = o.blurM / metresPerPixel(tileCentreLat(y, o.maxZoom), o.maxZoom, o.tileSize);
    const blurred = crop(gaussianBlur(coverageGrid(local, size, size), size, size, sigma), o.tileSize, pad);
    const heights = blurred.map((c) => o.heightM * reliefProfile(c, o.knee));
    return maxOf(heights) < ENCODING_STEP_M / 2 ? null : heights;
  };

  const build = async (z, x, y) => {
    let heights;
    if (z === o.maxZoom) heights = leaf(x, y);
    else {
      const children = [];
      for (const [dx, dy] of CHILDREN) {
        const [cx, cy] = [2 * x + dx, 2 * y + dy];
        children.push(inRange(ranges[z + 1], cx, cy) ? await build(z + 1, cx, cy) : null);
      }
      heights = downsample(children, o.tileSize);
    }
    await writeTile(z, x, y, heights);
    perZoom[z].tiles++;
    if (heights) perZoom[z].relief++;
    return heights;
  };

  const top = ranges[o.minZoom];
  for (let x = top.minX; x < top.maxX; x++) for (let y = top.minY; y < top.maxY; y++) await build(o.minZoom, x, y);

  const counts = Object.values(perZoom);
  return {
    bounds,
    tiles: counts.reduce((s, c) => s + c.tiles, 0),
    reliefTiles: counts.reduce((s, c) => s + c.relief, 0),
    perZoom,
  };
};
