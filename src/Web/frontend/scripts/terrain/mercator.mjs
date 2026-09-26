// Web Mercator tile math for the relief tiles, written to match MapLibre GL JS 6 exactly
// (src/geo/mercator_coordinate.ts, src/tile/tile_bounds.ts): the tiles this script writes are the tiles MapLibre asks for.

const EARTH_CIRCUMFERENCE_M = 2 * Math.PI * 6378137;

/** Mercator x in [0, 1] (MapLibre `mercatorXfromLng`). */
export const mercatorX = (lon) => (180 + lon) / 360;

/** Mercator y in [0, 1], 0 at the north edge (MapLibre `mercatorYfromLat`). */
export const mercatorY = (lat) => (180 - (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))) / 360;

/** Latitude of a mercator y (inverse of `mercatorY`). */
export const latFromMercatorY = (y) => (360 / Math.PI) * Math.atan(Math.exp(((180 - y * 360) * Math.PI) / 180)) - 90;

/** Global pixel coordinates of a lon/lat at zoom `z` for tiles of `tileSize` px. */
export const toPixel = ([lon, lat], z, tileSize) => {
  const world = tileSize * 2 ** z;
  return [mercatorX(lon) * world, mercatorY(lat) * world];
};

/**
 * Tiles MapLibre requests for a source with `bounds` = [w, s, e, n] at zoom z (TileBounds.contains):
 * x in [minX, maxX), y in [minY, maxY) — max bounds exclusive.
 */
export const tileRange = ([w, s, e, n], z) => {
  const world = 2 ** z;
  return {
    minX: Math.floor(mercatorX(w) * world),
    minY: Math.floor(mercatorY(n) * world),
    maxX: Math.ceil(mercatorX(e) * world),
    maxY: Math.ceil(mercatorY(s) * world),
  };
};

export const inRange = (range, x, y) => x >= range.minX && x < range.maxX && y >= range.minY && y < range.maxY;

export const rangeSize = (range) => Math.max(0, range.maxX - range.minX) * Math.max(0, range.maxY - range.minY);

/** Ground size of one pixel (metres) at latitude `lat`, zoom `z`, tiles of `tileSize` px. */
export const metresPerPixel = (lat, z, tileSize) => (EARTH_CIRCUMFERENCE_M * Math.cos((lat * Math.PI) / 180)) / (tileSize * 2 ** z);

/** Latitude of the centre of tile row `y` at zoom `z`. */
export const tileCentreLat = (y, z) => latFromMercatorY((y + 0.5) / 2 ** z);
