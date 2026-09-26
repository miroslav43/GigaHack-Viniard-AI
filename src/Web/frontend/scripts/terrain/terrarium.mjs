// Terrarium height encoding (MapLibre raster-dem `encoding: "terrarium"`): h = R·256 + G + B/256 − 32768 metres.
// Step 1/256 m (~4 mm): smooth enough for a ~1 m relief even at 2× exaggeration. The "mapbox" encoding steps by
// 0.1 m, which would show a ~1 m canopy as ~10 visible terraces.

const BASE_SHIFT = 32768;
const STEPS_PER_M = 256;
const MAX_CODE = 256 ** 3 - 1;

/** RGB bytes for a height in metres, rounded to the nearest 1/256 m (MapLibre `packDEMData`). */
export const encodeHeight = (h) => {
  const code = Math.min(MAX_CODE, Math.max(0, Math.round((h + BASE_SHIFT) * STEPS_PER_M)));
  return [Math.floor(code / 65536) % 256, Math.floor(code / 256) % 256, code % 256];
};

/** Height in metres of a terrarium pixel, as MapLibre's DEMData unpacks it. */
export const decodeHeight = (r, g, b) => r * 256 + g + b / 256 - BASE_SHIFT;

/** Largest rounding error of `encodeHeight` (metres). */
export const ENCODING_STEP_M = 1 / STEPS_PER_M;

/** Heights (metres, row-major) → interleaved RGB bytes for a PNG. */
export const encodeTile = (heights) => {
  const rgb = Buffer.alloc(heights.length * 3);
  for (let i = 0; i < heights.length; i++) {
    const [r, g, b] = encodeHeight(heights[i]);
    rgb[i * 3] = r;
    rgb[i * 3 + 1] = g;
    rgb[i * 3 + 2] = b;
  }
  return rgb;
};
