// Raster helpers for the synthetic canopy relief: polygon area coverage, gaussian blur, height profile, 2× downsampling.
// Grids are Float32Array, row-major, width × height. Every exported function returns a new grid.

/** Sub-scanlines per pixel row; horizontal coverage is computed exactly, so 4 is plenty for 20 cm pixels. */
export const SUBSAMPLES = 4;

// adds `weight` × the covered length of [xa, xb) to each pixel of one grid row (mutates only the caller's grid)
const addSpan = (grid, rowStart, width, xa, xb, weight) => {
  const a = Math.max(0, xa), b = Math.min(width, xb);
  if (b <= a) return;
  const ia = Math.floor(a), ib = Math.floor(b);
  if (ia === ib) {
    grid[rowStart + ia] += (b - a) * weight;
    return;
  }
  grid[rowStart + ia] += (ia + 1 - a) * weight;
  for (let i = ia + 1; i < ib; i++) grid[rowStart + i] += weight;
  if (ib < width) grid[rowStart + ib] += (b - ib) * weight;
};

// x of every edge crossing the horizontal line y (even-odd rule over all rings), sorted
const crossings = (rings, y) => {
  const xs = [];
  for (const ring of rings) {
    for (let i = 0, n = ring.length; i < n; i++) {
      const [xa, ya] = ring[i];
      const [xb, yb] = ring[(i + 1) % n];
      if ((ya <= y && y < yb) || (yb <= y && y < ya)) xs.push(xa + ((y - ya) * (xb - xa)) / (yb - ya));
    }
  }
  return xs.sort((p, q) => p - q);
};

const ringsBBoxY = (rings) => {
  let min = Infinity, max = -Infinity;
  for (const ring of rings) for (const [, y] of ring) {
    if (y < min) min = y;
    if (y > max) max = y;
  }
  return [min, max];
};

/**
 * Area coverage (0–1 per pixel) of `polygons` — each an array of rings in grid pixel coordinates, holes by even-odd —
 * clipped to the grid. Overlapping polygons are clamped at 1.
 */
export const coverageGrid = (polygons, width, height, samples = SUBSAMPLES) => {
  const grid = new Float32Array(width * height);
  const weight = 1 / samples;
  for (const rings of polygons) {
    const [yMin, yMax] = ringsBBoxY(rings);
    const j0 = Math.max(0, Math.floor(yMin)), j1 = Math.min(height - 1, Math.floor(yMax));
    for (let j = j0; j <= j1; j++) {
      for (let k = 0; k < samples; k++) {
        const xs = crossings(rings, j + (k + 0.5) / samples);
        for (let s = 0; s + 1 < xs.length; s += 2) addSpan(grid, j * width, width, xs[s], xs[s + 1], weight);
      }
    }
  }
  for (let i = 0; i < grid.length; i++) if (grid[i] > 1) grid[i] = 1;
  return grid;
};

const gaussianKernel = (sigma) => {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const raw = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)));
  const sum = raw.reduce((s, v) => s + v, 0);
  return { radius, weights: raw.map((v) => v / sum) };
};

// one separable pass, along rows (`horizontal`) or columns; edges clamp to the border pixel
const blurPass = (src, width, height, { radius, weights }, horizontal) => {
  const out = new Float32Array(src.length);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const ii = horizontal ? Math.min(width - 1, Math.max(0, i + k)) : i;
        const jj = horizontal ? j : Math.min(height - 1, Math.max(0, j + k));
        acc += src[jj * width + ii] * weights[k + radius];
      }
      out[j * width + i] = acc;
    }
  }
  return out;
};

/** Gaussian blur with standard deviation `sigma` pixels (a copy when sigma ≤ 0). */
export const gaussianBlur = (src, width, height, sigma) => {
  if (!(sigma > 0)) return Float32Array.from(src);
  const kernel = gaussianKernel(sigma);
  return blurPass(blurPass(src, width, height, kernel, true), width, height, kernel, false);
};

/**
 * Blurred coverage → relief in [0, 1]: coverage ≥ `knee` is the flat top of the canopy, below it a smoothstep
 * shoulder down to the ground, so a narrow canopy (0.5 m) still reaches ~90% of the height and rows stay distinct.
 */
export const reliefProfile = (coverage, knee) => {
  const t = Math.min(1, Math.max(0, coverage / knee));
  return t * t * (3 - 2 * t);
};

/** Centre `size` × `size` window of a (size + 2·pad)² grid. */
export const crop = (grid, size, pad) => {
  const full = size + 2 * pad;
  const out = new Float32Array(size * size);
  for (let j = 0; j < size; j++) out.set(grid.subarray((j + pad) * full + pad, (j + pad) * full + pad + size), j * size);
  return out;
};

/**
 * Parent tile from its four children [north-west, north-east, south-west, south-east] (null = flat ground),
 * each pixel the mean of the 2 × 2 child pixels it covers. Null when every child is flat.
 */
export const downsample = (children, size) => {
  if (children.every((c) => c == null)) return null;
  const out = new Float32Array(size * size);
  const half = size / 2;
  children.forEach((child, q) => {
    if (!child) return;
    const ox = (q % 2) * half, oy = Math.floor(q / 2) * half;
    for (let j = 0; j < half; j++) {
      for (let i = 0; i < half; i++) {
        const a = 2 * j * size + 2 * i;
        out[(oy + j) * size + ox + i] = (child[a] + child[a + 1] + child[a + size] + child[a + size + 1]) / 4;
      }
    }
  });
  return out;
};

export const maxOf = (grid) => grid.reduce((m, v) => (v > m ? v : m), 0);
