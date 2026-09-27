// Rows (rows.json + rows.geojson) → length and max-gap histograms, and the orientation rose.
import { histogram, quantile, round, share, sum } from "./util.mjs";

export const LENGTH_EDGES_M = [0, 10, 25, 50, 75, 100, 150, 200, 300];
export const GAP_EDGES_M = [0, 1, 2, 3, 5, 10, 20];
export const ROSE_BIN_DEG = 10;
/** The dominant axis share counts the dominant bin and its two neighbours (a 30° window). */
const DOMINANT_HALF_WINDOW = 1;

const DEG = Math.PI / 180;

/**
 * Axis of a row, degrees clockwise from north in [0, 180): the length-weighted axial mean of its segments in local
 * metres (longitude scaled by cos(latitude)). Angles are doubled before averaging, so a segment and its reverse count
 * the same and near east–west rows do not cancel out.
 */
export const rowAxisDeg = (geometry) => {
  const lines = geometry.type === "LineString" ? [geometry.coordinates] : geometry.coordinates;
  let c = 0;
  let s = 0;
  for (const line of lines) {
    for (let i = 1; i < line.length; i += 1) {
      const [x0, y0] = line[i - 1];
      const [x1, y1] = line[i];
      const dx = (x1 - x0) * Math.cos(y0 * DEG);
      const dy = y1 - y0;
      const len = Math.hypot(dx, dy);
      if (len === 0) continue;
      const az2 = 2 * Math.atan2(dx, dy);
      c += len * Math.cos(az2);
      s += len * Math.sin(az2);
    }
  }
  if (c === 0 && s === 0) return null;
  return ((Math.atan2(s, c) / 2 / DEG) % 180 + 180) % 180;
};

/** Row length (m) summed per ROSE_BIN_DEG bin of the row axis; the dominant axis and the share of length around it. */
export const orientationRose = (rowsGeo) => {
  const bins = Array.from({ length: 180 / ROSE_BIN_DEG }, () => 0);
  for (const { geometry, properties } of rowsGeo.features) {
    const axis = geometry ? rowAxisDeg(geometry) : null;
    if (axis == null) continue;
    bins[Math.floor(axis / ROSE_BIN_DEG) % bins.length] += properties.length_m ?? 0;
  }
  const total = sum(bins);
  const top = bins.indexOf(Math.max(...bins));
  let around = 0;
  for (let k = -DOMINANT_HALF_WINDOW; k <= DOMINANT_HALF_WINDOW; k += 1) around += bins[(top + k + bins.length) % bins.length];
  return {
    bin_deg: ROSE_BIN_DEG,
    length_m: bins.map((b) => round(b)),
    dominant_deg: top * ROSE_BIN_DEG + ROSE_BIN_DEG / 2,
    dominant_share: share(around, total),
    window_deg: (2 * DOMINANT_HALF_WINDOW + 1) * ROSE_BIN_DEG,
  };
};

export const rowStats = (rows, rowsGeo) => {
  const lengths = rows.map((r) => r.length_m ?? 0);
  const gaps = rows.map((r) => r.max_gap_m ?? 0);
  return {
    count: rows.length,
    length_median_m: round(quantile(lengths, 0.5)),
    length_max_m: round(Math.max(0, ...lengths)),
    gap_median_m: round(quantile(gaps, 0.5)),
    gap_max_m: round(Math.max(0, ...gaps)),
    length_hist: histogram(lengths, LENGTH_EDGES_M),
    gap_hist: histogram(gaps, GAP_EDGES_M),
    orientation: orientationRose(rowsGeo),
  };
};
