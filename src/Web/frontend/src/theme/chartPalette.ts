// Chart colours of /statistici, derived from tokens (the only place colours live).
// The categorical order is fixed and was checked with the dataviz palette validator (light surface: lightness band,
// chroma floor, colour-blind separation of adjacent slots, contrast ≥ 3:1) — keep the order, never cycle it.
import { color } from "./tokens";
import { mapPalette } from "./mapPalette";

export const chartPalette = {
  categorical: [color.primary.main, color.warning.main, color.success.main, color.map.farmRoute, color.map.farmLine],
  /** single-series magnitude (bars of one measure) */
  primary: color.primary.main,
  /** the "before" of a comparison (the route without optimisation) */
  muted: color.grey[400],
  /** "no data" segments (e.g. road surface OSM does not know) */
  unknown: color.grey[300],
  /** track of gauges and progress bars */
  track: color.grey[100],
  /** sequential ramp (tile vegetation share), light → dark, one hue */
  sequential: { from: color.success[50], to: color.success.dark },
  /** health index: below / above the threshold of attention */
  healthLow: color.warning.main,
  healthOk: color.success.main,
  /** limit markers (e.g. the 2% outside-route rule) */
  limit: color.error.main,
  tile: mapPalette.tile,
  interrow: mapPalette.interrow,
  heroFrom: color.brandNight,
  heroTo: color.primary.dark,
  heroText: color.white,
} as const;
