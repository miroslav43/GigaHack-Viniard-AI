// Farms (summary.farms), roads (roads.geojson) and survey tiles (tiles.geojson) → the stats page sections C, D, F.
// Each returns null when the survey has no such data (the mock and older bundles).
import { round, share, sum } from "./util.mjs";

const M2_PER_HA = 1e4;
export const UNKNOWN_SURFACE = "unknown";

/** Farms by area, largest first, with the cumulative share (Pareto) and targets per hectare. */
export const farmStats = (farms, uatAreaHa) => {
  if (!farms?.length) return null;
  const total = sum(farms.map((f) => f.area_m2));
  let cumulative = 0;
  const items = [...farms]
    .sort((a, b) => b.area_m2 - a.area_m2)
    .map((f) => {
      cumulative += f.area_m2;
      return {
        farm_id: f.farm_id,
        area_m2: round(f.area_m2),
        n_blocks: f.n_blocks,
        n_parcels: f.n_parcels ?? null,
        target_count: f.target_count ?? 0,
        targets_per_ha: f.area_m2 > 0 ? round((f.target_count ?? 0) / (f.area_m2 / M2_PER_HA), 1) : 0,
        cumulative_share: share(cumulative, total),
      };
    });
  const parcels = farms.map((f) => f.n_parcels).filter((n) => n != null);
  return {
    count: farms.length,
    total_area_m2: round(total),
    commune_share: uatAreaHa > 0 ? round(total / M2_PER_HA / uatAreaHa, 5) : 0,
    parcels_total: parcels.length ? sum(parcels) : null,
    items,
  };
};

/** Road metres per class, each split by OSM surface ("unknown" where OSM has none). */
export const roadStats = (roads) => {
  if (!roads?.features.length) return null;
  const metres = new Map(); // road_class → Map(surface → m)
  for (const { properties: p } of roads.features) {
    const surfaces = metres.get(p.road_class) ?? new Map();
    const surface = p.surface ?? UNKNOWN_SURFACE;
    metres.set(p.road_class, surfaces.set(surface, (surfaces.get(surface) ?? 0) + (p.length_m ?? 0)));
  }
  const byClass = Object.fromEntries(
    [...metres].map(([cls, surfaces]) => [
      cls,
      {
        total_m: round(sum([...surfaces.values()])),
        by_surface: Object.fromEntries([...surfaces].sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, round(v)])),
      },
    ]),
  );
  return { by_class: byClass };
};

const TILE_RE = /_r(\d+)_c(\d+)$/;

/** The survey tile grid (row / column from the tile id, e.g. siret3_r005_c004), with status and vegetation share. */
export const tileStats = (tiles) => {
  if (!tiles?.features.length) return null;
  const items = tiles.features
    .map(({ properties: p }) => {
      const m = TILE_RE.exec(p.tile);
      if (!m) return null;
      return {
        tile: p.tile,
        r: Number(m[1]),
        c: Number(m[2]),
        status: p.status,
        veg_frac: round(p.veg_frac ?? 0, 3),
        nodata_frac: round(p.nodata_frac ?? 0, 3),
        to_complete: p.review_status != null,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.r - b.r || a.c - b.c);
  return {
    rows: Math.max(...items.map((t) => t.r)) + 1,
    cols: Math.max(...items.map((t) => t.c)) + 1,
    veg_frac_mean: round(sum(items.map((t) => t.veg_frac)) / items.length, 3),
    items,
  };
};
