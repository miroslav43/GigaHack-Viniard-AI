// stats.json of the /statistici page, derived from the files a survey already publishes (docs/STATISTICI.md).
// Pure: takes the parsed files, returns the object; scripts/build-stats.mjs does the I/O.
import { healthStats } from "./health.mjs";
import { farmStats, roadStats, tileStats } from "./land.mjs";
import { rowStats } from "./rows.mjs";
import { damageByBlock, targetStats } from "./targets.mjs";
import { round, share, sum } from "./util.mjs";

export const STATS_VERSION = 1;
/** Rule of the challenge: a route with more than 2% of its length outside the study area scores 0. */
export const ROUTE_OUTSIDE_LIMIT = 0.02;

const structureStats = (totals, interrows) => {
  const byCover = {};
  for (const { properties: p } of interrows.features) {
    const k = p.interrow_cover ?? "unassessable";
    byCover[k] = (byCover[k] ?? 0) + (p.area_m2 ?? 0);
  }
  return {
    canopy_cover_share: share(totals.canopy_area_m2, totals.canopy_area_m2 + totals.interrow_area_m2),
    interrow_width_m: totals.row_length_m > 0 ? round(totals.interrow_area_m2 / totals.row_length_m) : 0,
    canopy_mean_m2: totals.canopy_count > 0 ? round(totals.canopy_area_m2 / totals.canopy_count) : 0,
    row_m_per_canopy: totals.canopy_count > 0 ? round(totals.row_length_m / totals.canopy_count) : 0,
    interrow_area_by_cover: Object.fromEntries(Object.entries(byCover).map(([k, v]) => [k, round(v)])),
    interrow_area_total_m2: round(sum(Object.values(byCover))),
  };
};

const routeStats = (route, routeGeo) => {
  const props = routeGeo?.features[0]?.properties ?? {};
  const baseline = route.baseline_length_m > 0 ? route.baseline_length_m : null;
  const saved = baseline != null ? baseline - route.length_m : null;
  const speed = route.speed_kmh > 0 ? route.speed_kmh : null;
  return {
    length_m: round(route.length_m),
    duration_min: round(route.duration_min, 1),
    baseline_length_m: baseline != null ? round(baseline) : null,
    saved_m: saved != null ? round(saved) : null,
    saved_share: saved != null ? share(saved, baseline) : null,
    saved_min: saved != null && speed != null ? round((saved / 1000 / speed) * 60, 1) : null,
    outside_share: props.outside_share != null ? round(props.outside_share, 5) : null,
    outside_limit: ROUTE_OUTSIDE_LIMIT,
  };
};

/**
 * files: { summary, rows (rows.json), rowsGeo, targets, interrows, route (route.geojson),
 *          roads?, tiles?, farms? (farms.geojson, for the block → farm link) } — optional ones may be null.
 */
export const buildStats = ({ summary, rows, rowsGeo, targets, interrows, route, roads = null, tiles = null, farms = null }) => {
  const farmOf = new Map(
    (farms?.features ?? []).flatMap(({ properties: p }) => (p.vineyard_ids ?? []).map((v) => [v, p.farm_id])),
  );
  return {
    version: STATS_VERSION,
    survey_id: summary.survey.id,
    targets: targetStats(targets),
    rows: rowStats(rows, rowsGeo),
    structure: structureStats(summary.totals, interrows),
    health: healthStats(summary.blocks, damageByBlock(targets), farmOf),
    farms: farmStats(summary.farms, summary.uat.area_ha),
    roads: roadStats(roads),
    tiles: tileStats(tiles),
    route: routeStats(summary.route, route),
  };
};
