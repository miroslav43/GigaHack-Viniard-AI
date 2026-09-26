// summary.json (SurveySummary, src/lib/types.ts) and rows.json (RowRecord[]) of a real survey.
// Measurements come from the measurements.csv survey / block lines (§6.5); counts come from the layers.
import { TILE_M, TILES_TOTAL } from "./contract.mjs";
import { isNum, naturalCompare, polygonArea, r2, r4 } from "./geo.mjs";

// same Sireți block as the mock (build-data.mjs); area_ha comes from public/data/uats.json
export const UAT = { name: "Primăria Sireți", district: "raionul Strășeni", country: "Republica Moldova", osm_relation_id: 19100171 };
export const DEFAULT_SPEED_KMH = 4;
const PROVENANCE = ["model_version", "run_id", "generated_at", "pipeline_version"];

const countBy = (items, key) => {
  const counts = {};
  for (const item of items) {
    const k = key(item);
    if (k !== null && k !== undefined) counts[k] = (counts[k] ?? 0) + 1;
  }
  return counts;
};
const props = (layer) => layer.features.map((f) => f.properties);
const isDisrupted = (p) => p.row_structure === "disrupted";

/** manifest.tiles_with_objects (newer bundles), else the distinct tiles referenced by any layer. */
export const tilesAnnotated = (bundle) => {
  if (Number.isInteger(bundle.manifest.tiles_with_objects)) return bundle.manifest.tiles_with_objects;
  const tiles = [bundle.canopies, bundle.interrows, bundle.waste, bundle.targets].flatMap((l) => props(l).map((p) => p.tile));
  const rowTiles = props(bundle.rows).flatMap((p) => Object.keys(p.tile_structures ?? {}));
  return new Set([...tiles, ...rowTiles].filter(Boolean)).size;
};

/** Inter-rows per distinct interrow_id, each counted under the cover of its largest piece. */
export const coverCounts = (interrows) => {
  const best = new Map();
  for (const f of interrows.features) {
    const { interrow_id: id, interrow_cover: cover, area_m2 } = f.properties;
    const area = isNum(area_m2) ? area_m2 : polygonArea(f.geometry);
    if (!best.has(id) || area > best.get(id).area) best.set(id, { cover, area });
  }
  return countBy([...best.values()], (b) => b.cover);
};

/** Route KPIs; speed_kmh from the bundle (newer bundles), else length / duration; no baseline → no savings KPI. */
export const routeSummary = (p) => {
  const length = p.length_m;
  const speed = isNum(p.speed_kmh) ? p.speed_kmh
    : isNum(p.duration_min) ? length / 1000 / (p.duration_min / 60) : DEFAULT_SPEED_KMH;
  const duration = isNum(p.duration_min) ? p.duration_min : (length / 1000 / speed) * 60;
  return {
    mock: false,
    length_m: r2(length),
    duration_min: r2(duration),
    baseline_length_m: r2(isNum(p.baseline_length_m) ? p.baseline_length_m : length),
    speed_kmh: r2(speed),
  };
};

const surveyInfo = (bundle) => {
  const m = bundle.manifest;
  return {
    id: m.survey_id,
    name: m.name,
    captured_at: m.captured_at,
    gsd_m: m.gsd_m,
    crs: m.crs,
    source: m.source,
    license: m.license,
    stage: m.stage,
    mock: false,
    tiles_total: TILES_TOTAL,
    tiles_annotated: tilesAnnotated(bundle),
    surveyed_area_ha: r4((TILES_TOTAL * TILE_M * TILE_M) / 1e4),
    ...Object.fromEntries(PROVENANCE.filter((k) => m[k] !== undefined && m[k] !== null).map((k) => [k, m[k]])),
  };
};

const totals = (bundle, survey, targetCount) => ({
  block_count: survey.block_count,
  row_count: survey.row_count,
  row_length_m: survey.row_length_m,
  canopy_count: bundle.canopies.features.length,
  canopy_area_m2: survey.canopy_area_m2,
  canopy_area_ha: survey.canopy_area_ha,
  interrow_count: new Set(props(bundle.interrows).map((p) => p.interrow_id)).size,
  interrow_area_m2: survey.interrow_area_m2,
  interrow_area_ha: survey.interrow_area_ha,
  plant_count: survey.plant_count,
  disrupted_rows: props(bundle.rows).filter(isDisrupted).length,
  waste_count: bundle.waste.features.length,
  target_count: targetCount,
});

const blockSummaries = (bundle, csvBlocks) => {
  const outline = new Map(bundle.blocks.features.map((f) =>
    [f.properties.vineyard_id, isNum(f.properties.area_m2) ? f.properties.area_m2 : polygonArea(f.geometry)]));
  const canopies = countBy(props(bundle.canopies), (p) => p.vineyard_id);
  const disrupted = countBy(props(bundle.rows).filter(isDisrupted), (p) => p.vineyard_id);
  return csvBlocks.map((l) => ({
    vineyard_id: l.vineyard_id,
    outline_area_m2: r2(outline.get(l.vineyard_id)),
    row_count: l.row_count,
    row_length_m: l.row_length_m,
    canopy_count: canopies[l.vineyard_id] ?? 0,
    canopy_area_m2: l.canopy_area_m2,
    interrow_area_m2: l.interrow_area_m2,
    plant_count: l.plant_count,
    disrupted_rows: disrupted[l.vineyard_id] ?? 0,
  }));
};

/** summary.json; targetCount = targets written (all, or only routed ones with --targets route). */
export const buildSummary = ({ bundle, targetCount, uatAreaHa }) => {
  const lines = bundle.csv.lines;
  return {
    survey: surveyInfo(bundle),
    uat: { ...UAT, area_ha: uatAreaHa },
    totals: totals(bundle, lines.find((l) => l.level === "survey"), targetCount),
    structure_counts: countBy(props(bundle.rows), (p) => p.row_structure),
    cover_counts: coverCounts(bundle.interrows),
    route: routeSummary(bundle.route.features[0].properties),
    blocks: blockSummaries(bundle, lines.filter((l) => l.level === "block")),
  };
};

/** rows.json: the 4326 rows layer's properties, natural sort by row_id. */
export const buildRowsJson = (rowsLayer) =>
  rowsLayer.features.map((f) => f.properties).sort((a, b) => naturalCompare(a.row_id, b.row_id));
