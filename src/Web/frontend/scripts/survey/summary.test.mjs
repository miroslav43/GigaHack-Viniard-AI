import assert from "node:assert/strict";
import test from "node:test";
import { copyBundle, editJson, load, MINI_BUNDLE, NEW } from "./fixtures/kit.mjs";
import { buildLayers } from "./layers.mjs";
import { buildRowsJson, buildSummary, coverCounts, routeSummary, UAT } from "./summary.mjs";

const UAT_AREA_HA = 2728.72;
const summarize = (dir, targets = "all") => {
  const { bundle } = load(dir);
  const { files } = buildLayers(bundle, { interrowTol: 0.0125, targets });
  return { bundle, files, summary: buildSummary({ bundle, targetCount: files["targets.geojson"].features.length, uatAreaHa: UAT_AREA_HA }) };
};
const CSV_TOTALS = ["block_count", "row_count", "row_length_m", "canopy_area_m2", "canopy_area_ha", "interrow_area_m2", "interrow_area_ha", "plant_count"];

test("summary.totals equal the measurements.csv survey line; counts come from the layers", () => {
  const { bundle, summary } = summarize(MINI_BUNDLE);
  const survey = bundle.csv.lines.find((l) => l.level === "survey");
  for (const key of CSV_TOTALS) assert.equal(summary.totals[key], survey[key], key);
  assert.deepEqual(summary.totals, {
    block_count: 2, row_count: 3, row_length_m: 66.44, canopy_count: 4, canopy_area_m2: 2.1, canopy_area_ha: 0.0002,
    interrow_count: 2, interrow_area_m2: 573, interrow_area_ha: 0.0573, plant_count: 4, disrupted_rows: 1, waste_count: 1, target_count: 5,
  });
});

test("summary has exactly the SurveySummary shape (src/lib/types.ts)", () => {
  const { summary } = summarize(MINI_BUNDLE);
  assert.deepEqual(Object.keys(summary), ["survey", "uat", "totals", "structure_counts", "cover_counts", "route", "blocks"]);
  assert.deepEqual(Object.keys(summary.route), ["mock", "length_m", "duration_min", "baseline_length_m", "speed_kmh"]);
  assert.deepEqual(Object.keys(summary.blocks[0]), ["vineyard_id", "outline_area_m2", "row_count", "row_length_m", "canopy_count",
    "canopy_area_m2", "interrow_area_m2", "plant_count", "disrupted_rows"]);
  assert.deepEqual(summary.uat, { ...UAT, area_ha: UAT_AREA_HA });
});

test("old schema survey block: manifest fields, mock false, 311 tiles, 81.5268 ha, tiles_annotated from the layers", () => {
  const { summary } = summarize(MINI_BUNDLE);
  assert.deepEqual(summary.survey, {
    id: "siret3", name: "Sireț3", captured_at: "2025-05-20", gsd_m: 0.025, crs: "EPSG:32635", source: "3DATA COLLECT / OpenAerialMap",
    license: "CC BY 4.0", stage: "model", mock: false, tiles_total: 311, tiles_annotated: 2, surveyed_area_ha: 81.5268,
    run_id: "post-mini", generated_at: "2026-09-26T04:37:13+03:00", pipeline_version: "0000000000000000000000000000000000000000",
  });
});

test("new schema survey block: model_version and tiles_with_objects from the manifest", (t) => {
  const { summary } = summarize(copyBundle(t, { schema: "new" }));
  assert.equal(summary.survey.model_version, NEW.modelVersion);
  assert.equal(summary.survey.tiles_annotated, NEW.tilesWithObjects);
});

test("route: old schema derives speed from length / duration; new schema takes speed_kmh; no baseline → length", (t) => {
  const { bundle, summary } = summarize(MINI_BUNDLE);
  const p = bundle.route.features[0].properties;
  assert.equal(summary.route.speed_kmh, 4);
  assert.equal(summary.route.length_m, Math.round(p.length_m * 100) / 100);
  assert.equal(summary.route.baseline_length_m, 400);
  assert.equal(summary.route.mock, false);
  assert.equal(summarize(copyBundle(t, { schema: "new" })).summary.route.speed_kmh, NEW.speedKmh);
  assert.deepEqual(routeSummary({ length_m: 1000, duration_min: 15, baseline_length_m: null }),
    { mock: false, length_m: 1000, duration_min: 15, baseline_length_m: 1000, speed_kmh: 4 });
  assert.deepEqual(routeSummary({ length_m: 1000 }), { mock: false, length_m: 1000, duration_min: 15, baseline_length_m: 1000, speed_kmh: 4 });
});

test("structure_counts from rows; cover_counts per interrow_id by the cover of the largest piece", (t) => {
  const { summary } = summarize(MINI_BUNDLE);
  assert.deepEqual(summary.structure_counts, { regular: 2, disrupted: 1 });
  assert.deepEqual(summary.cover_counts, { bare_soil: 1, vegetation: 1 },
    "V01-I001 = bare_soil 280.6 m² + mixed 118.4 m²; V02-I001 = two overlapping vegetation pieces in one tile");
  const dir = copyBundle(t);
  editJson(dir, "interrows.geojson", (fc) => ({ ...fc, features: fc.features.map((f, i) => (i === 1 ? { ...f, properties: { ...f.properties, area_m2: 999 } } : f)) }));
  assert.deepEqual(coverCounts(load(dir).bundle.interrows), { mixed: 1, vegetation: 1 });
});

test("blocks[]: CSV block lines + outline area + canopy and disrupted counts per vineyard_id", () => {
  const { summary } = summarize(MINI_BUNDLE);
  assert.deepEqual(summary.blocks, [
    { vineyard_id: "V01", outline_area_m2: 1200, row_count: 2, row_length_m: 46.44, canopy_count: 3, canopy_area_m2: 1.1,
      interrow_area_m2: 399, plant_count: 3, disrupted_rows: 1 },
    { vineyard_id: "V02", outline_area_m2: 510, row_count: 1, row_length_m: 20, canopy_count: 1, canopy_area_m2: 1,
      interrow_area_m2: 174, plant_count: 1, disrupted_rows: 0 },
  ]);
});

test("target_count follows --targets (the file the site reads)", () => {
  assert.equal(summarize(MINI_BUNDLE, "route").summary.totals.target_count, 2);
});

test("rows.json: RowRecord per row, natural sort by row_id, CSV lengths", () => {
  const { files } = summarize(MINI_BUNDLE);
  const rows = buildRowsJson(files["rows.geojson"]);
  assert.deepEqual(rows.map((r) => r.row_id), ["V01-R002", "V01-R010", "V02-R001"]);
  assert.deepEqual(rows[1], { row_id: "V01-R010", vineyard_id: "V01", row_structure: "regular", length_m: 16.44, plant_count: 1,
    max_gap_m: 0, tile_structures: { siret3_r018_c010: "regular" } });
});
