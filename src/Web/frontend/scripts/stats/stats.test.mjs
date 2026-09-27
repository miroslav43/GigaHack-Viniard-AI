import assert from "node:assert/strict";
import test from "node:test";
import { buildStats, ROUTE_OUTSIDE_LIMIT } from "./build.mjs";
import { HEALTH, healthStats } from "./health.mjs";
import { farmStats, roadStats, tileStats } from "./land.mjs";
import { orientationRose, rowAxisDeg, rowStats } from "./rows.mjs";
import { damageByBlock, targetStats } from "./targets.mjs";
import { histogram, quantile } from "./util.mjs";

const fc = (features) => ({ type: "FeatureCollection", features });
const point = (properties) => ({ type: "Feature", geometry: { type: "Point", coordinates: [28.6, 47.1] }, properties });
const line = (coordinates, properties = {}) => ({ type: "Feature", geometry: { type: "LineString", coordinates }, properties });

const TARGETS = fc([
  point({ target_id: "T1", type: "gap", kind: "row_gap", vineyard_id: "V01", gap_length_m: 4, priority: 2, route_order: 1, skip_reason: null }),
  point({ target_id: "T2", type: "missing", kind: "missing_plant", vineyard_id: "V01", gap_length_m: null, priority: 3, route_order: null, skip_reason: "disconnected" }),
  point({ target_id: "T3", type: "missing", kind: "missing_plant", vineyard_id: "V02", priority: 3, route_order: 2, skip_reason: null }),
  point({ target_id: "T4", type: "waste", kind: "waste", vineyard_id: "V02", priority: 1, route_order: null, skip_reason: "too_far" }),
]);

test("histogram: left-closed bins, last bin open-ended, values below the first edge go to the first bin", () => {
  assert.deepEqual(histogram([-1, 0, 0.5, 1, 9, 10, 500], [0, 1, 10]).counts, [3, 2, 2]);
  assert.deepEqual(histogram([1, 5], [0, 2], [10, 20]).counts, [10, 20]);
  assert.equal(quantile([], 0.5), 0);
  assert.equal(quantile([3, 1, 2], 0.5), 2);
  assert.equal(quantile([0, 10], 0.95), 9.5);
});

test("targets: counts by type / kind / priority / skip reason, route split and summed gap", () => {
  const s = targetStats(TARGETS);
  assert.equal(s.total, 4);
  assert.deepEqual(s.by_type, { gap: 1, missing: 2, waste: 1 });
  assert.deepEqual(s.by_kind, { row_gap: 1, missing_plant: 2, waste: 1 });
  assert.deepEqual(s.by_priority, { 2: 1, 3: 2, 1: 1 });
  assert.deepEqual(s.skip_reasons, { disconnected: 1, too_far: 1 });
  assert.equal(s.on_route, 2);
  assert.equal(s.off_route, 2);
  assert.equal(s.gap_total_m, 4);
});

test("targets: kind / priority / skip_reason sections are null on a survey without them (the mock)", () => {
  const s = targetStats(fc([point({ type: "gap", vineyard_id: "V01", gap_length_m: 2, route_order: 1 })]));
  assert.equal(s.by_kind, null);
  assert.equal(s.by_priority, null);
  assert.equal(s.skip_reasons, null);
});

test("damageByBlock: missing plants by kind (or type on older bundles) and summed gap metres per block", () => {
  const d = damageByBlock(TARGETS);
  assert.deepEqual(d.get("V01"), { missing: 1, gap_m: 4 });
  assert.deepEqual(d.get("V02"), { missing: 1, gap_m: 0 });
  assert.deepEqual(damageByBlock(fc([point({ type: "missing", vineyard_id: "V09" })])).get("V09"), { missing: 1, gap_m: 0 });
});

test("rowAxisDeg: axis from north in [0, 180), direction-free, east–west rows do not cancel", () => {
  const near = (a, b) => assert.ok(Math.abs(a - b) < 0.5, `${a} ≉ ${b}`);
  near(rowAxisDeg({ type: "LineString", coordinates: [[0, 0], [0, 1]] }), 0);
  near(rowAxisDeg({ type: "LineString", coordinates: [[0, 1], [0, 0]] }), 0);
  near(rowAxisDeg({ type: "LineString", coordinates: [[0, 0], [1, 0]] }), 90);
  // zig-zag around east–west: 89° then 91° must stay ~90°, not collapse
  near(rowAxisDeg({ type: "LineString", coordinates: [[0, 0], [1, 0.0175], [2, 0]] }), 90);
  near(rowAxisDeg({ type: "MultiLineString", coordinates: [[[0, 0], [1, 1]], [[5, 5], [4, 4]]] }), 45);
  assert.equal(rowAxisDeg({ type: "LineString", coordinates: [[1, 1], [1, 1]] }), null);
});

test("orientationRose: length per 10° bin, dominant axis and the share of length in a 30° window around it", () => {
  const rose = orientationRose(
    fc([line([[0, 0], [0, 1]], { length_m: 60 }), line([[0, 0], [0.001, 1]], { length_m: 30 }), line([[0, 0], [1, 0]], { length_m: 10 })]),
  );
  assert.equal(rose.length_m.length, 18);
  assert.equal(rose.length_m[0], 90);
  assert.equal(rose.length_m[9], 10);
  assert.equal(rose.dominant_deg, 5);
  assert.equal(rose.dominant_share, 0.9);
  assert.equal(rose.window_deg, 30);
});

test("rowStats: medians, maxima and histograms of row length and max gap", () => {
  const rows = [
    { length_m: 5, max_gap_m: 0 },
    { length_m: 60, max_gap_m: 2.5 },
    { length_m: 400, max_gap_m: 30 },
  ];
  const s = rowStats(rows, fc([]));
  assert.equal(s.count, 3);
  assert.equal(s.length_median_m, 60);
  assert.equal(s.length_max_m, 400);
  assert.equal(s.gap_max_m, 30);
  assert.equal(s.length_hist.counts.at(-1), 1);
  assert.equal(s.gap_hist.counts[0], 1);
});

test("healthStats: 100 for a clean block, lower with damage, null under the minimum row length", () => {
  const blocks = [
    { vineyard_id: "A", row_count: 10, disrupted_rows: 0, row_length_m: 500 },
    { vineyard_id: "B", row_count: 10, disrupted_rows: 5, row_length_m: 500 },
    { vineyard_id: "C", row_count: 2, disrupted_rows: 2, row_length_m: HEALTH.min_row_length_m - 1 },
  ];
  const damage = new Map([["B", { missing: 10, gap_m: 50 }]]);
  const h = healthStats(blocks, damage, new Map([["A", "F01"]]));
  const by = Object.fromEntries(h.blocks.map((b) => [b.vineyard_id, b]));
  assert.equal(by.A.score, 100);
  assert.equal(by.A.farm_id, "F01");
  assert.equal(by.C.score, null);
  // B: d = 0.5; m and g are the largest of the eligible blocks, so both terms are near their full weight
  assert.ok(by.B.score < 100 - 100 * HEALTH.weights.disrupted * 0.5, `B = ${by.B.score}`);
  assert.ok(by.B.score >= 0);
  assert.equal(by.B.missing_per_100m, 2);
  assert.equal(by.B.gap_m_per_100m, 10);
});

test("farmStats: largest first, cumulative share, targets per hectare, commune share; null without farms", () => {
  const s = farmStats(
    [
      { farm_id: "F01", area_m2: 10000, n_blocks: 1, n_parcels: 3, target_count: 5 },
      { farm_id: "F02", area_m2: 30000, n_blocks: 2, n_parcels: 1, target_count: 30 },
    ],
    400,
  );
  assert.deepEqual(s.items.map((f) => f.farm_id), ["F02", "F01"]);
  assert.deepEqual(s.items.map((f) => f.cumulative_share), [0.75, 1]);
  assert.equal(s.items[0].targets_per_ha, 10);
  assert.equal(s.commune_share, 0.01);
  assert.equal(s.parcels_total, 4);
  assert.equal(farmStats(undefined, 400), null);
});

test("roadStats: metres per class and surface, unknown surface kept apart; null without roads", () => {
  const s = roadStats(
    fc([
      line([], { road_class: "public", surface: "paved", length_m: 100 }),
      line([], { road_class: "public", length_m: 50 }),
      line([], { road_class: "field", surface: "dirt", length_m: 20 }),
    ]),
  );
  assert.deepEqual(s.by_class.public, { total_m: 150, by_surface: { paved: 100, unknown: 50 } });
  assert.equal(s.by_class.field.total_m, 20);
  assert.equal(roadStats(null), null);
});

test("tileStats: grid position from the tile id, sorted, grid size and mean vegetation; null without tiles", () => {
  const tile = (id, status, veg, review = null) => ({ type: "Feature", geometry: null, properties: { tile: id, status, veg_frac: veg, nodata_frac: 0, review_status: review } });
  const s = tileStats(fc([tile("s_r001_c002", "vineyard", 0.5, "missed"), tile("s_r000_c000", "no_vineyard", 0.1)]));
  assert.deepEqual(s.items.map((t) => [t.r, t.c, t.to_complete]), [[0, 0, false], [1, 2, true]]);
  assert.equal(s.rows, 2);
  assert.equal(s.cols, 3);
  assert.equal(s.veg_frac_mean, 0.3);
  assert.equal(tileStats(null), null);
});

test("buildStats: every section present, route saving and the 2% outside limit", () => {
  const summary = {
    survey: { id: "t" },
    uat: { area_ha: 100 },
    totals: { canopy_area_m2: 10, interrow_area_m2: 90, row_length_m: 45, canopy_count: 5 },
    blocks: [{ vineyard_id: "V01", row_count: 1, disrupted_rows: 0, row_length_m: 45 }],
    route: { length_m: 250, duration_min: 3.75, baseline_length_m: 1000, speed_kmh: 4 },
  };
  const s = buildStats({
    summary,
    rows: [{ length_m: 45, max_gap_m: 1 }],
    rowsGeo: fc([line([[0, 0], [0, 1]], { length_m: 45 })]),
    targets: TARGETS,
    interrows: fc([point({ interrow_cover: "bare_soil", area_m2: 60 }), point({ interrow_cover: "mixed", area_m2: 30 })]),
    route: fc([line([[0, 0], [0, 1]], { outside_share: 0.013 })]),
  });
  assert.deepEqual(Object.keys(s), ["version", "survey_id", "targets", "rows", "structure", "health", "farms", "roads", "tiles", "route"]);
  assert.equal(s.structure.canopy_cover_share, 0.1);
  assert.equal(s.structure.interrow_width_m, 2);
  assert.deepEqual(s.structure.interrow_area_by_cover, { bare_soil: 60, mixed: 30 });
  assert.equal(s.route.saved_m, 750);
  assert.equal(s.route.saved_share, 0.75);
  assert.equal(s.route.saved_min, 11.3);
  assert.equal(s.route.outside_share, 0.013);
  assert.equal(s.route.outside_limit, ROUTE_OUTSIDE_LIMIT);
  assert.equal(s.farms, null);
});
