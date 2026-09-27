// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  errorStatus,
  isJobId,
  isTiffMagic,
  isTiffName,
  jobFileType,
  orderedTimings,
  parseResult,
  parseStatus,
  progressOf,
  queuedStatus,
  type JobStatus,
} from "./contract.ts";

const running = (stage_index: number): JobStatus => ({
  state: "running",
  stage: "rows_detect",
  stage_index,
  n_stages: 11,
  error: null,
  updated_at: "2026-09-27T04:27:00+00:00",
});

const RESULT = {
  tile_id: "siret3_r021_c012",
  corners: [
    [28.708688, 47.1220821],
    [28.7093627, 47.122072],
    [28.7093479, 47.1216115],
    [28.7086732, 47.1216216],
  ],
  counts: { canopies: 395, rows: 25, interrows: 24, waste: 0 },
  canopy_area_m2: 236.34,
  interrow_area_m2: 2059.41,
  row_length_m: 909.68,
  timings_s: { ingest: 0.3, waste: 10.05, bogus: "x" },
  total_s: 17.17,
  model_version: "pipe@0.1.0+83ecdd3;nn=none;waste=none",
  waste_verify_level: "L1",
};

test("job ids: random UUIDs and hand-made sample folders, nothing that walks the file system", () => {
  assert.ok(isJobId("3f2b8c1e-9d4a-4c7e-8b1a-2f6d9e0c4a7b"));
  assert.ok(isJobId("sample-job"));
  for (const bad of ["", "abc", "../etc", "a/b/c/d/e/f", "Sample-Job", "job id!", "x".repeat(41), "..%2f..%2f"]) {
    assert.equal(isJobId(bad), false, bad);
  }
});

test("only the whitelisted outputs of a job folder are served, with their content type", () => {
  assert.equal(jobFileType("preview.jpg"), "image/jpeg");
  assert.equal(jobFileType("veg_mask.png"), "image/png");
  for (const f of ["canopies", "rows", "interrows", "waste"]) assert.equal(jobFileType(`${f}.geojson`), "application/geo+json");
  for (const bad of ["input.tif", "status.json", "result.json", "log.txt", "overrides.yaml", "../status.json", "toString", "__proto__", "hasOwnProperty"]) {
    assert.equal(jobFileType(bad), null, bad);
  }
});

test("uploads: .tif/.tiff names and the TIFF / BigTIFF magic numbers", () => {
  assert.ok(isTiffName("siret3_r021_c012.tif"));
  assert.ok(isTiffName("TILE.TIFF"));
  assert.equal(isTiffName("tile.tif.zip"), false);
  assert.equal(isTiffName("tile.jpg"), false);
  assert.ok(isTiffMagic(new Uint8Array([0x49, 0x49, 0x2a, 0x00])));
  assert.ok(isTiffMagic(new Uint8Array([0x4d, 0x4d, 0x00, 0x2a])));
  assert.ok(isTiffMagic(new Uint8Array([0x49, 0x49, 0x2b, 0x00])));
  assert.equal(isTiffMagic(new Uint8Array([0xff, 0xd8, 0xff, 0xe0])), false);
  assert.equal(isTiffMagic(new Uint8Array([0x49, 0x49])), false);
});

test("status.json: states, defaults, and an unknown error code read as pipeline_failed", () => {
  assert.deepEqual(parseStatus(running(4)), running(4));
  assert.equal(parseStatus({ state: "exploded" }), null);
  assert.equal(parseStatus(null), null);
  assert.equal(parseStatus([1, 2]), null);
  const err = parseStatus({ state: "error", error: { code: "not_geotiff", message: "bad.tif: not a readable GeoTIFF" } });
  assert.deepEqual(err?.error, { code: "not_geotiff", message: "bad.tif: not a readable GeoTIFF" });
  assert.equal(err?.stage_index, 0);
  assert.equal(parseStatus({ state: "error", error: { code: "segfault" } })?.error?.code, "pipeline_failed");
});

test("progress: 1-based stage_index while running counts the stages already finished", () => {
  assert.equal(progressOf(queuedStatus(new Date())), 0);
  assert.equal(progressOf(running(1)), 0);
  assert.equal(progressOf(running(6)), 5 / 11);
  assert.equal(progressOf({ ...running(11), state: "done" }), 1);
  assert.equal(progressOf({ ...running(3), n_stages: 0 }), 0);
});

test("errorStatus keeps the stage the job stopped at", () => {
  const now = new Date("2026-09-27T05:00:00Z");
  const s = errorStatus(running(4), { code: "pipeline_failed", message: "exit 1" }, now);
  assert.equal(s.state, "error");
  assert.equal(s.stage, "rows_detect");
  assert.equal(s.stage_index, 4);
  assert.equal(s.updated_at, now.toISOString());
  assert.equal(errorStatus(null, { code: "busy", message: "" }, now).stage, null);
});

test("result.json: the sample job parses; timings drop non-numbers; missing fields reject it", () => {
  const r = parseResult(RESULT);
  assert.ok(r);
  assert.equal(r.tile_id, "siret3_r021_c012");
  assert.equal(r.counts.canopies, 395);
  assert.deepEqual(r.timings_s, { ingest: 0.3, waste: 10.05 });
  assert.equal(parseResult({ ...RESULT, corners: RESULT.corners.slice(0, 3) }), null);
  assert.equal(parseResult({ ...RESULT, counts: { canopies: 1 } }), null);
  assert.equal(parseResult({ ...RESULT, total_s: "17" }), null);
  assert.equal(parseResult({ ...RESULT, tile_id: undefined }), null);
});

test("timings follow the pipeline order, unknown stages last", () => {
  const order = orderedTimings({ assemble: 0.6, blocks: 0.4, ingest: 0.3, zeta: 1, waste: 10, alpha: 2 }).map(([s]) => s);
  assert.deepEqual(order, ["ingest", "blocks", "waste", "assemble", "alpha", "zeta"]);
});
