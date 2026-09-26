// End-to-end: runs scripts/build-survey.mjs on the fixture (needs public/data/uats.json from `pnpm data`).
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { copyBundle, editJson, editTile, FRONTEND, MINI_BUNDLE, TILE_A, TILE_B, tempDir } from "./fixtures/kit.mjs";

const SCRIPT = path.join(FRONTEND, "scripts", "build-survey.mjs");
const NEEDS_DATA = !fs.existsSync(path.join(FRONTEND, "public", "data", "uats.json")) && "public/data/uats.json missing (run pnpm data)";
const OUTPUTS = ["blocks.geojson", "canopies.geojson", "interrows.geojson", "measurements.csv", "route.geojson", "route.gpx",
  "route_EPSG32635.geojson", "rows.geojson", "rows.json", "summary.json", "targets.geojson", "waste.geojson"];
// web bundle v3 (tiles.geojson + masks/) adds the tile layer and the mask folder
const OUTPUTS_WITH_TILES = [...OUTPUTS, "masks", "tiles.geojson"].sort();

const run = (args) => spawnSync(process.execPath, [SCRIPT, ...args], { cwd: FRONTEND, encoding: "utf8" });
const readOut = (dir, file) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));

// old schema = an older bundle (no tiles.geojson / masks); new schema = enriched layers + the tile layer and masks
for (const [schema, tiles] of [["old", false], ["new", true]]) {
  test(`${schema} schema: writes the survey files${tiles ? " + tiles and masks" : ""}; CSV and 32635 route are byte copies`, { skip: NEEDS_DATA }, (t) => {
    const bundle = copyBundle(t, { schema, tiles });
    const out = tempDir(t);
    const res = run(["--survey", "siret3", "--bundle", bundle, "--out", out]);
    assert.equal(res.status, 0, res.stderr);
    assert.match(res.stdout, /siret3 \(model, run post-mini\): 2 blocks, 3 rows \(66\.44 m\)/);
    assert.match(res.stdout, tiles
      ? /tiles: 311 \(2 vineyard, 309 no_vineyard, 2 to complete in Marcaj\), 2 vegetation masks/
      : /tiles: none in this bundle/);
    const dir = path.join(out, "siret3");
    assert.deepEqual(fs.readdirSync(out), ["siret3"], "no .tmp / .old left");
    assert.deepEqual(fs.readdirSync(dir).sort(), tiles ? OUTPUTS_WITH_TILES : OUTPUTS);
    assert.deepEqual(readOut(dir, "summary.json").tiles, tiles ? { total: 311, vineyard: 2, no_vineyard: 309, to_complete: 2 } : undefined);
    assert.ok(fs.readFileSync(path.join(dir, "measurements.csv")).equals(fs.readFileSync(path.join(bundle, "measurements.csv"))));
    assert.ok(fs.readFileSync(path.join(dir, "route_EPSG32635.geojson")).equals(fs.readFileSync(path.join(bundle, "route.geojson"))));
    assert.equal(readOut(dir, "summary.json").survey.mock, false);
    const targets = readOut(dir, "targets.geojson").features.map((f) => f.properties);
    assert.deepEqual(targets.map((p) => p.route_order), [1, 2, null, null, null]);
    assert.deepEqual(targets.map((p) => p.vineyard_id), ["V01", "V02", "V01", "V01", null], "waste target without a block kept");
  });
}

test("--targets route writes only routed targets and counts them in the summary", { skip: NEEDS_DATA }, (t) => {
  const out = tempDir(t);
  const res = run(["--bundle", MINI_BUNDLE, "--out", out, "--targets", "route"]);
  assert.equal(res.status, 0, res.stderr);
  assert.equal(readOut(path.join(out, "siret3"), "targets.geojson").features.length, 2);
  assert.equal(readOut(path.join(out, "siret3"), "summary.json").totals.target_count, 2);
  assert.match(res.stdout, /2 targets \(routed only, of 5\)/);
});

test("--check validates without writing", { skip: NEEDS_DATA }, (t) => {
  const out = tempDir(t);
  const res = run(["--bundle", MINI_BUNDLE, "--out", out, "--check"]);
  assert.equal(res.status, 0, res.stderr);
  assert.match(res.stdout, /is a valid "siret3" bundle \(0 warnings\); nothing written/);
  assert.deepEqual(fs.readdirSync(out), []);
});

test("an invalid bundle exits 1 naming the file and the problem, and keeps the published survey", { skip: NEEDS_DATA }, (t) => {
  const out = tempDir(t);
  assert.equal(run(["--bundle", MINI_BUNDLE, "--out", out]).status, 0);
  const bundle = copyBundle(t);
  editJson(bundle, "waste.geojson", (fc) => ({ ...fc, crs: { type: "name", properties: { name: "urn:ogc:def:crs:EPSG::4326" } } }));
  const res = run(["--bundle", bundle, "--out", out]);
  assert.equal(res.status, 1);
  assert.match(res.stderr, /✖ build-survey: bundle .* is invalid \(1 problem\):\n {2}- waste\.geojson: crs is "urn:ogc:def:crs:EPSG::4326"/);
  assert.deepEqual(fs.readdirSync(path.join(out, "siret3")).sort(), OUTPUTS_WITH_TILES);
});

test("masks: palette PNGs of the has_mask tiles and an index with their image corners", { skip: NEEDS_DATA }, (t) => {
  const out = tempDir(t);
  assert.equal(run(["--bundle", MINI_BUNDLE, "--out", out]).status, 0);
  const dir = path.join(out, "siret3");
  assert.deepEqual(fs.readdirSync(path.join(dir, "masks")).sort(), ["index.json", `${TILE_A}.png`, `${TILE_B}.png`]);
  const index = readOut(dir, "masks/index.json");
  assert.deepEqual(Object.keys(index), [TILE_A, TILE_B]);
  const tile = readOut(dir, "tiles.geojson").features.find((f) => f.properties.tile === TILE_A);
  const [sw, se, ne, nw] = tile.geometry.coordinates[0]; // CCW from the south-west corner
  assert.deepEqual(index[TILE_A], [nw, ne, se, sw], "TL, TR, BR, BL = the footprint corners");
  const png = fs.readFileSync(path.join(dir, "masks", `${TILE_A}.png`));
  assert.deepEqual([png[24], png[25]], [1, 3], "1-bit palette PNG");
});

test("a has_mask tile without its PNG exits 1 naming masks/", { skip: NEEDS_DATA }, (t) => {
  const bundle = copyBundle(t);
  fs.rmSync(path.join(bundle, "masks", `${TILE_B}.png`));
  const res = run(["--bundle", bundle, "--check"]);
  assert.equal(res.status, 1);
  assert.match(res.stderr, new RegExp(`- masks: missing PNG for 1 has_mask tile \\(${TILE_B}\\)`));
  editTile(bundle, TILE_B, (p) => ({ ...p, has_mask: false }));
  const ok = run(["--bundle", bundle, "--check"]);
  assert.equal(ok.status, 0, ok.stderr);
  assert.match(ok.stderr, /! manifest\.json: masks\.n is 2, but has_mask is true on 1 tile\n/);
  editTile(bundle, TILE_A, (p) => ({ ...p, has_mask: false }));
  const extra = run(["--bundle", bundle, "--check"]);
  assert.equal(extra.status, 0, extra.stderr);
  assert.match(extra.stderr, new RegExp(`! masks: 1 PNG without has_mask in tiles\\.geojson, ignored \\(${TILE_A}\\)`));
});

test("bad arguments exit 1 with a clear message", (t) => {
  const out = tempDir(t);
  for (const [args, message] of [
    [["--survey", "../etc"], /--survey must match/],
    [["--interrow-tol=-1"], /--interrow-tol must be a number of metres >= 0/],
    [["--interrow-tol", "abc"], /--interrow-tol must be a number of metres >= 0 \(got "abc"\)/],
    [["--targets", "some"], /--targets must be all\|route/],
    [["--bogus"], /Unknown option '--bogus'[\s\S]*usage: node scripts\/build-survey\.mjs/],
  ]) {
    const res = run([...args, "--out", out]);
    assert.equal(res.status, 1, args.join(" "));
    assert.match(res.stderr, message);
  }
  const missing = run(["--bundle", path.join(out, "none"), "--out", out]);
  assert.equal(missing.status, 1);
  assert.match(missing.stderr, NEEDS_DATA ? /uats\.json is missing — run pnpm data first/ : /bundle directory does not exist/);
});
