// End-to-end: runs scripts/build-survey.mjs on the fixture (needs public/data/uats.json from `pnpm data`).
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { copyBundle, editJson, FRONTEND, MINI_BUNDLE, tempDir } from "./fixtures/kit.mjs";

const SCRIPT = path.join(FRONTEND, "scripts", "build-survey.mjs");
const NEEDS_DATA = !fs.existsSync(path.join(FRONTEND, "public", "data", "uats.json")) && "public/data/uats.json missing (run pnpm data)";
const OUTPUTS = ["blocks.geojson", "canopies.geojson", "interrows.geojson", "measurements.csv", "route.geojson", "route.gpx",
  "route_EPSG32635.geojson", "rows.geojson", "rows.json", "summary.json", "targets.geojson", "waste.geojson"];

const run = (args) => spawnSync(process.execPath, [SCRIPT, ...args], { cwd: FRONTEND, encoding: "utf8" });
const readOut = (dir, file) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));

for (const schema of ["old", "new"]) {
  test(`${schema} schema: writes the 12 survey files; CSV and 32635 route are byte copies`, { skip: NEEDS_DATA }, (t) => {
    const bundle = copyBundle(t, { schema });
    const out = tempDir(t);
    const res = run(["--survey", "siret3", "--bundle", bundle, "--out", out]);
    assert.equal(res.status, 0, res.stderr);
    assert.match(res.stdout, /siret3 \(model, run post-mini\): 2 blocks, 3 rows \(66\.44 m\)/);
    const dir = path.join(out, "siret3");
    assert.deepEqual(fs.readdirSync(out), ["siret3"], "no .tmp / .old left");
    assert.deepEqual(fs.readdirSync(dir).sort(), OUTPUTS);
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
  assert.deepEqual(fs.readdirSync(path.join(out, "siret3")).sort(), OUTPUTS);
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
