// Synthetic relief (scripts/build-terrain.mjs): terrarium round-trip, tile coverage math, rasterisation, and the
// script end to end on a tiny fixture (two canopies near START, written to a temp dir).
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { inRange, latFromMercatorY, mercatorX, mercatorY, metresPerPixel, rangeSize, tileRange } from "./mercator.mjs";
import { coverageGrid, downsample, gaussianBlur, maxOf, reliefProfile } from "./raster.mjs";
import { buildRelief, canopyPolygons, reliefBounds, reliefTileRanges } from "./relief.mjs";
import { decodeHeight, encodeHeight, encodeTile, ENCODING_STEP_M } from "./terrarium.mjs";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SCRIPT = path.join(FRONTEND, "scripts", "build-terrain.mjs");
const START = [28.7073776, 47.1230335]; // src/Web/CLAUDE.md §6.6

const tempDir = (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "build-terrain-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
};
// a w × h metre rectangle south-east of `origin`, as a closed 4326 ring
const rect = ([lon, lat], dxM, dyM, w, h) => {
  const mLon = 111_320 * Math.cos((lat * Math.PI) / 180), mLat = 110_574;
  const [x0, y0] = [lon + dxM / mLon, lat - dyM / mLat];
  const [x1, y1] = [x0 + w / mLon, y0 - h / mLat];
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
};
const fixture = () => ({
  type: "FeatureCollection",
  features: [
    { type: "Feature", properties: { canopy_id: "a" }, geometry: { type: "Polygon", coordinates: [rect(START, 0, 0, 0.8, 3)] } },
    { type: "Feature", properties: { canopy_id: "b" }, geometry: { type: "Polygon", coordinates: [rect(START, 2.5, 0, 0.8, 3)] } },
    { type: "Feature", properties: { canopy_id: "c" }, geometry: null },
  ],
});

test("terrarium: heights round-trip within half a step (1/512 m), as MapLibre unpacks them", () => {
  for (const h of [0, 0.001, 0.5, 1.1, 1.0999, 2, 17.3, -3.25, 300.123]) {
    const [r, g, b] = encodeHeight(h);
    assert.ok([r, g, b].every((v) => Number.isInteger(v) && v >= 0 && v <= 255));
    assert.ok(Math.abs(decodeHeight(r, g, b) - h) <= ENCODING_STEP_M / 2 + 1e-9, `h=${h}`);
  }
  assert.deepEqual(encodeHeight(0), [128, 0, 0], "ground = the terrarium sea level");
  assert.deepEqual([...encodeTile(Float32Array.from([0, 1.1]))], [128, 0, 0, ...encodeHeight(1.1)]);
});

test("mercator helpers match MapLibre and invert", () => {
  assert.equal(mercatorX(-180), 0);
  assert.equal(mercatorX(180), 1);
  assert.ok(Math.abs(mercatorY(0) - 0.5) < 1e-12);
  assert.ok(Math.abs(latFromMercatorY(mercatorY(START[1])) - START[1]) < 1e-9);
  // ~0.2 m/px at z19 for 256 px tiles at 47° N
  assert.ok(Math.abs(metresPerPixel(47.12, 19, 256) - 0.2) < 0.01);
});

test("tile ranges: the tiles MapLibre requests inside bounds (max exclusive), halving per zoom", () => {
  const bounds = reliefBounds([[rect(START, 0, 0, 100, 100)]], 2);
  const { ranges, total } = reliefTileRanges(bounds, 14, 19);
  assert.equal(total, Object.values(ranges).reduce((s, r) => s + rangeSize(r), 0));
  for (let z = 15; z <= 19; z++) {
    const [child, parent] = [ranges[z], ranges[z - 1]];
    assert.equal(Math.floor(child.minX / 2), parent.minX);
    assert.equal(Math.floor((child.maxX - 1) / 2), parent.maxX - 1);
  }
  // a point exactly on the north-west corner falls in the first tile
  const r = tileRange([START[0], START[1] - 0.001, START[0] + 0.001, START[1]], 19);
  const x = Math.floor(mercatorX(START[0]) * 2 ** 19), y = Math.floor(mercatorY(START[1]) * 2 ** 19);
  assert.ok(inRange(r, x, y));
  assert.ok(!inRange(r, r.maxX, r.minY));
});

test("coverage is exact for an axis-aligned square and respects holes", () => {
  const square = [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]];
  const g = coverageGrid([square], 4, 4);
  assert.deepEqual([...g], [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
  const half = coverageGrid([[[[0.5, 0], [1, 0], [1, 1], [0.5, 1], [0.5, 0]]]], 2, 1);
  assert.deepEqual([...half], [0.5, 0]);
  const withHole = coverageGrid([[[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]]], 4, 4);
  assert.equal(withHole.reduce((s, v) => s + v, 0), 12);
});

test("blur keeps the volume, the profile is flat-topped, downsampling averages", () => {
  const g = new Float32Array(21 * 21);
  g[10 * 21 + 10] = 1;
  const b = gaussianBlur(g, 21, 21, 1.5);
  assert.ok(Math.abs(b.reduce((s, v) => s + v, 0) - 1) < 1e-5);
  assert.ok(maxOf(b) < 1);
  assert.equal(reliefProfile(0, 0.6), 0);
  assert.equal(reliefProfile(0.6, 0.6), 1);
  assert.equal(reliefProfile(1, 0.6), 1);
  const child = Float32Array.from({ length: 16 }, () => 1);
  const parent = downsample([child, null, null, null], 4);
  assert.deepEqual([...parent], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(downsample([null, null, null, null], 4), null);
});

test("buildRelief writes every tile in bounds, relief only near the canopies, at most the chosen height", async () => {
  const { polygons, skipped } = canopyPolygons(fixture());
  assert.equal(polygons.length, 2);
  assert.equal(skipped, 1);
  const written = new Map();
  const res = await buildRelief(polygons, { heightM: 1.1, minZoom: 17, maxZoom: 19 }, async (z, x, y, h) => written.set(`${z}/${x}/${y}`, h));
  const { ranges, total } = reliefTileRanges(res.bounds, 17, 19);
  assert.equal(res.tiles, total);
  assert.equal(written.size, total, "no tile written twice");
  for (let z = 17; z <= 19; z++)
    for (let x = ranges[z].minX; x < ranges[z].maxX; x++) for (let y = ranges[z].minY; y < ranges[z].maxY; y++) assert.ok(written.has(`${z}/${x}/${y}`));
  const leaves = [...written.entries()].filter(([k, h]) => k.startsWith("19/") && h);
  assert.ok(leaves.length >= 1);
  const peak = Math.max(...leaves.map(([, h]) => maxOf(h)));
  assert.ok(peak > 0.9 && peak <= 1.1 + 1e-6, `peak ${peak}`);
});

test("build-terrain.mjs publishes terrain/ with terrain.json and decodable PNG tiles", async (t) => {
  const dataDir = tempDir(t);
  fs.mkdirSync(path.join(dataDir, "fix"));
  fs.writeFileSync(path.join(dataDir, "fix", "canopies.geojson"), JSON.stringify(fixture()));
  const res = spawnSync(process.execPath, [SCRIPT, "--survey", "fix", "--data-dir", dataDir, "--min-zoom", "18"], { encoding: "utf8" });
  assert.equal(res.status, 0, res.stderr);
  assert.match(res.stderr, /1 canopy feature\(s\) without a polygon geometry skipped/);
  assert.deepEqual(fs.readdirSync(path.join(dataDir, "fix")).sort(), ["canopies.geojson", "terrain"], "no .tmp / .old left");
  const meta = JSON.parse(fs.readFileSync(path.join(dataDir, "fix", "terrain", "terrain.json"), "utf8"));
  assert.equal(meta.encoding, "terrarium");
  assert.deepEqual([meta.minzoom, meta.maxzoom, meta.tile_size, meta.height_m], [18, 19, 256, 1.1]);
  assert.equal(meta.canopy_count, 2);
  let peak = 0, pngs = 0;
  for (const z of ["18", "19"]) for (const x of fs.readdirSync(path.join(dataDir, "fix", "terrain", z))) {
    for (const f of fs.readdirSync(path.join(dataDir, "fix", "terrain", z, x))) {
      const { data, info } = await sharp(path.join(dataDir, "fix", "terrain", z, x, f)).raw().toBuffer({ resolveWithObject: true });
      assert.deepEqual([info.width, info.height, info.channels], [256, 256, 3]);
      for (let i = 0; i < data.length; i += 3) peak = Math.max(peak, decodeHeight(data[i], data[i + 1], data[i + 2]));
      pngs++;
    }
  }
  assert.equal(pngs, meta.tile_count);
  assert.ok(peak > 0.9 && peak <= 1.1 + ENCODING_STEP_M, `peak ${peak}`);
});

test("bad arguments and a missing survey exit 1 with a clear message", (t) => {
  const dataDir = tempDir(t);
  for (const [args, message] of [
    [["--survey", "../etc"], /--survey must match/],
    [["--height", "9"], /--height must be a number in \[0.1, 5\]/],
    [["--min-zoom", "19", "--max-zoom", "18"], /--min-zoom \(19\) must not exceed --max-zoom \(18\)/],
    [["--survey", "none"], /canopies\.geojson is missing/],
    [["--bogus"], /Unknown option '--bogus'[\s\S]*usage: node scripts\/build-terrain\.mjs/],
  ]) {
    const res = spawnSync(process.execPath, [SCRIPT, ...args, "--data-dir", dataDir], { encoding: "utf8" });
    assert.equal(res.status, 1, args.join(" "));
    assert.match(res.stderr, message);
  }
});
