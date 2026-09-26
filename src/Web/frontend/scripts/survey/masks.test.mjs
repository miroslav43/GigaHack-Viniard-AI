import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import sharp from "sharp";
import { copyBundle, editJson, load, MINI_BUNDLE, TILE_A, TILE_B, tempDir } from "./fixtures/kit.mjs";
import { buildMaskFiles, colourMask, maskedFeatures, validateMasks } from "./masks.mjs";
import { encodeBitPng, rgbaOf } from "./png.mjs";

const RGBA = [217, 70, 239, 140];
const decodeRgba = async (png) => sharp(png).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
/** Distinct RGBA values of a decoded image, as "r,g,b,a" strings (transparent pixels collapse to their alpha). */
const distinct = (data) => {
  const seen = new Set();
  for (let i = 0; i < data.length; i += 4) seen.add(data[i + 3] === 0 ? "transparent" : [...data.subarray(i, i + 4)].join(","));
  return [...seen].sort();
};
/** PNG chunk payloads by type. */
const chunks = (png) => {
  const out = {};
  for (let off = 8; off < png.length; ) {
    const len = png.readUInt32BE(off), type = png.toString("ascii", off + 4, off + 8);
    out[type] = [...png.subarray(off + 8, off + 8 + len)];
    off += 12 + len;
  }
  return out;
};

test("encodeBitPng: a 1-bit palette PNG, index 0 transparent, index 1 the exact colour; sharp decodes it", async () => {
  const [w, h] = [13, 3]; // a width that is not a multiple of 8
  const bits = Uint8Array.from({ length: w * h }, (_, i) => (i % w === 0 || i % w === 12 ? 1 : 0));
  const png = encodeBitPng(bits, w, h, [[0, 0, 0, 0], RGBA]);
  assert.deepEqual([png[24], png[25]], [1, 3], "bit depth 1, colour type 3");
  const c = chunks(png);
  assert.deepEqual(c.PLTE, [0, 0, 0, 217, 70, 239]);
  assert.deepEqual(c.tRNS, [0, 140]);
  const { data, info } = await decodeRgba(png);
  assert.deepEqual([info.width, info.height], [w, h]);
  assert.deepEqual(distinct(data), ["217,70,239,140", "transparent"]);
  for (let i = 0; i < w * h; i++) assert.equal(data[i * 4 + 3] > 0, bits[i] === 1, `pixel ${i}`);
});

test("encodeBitPng: greyscale 1-bit (as the pipeline writes masks), and a size mismatch throws", async () => {
  const png = encodeBitPng(Uint8Array.from([1, 0, 0, 1]), 2, 2);
  assert.deepEqual([png[24], png[25]], [1, 0]);
  const { data } = await sharp(png).extractChannel(0).raw().toBuffer({ resolveWithObject: true });
  assert.deepEqual([...data], [255, 0, 0, 255]);
  assert.throws(() => encodeBitPng(Uint8Array.from([1]), 2, 2), /1 pixels for 2×2/);
});

test("rgbaOf: theme hex + alpha → bytes; rejects anything else", () => {
  assert.deepEqual(rgbaOf("#D946EF", 0.55), [217, 70, 239, 140]);
  assert.throws(() => rgbaOf("#D946E", 0.5), /bad mask colour/);
  assert.throws(() => rgbaOf("#D946EF", 1.5), /bad mask colour/);
  assert.throws(() => rgbaOf(undefined, 0.5), /bad mask colour/);
});

test("colourMask keeps every vegetation pixel of the bundle mask and nothing else", async () => {
  const file = path.join(MINI_BUNDLE, "masks", `${TILE_A}.png`);
  const source = await sharp(file).extractChannel(0).raw().toBuffer();
  const { data, info } = await decodeRgba(await colourMask(file, RGBA));
  assert.deepEqual([info.width, info.height], [1024, 1024]);
  assert.deepEqual(distinct(data), ["217,70,239,140", "transparent"]);
  let ones = 0;
  for (let i = 0; i < source.length; i++) {
    assert.equal(data[i * 4 + 3] > 0, source[i] > 0);
    ones += source[i] > 0;
  }
  const veg = load(MINI_BUNDLE).bundle.tiles.features.find((f) => f.properties.tile === TILE_A).properties.veg_frac;
  assert.equal(Math.round((ones / source.length) * 1000) / 1000, veg, "tiles.geojson veg_frac matches the mask");
});

test("an 8-bit 0/1 mask (the other allowed encoding) reads the same as 0/255", async (t) => {
  const bits = Uint8Array.from({ length: 16 }, (_, i) => i % 3 === 0);
  const file = path.join(tempDir(t), "m.png");
  fs.writeFileSync(file, await sharp(Buffer.from(bits), { raw: { width: 4, height: 4, channels: 1 } }).toColourspace("b-w").png().toBuffer());
  const { data } = await decodeRgba(await colourMask(file, RGBA));
  for (let i = 0; i < 16; i++) assert.equal(data[i * 4 + 3] > 0, bits[i] === 1);
});

test("the fixture masks validate; buildMaskFiles writes a PNG per has_mask tile and the corner index", async () => {
  const { bundle } = load(MINI_BUNDLE);
  assert.deepEqual(await validateMasks(bundle), { errors: [], warnings: [] });
  assert.deepEqual(maskedFeatures(bundle.tiles).map((f) => f.properties.tile), [TILE_A, TILE_B]);
  const files = await buildMaskFiles(bundle, RGBA);
  assert.deepEqual(Object.keys(files), [`masks/${TILE_A}.png`, `masks/${TILE_B}.png`, "masks/index.json"]);
  assert.ok(Buffer.isBuffer(files[`masks/${TILE_A}.png`].buffer));
  assert.equal(files["masks/index.json"].json[TILE_B].length, 4);
});

test("an older bundle: no masks, nothing to build; stray masks/ without tiles.geojson is ignored with a warning", async (t) => {
  const dir = copyBundle(t, { tiles: false });
  const { bundle } = load(dir);
  assert.deepEqual(await validateMasks(bundle), { errors: [], warnings: [] });
  assert.deepEqual(await buildMaskFiles(bundle, RGBA), {});
  fs.cpSync(path.join(MINI_BUNDLE, "masks"), path.join(dir, "masks"), { recursive: true });
  assert.deepEqual((await validateMasks(load(dir).bundle)).warnings, [{ file: "masks", message: "ignored: the bundle has no tiles.geojson" }]);
});

test("rejects a mask of the wrong size, an unreadable PNG and a bad manifest block", async (t) => {
  const dir = copyBundle(t);
  fs.writeFileSync(path.join(dir, "masks", `${TILE_A}.png`), encodeBitPng(new Uint8Array(2048 * 2048), 2048, 2048));
  fs.writeFileSync(path.join(dir, "masks", `${TILE_B}.png`), "not a png");
  editJson(dir, "manifest.json", (m) => ({ ...m, masks: { ...m.masks, dir: "veg", px: 2048 } }));
  const { errors } = await validateMasks(load(dir).bundle);
  assert.deepEqual(errors.map((e) => e.file), ["manifest.json", "manifest.json", `masks/${TILE_A}.png`, `masks/${TILE_B}.png`]);
  assert.match(errors[0].message, /masks\.dir is "veg", expected "masks"/);
  assert.match(errors[1].message, /masks\.px is 2048, expected 1024/);
  assert.match(errors[2].message, /2048×2048 px, expected 1024×1024/);
  assert.match(errors[3].message, /not a readable PNG/);
});

test("warns when has_mask tiles ship without a manifest masks block", async (t) => {
  const dir = copyBundle(t);
  editJson(dir, "manifest.json", (m) => Object.fromEntries(Object.entries(m).filter(([k]) => k !== "masks")));
  assert.deepEqual((await validateMasks(load(dir).bundle)).warnings.map((w) => w.message), ["no \"masks\" block, but has_mask is true on 2 tiles"]);
});
