// masks/<tile>.png of the bundle (web bundle v3): the pipeline's vegetation mask per tile (tile_prep, Lab a*),
// 1024 px, non-zero = vegetation, pixel (0,0) = the tile's north-west corner. Published as
//   masks/<tile>.png   1-bit PNG, 2-entry palette: index 0 fully transparent, index 1 = theme vegMask colour + alpha
//   masks/index.json   { "<tile>": [[lon, lat] × 4] } in MapLibre image-source order TL, TR, BR, BL
// Only tiles whose tiles.geojson feature has has_mask = true are read; their ids passed TILE_ID_RE already.
import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";
import { MANIFEST_FILE, MASK_PX, MASKS_DIR } from "./contract.mjs";
import { encodeBitPng } from "./png.mjs";
import { imageCorners } from "./tiles.mjs";

const PNG_RE = /\.png$/i;
const BATCH = 16; // masks decoded at once (~1 MB each)
const SAMPLE = 3;
const problem = (file, message) => ({ file, message });
const sample = (ids) => `${ids.slice(0, SAMPLE).join(", ")}${ids.length > SAMPLE ? ", …" : ""}`;
const maskPath = (dir, tile) => path.join(dir, MASKS_DIR, `${tile}.png`);
const maskName = (tile) => `${MASKS_DIR}/${tile}.png`;

/** Features whose tile has a mask (has_mask = true), in bundle order. */
export const maskedFeatures = (tiles) => (tiles ? tiles.features.filter((f) => f.properties.has_mask === true) : []);

const pngsOnDisk = (dir) => {
  const masksDir = path.join(dir, MASKS_DIR);
  return fs.existsSync(masksDir) ? fs.readdirSync(masksDir).filter((f) => PNG_RE.test(f)).map((f) => f.replace(PNG_RE, "")) : [];
};

const onTiles = (n) => `has_mask is true on ${n} tile${n === 1 ? "" : "s"}`;
/** manifest.masks: { dir: "masks", px: 1024, n, source } — optional, but it must describe what ships. */
const manifestProblems = (block, expectedN) => {
  if (block === undefined || block === null)
    return { errors: [], warnings: expectedN ? [problem(MANIFEST_FILE, `no "masks" block, but ${onTiles(expectedN)}`)] : [] };
  if (typeof block !== "object" || Array.isArray(block)) return { errors: [problem(MANIFEST_FILE, "masks must be an object")], warnings: [] };
  const errors = [
    ...(block.dir !== MASKS_DIR ? [problem(MANIFEST_FILE, `masks.dir is ${JSON.stringify(block.dir)}, expected "${MASKS_DIR}"`)] : []),
    ...(block.px !== MASK_PX ? [problem(MANIFEST_FILE, `masks.px is ${JSON.stringify(block.px)}, expected ${MASK_PX}`)] : []),
  ];
  const warnings = block.n === expectedN ? [] : [problem(MANIFEST_FILE, `masks.n is ${JSON.stringify(block.n)}, but ${onTiles(expectedN)}`)];
  return { errors, warnings };
};

const headerProblems = async (dir, tile) => {
  try {
    const { width, height, channels } = await sharp(maskPath(dir, tile)).metadata();
    return {
      errors: width === MASK_PX && height === MASK_PX ? [] : [problem(maskName(tile), `${width}×${height} px, expected ${MASK_PX}×${MASK_PX}`)],
      warnings: channels === 1 ? [] : [problem(maskName(tile), `${channels} channels, expected 1 (the first one is used)`)],
    };
  } catch (err) {
    return { errors: [problem(maskName(tile), `not a readable PNG (${err.message})`)], warnings: [] };
  }
};

/**
 * Mask checks → { errors, warnings }: the manifest block, a PNG for every has_mask tile (readable, MASK_PX square),
 * PNGs without has_mask (ignored), masks without tiles.geojson (ignored). Reads headers only.
 */
export const validateMasks = async (bundle) => {
  const onDisk = pngsOnDisk(bundle.dir);
  if (!bundle.tiles) {
    const ignored = onDisk.length || bundle.manifest?.masks;
    return { errors: [], warnings: ignored ? [problem(MASKS_DIR, "ignored: the bundle has no tiles.geojson")] : [] };
  }
  const wanted = maskedFeatures(bundle.tiles).map((f) => f.properties.tile);
  const present = new Set(onDisk);
  const missing = wanted.filter((t) => !present.has(t));
  const wantedSet = new Set(wanted);
  const extra = onDisk.filter((t) => !wantedSet.has(t));
  const manifest = manifestProblems(bundle.manifest?.masks, wanted.length);
  const headers = await Promise.all(wanted.filter((t) => present.has(t)).map((t) => headerProblems(bundle.dir, t)));
  return {
    errors: [
      ...manifest.errors,
      ...(missing.length ? [problem(MASKS_DIR, `missing PNG for ${missing.length} has_mask tile${missing.length === 1 ? "" : "s"} (${sample(missing)})`)] : []),
      ...headers.flatMap((h) => h.errors),
    ],
    warnings: [
      ...manifest.warnings,
      ...headers.flatMap((h) => h.warnings),
      ...(extra.length ? [problem(MASKS_DIR, `${extra.length} PNG${extra.length === 1 ? "" : "s"} without has_mask in tiles.geojson, ignored (${sample(extra)})`)] : []),
    ],
  };
};

/** A bundle mask (any PNG; first channel, non-zero = vegetation) → the published 2-entry palette PNG. */
export const colourMask = async (file, rgba) => {
  const { data, info } = await sharp(file).extractChannel(0).raw().toBuffer({ resolveWithObject: true });
  return encodeBitPng(data, info.width, info.height, [[0, 0, 0, 0], rgba]);
};

const inBatches = async (items, fn) => {
  const out = [];
  for (let i = 0; i < items.length; i += BATCH) out.push(...(await Promise.all(items.slice(i, i + BATCH).map(fn))));
  return out;
};

/**
 * Published mask files, keyed by path under the survey folder: masks/<tile>.png ({ buffer }) and masks/index.json
 * ({ json }). rgba = [r, g, b, a] of the vegetation colour. Nothing when the bundle has no masked tiles.
 */
export const buildMaskFiles = async (bundle, rgba) => {
  const features = maskedFeatures(bundle.tiles);
  if (!features.length) return {};
  const pngs = await inBatches(features, async (f) => [maskName(f.properties.tile), { buffer: await colourMask(maskPath(bundle.dir, f.properties.tile), rgba) }]);
  const index = Object.fromEntries(features.map((f) => [f.properties.tile, imageCorners(f.geometry)]));
  return { ...Object.fromEntries(pngs), [`${MASKS_DIR}/index.json`]: { json: index } };
};
