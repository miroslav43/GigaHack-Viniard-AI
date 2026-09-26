// Synthetic canopy relief for the 3D map view: public/data/<survey>/canopies.geojson (EPSG:4326) →
//   public/data/<survey>/terrain/{z}/{x}/{y}.png   (MapLibre raster-dem, terrarium encoding, 256 px, every tile in bounds)
//   public/data/<survey>/terrain/terrain.json      (bounds, zooms, encoding, height — read by the /harta page)
// The orthophoto tiles are RGB only (no altitude): the relief is made up from the canopy annotations (docs/DECISIONS.md).
// Run after the survey itself (`pnpm data` builds the mock and its relief; after `pnpm data:survey`, run this again:
// build-survey replaces the survey folder as a whole, relief included).
// Usage: node scripts/build-terrain.mjs [--survey siret3-mock] [--height 1.1] [--blur 0.45] [--min-zoom 14] [--max-zoom 19]
//                                       [--data-dir public/data]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import sharp from "sharp";
import { buildRelief, canopyPolygons, RELIEF_DEFAULTS } from "./terrain/relief.mjs";
import { encodeTile } from "./terrain/terrarium.mjs";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SURVEY_ID_RE = /^[a-z0-9-]+$/;
const CANOPIES = "canopies.geojson";
const TERRAIN_DIR = "terrain";
const META_FILE = "terrain.json";
const LIMITS = { height: [0.1, 5], blur: [0, 2], zoom: [10, 22] };
const USAGE = "usage: node scripts/build-terrain.mjs [--survey siret3-mock] [--height 1.1] [--blur 0.45] " +
  "[--min-zoom 14] [--max-zoom 19] [--data-dir public/data]";

const fail = (message) => {
  throw new Error(message);
};
const rel = (p) => path.relative(process.cwd(), p) || ".";

const numberIn = (name, raw, [min, max], integer = false) => {
  const v = Number(raw);
  if (!Number.isFinite(v) || v < min || v > max || (integer && !Number.isInteger(v)))
    fail(`--${name} must be ${integer ? "an integer" : "a number"} in [${min}, ${max}] (got "${raw}")`);
  return v;
};

const parseOptions = (argv) => {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    options: {
      survey: { type: "string", default: process.env.NEXT_PUBLIC_SURVEY_ID ?? "siret3-mock" },
      height: { type: "string", default: String(RELIEF_DEFAULTS.heightM) },
      blur: { type: "string", default: String(RELIEF_DEFAULTS.blurM) },
      "min-zoom": { type: "string", default: String(RELIEF_DEFAULTS.minZoom) },
      "max-zoom": { type: "string", default: String(RELIEF_DEFAULTS.maxZoom) },
      "data-dir": { type: "string", default: path.join(FRONTEND, "public", "data") },
    },
  });
  if (!SURVEY_ID_RE.test(values.survey)) fail(`--survey must match ${SURVEY_ID_RE} (got "${values.survey}")`);
  const minZoom = numberIn("min-zoom", values["min-zoom"], LIMITS.zoom, true);
  const maxZoom = numberIn("max-zoom", values["max-zoom"], LIMITS.zoom, true);
  if (minZoom > maxZoom) fail(`--min-zoom (${minZoom}) must not exceed --max-zoom (${maxZoom})`);
  return {
    surveyId: values.survey,
    surveyDir: path.resolve(values["data-dir"], values.survey),
    heightM: numberIn("height", values.height, LIMITS.height),
    blurM: numberIn("blur", values.blur, LIMITS.blur),
    minZoom,
    maxZoom,
  };
};

const readCanopies = (surveyDir) => {
  const file = path.join(surveyDir, CANOPIES);
  if (!fs.existsSync(file)) fail(`${rel(file)} is missing — build the survey first (pnpm data, or pnpm data:survey)`);
  let fc;
  try {
    fc = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (err) {
    fail(`${rel(file)}: invalid JSON (${err.message})`);
  }
  if (fc?.type !== "FeatureCollection" || !Array.isArray(fc.features)) fail(`${rel(file)}: not a GeoJSON FeatureCollection`);
  return fc;
};

/** PNG writer into `dir`; flat tiles share one pre-encoded buffer. Returns the writer and a byte counter. */
const pngWriter = async (dir, tileSize) => {
  const raw = { width: tileSize, height: tileSize, channels: 3 };
  const png = (heights) => sharp(encodeTile(heights), { raw }).png({ compressionLevel: 9, adaptiveFiltering: true }).toBuffer();
  const flat = await png(new Float32Array(tileSize * tileSize));
  const stats = { bytes: 0 };
  const write = async (z, x, y, heights) => {
    const buf = heights ? await png(heights) : flat;
    fs.mkdirSync(path.join(dir, String(z), String(x)), { recursive: true });
    fs.writeFileSync(path.join(dir, String(z), String(x), `${y}.png`), buf);
    stats.bytes += buf.length;
  };
  return { write, stats };
};

/** Replaces <surveyDir>/terrain with the freshly written <surveyDir>/terrain.tmp (previous kept on failure). */
const swapIn = (surveyDir) => {
  const final = path.join(surveyDir, TERRAIN_DIR), tmp = `${final}.tmp`, old = `${final}.old`;
  fs.rmSync(old, { recursive: true, force: true });
  const hadPrevious = fs.existsSync(final);
  if (hadPrevious) fs.renameSync(final, old);
  try {
    fs.renameSync(tmp, final);
  } catch (err) {
    if (hadPrevious) fs.renameSync(old, final);
    throw new Error(`replacing ${rel(final)} failed (previous relief kept): ${err.message}`, { cause: err });
  }
  fs.rmSync(old, { recursive: true, force: true });
  return final;
};

const main = async (argv) => {
  const opts = parseOptions(argv);
  const t0 = Date.now();
  const { polygons, skipped } = canopyPolygons(readCanopies(opts.surveyDir));
  if (skipped) console.warn(`! ${skipped} canopy feature(s) without a polygon geometry skipped`);
  if (!polygons.length) fail(`${rel(path.join(opts.surveyDir, CANOPIES))}: no canopy polygons, nothing to raise`);

  const tmp = path.join(opts.surveyDir, `${TERRAIN_DIR}.tmp`);
  fs.rmSync(tmp, { recursive: true, force: true });
  const { write, stats } = await pngWriter(tmp, RELIEF_DEFAULTS.tileSize);
  let result;
  try {
    result = await buildRelief(polygons, opts, write);
    const meta = {
      version: 1,
      survey_id: opts.surveyId,
      source: CANOPIES,
      encoding: "terrarium",
      tile_size: RELIEF_DEFAULTS.tileSize,
      minzoom: opts.minZoom,
      maxzoom: opts.maxZoom,
      bounds: result.bounds,
      height_m: opts.heightM,
      blur_m: opts.blurM,
      tiles: "{z}/{x}/{y}.png",
      canopy_count: polygons.length,
      tile_count: result.tiles,
      relief_tile_count: result.reliefTiles,
      bytes: stats.bytes,
      generated_at: new Date().toISOString(),
    };
    fs.writeFileSync(path.join(tmp, META_FILE), JSON.stringify(meta, null, 2) + "\n");
  } catch (err) {
    fs.rmSync(tmp, { recursive: true, force: true });
    throw err;
  }
  const dir = swapIn(opts.surveyDir);

  const zooms = Object.entries(result.perZoom).map(([z, c]) => `z${z} ${c.tiles}${c.relief ? ` (${c.relief} relief)` : ""}`);
  console.log(
    `${opts.surveyId}: relief ${opts.heightM} m from ${polygons.length} canopies → ${result.tiles} tiles ` +
      `(${result.reliefTiles} with relief), ${(stats.bytes / 1024 / 1024).toFixed(2)} MB in ${((Date.now() - t0) / 1000).toFixed(1)} s`,
  );
  console.log(`  ${zooms.join(" · ")}`);
  console.log(`wrote ${rel(dir)}/`);
};

main(process.argv.slice(2)).catch((err) => {
  console.error(`✖ build-terrain: ${err.message}`);
  if (err.code?.startsWith?.("ERR_PARSE_ARGS")) console.error(USAGE);
  process.exit(1);
});
