// Converts an AI survey bundle (EPSG:32635, src/Web/CLAUDE.md §6.2–6.4) into the static files the site reads:
//   public/data/<id>/{summary.json, rows.json, rows|blocks|canopies|interrows|waste|targets|route.geojson,
//                     route_EPSG32635.geojson, route.gpx, measurements.csv}          (EPSG:4326, 7 decimals)
// Run after `pnpm data` (public/data/uats.json; --emit-seed also needs public/data/ref/study_area.geojson).
// Usage: node scripts/build-survey.mjs [--survey siret3] [--bundle <dir>] [--out <dir>] [--interrow-tol 0.0125]
//                                      [--targets all|route] [--check] [--emit-seed]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { CANOPY_GUARD } from "./survey/contract.mjs";
import { buildGpx, buildSeedSql, VERBATIM } from "./survey/exports.mjs";
import { buildLayers, TARGET_MODES } from "./survey/layers.mjs";
import { publishDir } from "./survey/publish.mjs";
import { readBundle } from "./survey/read-bundle.mjs";
import { buildRowsJson, buildSummary } from "./survey/summary.mjs";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const WEB_DATA = process.env.WEB_DATA_DIR ?? path.resolve(FRONTEND, "../data");
const PUBLIC_DATA = path.join(FRONTEND, "public", "data");
const UATS_FILE = path.join(PUBLIC_DATA, "uats.json");
const STUDY_AREA_FILE = path.join(PUBLIC_DATA, "ref", "study_area.geojson");
const GEOFENCE_FILE = path.join(FRONTEND, "data", "osm", "sireti_19100171.geojson");
const SEED_DIR = path.resolve(FRONTEND, "../supabase/seed");
const SURVEY_ID_RE = /^[a-z0-9-]+$/;
const DEFAULTS = { survey: "siret3", interrowTol: 0.0125, targets: "all" };
const USAGE = "usage: node scripts/build-survey.mjs [--survey siret3] [--bundle <dir>] [--out <dir>] " +
  "[--interrow-tol 0.0125] [--targets all|route] [--check] [--emit-seed]";

const rel = (p) => path.relative(process.cwd(), p) || ".";
const fail = (message) => {
  throw new Error(message);
};

const parseOptions = (argv) => {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    options: {
      survey: { type: "string", default: DEFAULTS.survey },
      bundle: { type: "string" },
      out: { type: "string", default: PUBLIC_DATA },
      "interrow-tol": { type: "string", default: String(DEFAULTS.interrowTol) },
      targets: { type: "string", default: DEFAULTS.targets },
      check: { type: "boolean", default: false },
      "emit-seed": { type: "boolean", default: false },
    },
  });
  const interrowTol = Number(values["interrow-tol"]);
  if (!SURVEY_ID_RE.test(values.survey)) fail(`--survey must match ${SURVEY_ID_RE} (got "${values.survey}")`);
  if (!(Number.isFinite(interrowTol) && interrowTol >= 0)) fail(`--interrow-tol must be a number of metres >= 0 (got "${values["interrow-tol"]}")`);
  if (!TARGET_MODES.includes(values.targets)) fail(`--targets must be ${TARGET_MODES.join("|")} (got "${values.targets}")`);
  return {
    surveyId: values.survey,
    bundleDir: path.resolve(values.bundle ?? path.join(WEB_DATA, "surveys", values.survey, "pipeline")),
    outRoot: path.resolve(values.out),
    interrowTol,
    targets: values.targets,
    check: values.check,
    emitSeed: values["emit-seed"],
  };
};

const readJsonFile = (file, hint) => {
  if (!fs.existsSync(file)) fail(`${rel(file)} is missing — ${hint}`);
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (err) {
    return fail(`${rel(file)}: invalid JSON (${err.message})`);
  }
};

const firstGeometry = (file, hint) =>
  readJsonFile(file, hint).features?.[0]?.geometry ?? fail(`${rel(file)}: the first feature has no geometry`);

/** Inputs made by other scripts, checked before any work: UAT area, geofence, and the footprint for the seed. */
const prerequisites = ({ emitSeed }) => {
  const uatAreaHa = readJsonFile(UATS_FILE, "run pnpm data first").sireti?.area_ha;
  if (!Number.isFinite(uatAreaHa)) fail(`${rel(UATS_FILE)}: sireti.area_ha is missing — run pnpm data first`);
  return {
    uatAreaHa,
    geofence: firstGeometry(GEOFENCE_FILE, "the committed OSM boundary of Sireți"),
    footprint: emitSeed ? firstGeometry(STUDY_AREA_FILE, "run pnpm data first") : null,
  };
};

const buildFiles = (bundle, opts, uatAreaHa) => {
  const { files: layers, stats } = buildLayers(bundle, opts);
  const summary = buildSummary({ bundle, targetCount: layers["targets.geojson"].features.length, uatAreaHa });
  const files = {
    "summary.json": { json: summary },
    "rows.json": { json: buildRowsJson(layers["rows.geojson"]) },
    ...Object.fromEntries(Object.entries(layers).map(([name, fc]) => [name, { json: fc }])),
    "route.gpx": { text: buildGpx(layers["targets.geojson"], layers["route.geojson"]) },
    ...Object.fromEntries(Object.entries(VERBATIM).map(([name, src]) => [name, { copy: path.join(bundle.dir, src) }])),
  };
  return { files, summary, stats };
};

const MB = 1024 * 1024;
const size = (bytes) =>
  bytes >= MB ? `${(bytes / MB).toFixed(2)} MB` : bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${bytes} B`;
const printSizes = (dir) => {
  const sizes = fs.readdirSync(dir).sort().map((f) => [f, fs.statSync(path.join(dir, f)).size]);
  for (const [f, bytes] of sizes) console.log(`  ${f.padEnd(26)} ${size(bytes).padStart(10)}`);
  console.log(`  ${"total".padEnd(26)} ${size(sizes.reduce((s, [, bytes]) => s + bytes, 0)).padStart(10)}`);
  const canopyBytes = sizes.find(([f]) => f === "canopies.geojson")?.[1] ?? 0;
  if (canopyBytes > CANOPY_GUARD.bytes)
    console.warn(`! canopies.geojson is ${size(canopyBytes)} (> ${size(CANOPY_GUARD.bytes)}): split canopies per tile or move them to PMTiles`);
};

const printCounts = (bundle, summary, stats, opts) => {
  const t = summary.totals, s = stats.interrows;
  const change = s.areaBefore ? ((s.areaAfter - s.areaBefore) / s.areaBefore) * 100 : 0;
  console.log(
    `${summary.survey.id} (${summary.survey.stage}${summary.survey.run_id ? `, run ${summary.survey.run_id}` : ""}): ` +
      `${t.block_count} blocks, ${t.row_count} rows (${t.row_length_m} m), ${t.canopy_count} canopies (${t.canopy_area_m2} m²), ` +
      `${t.interrow_count} inter-rows / ${bundle.interrows.features.length} pieces (${t.interrow_area_m2} m²), ` +
      `${t.waste_count} waste, ${t.target_count} targets${opts.targets === "route" ? ` (routed only, of ${bundle.targets.features.length})` : ""}, ` +
      `route ${summary.route.length_m} m / ${summary.route.duration_min} min`,
  );
  console.log(
    `  inter-rows simplified at ${opts.interrowTol} m: ${s.coordsBefore} → ${s.coordsAfter} vertices, ` +
      `area ${change >= 0 ? "+" : ""}${change.toFixed(4)}% (max piece ${(s.maxPieceChange * 100).toFixed(2)}%)`,
  );
};

const main = (argv) => {
  const opts = parseOptions(argv);
  const { uatAreaHa, geofence, footprint } = prerequisites(opts);
  const { bundle, warnings } = readBundle(opts.bundleDir, { surveyId: opts.surveyId, geofence });
  for (const w of warnings) console.warn(`! ${w.file}: ${w.message}`);
  if (opts.check) {
    console.log(`✔ ${rel(opts.bundleDir)} is a valid "${opts.surveyId}" bundle (${warnings.length} warning${warnings.length === 1 ? "" : "s"}); nothing written`);
    return;
  }
  const { files, summary, stats } = buildFiles(bundle, opts, uatAreaHa);
  const dir = publishDir(opts.outRoot, opts.surveyId, files);
  printCounts(bundle, summary, stats, opts);
  console.log(`wrote ${rel(dir)}/`);
  printSizes(dir);
  if (opts.emitSeed) {
    const seed = path.join(SEED_DIR, `survey_${opts.surveyId}.sql`);
    fs.writeFileSync(seed, buildSeedSql(summary.survey, footprint));
    console.log(`wrote ${rel(seed)}`);
  }
};

try {
  main(process.argv.slice(2));
} catch (err) {
  console.error(`✖ build-survey: ${err.message}`);
  if (err.code?.startsWith?.("ERR_PARSE_ARGS")) console.error(USAGE);
  process.exit(1);
}
