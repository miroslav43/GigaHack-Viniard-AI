// Derived metrics of the /statistici page: public/data/<survey>/*.{json,geojson} → public/data/<survey>/stats.json.
// Reads only what the survey already publishes (no pipeline, no measurements.csv change; docs/STATISTICI.md).
// Run after the survey itself (`pnpm data` does it for the mock; after `pnpm data:survey`, run this again:
// build-survey replaces the survey folder as a whole, stats.json included).
// Usage: node scripts/build-stats.mjs [--survey siret3-mock] [--data-dir public/data]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { buildStats } from "./stats/build.mjs";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SURVEY_ID_RE = /^[a-z0-9-]+$/;
const OUT_FILE = "stats.json";
const REQUIRED = { summary: "summary.json", rows: "rows.json", rowsGeo: "rows.geojson", targets: "targets.geojson", interrows: "interrows.geojson", route: "route.geojson" };
const OPTIONAL = { roads: "roads.geojson", tiles: "tiles.geojson", farms: "farms.geojson" };

const parseOptions = (argv) => {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    options: {
      survey: { type: "string", default: process.env.NEXT_PUBLIC_SURVEY_ID ?? "siret3-mock" },
      "data-dir": { type: "string", default: path.join(FRONTEND, "public", "data") },
    },
  });
  if (!SURVEY_ID_RE.test(values.survey)) throw new Error(`--survey must match ${SURVEY_ID_RE} (got "${values.survey}")`);
  return { surveyId: values.survey, surveyDir: path.resolve(values["data-dir"], values.survey) };
};

const readJson = (dir, file) => {
  const p = path.join(dir, file);
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch (err) {
    throw new Error(`${path.relative(process.cwd(), p)}: ${err.code === "ENOENT" ? "missing (build the survey first)" : err.message}`);
  }
};

const readFiles = (dir) => ({
  ...Object.fromEntries(Object.entries(REQUIRED).map(([k, f]) => [k, readJson(dir, f)])),
  ...Object.fromEntries(Object.entries(OPTIONAL).map(([k, f]) => [k, fs.existsSync(path.join(dir, f)) ? readJson(dir, f) : null])),
});

const main = () => {
  const opts = parseOptions(process.argv.slice(2));
  const stats = buildStats(readFiles(opts.surveyDir));
  const out = path.join(opts.surveyDir, OUT_FILE);
  fs.writeFileSync(out, `${JSON.stringify(stats)}\n`);
  const missing = ["farms", "roads", "tiles"].filter((k) => stats[k] == null);
  console.log(
    `${opts.surveyId}: ${path.relative(process.cwd(), out)} (${(fs.statSync(out).size / 1024).toFixed(1)} KB) — ` +
      `${stats.targets.total} targets, ${stats.rows.count} rows, ` +
      `${stats.health.blocks.filter((b) => b.score != null).length}/${stats.health.blocks.length} blocks scored` +
      (missing.length ? `; no ${missing.join(", ")} on this survey` : ""),
  );
};

try {
  main();
} catch (err) {
  console.error(`build-stats: ${err.message}`);
  process.exit(1);
}
