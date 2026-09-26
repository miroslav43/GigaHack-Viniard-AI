// measurements.csv checks (src/Web/CLAUDE.md §6.4) and its consistency with the bundle layers.
import { CSV_FILE, CSV_HEADER, ROW_STRUCTURES } from "./contract.mjs";

const FIELDS = CSV_HEADER.split(",").length;
const AREAS = ["canopy_area_m2", "canopy_area_ha", "interrow_area_m2", "interrow_area_ha"];
// cells that must hold a number, per level
const NUMERIC = {
  survey: ["block_count", "row_count", "row_length_m", ...AREAS, "plant_count"],
  block: ["row_count", "row_length_m", ...AREAS, "plant_count"],
  row: ["row_length_m"],
};
const TEXT = { survey: [], block: ["vineyard_id"], row: ["vineyard_id", "row_id", "row_structure"] };
const SUM_TOLERANCE = 0.005; // block lines vs the survey line (the §6.2 0.5% import warning)

const lineList = (lines) => `line${lines.length === 1 ? "" : "s"} ${lines.slice(0, 5).map((l) => l.line).join(", ")}${lines.length > 5 ? ", …" : ""}`;
const problem = (message) => ({ file: CSV_FILE, message });
const setDiff = (a, b) => [...a].filter((v) => !b.has(v));

/** Structural checks: header, field count, levels, one survey line, numeric and text cells, enums. */
const structureProblems = (lines) => {
  const bad = (pred) => lines.filter(pred);
  const checks = [
    [bad((l) => l.cellCount !== FIELDS), `expected ${FIELDS} fields`],
    [bad((l) => !(l.level in NUMERIC)), "level must be survey|block|row"],
    [bad((l) => (NUMERIC[l.level] ?? []).some((c) => !Number.isFinite(l[c]))), "missing or non-numeric measurement cells"],
    [bad((l) => (TEXT[l.level] ?? []).some((c) => !l[c])), "missing vineyard_id / row_id / row_structure"],
    [bad((l) => l.level === "row" && l.row_structure && !ROW_STRUCTURES.includes(l.row_structure)),
      `row_structure outside ${ROW_STRUCTURES.join("|")} (lowercase)`],
  ];
  const surveys = lines.filter((l) => l.level === "survey").length;
  return [
    ...checks.filter(([ls]) => ls.length).map(([ls, what]) => problem(`${lineList(ls)}: ${what}`)),
    ...(surveys === 1 ? [] : [problem(`expected exactly 1 survey line, found ${surveys}`)]),
  ];
};

const unless = (ok, message) => (ok ? [] : [problem(message)]);
const sample = (ids) => `${ids.slice(0, 5).join(", ")}${ids.length > 5 ? ", …" : ""}`;
const idProblems = (kind, csvIds, layerIds, layerFile) => {
  const [onlyLayer, onlyCsv] = [setDiff(layerIds, csvIds), setDiff(csvIds, layerIds)];
  return [
    ...unless(!onlyLayer.length, `${kind} ids in ${layerFile} but not in the CSV: ${sample(onlyLayer)}`),
    ...unless(!onlyCsv.length, `${kind} ids in the CSV but not in ${layerFile}: ${sample(onlyCsv)}`),
  ];
};

/** The CSV and the layers must describe the same survey: counts and block / row id sets. */
const crossProblems = (lines, bundle) => {
  const survey = lines.find((l) => l.level === "survey");
  const [blocks, rows] = [bundle.blocks.features, bundle.rows.features];
  const csvBlocks = new Set(lines.filter((l) => l.level === "block").map((l) => l.vineyard_id));
  const csvRows = new Set(lines.filter((l) => l.level === "row").map((l) => l.row_id));
  return [
    ...unless(survey.block_count === blocks.length, `survey block_count ${survey.block_count} ≠ ${blocks.length} features in blocks.geojson`),
    ...unless(survey.row_count === rows.length, `survey row_count ${survey.row_count} ≠ ${rows.length} features in rows.geojson`),
    ...unless(csvBlocks.size === survey.block_count, `${csvBlocks.size} block lines but survey block_count is ${survey.block_count}`),
    ...idProblems("block", csvBlocks, new Set(blocks.map((f) => f.properties.vineyard_id)), "blocks.geojson"),
    ...idProblems("row", csvRows, new Set(rows.map((f) => f.properties.row_id)), "rows.geojson"),
  ];
};

/** Block lines that do not add up to the survey line (e.g. interrows overlapping across blocks). */
const sumWarnings = (lines) => {
  const survey = lines.find((l) => l.level === "survey");
  const blocks = lines.filter((l) => l.level === "block");
  return ["canopy_area_m2", "interrow_area_m2", "row_length_m"].flatMap((col) => {
    const sum = blocks.reduce((s, l) => s + l[col], 0);
    const rel = survey[col] ? (sum - survey[col]) / survey[col] : 0;
    const delta = `${rel > 0 ? "+" : ""}${(rel * 100).toFixed(2)}%`;
    return unless(Math.abs(rel) <= SUM_TOLERANCE,
      `block ${col} lines sum to ${sum.toFixed(2)}, survey line says ${survey[col].toFixed(2)} (${delta})`);
  });
};

/** → { errors, warnings }; cross checks against the layers only when those passed their own checks. */
export const validateMeasurements = (csv, bundle, { crossCheck }) => {
  if (!csv) return { errors: [], warnings: [] };
  if (csv.header !== CSV_HEADER) return { errors: [problem(`header differs from §6.4 (got "${csv.header.slice(0, 80)}")`)], warnings: [] };
  const structure = structureProblems(csv.lines);
  if (structure.length) return { errors: structure, warnings: [] };
  return { errors: crossCheck ? crossProblems(csv.lines, bundle) : [], warnings: sumWarnings(csv.lines) };
};
