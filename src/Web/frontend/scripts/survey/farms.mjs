// The optional farm and road layers (farms.geojson, roads.geojson; older bundles have neither): their checks,
// the helpers of their 4326 map layers (layers.mjs) and the summary.json "farms" / "roads" blocks.
// A farm groups vineyard blocks; its vineyard_ids are the source of truth for membership. Farm figures are summed
// from the measurement-based block summaries (measurements.csv block lines), so the farms add up to the survey line.
// validate.mjs never imports this module (read-bundle.mjs runs both), so there is no import cycle.
import { FARMS, ROAD_CLASSES, ROADS } from "./contract.mjs";
import { bbox, isNum, naturalCompare, pointInPolygonal, polygonArea, polygonsOf, r2 } from "./geo.mjs";
import { layerProblems } from "./validate.mjs";

const SAMPLE = 3;
const isNil = (v) => v === undefined || v === null;
const sample = (ids) => `${ids.slice(0, SAMPLE).join(", ")}${ids.length > SAMPLE ? ", …" : ""}`;
const features = (what, ids) => `${ids.length} ${what}${ids.length === 1 ? "" : "s"} (${sample(ids)})`;
const problem = (file, message) => ({ file, message });
/** Integer cents: sums of 2-decimal figures stay exact. */
const cents = (v) => Math.round((isNum(v) ? v : 0) * 100);

const duplicates = (values) => {
  const seen = new Set();
  return [...new Set(values.filter((v) => (seen.has(v) ? true : (seen.add(v), false))))];
};
const isIdList = (v) => Array.isArray(v) && v.length > 0 && v.every((id) => typeof id === "string" && id !== "");

/** vineyard_id → farm_id, from the farms' vineyard_ids (the source of truth for membership). */
export const farmOfBlock = (farms) =>
  new Map(farms.features.flatMap((f) => (isIdList(f.properties.vineyard_ids) ? f.properties.vineyard_ids : [])
    .map((id) => [id, f.properties.farm_id])));

// ---------- checks ----------
const farmProblems = (farms, blocks) => {
  const generic = layerProblems(FARMS, farms);
  if (generic.length) return { errors: generic, warnings: [] };
  const file = FARMS.file;
  const props = farms.features.map((f) => f.properties);
  const badLists = props.filter((p) => !isIdList(p.vineyard_ids)).map((p) => p.farm_id);
  if (badLists.length) return { errors: [problem(file, `"vineyard_ids" not a non-empty list of block ids: ${features("feature", badLists)}`)], warnings: [] };
  const blockIds = blocks.features.map((f) => f.properties.vineyard_id);
  const known = new Set(blockIds);
  const members = props.flatMap((p) => p.vineyard_ids);
  const inFarm = new Set(members);
  const unknown = [...new Set(members.filter((id) => !known.has(id)))];
  const twice = duplicates(members);
  const badArea = props.filter((p) => !isNil(p.area_m2) && !(isNum(p.area_m2) && p.area_m2 >= 0)).map((p) => p.farm_id);
  const orphans = blockIds.filter((id) => !inFarm.has(id));
  const badCount = props.filter((p) => !isNil(p.n_blocks) && p.n_blocks !== p.vineyard_ids.length).map((p) => p.farm_id);
  const farmOf = farmOfBlock(farms);
  const disagree = blocks.features.map((f) => f.properties)
    .filter((p) => !isNil(p.farm_id) && farmOf.has(p.vineyard_id) && farmOf.get(p.vineyard_id) !== p.farm_id)
    .map((p) => p.vineyard_id);
  return {
    errors: [
      ...(unknown.length ? [problem(file, `vineyard_ids name blocks missing from blocks.geojson: ${sample(unknown)}`)] : []),
      ...(twice.length ? [problem(file, `blocks listed in more than one farm: ${sample(twice)}`)] : []),
      ...(badArea.length ? [problem(file, `"area_m2" not a non-negative number: ${features("feature", badArea)}`)] : []),
    ],
    warnings: [
      ...(orphans.length ? [problem(file, `${features("block", orphans)} in no farm: the farms do not add up to the survey`)] : []),
      ...(badCount.length ? [problem(file, `n_blocks differs from the vineyard_ids count (the list wins): ${features("feature", badCount)}`)] : []),
      ...(disagree.length ? [problem("blocks.geojson", `farm_id disagrees with farms.geojson vineyard_ids (farms win): ${features("block", disagree)}`)] : []),
    ],
  };
};

const roadProblems = (roads, farms) => {
  const generic = layerProblems(ROADS, roads);
  if (generic.length) return { errors: generic, warnings: [] };
  const file = ROADS.file;
  const props = roads.features.map((f) => f.properties);
  const badLength = props.filter((p) => !isNil(p.length_m) && !(isNum(p.length_m) && p.length_m >= 0)).map((p) => p.road_id);
  const farmIds = new Set(farms?.features.map((f) => f.properties.farm_id) ?? []);
  const unknownFarm = props.filter((p) => !isNil(p.farm_id) && !farmIds.has(p.farm_id)).map((p) => p.road_id);
  return {
    errors: badLength.length ? [problem(file, `"length_m" not a non-negative number: ${features("feature", badLength)}`)] : [],
    warnings: unknownFarm.length ? [problem(file, `farm_id missing from farms.geojson: ${features("road", unknownFarm)}`)] : [],
  };
};

const isCount = (v) => Number.isInteger(v) && v >= 0;
const isCountMap = (v) => typeof v === "object" && !Array.isArray(v) && Object.values(v).every(isCount);
const CADASTRE_CHECKS = [
  ["n_parcels", isCount, "a non-negative integer"],
  ["cadastral_codes", (v) => Array.isArray(v) && v.every((c) => typeof c === "string"), "a list of strings"],
  ["landuse_counts", isCountMap, "an object of counts"],
];
/** The optional cadastre snapshot properties (farms, blocks): absent, null, or of their type. */
const cadastreProblems = (file, fc, idKey) => CADASTRE_CHECKS.flatMap(([key, ok, what]) => {
  const bad = fc.features.map((f) => f.properties ?? {}).filter((p) => !isNil(p[key]) && !ok(p[key])).map((p) => p[idKey]);
  return bad.length ? [problem(file, `"${key}" not ${what} or null: ${features("feature", bad)}`)] : [];
});
const cadastralProblems = (roads) => {
  const bad = roads.features.map((f) => f.properties).filter((p) => !isNil(p.cadastral) && typeof p.cadastral !== "boolean").map((p) => p.road_id);
  return bad.length ? [problem(ROADS.file, `"cadastral" not a boolean or null: ${features("feature", bad)}`)] : [];
};

/** farms.geojson / roads.geojson checks, for the files the bundle ships → { errors, warnings }. */
export const validateFarmsRoads = (bundle) => {
  const farms = bundle.farms ? farmProblems(bundle.farms, bundle.blocks) : null;
  const roads = bundle.roads ? roadProblems(bundle.roads, bundle.farms) : null;
  const parts = [
    { errors: cadastreProblems("blocks.geojson", bundle.blocks, "vineyard_id"), warnings: [] },
    ...(farms ? [farms, { errors: farms.errors.length ? [] : cadastreProblems(FARMS.file, bundle.farms, "farm_id"), warnings: [] }] : []),
    ...(roads ? [roads, { errors: roads.errors.length ? [] : cadastralProblems(bundle.roads), warnings: [] }] : []),
  ];
  return { errors: parts.flatMap((p) => p.errors), warnings: parts.flatMap((p) => p.warnings) };
};

// ---------- geometry (EPSG:32635 metres) ----------
/** Planar length of a LineString / MultiLineString. */
export const lineLength = (geometry) => {
  const lines = geometry.type === "LineString" ? [geometry.coordinates] : geometry.type === "MultiLineString" ? geometry.coordinates : [];
  return lines.reduce((sum, line) => sum + line.slice(1).reduce((s, [x, y], i) => s + Math.hypot(x - line[i][0], y - line[i][1]), 0), 0);
};
/** A road's length: the bundle's length_m, else measured in UTM. */
export const roadLength = (p, geometry) => (isNum(p.length_m) ? p.length_m : lineLength(geometry));

const ringCentroid = (ring) => {
  const [x0, y0] = ring[0];
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [xi, yi, xj, yj] = [ring[i][0] - x0, ring[i][1] - y0, ring[i + 1][0] - x0, ring[i + 1][1] - y0];
    const c = xi * yj - xj * yi;
    a += c;
    cx += (xi + xj) * c;
    cy += (yi + yj) * c;
  }
  return a === 0 ? ring[0] : [x0 + cx / (3 * a), y0 + cy / (3 * a)];
};

/** Middle of the widest stretch of the horizontal line at `y` inside the polygon (its rings, holes included). */
const scanlineMiddle = (rings, y) => {
  const xs = rings.flatMap((ring) => ring.slice(0, -1).flatMap(([xa, ya], i) => {
    const [xb, yb] = ring[i + 1];
    return ya > y !== yb > y ? [xa + ((y - ya) * (xb - xa)) / (yb - ya)] : [];
  })).sort((a, b) => a - b);
  const spans = Array.from({ length: Math.floor(xs.length / 2) }, (_, k) => [xs[2 * k], xs[2 * k + 1]]);
  const widest = spans.reduce((best, s) => (!best || s[1] - s[0] > best[1] - best[0] ? s : best), null);
  return widest ? [(widest[0] + widest[1]) / 2, y] : null;
};

/** A point inside the largest polygon of a farm outline, for its label: the centroid, or a scanline middle. */
export const labelPoint = (geometry) => {
  const polygons = polygonsOf(geometry);
  const largest = polygons.reduce((best, rings) => {
    const area = polygonArea({ type: "Polygon", coordinates: rings });
    return !best || area > best.area ? { rings, area } : best;
  }, null);
  if (!largest) return null;
  const outline = { type: "Polygon", coordinates: largest.rings };
  const centroid = ringCentroid(largest.rings[0]);
  if (pointInPolygonal(centroid, outline)) return centroid;
  const [, y0, , y1] = bbox(largest.rings[0]);
  return scanlineMiddle(largest.rings, (y0 + y1) / 2) ?? largest.rings[0][0];
};

// ---------- summary.json ----------
// block figures summed per farm, in summary.json order
const FIGURES = ["row_count", "row_length_m", "canopy_area_m2", "interrow_area_m2", "plant_count"];
// 2-decimal figures of the CSV block lines: their rounding residual is balanced so the farms add up to the survey line
const MEASURES = ["row_length_m", "canopy_area_m2", "interrow_area_m2"];

/**
 * Hands the residual `survey − Σ farms` (cents) out one cent at a time, largest farm first, when it is a rounding
 * residual of the block lines (every block in a farm, at most half a cent per block line and the survey line).
 */
const balance = (farms, key, surveyCents, blockCount) => {
  const residual = surveyCents - farms.reduce((s, f) => s + f[key], 0);
  if (residual === 0 || !farms.length || Math.abs(residual) > (blockCount + 1) / 2) return farms;
  const order = farms.map((f, i) => [f[key], i]).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
  const step = Math.sign(residual);
  const extra = new Map();
  for (let k = 0; k < Math.abs(residual); k++) extra.set(order[k % order.length], (extra.get(order[k % order.length]) ?? 0) + step);
  return farms.map((f, i) => ({ ...f, [key]: f[key] + (extra.get(i) ?? 0) }));
};

/**
 * summary.json "farms": per farm (natural farm_id order) the outline area and the sums of its blocks' figures
 * (summary.blocks, i.e. the measurements.csv block lines) plus its targets (targets layer, by vineyard_id).
 */
export const farmsSummary = ({ farms, blocks, survey, targets }) => {
  const byBlock = new Map(blocks.map((b) => [b.vineyard_id, b]));
  const targetsOf = Map.groupBy(targets.features, (f) => f.properties.vineyard_id);
  const inCents = farms.features.map((f) => {
    const p = f.properties;
    const ids = [...p.vineyard_ids].sort(naturalCompare);
    const members = ids.flatMap((id) => (byBlock.has(id) ? [byBlock.get(id)] : []));
    return {
      farm_id: p.farm_id,
      vineyard_ids: ids,
      n_blocks: ids.length,
      area_m2: r2(isNum(p.area_m2) ? p.area_m2 : polygonArea(f.geometry)),
      ...Object.fromEntries(FIGURES.map((k) => [k, members.reduce((s, b) => s + cents(b[k]), 0)])),
      target_count: ids.reduce((s, id) => s + (targetsOf.get(id)?.length ?? 0), 0),
      // cadastre snapshot (optional): parcels under the farm outline
      ...("n_parcels" in p ? { n_parcels: p.n_parcels ?? null } : {}),
    };
  });
  const inFarm = new Set(inCents.flatMap((f) => f.vineyard_ids));
  const balanced = blocks.every((b) => inFarm.has(b.vineyard_id))
    ? MEASURES.reduce((acc, k) => balance(acc, k, cents(survey[k]), blocks.length), inCents)
    : inCents;
  return balanced
    .map((f) => ({ ...f, ...Object.fromEntries(FIGURES.map((k) => [k, f[k] / 100])) }))
    .sort((a, b) => naturalCompare(a.farm_id, b.farm_id));
};

/** summary.json "roads": total length per road class, in metres (2 decimals, summed from the rounded road lengths). */
export const roadsSummary = (roads) => {
  const byClass = Object.fromEntries(ROAD_CLASSES.map((c) => [c, 0]));
  const totals = roads.features.reduce((acc, f) => {
    const c = f.properties.road_class;
    return { ...acc, [c]: acc[c] + cents(r2(roadLength(f.properties, f.geometry))) };
  }, byClass);
  return Object.fromEntries(ROAD_CLASSES.map((c) => [`${c}_m`, totals[c] / 100]));
};
