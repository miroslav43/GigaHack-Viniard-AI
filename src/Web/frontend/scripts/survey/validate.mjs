// Contract checks for a loaded bundle (src/Web/CLAUDE.md §6.2–6.4).
// validateBundle(bundle, { surveyId, geofence }) → { errors, warnings }, each [{ file, message }].
import {
  BUNDLE_CRS, CANOPY_GUARD, CRS_URN, LAYERS, MANIFEST_FILE, MANIFEST_REQUIRED, STAGES, TILES_TOTAL,
} from "./contract.mjs";
import { isNum, ll, pointInPolygonal, positions } from "./geo.mjs";
import { validateMeasurements } from "./validate-csv.mjs";

const SAMPLE = 3;
const isNil = (v) => v === undefined || v === null;
const isSet = (v) => !isNil(v) && v !== "";
// a UTM 35N position in metres (catches 4326 degrees labelled as EPSG:32635)
const isUtmPosition = (p) => Array.isArray(p) && isNum(p[0]) && isNum(p[1]) && p[0] > 1e5 && p[0] < 9e5 && p[1] > 0 && p[1] < 1e7;
const quote = (v) => JSON.stringify(v);

/** One aggregated problem for every feature matching `bad` ("missing property \"tile\": 12 features (V01-I001, …)"). */
const aggregate = (file, features, idKey, bad, what) => {
  const idx = features.flatMap((f, i) => (bad(f) ? [i] : []));
  if (!idx.length) return [];
  const names = idx.slice(0, SAMPLE).map((i) => features[i]?.properties?.[idKey] ?? `feature #${i + 1}`);
  return [{ file, message: `${what}: ${idx.length} feature${idx.length === 1 ? "" : "s"} (${names.join(", ")}${idx.length > SAMPLE ? ", …" : ""})` }];
};

const duplicates = (values) => {
  const seen = new Set(), dups = new Set();
  for (const v of values) if (isSet(v)) (seen.has(v) ? dups : seen).add(v);
  return [...dups];
};

const crsProblems = (file, fc) => {
  const name = fc.crs?.properties?.name;
  return name === CRS_URN ? [] : [{ file, message: `crs is ${quote(name ?? "missing")}, expected ${quote(CRS_URN)}` }];
};

/** Generic §6.2/§6.3 checks of one layer: FeatureCollection, crs URN, geometry types, properties, enums, unique ids. */
export const layerProblems = (spec, fc, { file = spec.file, checkCrs = true } = {}) => {
  if (fc?.type !== "FeatureCollection" || !Array.isArray(fc.features))
    return [{ file, message: "not a GeoJSON FeatureCollection" }];
  const features = fc.features;
  const id = spec.required[0];
  const geometryOk = (f) => spec.geometry.includes(f.geometry?.type);
  return [
    ...(checkCrs ? crsProblems(file, fc) : []),
    ...aggregate(file, features, id, (f) => !geometryOk(f), `geometry other than ${spec.geometry.join("|")}`),
    ...aggregate(file, features, id, (f) => geometryOk(f) && !isUtmPosition(positions(f.geometry)[0]),
      "coordinates not in EPSG:32635 metres"),
    ...spec.required.flatMap((k) => aggregate(file, features, id, (f) => !isSet(f.properties?.[k]), `missing property ${quote(k)}`)),
    ...(spec.nullable ?? []).flatMap((k) =>
      aggregate(file, features, id, (f) => !(f.properties && k in f.properties), `missing property ${quote(k)} (null allowed)`)),
    ...Object.entries(spec.enums ?? {}).flatMap(([k, allowed]) =>
      aggregate(file, features, id, (f) => isSet(f.properties?.[k]) && !allowed.includes(f.properties[k]),
        `${quote(k)} outside ${allowed.join("|")} (lowercase)`)),
    ...(spec.unique ? duplicateProblems(file, spec.unique, features.map((f) => f.properties?.[spec.unique])) : []),
  ];
};
const duplicateProblems = (file, key, values) => {
  const dups = duplicates(values);
  return dups.length ? [{ file, message: `duplicate ${key}: ${dups.slice(0, SAMPLE * 2).join(", ")}${dups.length > SAMPLE * 2 ? ", …" : ""}` }] : [];
};

const manifestProblems = (m, surveyId) => {
  const file = MANIFEST_FILE;
  if (!m || typeof m !== "object") return { errors: [{ file, message: "not a JSON object" }], warnings: [] };
  const errors = [
    ...MANIFEST_REQUIRED.filter((k) => !isSet(m[k])).map((k) => ({ file, message: `missing ${quote(k)}` })),
    ...(isSet(m.survey_id) && m.survey_id !== surveyId ? [{ file, message: `survey_id is ${quote(m.survey_id)}, expected ${quote(surveyId)}` }] : []),
    ...(isSet(m.crs) && m.crs !== BUNDLE_CRS ? [{ file, message: `crs is ${quote(m.crs)}, expected ${quote(BUNDLE_CRS)}` }] : []),
    ...(isSet(m.stage) && !STAGES.includes(m.stage) ? [{ file, message: `stage is ${quote(m.stage)}, expected ${STAGES.join("|")}` }] : []),
    ...("tiles_with_objects" in m && !(Number.isInteger(m.tiles_with_objects) && m.tiles_with_objects >= 0)
      ? [{ file, message: "tiles_with_objects must be a non-negative integer" }] : []),
    ...(!isNil(m.model_version) && typeof m.model_version !== "string"
      ? [{ file, message: "model_version must be a string" }] : []),
  ];
  const warnings = [
    ...(m.tiles !== TILES_TOTAL ? [{ file, message: `tiles is ${quote(m.tiles ?? "missing")}, expected ${TILES_TOTAL}` }] : []),
    ...(!isNil(m.bbox_32635) && !(Array.isArray(m.bbox_32635) && m.bbox_32635.length === 4 && m.bbox_32635.every(isNum))
      ? [{ file, message: "bbox_32635 is not [minx, miny, maxx, maxy]; ignored" }] : []),
  ];
  return { errors, warnings };
};

/** manifest.counts vs the feature counts actually shipped (warnings only). */
const countWarnings = (bundle) =>
  Object.entries(bundle.manifest?.counts ?? {}).flatMap(([key, n]) => {
    const actual = bundle[key]?.features?.length;
    return actual === undefined || actual === n ? [] : [{ file: MANIFEST_FILE, message: `counts.${key} is ${n} but the layer has ${actual} features` }];
  });

const targetProblems = (fc, geofence) => {
  const { file } = LAYERS.targets;
  const features = fc.features;
  // JSON 1.0 parses to the integer 1, so float route_order from older bundles passes; 1.5 does not
  const orders = features.map((f) => f.properties?.route_order).filter((v) => !isNil(v));
  const sorted = [...orders].sort((a, b) => a - b);
  const badOrder = (f) => {
    const v = f.properties?.route_order;
    return !isNil(v) && !(Number.isInteger(v) && v >= 1);
  };
  const outside = (f) => !pointInPolygonal(ll(f.geometry.coordinates), geofence);
  // gap / missing targets come from rows, so only a waste target may lack a block
  const noBlock = (f) => f.properties.type !== "waste" && !isSet(f.properties.vineyard_id);
  const errors = [
    ...aggregate(file, features, "target_id", noBlock, "missing property \"vineyard_id\" (null only on waste targets)"),
    ...aggregate(file, features, "target_id", badOrder, "route_order not a positive integer or null"),
    ...aggregate(file, features, "target_id", (f) => typeof f.properties?.reachable !== "boolean", "missing boolean property \"reachable\""),
    ...duplicateProblems(file, "route_order", orders),
    ...(geofence ? aggregate(file, features, "target_id", outside, "outside the UAT geofence (task inserts would fail with 23514)") : []),
  ];
  const warnings = sorted.every((v, i) => v === i + 1) ? [] : [{ file, message: `route_order is not 1..${sorted.length} without gaps` }];
  return { errors, warnings };
};

const routeProblems = (fc) => {
  const { file } = LAYERS.route;
  if (fc.features.length !== 1) return [{ file, message: `expected exactly 1 route feature, found ${fc.features.length}` }];
  const p = fc.features[0].properties ?? {};
  const positive = (k) => (k in p && p[k] !== null && !(isNum(p[k]) && p[k] > 0) ? [{ file, message: `${k} must be a positive number` }] : []);
  return [
    ...positive("length_m"), ...positive("duration_min"), ...positive("speed_kmh"),
    ...("baseline_length_m" in p && p.baseline_length_m !== null && !isNum(p.baseline_length_m)
      ? [{ file, message: "baseline_length_m must be a number or null" }] : []),
  ];
};

/** Every check; the layer-specific ones only run on layers that passed the generic checks. */
export const validateBundle = (bundle, { surveyId, geofence = null }) => {
  const manifest = manifestProblems(bundle.manifest, surveyId);
  const generic = Object.fromEntries(Object.entries(LAYERS).map(([key, spec]) => [key,
    layerProblems(spec, bundle[key], key === "canopies"
      ? { file: bundle.canopyFile, checkCrs: !bundle.canopies?.geojsonl } : {})]));
  const ok = (key) => generic[key].length === 0;
  const targets = ok("targets") ? targetProblems(bundle.targets, geofence) : { errors: [], warnings: [] };
  const csv = validateMeasurements(bundle.csv, bundle, { crossCheck: Object.values(generic).every((p) => p.length === 0) });
  const canopyCount = bundle.canopies?.features?.length ?? 0;
  return {
    errors: [
      ...manifest.errors, ...Object.values(generic).flat(), ...targets.errors,
      ...(ok("route") ? routeProblems(bundle.route) : []), ...csv.errors,
    ],
    warnings: [
      ...manifest.warnings, ...countWarnings(bundle), ...targets.warnings, ...csv.warnings,
      ...(canopyCount > CANOPY_GUARD.features
        ? [{ file: bundle.canopyFile, message: `${canopyCount} canopies (> ${CANOPY_GUARD.features}): split canopies per tile or move them to PMTiles` }] : []),
    ],
  };
};
