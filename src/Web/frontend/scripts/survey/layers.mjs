// The EPSG:4326 layers the map reads (public/data/<id>/*.geojson), built from the bundle's EPSG:32635 layers:
// reprojected at 7 decimals, bundle properties kept, numeric feature id = i+1 (same shapes as build-data.mjs).
// Areas missing from the bundle are measured in UTM before reprojection.
import { farmOfBlock, labelPoint, roadLength } from "./farms.mjs";
import { isNum, ll, polygonArea, positions, r2, r4, simplifyPolygonal, toWgs84 } from "./geo.mjs";

export const TARGET_MODES = ["all", "route"];

const feature = (properties, geometry, i) => ({ type: "Feature", id: i + 1, properties, geometry: toWgs84(geometry) });
const fc = (features) => ({ type: "FeatureCollection", features });
/** A bundle layer in EPSG:4326: 7 decimals, properties kept (or mapped by `props`), numeric id = i + 1. */
export const reproject = (layer, props = (p) => p) =>
  fc(layer.features.map((f, i) => feature(props(f.properties ?? {}, f), f.geometry, i)));

/** Row properties; length_m is the CSV row figure (2 decimals), so the map, the tables and the CSV agree. */
export const rowProperties = (p, csvLength) => ({
  ...p,
  length_m: isNum(csvLength) ? csvLength : r2(p.length_m),
  ...(isNum(p.max_gap_m) ? { max_gap_m: r2(p.max_gap_m) } : {}),
});

const rowsLayer = (rows, csv) => {
  const lengths = new Map(csv.lines.filter((l) => l.level === "row").map((l) => [l.row_id, l.row_length_m]));
  return reproject(rows, (p) => rowProperties(p, lengths.get(p.row_id)));
};

/** Canopy area: the bundle's area_m2 when present (newer bundles), else the shoelace area in UTM. */
export const canopyArea = (p, geometry) => r4(isNum(p.area_m2) ? p.area_m2 : polygonArea(geometry));
const canopiesLayer = (canopies) => reproject(canopies, (p, f) => ({ ...p, area_m2: canopyArea(p, f.geometry) }));

const pieceArea = (f) => (isNum(f.properties.area_m2) ? f.properties.area_m2 : polygonArea(f.geometry));

/**
 * interrow_total_m2 for older bundles, by interrow_id: the sum of the piece areas, only when every piece is in a
 * different tile (tiles do not overlap, so the sum is the union). Pieces cut twice in one tile ("…#2") can overlap
 * almost completely, and their sum would overstate the union up to 2×: those inter-rows get no total.
 */
export const summedInterrowTotals = (features) => {
  const byId = Map.groupBy(features, (f) => f.properties.interrow_id);
  const oneTileEach = (pieces) => new Set(pieces.map((f) => f.properties.tile)).size === pieces.length;
  return new Map([...byId]
    .filter(([, pieces]) => oneTileEach(pieces))
    .map(([id, pieces]) => [id, r2(pieces.reduce((s, f) => s + pieceArea(f), 0))]));
};

/**
 * Inter-rows simplified in UTM (Douglas-Peucker, `tol` metres), keeping the bundle's piece area_m2 and
 * interrow_total_m2 (newer bundles: the union); when absent, the UTM piece area and summedInterrowTotals.
 */
export const interrowsLayer = (interrows, tol) => {
  const totals = summedInterrowTotals(interrows.features);
  const simplified = interrows.features.map((f) => simplifyPolygonal(f.geometry, tol));
  const features = interrows.features.map((f, i) => {
    const p = f.properties;
    const total = isNum(p.interrow_total_m2) ? p.interrow_total_m2 : totals.get(p.interrow_id);
    const properties = {
      ...p,
      area_m2: isNum(p.area_m2) ? p.area_m2 : r2(pieceArea(f)),
      ...(isNum(total) ? { interrow_total_m2: total } : {}),
    };
    return feature(properties, simplified[i], i);
  });
  return { layer: fc(features), stats: simplifyStats(interrows.features.map((f) => f.geometry), simplified) };
};

/** Vertex counts and area change of a simplification (areas in UTM). */
export const simplifyStats = (before, after) => {
  const areas = (gs) => gs.map(polygonArea);
  const [a0, a1] = [areas(before), areas(after)];
  const total = (xs) => xs.reduce((s, v) => s + v, 0);
  const count = (gs) => gs.reduce((s, g) => s + positions(g).length, 0);
  return {
    coordsBefore: count(before),
    coordsAfter: count(after),
    areaBefore: total(a0),
    areaAfter: total(a1),
    maxPieceChange: a0.reduce((m, a, i) => Math.max(m, a > 0 ? Math.abs(a1[i] - a) / a : 0), 0),
  };
};

/** route_order as int | null (older bundles write 1.0). */
export const routeOrder = (p) => (isNum(p.route_order) ? Math.round(p.route_order) : null);
const byRouteOrder = (a, b) => {
  const [x, y] = [a.properties.route_order, b.properties.route_order];
  if (x === null || y === null) return (x === null) - (y === null);
  return x - y;
};

/** Targets sorted by route_order, unrouted (null) last in bundle order; mode "route" keeps only routed ones. */
export const targetsLayer = (targets, mode = "all") => {
  const normalised = targets.features.map((f) => ({ ...f, properties: { ...f.properties, route_order: routeOrder(f.properties) } }));
  const kept = mode === "route" ? normalised.filter((f) => f.properties.route_order !== null) : normalised;
  return fc([...kept].sort(byRouteOrder).map((f, i) => feature(f.properties, f.geometry, i)));
};

/** Blocks with farm_id when the bundle ships farms: from the farms' vineyard_ids (they win), else the block's own, else null. */
export const blocksLayer = (blocks, farms) => {
  if (!farms) return reproject(blocks);
  const farmOf = farmOfBlock(farms);
  return reproject(blocks, (p) => ({ ...p, farm_id: farmOf.get(p.vineyard_id) ?? p.farm_id ?? null }));
};

/** Farm outlines: n_blocks from vineyard_ids, area_m2 measured in UTM when absent, label_point = [lon, lat] inside the outline. */
export const farmsLayer = (farms) =>
  reproject(farms, (p, f) => {
    const label = labelPoint(f.geometry);
    return {
      ...p,
      n_blocks: p.vineyard_ids.length,
      area_m2: r2(isNum(p.area_m2) ? p.area_m2 : polygonArea(f.geometry)),
      label_point: label ? ll(label) : null,
    };
  });

/** Roads: every optional property present (null when absent), length_m rounded to 2 decimals (measured when absent). */
export const roadsLayer = (roads) =>
  reproject(roads, (p, f) => ({
    ...p,
    highway: p.highway ?? null,
    name: p.name ?? null,
    surface: p.surface ?? null,
    farm_id: p.farm_id ?? null,
    source: p.source ?? null,
    length_m: r2(roadLength(p, f.geometry)),
  }));

/** Every 4326 layer file, keyed by output file name, plus the inter-row simplification stats. */
export const buildLayers = (bundle, { interrowTol, targets = "all" }) => {
  const interrows = interrowsLayer(bundle.interrows, interrowTol);
  return {
    files: {
      "rows.geojson": rowsLayer(bundle.rows, bundle.csv),
      "blocks.geojson": blocksLayer(bundle.blocks, bundle.farms),
      "canopies.geojson": canopiesLayer(bundle.canopies),
      "interrows.geojson": interrows.layer,
      "waste.geojson": reproject(bundle.waste),
      "targets.geojson": targetsLayer(bundle.targets, targets),
      "route.geojson": reproject(bundle.route, (p) => ({ ...p, mock: false })),
      // optional (web bundle v3): the survey tile footprints
      ...(bundle.tiles ? { "tiles.geojson": reproject(bundle.tiles) } : {}),
      // optional: farms and road classes
      ...(bundle.farms ? { "farms.geojson": farmsLayer(bundle.farms) } : {}),
      ...(bundle.roads ? { "roads.geojson": roadsLayer(bundle.roads) } : {}),
    },
    stats: { interrows: interrows.stats },
  };
};
