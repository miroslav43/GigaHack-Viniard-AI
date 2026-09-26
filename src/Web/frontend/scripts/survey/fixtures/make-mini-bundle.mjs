// Regenerates fixtures/mini-bundle/: a synthetic 2-tile AI bundle in the contract format (src/Web/CLAUDE.md §6.2–6.4,
// EPSG:32635) around START, tiles siret3_r018_c010 (A) and siret3_r018_c011 (B). It uses the OLD schema, as the
// first real bundle did: no canopy area_m2, float route_order, no speed_kmh / priority / interrow_total_m2.
// It also has two shapes the pipeline produces: an inter-row cut twice in one tile (piece_id "…#2") whose pieces
// overlap, and a waste object more than 10 m from every block (vineyard_id null on the waste and on its target).
// measurements.csv is computed from the geometries below, so the fixture stays self-consistent.
// Usage: node scripts/survey/fixtures/make-mini-bundle.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { CRS_URN, CSV_HEADER } from "../contract.mjs";
import { polygonArea } from "../geo.mjs";

const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), "mini-bundle");
const A = "siret3_r018_c010", B = "siret3_r018_c011";
const START = [629504.7, 5220250.75];
const r3 = (v) => Math.round(v * 1000) / 1000;

const rect = (x0, y0, x1, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
const square = ([cx, cy], side) => rect(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2);
// dense rectangle ring: a vertex every 5 cm with ±3 mm deterministic jitter (like the pipeline's pixel-traced outlines)
const denseRect = (x0, y0, x1, y1) => {
  const corners = rect(x0, y0, x1, y1);
  const pts = corners.slice(0, -1).flatMap(([ax, ay], k) => {
    const [bx, by] = corners[k + 1];
    const n = Math.round(Math.hypot(bx - ax, by - ay) / 0.05);
    return Array.from({ length: n }, (_, i) => [ax + ((bx - ax) * i) / n, ay + ((by - ay) * i) / n]);
  });
  const jittered = pts.map(([x, y], i) => [r3(x + 0.003 * Math.sin(i * 12.9898)), r3(y + 0.003 * Math.cos(i * 78.233))]);
  return [...jittered, jittered[0]];
};
const hole = rect(629537, 5220284.5, 629538, 5220285.5).reverse();

const fc = (name, features) => ({ type: "FeatureCollection", name, crs: { type: "name", properties: { name: CRS_URN } }, features });
const feature = (properties, type, coordinates) => ({ type: "Feature", properties, geometry: { type, coordinates } });

const rows = [
  feature({ row_id: "V01-R010", vineyard_id: "V01", row_structure: "regular", length_m: 16.445, plant_count: 1, max_gap_m: 0.0,
    tile_structures: { [A]: "regular" } }, "LineString", [[629520, 5220290], [629536.4449, 5220290]]),
  feature({ row_id: "V02-R001", vineyard_id: "V02", row_structure: "regular", length_m: 20.0, plant_count: 1, max_gap_m: 12.5,
    tile_structures: { [B]: "regular" } }, "LineString", [[629580, 5220260], [629600, 5220260]]),
  feature({ row_id: "V01-R002", vineyard_id: "V01", row_structure: "disrupted", length_m: 30.0, plant_count: 2, max_gap_m: 7.25,
    tile_structures: { [A]: "regular", [B]: "disrupted" } }, "MultiLineString",
  [[[629540, 5220280], [629555.2, 5220280]], [[629555.2, 5220280], [629570, 5220280]]]),
];
const ROW_LENGTHS = { "V01-R010": 16.4449, "V01-R002": 30, "V02-R001": 20 }; // exact lengths (bundle rounds to 3 dp)

const canopies = [
  { canopy_id: `${A}#0001`, vineyard_id: "V01", tile: A, row_id: "V01-R010", ring: square([629525, 5220290], 0.5) },
  { canopy_id: `${A}#0002`, vineyard_id: "V01", tile: A, row_id: "V01-R002", ring: square([629545, 5220280], 0.6) },
  { canopy_id: `${B}#0001`, vineyard_id: "V01", tile: B, row_id: "V01-R002", ring: square([629562, 5220280], 0.7) },
  { canopy_id: `${B}#0002`, vineyard_id: "V02", tile: B, row_id: "V02-R001", ring: square([629590, 5220260], 1.0) },
].map(({ ring, ...p }) => feature(p, "Polygon", [ring]));

// piece_id as the pipeline writes it: "<interrow_id>@<tile>", then "#2", "#3" … for more pieces in the same tile
const interrowPiece = (id, cover, tile, rowIds, rings, n = 1) => {
  const geometry = { type: "Polygon", coordinates: rings };
  return feature({ interrow_id: id, vineyard_id: id.slice(0, 3), interrow_cover: cover, tile, row_ids: rowIds,
    area_m2: r3(polygonArea(geometry)), piece_id: `${id}@${tile}${n > 1 ? `#${n}` : ""}` }, "Polygon", rings);
};
// V02-I001 is cut twice in tile B: 160 + 164 m² of pieces, but a 174 m² union (the sum would overstate it 1.86×)
const V02_PIECES = [[629580, 5220261, 629600, 5220269], [629580, 5220261.5, 629600.5, 5220269.5]];
const rectOverlap = ([ax0, ay0, ax1, ay1], [bx0, by0, bx1, by1]) =>
  Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)) * Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
const interrows = [
  interrowPiece("V01-I001", "bare_soil", A, ["V01-R002", "V01-R010"], [denseRect(629520, 5220281, 629555.2, 5220289), hole]),
  interrowPiece("V01-I001", "mixed", B, ["V01-R002"], [rect(629555.2, 5220281, 629570, 5220289)]),
  ...V02_PIECES.map((box, i) => interrowPiece("V02-I001", "vegetation", B, ["V02-R001"], [rect(...box)], i + 1)),
];
// the only overlap between inter-row pieces, per block (the CSV areas are unions)
const INTERROW_OVERLAPS = { V02: rectOverlap(...V02_PIECES) };

const blocks = [
  feature({ vineyard_id: "V01", area_m2: 1200.0 }, "Polygon", [rect(629515, 5220275, 629575, 5220295)]),
  feature({ vineyard_id: "V02", area_m2: 510.0 }, "MultiPolygon", [[rect(629575, 5220255, 629605, 5220272)]]),
];
const waste = [feature({ waste_id: "W1", vineyard_id: null, tile: B, confidence: 0.87 }, "Polygon",
  [rect(629590, 5220285, 629592, 5220287)])];

const target = (id, type, vineyard, row, order, extra, pt) => feature({ target_id: id, type, vineyard_id: vineyard, row_id: row,
  waste_id: null, route_order: order, reachable: order !== null, gap_length_m: null, note: null, kind: null, tile: pt[0] < 629555.2 ? A : B,
  source_target_id: `T-${id}`, ...extra }, "Point", pt);
// file order is deliberately not route order: the converter sorts (nulls last, bundle order kept among them)
const targets = [
  target("T003", "gap", "V01", "V01-R010", null, { gap_length_m: 5.1, note: "optional_detour; gap 5.10 m", kind: "row_gap" }, [629530, 5220290]),
  target("T002", "missing", "V02", "V02-R001", 2, { note: "missing_plant", kind: "missing_plant" }, [629590.5, 5220260]),
  target("T001", "gap", "V01", "V01-R002", 1, { gap_length_m: 7.25, note: "row_gap; gap 7.25 m", kind: "row_gap" }, [629560, 5220280]),
  target("T004", "missing", "V01", null, null, { note: "disconnected; missing row", kind: "missing_row" }, [629545, 5220285]),
  // W1 is 13–15 m from both blocks: no vineyard_id on the waste, none on its target
  target("T005", "waste", null, null, null, { waste_id: "W1", note: "too_far", kind: "waste" }, [629591, 5220286]),
];

const routeCoords = [START, [629560, 5220280], [629590.5, 5220260], START];
const routeLength = routeCoords.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - routeCoords[i][0], p[1] - routeCoords[i][1]), 0);
const route = [feature({ length_m: r3(routeLength), duration_min: r3((routeLength / 1000 / 4) * 60), baseline_length_m: 400.0,
  outside_share: 0.0 }, "LineString", routeCoords)];

// ---------- measurements.csv (union areas = sums minus the one inter-row overlap; nothing else overlaps) ----------
const areaOf = (f) => polygonArea(f.geometry);
const sum = (xs, fn) => xs.reduce((s, x) => s + fn(x), 0);
const measure = (vid) => {
  const of = (xs) => (vid ? xs.filter((f) => f.properties.vineyard_id === vid) : xs);
  const overlap = sum(Object.entries(INTERROW_OVERLAPS).filter(([v]) => !vid || v === vid), ([, area]) => area);
  return { rows: of(rows), rowLength: sum(of(rows), (f) => ROW_LENGTHS[f.properties.row_id]), canopy: sum(of(canopies), areaOf),
    interrow: sum(of(interrows), areaOf) - overlap, plants: of(canopies).length };
};
const areaCells = (m) => `${m.canopy.toFixed(2)},${(m.canopy / 1e4).toFixed(4)},${m.interrow.toFixed(2)},${(m.interrow / 1e4).toFixed(4)}`;
const all = measure(null);
const csv = [
  CSV_HEADER,
  `survey,,,${blocks.length},${rows.length},${all.rowLength.toFixed(2)},${areaCells(all)},${all.plants},`,
  ...["V01", "V02"].map((v) => { const m = measure(v); return `block,${v},,,${m.rows.length},${m.rowLength.toFixed(2)},${areaCells(m)},${m.plants},`; }),
  ...[...rows].sort((a, b) => a.properties.row_id.localeCompare(b.properties.row_id)).map(({ properties: p }) =>
    `row,${p.vineyard_id},${p.row_id},,,${ROW_LENGTHS[p.row_id].toFixed(2)},,,,,${p.plant_count},${p.row_structure}`),
].join("\n") + "\n";

const manifest = {
  annset_run_id: "mini", annset_source: "model", captured_at: "2025-05-20",
  counts: { blocks: 2, canopies: 4, interrows: interrows.length, route: 1, rows: 3, targets: 5, waste: 1 },
  crs: "EPSG:32635", generated_at: "2026-09-26T04:37:13+03:00", gsd_m: 0.025, license: "CC BY 4.0", name: "Sireț3",
  pipeline_version: "0000000000000000000000000000000000000000", run_id: "post-mini", source: "3DATA COLLECT / OpenAerialMap",
  stage: "model", survey_id: "siret3", tiles: 311,
};

const write = (name, text) => fs.writeFileSync(path.join(OUT, name), text);
fs.mkdirSync(OUT, { recursive: true });
write("manifest.json", JSON.stringify(manifest, null, 2) + "\n");
write("blocks.geojson", JSON.stringify(fc("blocks", blocks)));
write("rows.geojson", JSON.stringify(fc("rows", rows)));
write("canopies.geojsonl", canopies.map((f) => JSON.stringify(f)).join("\n") + "\n");
write("interrows.geojson", JSON.stringify(fc("interrows", interrows)));
write("waste.geojson", JSON.stringify(fc("waste", waste)));
// the pipeline wrote route_order as a float (1.0); JSON.stringify cannot, so patch the text
write("targets.geojson", JSON.stringify(fc("targets", targets)).replace(/"route_order":(\d+)/g, '"route_order":$1.0'));
write("route.geojson", JSON.stringify(fc("route", route)));
write("measurements.csv", csv);
console.log(`wrote ${path.relative(process.cwd(), OUT)}/ (${fs.readdirSync(OUT).length} files)`);
