// Builds everything the site reads from public/:
//   public/ortho/overview.webp + public/ortho/tiles/<tile>.jpg   (orthophoto from the 311 GeoTIFF tiles)
//   public/data/tiles.json                                        (tile index with 4326 corners)
//   public/data/<survey>/*.geojson, summary.json, measurements.csv, route.gpx
// Survey "siret3-mock" = the two organizer example tiles (CVAT annotations), measured in EPSG:32635.
// Usage: node scripts/build-data.mjs [--skip-ortho]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import proj4 from "proj4";
import sharp from "sharp";
import { XMLParser } from "fast-xml-parser";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(HERE, "..");
const ORGANIZER = process.env.ORGANIZER_DATA_DIR ?? path.resolve(FRONTEND, "../../../data & info");
const PUBLIC = path.join(FRONTEND, "public");
const SURVEY = "siret3-mock";
const OUT = path.join(PUBLIC, "data", SURVEY);
const SKIP_ORTHO = process.argv.includes("--skip-ortho");

// ---------- geometry helpers (planar, EPSG:32635 metres) ----------
const UTM = "+proj=utm +zone=35 +datum=WGS84 +units=m +no_defs";
const toLonLat = proj4(UTM, "EPSG:4326");
const ll = ([x, y]) => toLonLat.forward([x, y]).map((v) => Math.round(v * 1e7) / 1e7);

const TILE_M = 51.2;
const PX_M = 0.025;
const tileOrigin = (r, c) => [628992 + c * TILE_M, 5220966.4 - (r - 5) * TILE_M];
const parseTile = (name) => {
  const m = /siret3_r(\d{3})_c(\d{3})/.exec(name);
  return { r: Number(m[1]), c: Number(m[2]) };
};
const START = [629504.7, 5220250.75];

const ringArea = (ring) => {
  let s = 0;
  for (let i = 0; i < ring.length - 1; i++) s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
  return Math.abs(s) / 2;
};
const lineLength = (pts) => pts.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]), 0);
const closeRing = (pts) =>
  pts[0][0] === pts.at(-1)[0] && pts[0][1] === pts.at(-1)[1] ? pts : [...pts, pts[0]];
const centroid = (ring) => {
  const n = ring.length - 1;
  return [ring.slice(0, n).reduce((s, p) => s + p[0], 0) / n, ring.slice(0, n).reduce((s, p) => s + p[1], 0) / n];
};
// project point onto segment a→b; returns distance along (t, metres) and perpendicular distance
const projectOnSegment = (p, a, b) => {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len;
  const d = Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / len;
  return { t, d, len };
};
const convexHull = (points) => {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), p) <= 0) lower.pop();
    lower.push(p);
  }
  for (const p of pts.reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), p) <= 0) upper.pop();
    upper.push(p);
  }
  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  return [...hull, hull[0]];
};
const omit = (o, keys) => Object.fromEntries(Object.entries(o).filter(([k]) => !keys.includes(k)));
const r2 = (v) => Math.round(v * 100) / 100;
const r4 = (v) => Math.round(v * 1e4) / 1e4;

// GeoJSON writers (reproject UTM → 4326)
const mapCoords = (coords, depth) => (depth === 0 ? ll(coords) : coords.map((c) => mapCoords(c, depth - 1)));
const DEPTH = { Point: 0, LineString: 1, MultiLineString: 2, Polygon: 2, MultiPolygon: 3 };
const feature = (type, coords, properties, id) => ({
  type: "Feature",
  ...(id !== undefined ? { id } : {}),
  properties,
  geometry: { type, coordinates: mapCoords(coords, DEPTH[type]) },
});
const writeJson = (file, obj) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(obj));
};
const fc = (features) => ({ type: "FeatureCollection", features });

// ---------- tile index + orthophoto ----------
// GeoTIFFs are git-ignored; without them (e.g. CI) the committed index keeps every figure identical and ortho is skipped.
const TILES_DIR = path.join(ORGANIZER, "01_tiles");
const tiffsOnDisk = fs.existsSync(TILES_DIR)
  ? fs
      .readdirSync(TILES_DIR, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .flatMap((d) =>
        fs
          .readdirSync(path.join(TILES_DIR, d.name))
          .filter((f) => f.endsWith(".tif"))
          .map((f) => path.join(TILES_DIR, d.name, f)),
      )
      .sort()
  : [];
// the folder exists in git (overview.png) but the GeoTIFFs do not — decide on the files, not the folder
const HAS_TIFFS = tiffsOnDisk.length > 0;
const tileFiles = HAS_TIFFS
  ? tiffsOnDisk
  : JSON.parse(fs.readFileSync(path.join(FRONTEND, "data", "tiles_index.json"), "utf8")).tiles.map((id) => `${id}.tif`);
if (!HAS_TIFFS) console.warn(`! no GeoTIFF tiles under ${TILES_DIR} — using data/tiles_index.json, orthophoto skipped`);
if (tileFiles.length !== 311) console.warn(`! expected 311 tiles, found ${tileFiles.length}`);

const tiles = tileFiles.map((file) => {
  const id = path.basename(file, ".tif");
  const { r, c } = parseTile(id);
  const [x0, y0] = tileOrigin(r, c);
  const x1 = x0 + TILE_M, y1 = y0 - TILE_M;
  // MapLibre image source corner order: top-left, top-right, bottom-right, bottom-left
  return { id, r, c, file, corners: [ll([x0, y0]), ll([x1, y0]), ll([x1, y1]), ll([x0, y1])] };
});
const rows = tiles.map((t) => t.r), cols = tiles.map((t) => t.c);
const grid = { rMin: Math.min(...rows), rMax: Math.max(...rows), cMin: Math.min(...cols), cMax: Math.max(...cols) };
const [gx0, gy0] = tileOrigin(grid.rMin, grid.cMin);
const gx1 = gx0 + (grid.cMax - grid.cMin + 1) * TILE_M, gy1 = gy0 - (grid.rMax - grid.rMin + 1) * TILE_M;
writeJson(path.join(PUBLIC, "data", "tiles.json"), {
  tile_m: TILE_M,
  overview: { url: "/ortho/overview.webp", corners: [ll([gx0, gy0]), ll([gx1, gy0]), ll([gx1, gy1]), ll([gx0, gy1])] },
  tiles: tiles.map(({ id, r, c, corners }) => ({ id, r, c, url: `/ortho/tiles/${id}.jpg`, corners })),
});

if (!SKIP_ORTHO && HAS_TIFFS) {
  const OV = 128; // px per tile in the overview mosaic (~0.4 m/px)
  const DETAIL = 1024; // px per tile for the detail layer (5 cm/px)
  fs.mkdirSync(path.join(PUBLIC, "ortho", "tiles"), { recursive: true });
  const t0 = Date.now();
  const composites = [];
  for (const [i, t] of tiles.entries()) {
    const out = path.join(PUBLIC, "ortho", "tiles", `${t.id}.jpg`);
    if (!fs.existsSync(out)) await sharp(t.file).resize(DETAIL).jpeg({ quality: 80, mozjpeg: true }).toFile(out);
    const small = await sharp(t.file).resize(OV).ensureAlpha().raw().toBuffer();
    // treat pure-black nodata as transparent in the overview
    for (let p = 0; p < small.length; p += 4) if (small[p] + small[p + 1] + small[p + 2] < 12) small[p + 3] = 0;
    composites.push({
      input: small,
      raw: { width: OV, height: OV, channels: 4 },
      left: (t.c - grid.cMin) * OV,
      top: (t.r - grid.rMin) * OV,
    });
    if ((i + 1) % 50 === 0) console.log(`  ortho ${i + 1}/${tiles.length}`);
  }
  await sharp({
    create: {
      width: (grid.cMax - grid.cMin + 1) * OV,
      height: (grid.rMax - grid.rMin + 1) * OV,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    },
  })
    .composite(composites)
    .webp({ quality: 72 })
    .toFile(path.join(PUBLIC, "ortho", "overview.webp"));
  console.log(`ortho: ${tiles.length} tiles in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}

// ---------- CVAT examples → mock survey ----------
const xml = fs.readFileSync(path.join(ORGANIZER, "05_examples/siret3_examples_cvat/annotations.xml"), "utf8");
const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "",
  isArray: (name) => ["image", "polygon", "polyline", "box", "attribute"].includes(name),
});
const images = parser.parse(xml).annotations.image;
const attrsOf = (el) => Object.fromEntries((el.attribute ?? []).map((a) => [a.name, String(a["#text"] ?? "")]));

const canopies = [], interrows = [], rowPieces = [], waste = [];
for (const im of images) {
  const tile = path.basename(im.name, ".tif");
  const { r, c } = parseTile(tile);
  const [x0, y0] = tileOrigin(r, c);
  const toUtm = (s) => s.split(";").map((p) => {
    const [px, py] = p.split(",").map(Number);
    return [x0 + px * PX_M, y0 - py * PX_M];
  });
  for (const el of im.polygon ?? []) {
    const ring = closeRing(toUtm(el.points));
    const a = attrsOf(el);
    if (el.label === "vineyard") canopies.push({ tile, ring, vineyard_id: a.vineyard_id, area: ringArea(ring) });
    if (el.label === "interrow_area")
      interrows.push({ tile, ring, vineyard_id: a.vineyard_id, cover: a.interrow_cover, area: ringArea(ring) });
  }
  for (const el of im.polyline ?? []) {
    if (el.label !== "row") continue;
    const a = attrsOf(el);
    rowPieces.push({ tile, x0, y0, pts: toUtm(el.points), ...a });
  }
  for (const el of im.box ?? []) {
    if (el.label !== "waste") continue;
    const [xa, ya] = [x0 + el.xtl * PX_M, y0 - el.ytl * PX_M], [xb, yb] = [x0 + el.xbr * PX_M, y0 - el.ybr * PX_M];
    waste.push({ tile, ring: [[xa, ya], [xb, ya], [xb, yb], [xa, yb], [xa, ya]], vineyard_id: attrsOf(el).vineyard_id });
  }
}

// assign canopies to the nearest row piece (≤ 0.35 m from the axis) and find gaps along each piece
const targets = [];
for (const piece of rowPieces) {
  const [a, b] = [piece.pts[0], piece.pts.at(-1)];
  const intervals = [];
  for (const can of canopies) {
    if (can.tile !== piece.tile) continue;
    const cp = projectOnSegment(centroid(can.ring), a, b);
    if (cp.d > 0.35) continue;
    can.row_id ??= piece.row_id;
    const ts = can.ring.map((p) => projectOnSegment(p, a, b).t);
    intervals.push([Math.max(0, Math.min(...ts)), Math.min(cp.len, Math.max(...ts))]);
  }
  intervals.sort((p, q) => p[0] - q[0]);
  const len = lineLength(piece.pts);
  let cursor = 0, maxGap = 0;
  const gaps = [];
  for (const [s, e] of [...intervals, [len, len]]) {
    if (s - cursor > maxGap) maxGap = s - cursor;
    if (s - cursor >= 5) gaps.push([cursor, s]);
    cursor = Math.max(cursor, e);
  }
  piece.length = len;
  piece.plants = intervals.length;
  piece.maxGap = maxGap;
  for (const [s, e] of gaps) {
    const t = (s + e) / 2 / len;
    targets.push({
      type: "gap",
      vineyard_id: piece.vineyard_id,
      row_id: piece.row_id,
      gap_length_m: r2(e - s),
      tile: piece.tile,
      pt: [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t],
    });
  }
}

// physical rows (pieces joined by row_id)
const rowsById = new Map();
for (const p of rowPieces) {
  const row = rowsById.get(p.row_id) ?? { row_id: p.row_id, vineyard_id: p.vineyard_id, pieces: [] };
  row.pieces.push(p);
  rowsById.set(p.row_id, row);
}
const physRows = [...rowsById.values()].map((row) => {
  const structures = row.pieces.map((p) => p.row_structure);
  const row_structure = structures.includes("disrupted")
    ? "disrupted"
    : structures.every((s) => s === "unassessable") ? "unassessable" : "regular";
  return {
    ...row,
    row_structure,
    length_m: r2(row.pieces.reduce((s, p) => s + p.length, 0)),
    plant_count: row.pieces.reduce((s, p) => s + p.plants, 0),
    max_gap_m: r2(Math.max(...row.pieces.map((p) => p.maxGap))),
    tile_structures: Object.fromEntries(row.pieces.map((p) => [p.tile, p.row_structure])),
  };
});
physRows.sort((p, q) => p.row_id.localeCompare(q.row_id, "en", { numeric: true }));

// blocks: convex hull of every object of a vineyard_id (display outline; areas come from canopies/inter-rows)
const blockIds = [...new Set([...physRows, ...canopies, ...interrows].map((o) => o.vineyard_id).filter(Boolean))].sort();
const blocks = blockIds.map((id) => {
  const pts = [
    ...canopies.filter((o) => o.vineyard_id === id).flatMap((o) => o.ring),
    ...interrows.filter((o) => o.vineyard_id === id).flatMap((o) => o.ring),
  ];
  const hull = convexHull(pts);
  const rowsOf = physRows.filter((r) => r.vineyard_id === id);
  const cans = canopies.filter((o) => o.vineyard_id === id);
  const irs = interrows.filter((o) => o.vineyard_id === id);
  return {
    vineyard_id: id,
    hull,
    outline_area_m2: r2(ringArea(hull)),
    row_count: rowsOf.length,
    row_length_m: r2(rowsOf.reduce((s, r) => s + r.length_m, 0)),
    canopy_count: cans.length,
    canopy_area_m2: r2(cans.reduce((s, o) => s + o.area, 0)),
    interrow_area_m2: r2(irs.reduce((s, o) => s + o.area, 0)),
    plant_count: rowsOf.reduce((s, r) => s + r.plant_count, 0),
    disrupted_rows: rowsOf.filter((r) => r.row_structure === "disrupted").length,
    tiles: [...new Set(cans.map((o) => o.tile))],
  };
});

// targets: ids, nearest-neighbour route from START (MOCK: straight segments, not constrained to passages)
targets.forEach((t, i) => (t.target_id = `T${String(i + 1).padStart(3, "0")}`));
const tourLength = (order) => lineLength([START, ...order.map((t) => t.pt), START]);
const remaining = [...targets], ordered = [];
let here = START;
while (remaining.length) {
  remaining.sort((p, q) => Math.hypot(p.pt[0] - here[0], p.pt[1] - here[1]) - Math.hypot(q.pt[0] - here[0], q.pt[1] - here[1]));
  const next = remaining.shift();
  ordered.push(next);
  here = next.pt;
}
ordered.forEach((t, i) => (t.route_order = i + 1));
const routeCoords = [START, ...ordered.map((t) => t.pt), START];
const routeLength = r2(lineLength(routeCoords));
const baselineLength = r2(tourLength(targets)); // naive: visit in id order

// ---------- totals ----------
const totalCanopy = canopies.reduce((s, o) => s + o.area, 0);
const totalInterrow = interrows.reduce((s, o) => s + o.area, 0);
const totalRowLen = physRows.reduce((s, r) => s + r.length_m, 0);
// municipality boundaries (real OSM administrative relations, data/osm/*)
const toUtm = proj4("EPSG:4326", UTM);
const polygonAreaM2 = (geom) =>
  (geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates).reduce(
    (sum, poly) => sum + poly.reduce((s, ring, i) => s + (i === 0 ? 1 : -1) * ringArea(ring.map((p) => toUtm.forward(p))), 0),
    0,
  );
const UAT_FILES = { sireti: "sireti_19100171.geojson", cojusna: "cojusna_19100156.geojson" };
const uats = Object.fromEntries(
  Object.entries(UAT_FILES).map(([key, file]) => {
    const geo = JSON.parse(fs.readFileSync(path.join(FRONTEND, "data", "osm", file), "utf8"));
    writeJson(path.join(PUBLIC, "data", "ref", `geofence_${key}.geojson`), geo);
    const f = geo.features[0];
    return [key, { key, name: f.properties.name, osm_relation_id: f.properties.osm_relation_id, area_ha: r2(polygonAreaM2(f.geometry) / 1e4) }];
  }),
);
writeJson(path.join(PUBLIC, "data", "uats.json"), uats);
const geofenceArea = uats.sireti.area_ha * 1e4;
const coverCounts = interrows.reduce((m, o) => ((m[o.cover] = (m[o.cover] ?? 0) + 1), m), {});
const structureCounts = physRows.reduce((m, r) => ((m[r.row_structure] = (m[r.row_structure] ?? 0) + 1), m), {});

const summary = {
  survey: {
    id: SURVEY,
    name: "Sireț3 — date de test (2 tile-uri exemplu)",
    captured_at: "2025-05-20",
    gsd_m: 0.025,
    crs: "EPSG:32635",
    source: "3DATA COLLECT / OpenAerialMap",
    license: "CC BY 4.0",
    stage: "mock",
    mock: true,
    tiles_total: tiles.length,
    tiles_annotated: images.length,
    surveyed_area_ha: r4((tiles.length * TILE_M * TILE_M) / 1e4),
  },
  uat: {
    name: "Primăria Sireți",
    district: "raionul Strășeni",
    country: "Republica Moldova",
    osm_relation_id: 19100171,
    area_ha: r2(geofenceArea / 1e4),
  },
  totals: {
    block_count: blocks.length,
    row_count: physRows.length,
    row_length_m: r2(totalRowLen),
    canopy_count: canopies.length,
    canopy_area_m2: r2(totalCanopy),
    canopy_area_ha: r4(totalCanopy / 1e4),
    interrow_count: interrows.length,
    interrow_area_m2: r2(totalInterrow),
    interrow_area_ha: r4(totalInterrow / 1e4),
    plant_count: physRows.reduce((s, r) => s + r.plant_count, 0),
    disrupted_rows: structureCounts.disrupted ?? 0,
    waste_count: waste.length,
    target_count: targets.length,
  },
  structure_counts: structureCounts,
  cover_counts: coverCounts,
  route: {
    mock: true,
    length_m: routeLength,
    duration_min: r2((routeLength / 1000 / 4) * 60),
    baseline_length_m: baselineLength,
    speed_kmh: 4,
  },
  blocks: blocks.map((b) => omit(b, ["hull", "tiles"])),
};

// ---------- write layers ----------
writeJson(path.join(OUT, "summary.json"), summary);
writeJson(path.join(OUT, "rows.json"), physRows.map((r) => omit(r, ["pieces"])));
writeJson(
  path.join(OUT, "rows.geojson"),
  fc(physRows.map((r, i) => feature("MultiLineString", r.pieces.map((p) => p.pts), omit(r, ["pieces"]), i + 1))),
);
writeJson(
  path.join(OUT, "canopies.geojson"),
  fc(canopies.map((o, i) =>
    feature("Polygon", [o.ring], {
      canopy_id: `${o.tile}#${String(i + 1).padStart(4, "0")}`,
      vineyard_id: o.vineyard_id,
      row_id: o.row_id ?? null,
      tile: o.tile,
      area_m2: r4(o.area),
    }, i + 1),
  )),
);
writeJson(
  path.join(OUT, "interrows.geojson"),
  fc(interrows.map((o, i) =>
    feature("Polygon", [o.ring], {
      interrow_id: `${o.tile}#IR${String(i + 1).padStart(3, "0")}`,
      vineyard_id: o.vineyard_id,
      interrow_cover: o.cover,
      tile: o.tile,
      area_m2: r2(o.area),
    }, i + 1),
  )),
);
writeJson(
  path.join(OUT, "blocks.geojson"),
  fc(blocks.map((b, i) => feature("Polygon", [b.hull], omit(b, ["hull", "tiles"]), i + 1))),
);
writeJson(
  path.join(OUT, "waste.geojson"),
  fc(waste.map((o, i) => feature("Polygon", [o.ring], { waste_id: `W${i + 1}`, vineyard_id: o.vineyard_id ?? null, tile: o.tile }, i + 1))),
);
writeJson(
  path.join(OUT, "targets.geojson"),
  fc(ordered.map(({ pt, ...t }, i) => feature("Point", pt, t, i + 1))),
);
writeJson(
  path.join(OUT, "route.geojson"),
  fc([feature("LineString", routeCoords, { length_m: routeLength, duration_min: summary.route.duration_min, mock: true })]),
);
// route in EPSG:32635, same shape as the root deliverable
writeJson(path.join(OUT, "route_EPSG32635.geojson"), {
  type: "FeatureCollection",
  crs: { type: "name", properties: { name: "urn:ogc:def:crs:EPSG::32635" } },
  features: [{ type: "Feature", properties: { length_m: routeLength, mock: true }, geometry: { type: "LineString", coordinates: routeCoords.map((p) => p.map(r4)) } }],
});
fs.writeFileSync(
  path.join(OUT, "route.gpx"),
  `<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="Solemtrix" xmlns="http://www.topografix.com/GPX/1/1">\n` +
    ordered.map((t) => { const [lon, lat] = ll(t.pt); return `  <wpt lat="${lat}" lon="${lon}"><name>${t.route_order}. ${t.target_id}</name><desc>${t.type} ${t.row_id ?? ""}</desc></wpt>\n`; }).join("") +
    `  <trk><name>Ruta de inspecție (${routeLength} m)</name><trkseg>\n` +
    routeCoords.map((p) => { const [lon, lat] = ll(p); return `    <trkpt lat="${lat}" lon="${lon}"/>\n`; }).join("") +
    `  </trkseg></trk>\n</gpx>\n`,
);

// organizer layers + geofence
for (const name of ["passages", "forbidden", "study_area", "start"]) {
  const src = JSON.parse(fs.readFileSync(path.join(ORGANIZER, "02_route", `${name}.geojson`), "utf8"));
  writeJson(
    path.join(PUBLIC, "data", "ref", `${name}.geojson`),
    fc(src.features.map((f) => feature(f.geometry.type, f.geometry.coordinates, f.properties ?? {}))),
  );
}

// measurements.csv (format from src/Web/CLAUDE.md §6.4)
const csv = [
  "level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,interrow_area_m2,interrow_area_ha,plant_count,row_structure",
  `survey,,,${blocks.length},${physRows.length},${totalRowLen.toFixed(2)},${totalCanopy.toFixed(2)},${(totalCanopy / 1e4).toFixed(4)},${totalInterrow.toFixed(2)},${(totalInterrow / 1e4).toFixed(4)},${summary.totals.plant_count},`,
  ...blocks.map((b) => `block,${b.vineyard_id},,,${b.row_count},${b.row_length_m.toFixed(2)},${b.canopy_area_m2.toFixed(2)},${(b.canopy_area_m2 / 1e4).toFixed(4)},${b.interrow_area_m2.toFixed(2)},${(b.interrow_area_m2 / 1e4).toFixed(4)},${b.plant_count},`),
  ...physRows.map((r) => `row,${r.vineyard_id},${r.row_id},,,${r.length_m.toFixed(2)},,,,,${r.plant_count},${r.row_structure}`),
].join("\n");
fs.writeFileSync(path.join(OUT, "measurements.csv"), csv + "\n");

console.log(
  `${SURVEY}: ${blocks.length} blocks, ${physRows.length} rows (${totalRowLen.toFixed(1)} m), ` +
    `${canopies.length} canopies (${totalCanopy.toFixed(1)} m²), ${interrows.length} inter-rows (${totalInterrow.toFixed(1)} m²), ` +
    `${targets.length} targets, route ${routeLength} m (naive ${baselineLength} m); geofence ${(geofenceArea / 1e4).toFixed(1)} ha`,
);
