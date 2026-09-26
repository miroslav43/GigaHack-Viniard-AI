// Synthetic canopy relief (scripts/build-terrain.mjs → public/data/<survey>/terrain/terrain.json).
// The orthophoto has no altitude: the relief is raised from the canopy annotations, for the map's oblique 3D view.

export interface TerrainMeta {
  version: 1;
  survey_id: string;
  encoding: "terrarium";
  tile_size: number;
  minzoom: number;
  maxzoom: number;
  /** [west, south, east, north], EPSG:4326 — MapLibre requests tiles only inside it (every one exists) */
  bounds: [number, number, number, number];
  /** canopy height encoded in the tiles (m); the view rescales it with terrain exaggeration */
  height_m: number;
  /** tile path template, relative to the terrain folder */
  tiles: string;
}

export const TERRAIN_DIR = "terrain";
export const TERRAIN_META_FILE = `${TERRAIN_DIR}/terrain.json`;

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isZoom = (v: unknown): v is number => isNum(v) && Number.isInteger(v) && v >= 0 && v <= 24;

/** The metadata if `json` is a usable terrain.json, else null (a stale or hand-edited file disables 3D, never breaks the map). */
export function parseTerrainMeta(json: unknown): TerrainMeta | null {
  if (typeof json !== "object" || json === null) return null;
  const m = json as Record<string, unknown>;
  const b = m.bounds;
  const boundsOk =
    Array.isArray(b) && b.length === 4 && b.every(isNum) && b[0] < b[2] && b[1] < b[3] && b[0] >= -180 && b[2] <= 180 && b[1] >= -90 && b[3] <= 90;
  const ok =
    m.version === 1 &&
    typeof m.survey_id === "string" &&
    m.encoding === "terrarium" &&
    isNum(m.tile_size) && m.tile_size > 0 &&
    isZoom(m.minzoom) && isZoom(m.maxzoom) && m.minzoom <= m.maxzoom &&
    boundsOk &&
    isNum(m.height_m) && m.height_m > 0 &&
    typeof m.tiles === "string" && /^\{z\}\/\{x\}\/\{y\}\.png$/.test(m.tiles);
  if (!ok) return null;
  return {
    version: 1,
    survey_id: m.survey_id as string,
    encoding: "terrarium",
    tile_size: m.tile_size as number,
    minzoom: m.minzoom as number,
    maxzoom: m.maxzoom as number,
    bounds: b as TerrainMeta["bounds"],
    height_m: m.height_m as number,
    tiles: m.tiles as string,
  };
}
