// The live cadastre layer: AGCC (Agenția Servicii Publice, Departamentul Cadastru) public GeoServer, no login,
// CORS *, Fees=NONE. The only place its URL, layer name and limits live. Informative: the official extract is AGCC's.
import type { MultiPolygon, Polygon } from "geojson";

export const CADASTRE_ORIGIN = "https://geodata.gov.md";
export const CADASTRE_WMS_URL = `${CADASTRE_ORIGIN}/geoserver/wms`;
export const CADASTRE_LAYER = "cadastru_data:terenuri";
/** the WMS is drawn and queried from this zoom on (parcels are unreadable further out) */
export const CADASTRE_MIN_ZOOM = 16;
export const CADASTRE_MAX_ZOOM = 22;
export const CADASTRE_TILE_PX = 256;
export const CADASTRE_TIMEOUT_MS = 8000;
export const CADASTRE_ATTRIBUTION = "Cadastru © AGCC (geodata.gov.md), informativ";
/** half-size in degrees of the GetFeatureInfo box around the click (~4–5 m) */
const PICK_HALF_DEG = 0.00005;
const PICK_PX = 101;

const COMMON = `service=WMS&version=1.1.1&layers=${CADASTRE_LAYER}`;

/** MapLibre raster-source tile template (EPSG:3857 bbox filled in by MapLibre). */
export const CADASTRE_TILE_URL =
  `${CADASTRE_WMS_URL}?${COMMON}&request=GetMap&styles=&format=image/png&transparent=true&srs=EPSG:3857` +
  `&width=${CADASTRE_TILE_PX}&height=${CADASTRE_TILE_PX}&bbox={bbox-epsg-3857}`;

/** GetFeatureInfo at a point: the parcel under the centre pixel of a small box, as GeoJSON in EPSG:4326. */
export const cadastreInfoUrl = (lon: number, lat: number) => {
  const d = PICK_HALF_DEG;
  const bbox = [lon - d, lat - d, lon + d, lat + d].map((v) => v.toFixed(7)).join(",");
  const mid = Math.floor(PICK_PX / 2);
  return (
    `${CADASTRE_WMS_URL}?${COMMON}&request=GetFeatureInfo&query_layers=${CADASTRE_LAYER}&info_format=application/json` +
    `&feature_count=1&srs=EPSG:4326&bbox=${bbox}&width=${PICK_PX}&height=${PICK_PX}&x=${mid}&y=${mid}`
  );
};

/** One cadastral parcel. The server also sends an HTML `description`: it is dropped, never rendered. */
export interface CadastreParcel {
  codcadastral: string | null;
  cod_parcel: string | null;
  landuse: string | null;
  typeproperty: string | null;
  /** as the server writes it, e.g. "0.12 ha" */
  aria: string | null;
  /** EPSG:4326 outline for the highlight; null when absent or not in lon/lat */
  geometry: Polygon | MultiPolygon | null;
}

const text = (v: unknown) => (typeof v === "string" && v.trim() !== "" ? v.trim() : null);
const isLonLat = (p: unknown) => Array.isArray(p) && Math.abs(Number(p[0])) <= 180 && Math.abs(Number(p[1])) <= 90;
const firstPosition = (g: Polygon | MultiPolygon) => (g.type === "Polygon" ? g.coordinates[0]?.[0] : g.coordinates[0]?.[0]?.[0]);

const outline = (g: unknown): Polygon | MultiPolygon | null => {
  if (!g || typeof g !== "object") return null;
  const geom = g as Polygon | MultiPolygon;
  if ((geom.type !== "Polygon" && geom.type !== "MultiPolygon") || !Array.isArray(geom.coordinates)) return null;
  return isLonLat(firstPosition(geom)) ? { type: geom.type, coordinates: geom.coordinates } as Polygon | MultiPolygon : null;
};

/** The GetFeatureInfo response → the first parcel, or null when there is none at the point. Throws on anything else. */
export function parseParcel(json: unknown): CadastreParcel | null {
  const fc = json as { type?: unknown; features?: unknown };
  if (!fc || fc.type !== "FeatureCollection" || !Array.isArray(fc.features)) throw new Error("unexpected cadastre response");
  const feature = fc.features[0] as { properties?: Record<string, unknown>; geometry?: unknown } | undefined;
  if (!feature) return null;
  const p = feature.properties ?? {};
  return {
    codcadastral: text(p.codcadastral),
    cod_parcel: text(p.cod_parcel),
    landuse: text(p.landuse),
    typeproperty: text(p.typeproperty),
    aria: text(p.aria),
    geometry: outline(feature.geometry),
  };
}

/** Queries the parcel at [lon, lat]; aborts after CADASTRE_TIMEOUT_MS or when `signal` aborts. */
export async function fetchParcel(lon: number, lat: number, signal?: AbortSignal): Promise<CadastreParcel | null> {
  const timeout = AbortSignal.timeout(CADASTRE_TIMEOUT_MS);
  const res = await fetch(cadastreInfoUrl(lon, lat), { signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return parseParcel(await res.json());
}
