// Administrative boundaries from OpenStreetMap via Nominatim (server-side only; usage policy: identify the app,
// low volume, cache). Only relations tagged boundary=administrative in Moldova or Romania are accepted.
import "server-only";
import type { Geometry } from "geojson";

const NOMINATIM = "https://nominatim.openstreetmap.org";
const HEADERS = { "User-Agent": "Solemtrix/0.1 (GigaHack 2026; https://github.com/miroslav43/GigaHack-Viniard-AI)" };
const COUNTRIES = new Set(["md", "ro"]);

export interface OsmSearchHit {
  osmRelationId: number;
  name: string;
  displayName: string;
  country: "MD" | "RO";
  placeRank: number;
}

export interface OsmBoundary extends OsmSearchHit {
  district: string;
  geometry: Geometry;
}

interface NominatimPlace {
  osm_type: string;
  osm_id: number;
  class?: string;
  category?: string;
  type: string;
  place_rank: number;
  name?: string;
  display_name: string;
  address?: Record<string, string>;
  namedetails?: Record<string, string>;
  geojson?: Geometry;
}

const isAdminRelation = (p: NominatimPlace) =>
  p.osm_type.toLowerCase().startsWith("r") && (p.class ?? p.category) === "boundary" && p.type === "administrative";

const toHit = (p: NominatimPlace): OsmSearchHit | null => {
  const cc = p.address?.country_code?.toLowerCase();
  if (!cc || !COUNTRIES.has(cc) || !isAdminRelation(p)) return null;
  return {
    osmRelationId: p.osm_id,
    name: p.namedetails?.["name:ro"] ?? p.name ?? p.display_name.split(",")[0],
    displayName: p.display_name,
    country: cc.toUpperCase() as "MD" | "RO",
    placeRank: p.place_rank,
  };
};

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: HEADERS, next: { revalidate: 86400 } });
  if (!res.ok) throw new Error(`osm_http_${res.status}`);
  return (await res.json()) as T;
}

export async function searchOsmBoundaries(query: string): Promise<OsmSearchHit[]> {
  if (/^\d+$/.test(query)) {
    const b = await fetchOsmBoundary(Number(query)).catch(() => null);
    return b ? [{ osmRelationId: b.osmRelationId, name: b.name, displayName: b.displayName, country: b.country, placeRank: b.placeRank }] : [];
  }
  const params = new URLSearchParams({
    q: query,
    countrycodes: "md,ro",
    format: "jsonv2",
    addressdetails: "1",
    namedetails: "1",
    limit: "15",
  });
  const places = await get<NominatimPlace[]>(`${NOMINATIM}/search?${params}`);
  return places
    .map(toHit)
    .filter((h): h is OsmSearchHit => h !== null)
    .sort((a, b) => a.placeRank - b.placeRank);
}

export async function fetchOsmBoundary(osmRelationId: number): Promise<OsmBoundary> {
  if (!Number.isInteger(osmRelationId) || osmRelationId <= 0) throw new Error("invalid_osm_id");
  const params = new URLSearchParams({
    osm_ids: `R${osmRelationId}`,
    format: "json",
    polygon_geojson: "1",
    addressdetails: "1",
    namedetails: "1",
  });
  const [place] = await get<NominatimPlace[]>(`${NOMINATIM}/lookup?${params}`);
  const hit = place && toHit(place);
  if (!hit) throw new Error("osm_not_admin_boundary");
  const g = place.geojson;
  if (!g || (g.type !== "Polygon" && g.type !== "MultiPolygon")) throw new Error("osm_no_polygon");
  const a = place.address ?? {};
  // district (raion) / county (județ); for communes under a municipality (e.g. Chișinău) the parent sits in `municipality`/`city`
  return { ...hit, district: a.county ?? a.municipality ?? a.state_district ?? a.city ?? a.state ?? "", geometry: g };
}
