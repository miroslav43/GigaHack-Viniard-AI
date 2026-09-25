// Municipality (UAT) model. Signed-in users get their UAT from the database (public.uat_public, RLS);
// the public demo (and CI, without Supabase) uses the built-in Sireți fixture below.
import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Geometry } from "geojson";

export type Country = "MD" | "RO";

export interface UatInfo {
  key: string;
  name: string;
  district: string | null;
  country: Country;
  osmRelationId: number | null;
  areaHa: number;
  geofence: Geometry;
  /** survey ids whose footprint intersects the boundary */
  surveys: string[];
}

/** Row of the public.uat_public view. */
export interface UatRow {
  key: string;
  name: string;
  district: string | null;
  country: Country;
  osm_relation_id: number | null;
  area_ha: number;
  active: boolean;
  created_at: string;
  geofence: Geometry;
}

export const fromRow = (r: UatRow, surveys: string[]): UatInfo => ({
  key: r.key,
  name: r.name,
  district: r.district,
  country: r.country,
  osmRelationId: r.osm_relation_id,
  areaHa: Number(r.area_ha),
  geofence: r.geofence,
  surveys,
});

let demoUat: Promise<UatInfo> | null = null;

/** Sireți from the committed OSM fixture: the public demo municipality. */
export function getDemoUat(): Promise<UatInfo> {
  demoUat ??= (async () => {
    const root = process.cwd();
    const osm = JSON.parse(await readFile(path.join(root, "data", "osm", "sireti_19100171.geojson"), "utf8"));
    const areas = JSON.parse(await readFile(path.join(root, "public", "data", "uats.json"), "utf8"));
    return {
      key: "sireti",
      name: "Sireți",
      district: "raionul Strășeni",
      country: "MD",
      osmRelationId: 19100171,
      areaHa: areas.sireti.area_ha,
      geofence: osm.features[0].geometry,
      surveys: ["siret3-mock"],
    };
  })();
  return demoUat;
}
