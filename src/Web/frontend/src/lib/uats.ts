// Municipalities (UAT) known to the app. Boundaries are real OSM administrative relations (data/osm/*).
// Until the Supabase data layer exists, the survey list per UAT lives here; isolation is enforced by
// only ever serving a UAT the surveys inside its own boundary.
export type UatKey = "sireti" | "cojusna";

export interface Uat {
  key: UatKey;
  osmRelationId: number;
  geofenceUrl: string;
  surveys: string[];
}

export const UATS: Record<UatKey, Uat> = {
  sireti: { key: "sireti", osmRelationId: 19100171, geofenceUrl: "/data/ref/geofence_sireti.geojson", surveys: ["siret3-mock"] },
  cojusna: { key: "cojusna", osmRelationId: 19100156, geofenceUrl: "/data/ref/geofence_cojusna.geojson", surveys: [] },
};

export const DEFAULT_UAT: UatKey = "sireti";

export const isUatKey = (v: unknown): v is UatKey => typeof v === "string" && v in UATS;
