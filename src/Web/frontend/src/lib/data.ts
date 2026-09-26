// Server-side readers for the generated static data (public/data, built by scripts/build-data.mjs).
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import type { FeatureCollection, Point } from "geojson";
import type { OverlayFiles, RowRecord, SurveySummary, TargetProps } from "./types";
import { parseTerrainMeta, TERRAIN_META_FILE, type TerrainMeta } from "./terrain";

export const SURVEY_ID = process.env.NEXT_PUBLIC_SURVEY_ID ?? "siret3-mock";
const DATA_DIR = path.join(process.cwd(), "public", "data", SURVEY_ID);

const readJson = async <T>(file: string): Promise<T> => JSON.parse(await readFile(path.join(DATA_DIR, file), "utf8")) as T;
const exists = (file: string) => access(path.join(DATA_DIR, file)).then(() => true, () => false);

export const getSummary = () => readJson<SurveySummary>("summary.json");
export const getRows = () => readJson<RowRecord[]>("rows.json");
export const getTargets = () => readJson<FeatureCollection<Point, TargetProps>>("targets.geojson");
/** Relief of the 3D view (scripts/build-terrain.mjs); null when not generated or unusable — the map then stays 2D only. */
export const getTerrain = (): Promise<TerrainMeta | null> =>
  readJson<unknown>(TERRAIN_META_FILE).then(parseTerrainMeta, (err: NodeJS.ErrnoException) => {
    if (err.code !== "ENOENT") console.warn(`terrain: ${TERRAIN_META_FILE} of ${SURVEY_ID} unreadable (${err.message}); 3D view disabled`);
    return null;
  });

/** The optional layers scripts/build-survey.mjs writes for a bundle that ships them (none on the mock). */
export const getOverlayFiles = async (): Promise<OverlayFiles> => {
  const [tiles, masks] = await Promise.all([exists("tiles.geojson"), exists("masks/index.json")]);
  return { tiles, masks };
};

export const dataUrl = (file: string) => `/data/${SURVEY_ID}/${file}`;
