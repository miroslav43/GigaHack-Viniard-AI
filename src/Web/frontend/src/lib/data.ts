// Server-side readers for the generated static data (public/data, built by scripts/build-data.mjs).
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { FeatureCollection, Point } from "geojson";
import type { RowRecord, SurveySummary, TargetProps } from "./types";

export const SURVEY_ID = process.env.NEXT_PUBLIC_SURVEY_ID ?? "siret3-mock";
const DATA_DIR = path.join(process.cwd(), "public", "data", SURVEY_ID);

const readJson = async <T>(file: string): Promise<T> => JSON.parse(await readFile(path.join(DATA_DIR, file), "utf8")) as T;

export const getSummary = () => readJson<SurveySummary>("summary.json");
export const getRows = () => readJson<RowRecord[]>("rows.json");
export const getTargets = () => readJson<FeatureCollection<Point, TargetProps>>("targets.geojson");

export const dataUrl = (file: string) => `/data/${SURVEY_ID}/${file}`;
