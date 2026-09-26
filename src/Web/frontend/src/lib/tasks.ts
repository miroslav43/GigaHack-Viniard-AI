// Field tasks: shared types and the inspection targets of a survey (from the static AI bundle).
import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { FeatureCollection, Point } from "geojson";
import { SURVEY_ID } from "./data";
import type { TargetProps } from "./types";

export type TaskStatus = "open" | "in_progress" | "done" | "cancelled";
export type TaskKind = "gap" | "missing" | "waste" | "other";

/** Row of the public.task_public view. */
export interface TaskRow {
  id: number;
  uat_key: string;
  survey_id: string | null;
  target_id: string | null;
  kind: TaskKind;
  title: string;
  description: string | null;
  vineyard_id: string | null;
  row_id: string | null;
  gap_length_m: number | null;
  lon: number | null;
  lat: number | null;
  assignee: string | null;
  assignee_email: string | null;
  status: TaskStatus;
  priority: number;
  due_date: string | null;
  resolution_note: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface Target {
  survey_id: string;
  target_id: string;
  kind: TaskKind;
  vineyard_id: string | null;
  row_id: string | null;
  gap_length_m: number | null;
  route_order: number | null;
  lon: number;
  lat: number;
}

const KINDS: TaskKind[] = ["gap", "missing", "waste"];

/**
 * The survey the app serves (NEXT_PUBLIC_SURVEY_ID) if the municipality has it, else none. Tasks and targets
 * follow it: a municipality with several surveys (e.g. siret3-mock and siret3) never mixes their targets.
 */
export const activeSurveys = (uatSurveys: string[]) => (uatSurveys.includes(SURVEY_ID) ? [SURVEY_ID] : []);

/** PostgREST filter: tasks of the active survey, plus tasks not tied to any survey. */
export const ACTIVE_SURVEY_FILTER = `survey_id.eq.${SURVEY_ID},survey_id.is.null`;

/** Inspection targets of the given surveys (targets.geojson of each static bundle). */
export async function readTargets(surveyIds: string[]): Promise<Target[]> {
  const out: Target[] = [];
  for (const id of surveyIds) {
    if (!/^[a-z0-9-]+$/.test(id)) continue;
    const file = path.join(process.cwd(), "public", "data", id, "targets.geojson");
    const fc = JSON.parse(await readFile(file, "utf8").catch(() => '{"features":[]}')) as FeatureCollection<Point, TargetProps>;
    for (const f of fc.features) {
      const p = f.properties;
      out.push({
        survey_id: id,
        target_id: p.target_id,
        kind: KINDS.includes(p.type as TaskKind) ? (p.type as TaskKind) : "other",
        vineyard_id: p.vineyard_id ?? null,
        row_id: p.row_id ?? null,
        gap_length_m: p.gap_length_m ?? null,
        route_order: p.route_order ?? null,
        lon: f.geometry.coordinates[0],
        lat: f.geometry.coordinates[1],
      });
    }
  }
  return out;
}

/** Stored (Romanian) title; the UI renders a localized title from kind + fields. */
export function taskTitle(t: Pick<Target, "kind" | "row_id" | "vineyard_id" | "gap_length_m">) {
  if (t.kind === "gap") return `Gol ${t.gap_length_m ? `de ${t.gap_length_m.toFixed(1).replace(".", ",")} m ` : ""}în rândul ${t.row_id ?? "?"}`;
  if (t.kind === "missing") return `Plante lipsă în rândul ${t.row_id ?? "?"}`;
  if (t.kind === "waste") return `Deșeuri în blocul ${t.vineyard_id ?? "?"}`;
  return "Verificare pe teren";
}
