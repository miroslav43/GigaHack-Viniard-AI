// Live tile analysis (/analiza): the job-folder contract written by `vineyard demo tile` (src/AI) and read by the
// API routes and the page. Spec: src/Web/docs/design/2026-09-27-analiza-tif-spec.md. Pure module (no Node or React
// imports), so `node --test` can run its tests directly.

export type JobState = "queued" | "running" | "done" | "error";

export const ERROR_CODES = [
  "not_geotiff",
  "wrong_crs",
  "wrong_bands",
  "wrong_size",
  "wrong_gsd",
  "off_grid",
  "pipeline_failed",
  "busy",
  "too_large",
] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

export interface JobError {
  code: ErrorCode;
  message: string;
}

/** status.json */
export interface JobStatus {
  state: JobState;
  /** pipeline stage running now (ingest, tile_prep, rows_detect, …) */
  stage: string | null;
  /** 1-based position of `stage` among the job's `n_stages` stages while running (n_stages once done; 0 before the first) */
  stage_index: number;
  n_stages: number;
  error: JobError | null;
  updated_at: string;
}

export type Corners = [[number, number], [number, number], [number, number], [number, number]];

/** result.json (written once the job is done) */
export interface JobResult {
  tile_id: string;
  /** [lon, lat] × 4 in MapLibre image-source order: TL, TR, BR, BL */
  corners: Corners;
  counts: { canopies: number; rows: number; interrows: number; waste: number };
  canopy_area_m2: number;
  interrow_area_m2: number;
  row_length_m: number;
  timings_s: Record<string, number>;
  total_s: number;
  model_version: string;
  waste_verify_level: string;
}

/** the stages `vineyard demo tile` runs, in order (n_stages = 11) */
export const PIPELINE_STAGES = [
  "ingest",
  "tile_prep",
  "nn_infer",
  "rows_detect",
  "rows_link",
  "blocks",
  "canopy",
  "interrow",
  "row_attrs",
  "waste",
  "assemble",
] as const;

/** result.timings_s in pipeline order (unknown stages last, by name) */
export function orderedTimings(timings: Record<string, number>): [string, number][] {
  const rank = (s: string) => {
    const i = (PIPELINE_STAGES as readonly string[]).indexOf(s);
    return i < 0 ? PIPELINE_STAGES.length : i;
  };
  return Object.entries(timings).sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b));
}

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

/** random UUIDs from the API; also the hand-made sample job folders ("sample-job") */
export const JOB_ID_RE = /^[a-z0-9-]{6,40}$/;
export const isJobId = (id: string) => JOB_ID_RE.test(id);

/** the only files of a job folder the browser may fetch, with their content type */
export const JOB_FILES: Readonly<Record<string, string>> = {
  "preview.jpg": "image/jpeg",
  "veg_mask.png": "image/png",
  "canopies.geojson": "application/geo+json",
  "rows.geojson": "application/geo+json",
  "interrows.geojson": "application/geo+json",
  "waste.geojson": "application/geo+json",
};
export const jobFileType = (name: string): string | null => (Object.hasOwn(JOB_FILES, name) ? JOB_FILES[name] : null);

export const isTiffName = (name: string) => /\.tiff?$/i.test(name.trim());

/** TIFF / BigTIFF byte-order mark + magic number: "II*\0", "MM\0*", "II+\0", "MM\0+" */
export function isTiffMagic(head: Uint8Array): boolean {
  if (head.length < 4) return false;
  const [a, b, c, d] = head;
  if (a === 0x49 && b === 0x49) return (c === 0x2a || c === 0x2b) && d === 0x00;
  if (a === 0x4d && b === 0x4d) return c === 0x00 && (d === 0x2a || d === 0x2b);
  return false;
}

export const isErrorCode = (v: unknown): v is ErrorCode => typeof v === "string" && (ERROR_CODES as readonly string[]).includes(v);

const STATES: readonly JobState[] = ["queued", "running", "done", "error"];
const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** status.json as written by Python or the API, or null when it is not one; an unknown error code reads as pipeline_failed */
export function parseStatus(raw: unknown): JobStatus | null {
  if (!isRecord(raw) || !STATES.includes(raw.state as JobState)) return null;
  const e = isRecord(raw.error) ? raw.error : null;
  return {
    state: raw.state as JobState,
    stage: typeof raw.stage === "string" ? raw.stage : null,
    stage_index: finite(raw.stage_index) ? raw.stage_index : 0,
    n_stages: finite(raw.n_stages) ? raw.n_stages : 0,
    error: e ? { code: isErrorCode(e.code) ? e.code : "pipeline_failed", message: typeof e.message === "string" ? e.message : "" } : null,
    updated_at: typeof raw.updated_at === "string" ? raw.updated_at : "",
  };
}

const isLonLat = (p: unknown): p is [number, number] => Array.isArray(p) && p.length === 2 && finite(p[0]) && finite(p[1]);

/** result.json, or null when a required field is missing or malformed */
export function parseResult(raw: unknown): JobResult | null {
  if (!isRecord(raw) || typeof raw.tile_id !== "string") return null;
  const corners = raw.corners;
  if (!Array.isArray(corners) || corners.length !== 4 || !corners.every(isLonLat)) return null;
  const c = isRecord(raw.counts) ? raw.counts : null;
  if (!c || ![c.canopies, c.rows, c.interrows, c.waste].every(finite)) return null;
  if (![raw.canopy_area_m2, raw.interrow_area_m2, raw.row_length_m, raw.total_s].every(finite)) return null;
  const timings = isRecord(raw.timings_s)
    ? Object.fromEntries(Object.entries(raw.timings_s).filter((e): e is [string, number] => finite(e[1])))
    : {};
  return {
    tile_id: raw.tile_id,
    corners: corners as Corners,
    counts: { canopies: c.canopies as number, rows: c.rows as number, interrows: c.interrows as number, waste: c.waste as number },
    canopy_area_m2: raw.canopy_area_m2 as number,
    interrow_area_m2: raw.interrow_area_m2 as number,
    row_length_m: raw.row_length_m as number,
    timings_s: timings,
    total_s: raw.total_s as number,
    model_version: typeof raw.model_version === "string" ? raw.model_version : "",
    waste_verify_level: typeof raw.waste_verify_level === "string" ? raw.waste_verify_level : "",
  };
}

export const queuedStatus = (now: Date): JobStatus => ({
  state: "queued",
  stage: null,
  stage_index: 0,
  n_stages: 0,
  error: null,
  updated_at: now.toISOString(),
});

export const errorStatus = (prev: JobStatus | null, error: JobError, now: Date): JobStatus => ({
  state: "error",
  stage: prev?.stage ?? null,
  stage_index: prev?.stage_index ?? 0,
  n_stages: prev?.n_stages ?? 0,
  error,
  updated_at: now.toISOString(),
});

/** progress in [0, 1] for the bar: done = 1; running = stages finished so far (the one at stage_index still runs) */
export function progressOf(s: JobStatus): number {
  if (s.state === "done") return 1;
  if (s.state !== "running" || s.n_stages <= 0) return 0;
  return Math.min(1, Math.max(0, (s.stage_index - 1) / s.n_stages));
}

export const isFinished = (s: JobStatus) => s.state === "done" || s.state === "error";
