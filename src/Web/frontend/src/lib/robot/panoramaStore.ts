// The robot's panoramas, kept on this laptop: src/Web/data/robot/panoramas/<id>/ (ignored by git) with the three
// photos (0.jpg, 90.jpg, 180.jpg), the strip (strip.jpg) and meta.json — where it was taken and the grape / leaf
// detections of each photo. ROBOT_DATA_DIR moves it (tests). The detections run in the background after a save.
import "server-only";
import { mkdir, readdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { detectObjects, detectionConfigured, detectionModel } from "./detect";
import type { DetectionBox } from "./detections";

export const PANORAMA_ANGLES = [0, 90, 180] as const;
export type PanoramaFile = "0.jpg" | "90.jpg" | "180.jpg" | "strip.jpg";
export const PANORAMA_FILES: readonly PanoramaFile[] = ["0.jpg", "90.jpg", "180.jpg", "strip.jpg"];
/** one photo at most (the camera's VGA JPEGs are ~20–60 KB; the strip ~150 KB) */
export const MAX_FILE_BYTES = 5 * 1024 * 1024;
/** a detection still "running" after this long was cut short (a server restart): shown as failed */
const STALE_MS = 5 * 60 * 1000;

export type DetectionStatus = "off" | "pending" | "running" | "done" | "error";

export interface PanoramaMeta {
  id: string;
  station: number;
  atCm: number;
  takenAt: string;
  detection: {
    status: DetectionStatus;
    model: string | null;
    startedAt: string | null;
    finishedAt: string | null;
    error: string | null;
    /** boxes per photo angle ("0", "90", "180") */
    frames: Record<string, DetectionBox[]>;
  };
}

const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
export const validId = (id: string) => ID.test(id);

const root = () => path.join(process.env.ROBOT_DATA_DIR ?? path.resolve(process.cwd(), "..", "data", "robot"), "panoramas");
const dirOf = (id: string) => {
  if (!validId(id)) throw new Error("bad id");
  return path.join(root(), id);
};

async function writeMeta(meta: PanoramaMeta) {
  const file = path.join(dirOf(meta.id), "meta.json");
  await writeFile(`${file}.tmp`, JSON.stringify(meta, null, 2));
  await rename(`${file}.tmp`, file); // atomic: a reader never sees half a file
}

export async function readMeta(id: string): Promise<PanoramaMeta | null> {
  try {
    const meta = JSON.parse(await readFile(path.join(dirOf(id), "meta.json"), "utf8")) as PanoramaMeta;
    const d = meta.detection;
    if ((d.status === "running" || d.status === "pending") && d.startedAt && Date.now() - Date.parse(d.startedAt) > STALE_MS)
      return { ...meta, detection: { ...d, status: "error", error: "interrupted" } };
    return meta;
  } catch {
    return null;
  }
}

/** Every saved panorama, newest first. */
export async function listPanoramas(): Promise<PanoramaMeta[]> {
  const ids = await readdir(root()).catch(() => [] as string[]);
  const metas = await Promise.all(ids.filter(validId).map(readMeta));
  return metas.filter((m): m is PanoramaMeta => m !== null).sort((a, b) => b.takenAt.localeCompare(a.takenAt));
}

export async function readPanoramaFile(id: string, file: PanoramaFile): Promise<Uint8Array | null> {
  return readFile(path.join(dirOf(id), file)).catch(() => null);
}

export async function deletePanorama(id: string) {
  await rm(dirOf(id), { recursive: true, force: true });
}

/** Grape / leaf detection of every photo of a panorama, written into its meta as it goes. */
export async function runDetection(id: string): Promise<void> {
  const meta = await readMeta(id);
  if (!meta) return;
  if (!detectionConfigured()) {
    await writeMeta({ ...meta, detection: { ...meta.detection, status: "off" } });
    return;
  }
  const startedAt = new Date().toISOString();
  await writeMeta({ ...meta, detection: { status: "running", model: detectionModel(), startedAt, finishedAt: null, error: null, frames: {} } });
  try {
    const results = await Promise.all(
      PANORAMA_ANGLES.map(async (deg) => {
        const jpeg = await readPanoramaFile(id, `${deg}.jpg`);
        return [String(deg), jpeg ? await detectObjects(jpeg) : []] as const;
      }),
    );
    const now = await readMeta(id);
    if (!now) return; // deleted meanwhile
    await writeMeta({ ...now, detection: { status: "done", model: detectionModel(), startedAt, finishedAt: new Date().toISOString(), error: null, frames: Object.fromEntries(results) } });
  } catch (e) {
    console.error(`[robot] detection of ${id} failed:`, e instanceof Error ? e.message : e);
    const now = await readMeta(id);
    if (now)
      await writeMeta({
        ...now,
        detection: { ...now.detection, status: "error", finishedAt: new Date().toISOString(), error: e instanceof Error ? e.message.slice(0, 200) : "failed" },
      });
  }
}

/** Saves a new panorama (its photos checked as JPEGs) and starts its detection in the background. */
export async function savePanorama(input: { station: number; atCm: number; takenAt: string; files: Record<PanoramaFile, Uint8Array> }): Promise<PanoramaMeta> {
  const id = crypto.randomUUID();
  await mkdir(dirOf(id), { recursive: true });
  await Promise.all(PANORAMA_FILES.map((f) => writeFile(path.join(dirOf(id), f), input.files[f])));
  const meta: PanoramaMeta = {
    id,
    station: input.station,
    atCm: input.atCm,
    takenAt: input.takenAt,
    detection: { status: detectionConfigured() ? "pending" : "off", model: null, startedAt: new Date().toISOString(), finishedAt: null, error: null, frames: {} },
  };
  await writeMeta(meta);
  void runDetection(id); // not awaited: the page polls the list
  return meta;
}
