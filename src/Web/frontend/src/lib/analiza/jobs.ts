// Job folders of the live tile analysis (/analiza) and the Python process that fills them (`vineyard demo tile`).
// Contract: ./contract.ts and src/Web/docs/design/2026-09-27-analiza-tif-spec.md. One job at a time (CPU-heavy): an
// exclusive lock file holds the running job's id and pid, so a second upload gets 409 even across module reloads.
import "server-only";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, readdir, rename, rm, stat, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import rasterColors from "@/theme/rasterColors.json";
import {
  errorStatus,
  isJobId,
  jobFileType,
  parseResult,
  parseStatus,
  queuedStatus,
  type JobError,
  type JobResult,
  type JobStatus,
} from "./contract";

const INPUT = "input.tif";
const STATUS = "status.json";
const RESULT = "result.json";
const LOG = "log.txt";
const LOCK = "active.lock";
/** a stuck pipeline is stopped after this long (one tile takes ~25 s) */
const JOB_TIMEOUT_MS = 10 * 60 * 1000;
/** job folders older than this are removed when a new job starts */
const KEEP_JOBS_MS = 24 * 60 * 60 * 1000;
/** an active conda/GDAL install would shadow the wheels' PROJ/GDAL data (README §3.2) */
const STRIPPED_ENV = ["PROJ_DATA", "PROJ_LIB", "GDAL_DATA"];

export const jobsDir = () => process.env.ANALIZA_JOBS_DIR || path.join(os.tmpdir(), "vineyard-analiza");
/** src/AI (the uv project with the `vineyard` CLI); `pnpm dev|start` runs from src/Web/frontend */
export const aiDir = () => process.env.VINEYARD_AI_DIR || path.resolve(process.cwd(), "../../AI");
const uvBin = () => process.env.ANALIZA_UV_BIN || "uv";

const jobDir = (id: string) => path.join(jobsDir(), id);
const lockPath = () => path.join(jobsDir(), LOCK);
const log = (msg: string, extra?: unknown) => console.error(`[analiza] ${msg}`, extra ?? "");

async function readJson(file: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    return null;
  }
}

async function writeStatus(id: string, status: JobStatus) {
  const file = path.join(jobDir(id), STATUS);
  const tmp = `${file}.${process.pid}.tmp`;
  await writeFile(tmp, JSON.stringify(status));
  await rename(tmp, file);
}

const alive = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return (e as NodeJS.ErrnoException).code === "EPERM";
  }
};

async function readLock(): Promise<{ id: string; pid: number } | null> {
  const raw = (await readJson(lockPath())) as { id?: unknown; pid?: unknown } | null;
  return raw && typeof raw.id === "string" && typeof raw.pid === "number" ? { id: raw.id, pid: raw.pid } : null;
}

/** The running job's id, or null; a lock left by a process that died is removed. */
export async function activeJob(): Promise<string | null> {
  const lock = await readLock();
  if (!lock) return null;
  if (lock.pid > 0 && alive(lock.pid)) return lock.id;
  // pid 0: reserved by a request that is still creating its job folder (give it a minute)
  const age = Date.now() - (await stat(lockPath()).then((s) => s.mtimeMs, () => 0));
  if (lock.pid === 0 && age < 60_000) return lock.id;
  await unlink(lockPath()).catch(() => undefined);
  return null;
}

/** Reserves the single job slot for `id`; returns the id holding it when busy. */
async function reserve(id: string): Promise<string | null> {
  await mkdir(jobsDir(), { recursive: true });
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      await writeFile(lockPath(), JSON.stringify({ id, pid: 0 }), { flag: "wx" });
      return null;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== "EEXIST") throw e;
      const holder = await activeJob();
      if (holder) return holder;
    }
  }
  return (await readLock())?.id ?? "unknown";
}

async function release(id: string) {
  if ((await readLock())?.id === id) await unlink(lockPath()).catch(() => undefined);
}

async function pruneOldJobs() {
  const now = Date.now();
  const entries = await readdir(jobsDir(), { withFileTypes: true }).catch(() => []);
  await Promise.all(
    entries
      .filter((e) => e.isDirectory() && isJobId(e.name))
      .map(async (e) => {
        const dir = path.join(jobsDir(), e.name);
        const { mtimeMs } = await stat(dir);
        if (now - mtimeMs > KEEP_JOBS_MS) await rm(dir, { recursive: true, force: true });
      }),
  ).catch((err) => log("prune failed", err));
}

async function fail(id: string, error: JobError) {
  const prev = parseStatus(await readJson(path.join(jobDir(id), STATUS)));
  if (prev?.state === "error") return;
  await writeStatus(id, errorStatus(prev, error, new Date())).catch((err) => log(`cannot write status of ${id}`, err));
}

function run(id: string) {
  const dir = jobDir(id);
  const env = { ...process.env };
  for (const k of STRIPPED_ENV) delete env[k];
  const out = createWriteStream(path.join(dir, LOG), { flags: "a" });
  // the mask colour is baked into veg_mask.png from the single source of raster colours (like the survey masks)
  const { color, alpha } = rasterColors.vegMask;
  const args = ["run", "--no-sync", "vineyard", "demo", "tile", path.join(dir, INPUT), "--out", dir, "--mask-colour", color, "--mask-alpha", String(alpha)];
  const child = spawn(uvBin(), args, { cwd: aiDir(), env, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.pipe(out, { end: false });
  child.stderr.pipe(out, { end: false });

  let finished = false;
  const finish = async (error: JobError | null) => {
    if (finished) return;
    finished = true;
    clearTimeout(timer);
    out.end();
    const status = parseStatus(await readJson(path.join(dir, STATUS)));
    if (error) await fail(id, error);
    else if (status?.state !== "done") await fail(id, { code: "pipeline_failed", message: "the pipeline stopped without a result" });
    await release(id);
  };
  const timer = setTimeout(() => {
    log(`job ${id} timed out`);
    child.kill("SIGTERM");
  }, JOB_TIMEOUT_MS);

  child.on("error", (err) => {
    log(`cannot start ${uvBin()} in ${aiDir()}`, err);
    void finish({ code: "pipeline_failed", message: `cannot start the pipeline: ${err.message}` });
  });
  child.on("close", (code, signal) => {
    const error: JobError | null =
      code === 0 ? null : { code: "pipeline_failed", message: signal ? `the pipeline was stopped (${signal})` : `the pipeline exited with code ${code}` };
    void finish(error);
  });
  if (child.pid) void writeFile(lockPath(), JSON.stringify({ id, pid: child.pid })).catch((err) => log("cannot update the lock", err));
}

export type StartOutcome = { ok: true; id: string } | { ok: false; busyWith: string };

/** Saves the upload as a new job folder and starts the pipeline on it. */
export async function startJob(tif: Uint8Array): Promise<StartOutcome> {
  const id = randomUUID();
  const busyWith = await reserve(id);
  if (busyWith) return { ok: false, busyWith };
  try {
    await pruneOldJobs();
    await mkdir(jobDir(id), { recursive: true });
    await writeFile(path.join(jobDir(id), INPUT), tif);
    await writeStatus(id, queuedStatus(new Date()));
    run(id);
    return { ok: true, id };
  } catch (e) {
    await release(id);
    throw e;
  }
}

/** status.json (+ result.json once done) of a job; null when the job does not exist */
export async function readJob(id: string): Promise<{ status: JobStatus; result: JobResult | null } | null> {
  if (!isJobId(id)) return null;
  const status = parseStatus(await readJson(path.join(jobDir(id), STATUS)));
  if (!status) return null;
  const result = status.state === "done" ? parseResult(await readJson(path.join(jobDir(id), RESULT))) : null;
  return { status, result };
}

/** A whitelisted output file of a job, or null */
export async function readJobFile(id: string, name: string): Promise<{ body: Buffer; type: string } | null> {
  const type = jobFileType(name);
  if (!type || !isJobId(id)) return null;
  try {
    return { body: await readFile(path.join(jobDir(id), name)), type };
  } catch {
    return null;
  }
}
