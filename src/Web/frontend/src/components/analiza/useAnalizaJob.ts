"use client";

// Upload → poll → result of one live tile analysis (POST /api/analiza, GET /api/analiza/<job> every second).
// `?job=<id>` in the URL follows an existing job (a reload, or a job started from another tab).
import { useCallback, useEffect, useRef, useState } from "react";
import {
  MAX_UPLOAD_BYTES,
  isErrorCode,
  isFinished,
  isTiffName,
  parseResult,
  parseStatus,
  type ErrorCode,
  type JobResult,
  type JobStatus,
} from "@/lib/analiza/contract";

export type ClientErrorCode = ErrorCode | "sign_in" | "bad_request" | "network" | "not_found";

export type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; name: string }
  | { kind: "running"; job: string; status: JobStatus | null; startedAt: number }
  | { kind: "done"; job: string; result: JobResult }
  | { kind: "error"; code: ClientErrorCode; message: string; busyJob: string | null };

const POLL_MS = 1000;
const API = "/api/analiza";

const failed = (code: ClientErrorCode, message = "", busyJob: string | null = null): Phase => ({ kind: "error", code, message, busyJob });

const codeOf = (v: unknown, fallback: ClientErrorCode): ClientErrorCode =>
  isErrorCode(v) || v === "sign_in" || v === "bad_request" || v === "not_found" ? v : fallback;

function setJobParam(job: string | null) {
  const url = new URL(window.location.href);
  if (job) url.searchParams.set("job", job);
  else url.searchParams.delete("job");
  window.history.replaceState(window.history.state, "", url);
}

export function useAnalizaJob() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [now, setNow] = useState(() => Date.now());
  const busy = useRef(false);

  const follow = useCallback((job: string) => {
    setJobParam(job);
    setPhase({ kind: "running", job, status: null, startedAt: Date.now() });
  }, []);

  const reset = useCallback(() => {
    setJobParam(null);
    setPhase({ kind: "idle" });
  }, []);

  const upload = useCallback(
    async (file: File) => {
      if (busy.current) return;
      if (!isTiffName(file.name)) return setPhase(failed("not_geotiff"));
      if (file.size > MAX_UPLOAD_BYTES) return setPhase(failed("too_large"));
      busy.current = true;
      setPhase({ kind: "uploading", name: file.name });
      try {
        const body = new FormData();
        body.append("file", file);
        const res = await fetch(API, { method: "POST", body });
        const data = (await res.json().catch(() => ({}))) as { job?: unknown; error?: unknown };
        if (res.status === 202 && typeof data.job === "string") follow(data.job);
        else setPhase(failed(codeOf(data.error, "pipeline_failed"), "", res.status === 409 && typeof data.job === "string" ? data.job : null));
      } catch (e) {
        setPhase(failed("network", e instanceof Error ? e.message : String(e)));
      } finally {
        busy.current = false;
      }
    },
    [follow],
  );

  // a job named in the URL (reload, shared link)
  useEffect(() => {
    const job = new URLSearchParams(window.location.search).get("job");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-off read of the URL on mount
    if (job) follow(job);
  }, [follow]);

  const runningJob = phase.kind === "running" ? phase.job : null;
  useEffect(() => {
    if (!runningJob) return;
    let stopped = false;
    const tick = async () => {
      setNow(Date.now());
      try {
        const res = await fetch(`${API}/${encodeURIComponent(runningJob)}`, { cache: "no-store" });
        const data = (await res.json().catch(() => ({}))) as { status?: unknown; result?: unknown; error?: unknown };
        if (stopped) return;
        if (!res.ok) {
          stopped = true;
          setJobParam(null);
          return setPhase(failed(codeOf(data.error, res.status === 404 ? "not_found" : "network")));
        }
        const status = parseStatus(data.status);
        if (!status) return;
        if (!isFinished(status)) return setPhase((p) => (p.kind === "running" && p.job === runningJob ? { ...p, status } : p));
        stopped = true;
        if (status.state === "error") {
          setJobParam(null);
          return setPhase(failed(status.error?.code ?? "pipeline_failed", status.error?.message ?? ""));
        }
        const result = parseResult(data.result);
        setPhase(result ? { kind: "done", job: runningJob, result } : failed("pipeline_failed", "result.json"));
      } catch {
        // a missed poll (server restarting): try again on the next tick
      }
    };
    void tick();
    const timer = setInterval(() => void (stopped || tick()), POLL_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [runningJob]);

  const elapsedS = phase.kind === "running" ? Math.max(0, (now - phase.startedAt) / 1000) : 0;
  return { phase, elapsedS, upload, follow, reset };
}
