"use client";

// The panoramas saved on the laptop (/api/robot/panoramas): the list, reloaded every 2 s while a detection runs;
// saving a new one (upload), running the detection again, deleting.
import { useCallback, useEffect, useState } from "react";
import type { PanoramaMeta } from "@/lib/robot/panoramaStore";
import type { Panorama } from "./useRecorder";

const POLL_MS = 2000;
async function fetchList(): Promise<PanoramaMeta[] | null> {
  const res = await fetch("/api/robot/panoramas", { cache: "no-store" }).catch(() => null);
  return res?.ok ? ((await res.json()) as { panoramas: PanoramaMeta[] }).panoramas : null;
}

const busy = (p: PanoramaMeta) => p.detection.status === "pending" || p.detection.status === "running";

export function usePanoramas() {
  const [panoramas, setPanoramas] = useState<PanoramaMeta[]>([]);
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    const list = await fetchList();
    if (!list) return;
    setPanoramas(list);
    setLoaded(true);
  }, []);

  useEffect(() => {
    let alive = true;
    void fetchList().then((list) => {
      if (!alive || !list) return;
      setPanoramas(list);
      setLoaded(true);
    });
    return () => {
      alive = false;
    };
  }, []);

  const running = panoramas.some(busy);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [running, refresh]);

  /** Uploads a panorama just taken; true when it is saved. */
  const save = useCallback(
    async (p: Panorama): Promise<boolean> => {
      const form = new FormData();
      form.set("station", String(p.station));
      form.set("atCm", String(p.atCm));
      form.set("takenAt", p.takenAt.toISOString());
      for (const f of p.frames) form.set(`${f.deg}.jpg`, f.blob, `${f.deg}.jpg`);
      form.set("strip.jpg", p.strip.blob, "strip.jpg");
      const res = await fetch("/api/robot/panoramas", { method: "POST", body: form }).catch(() => null);
      if (!res?.ok) return false;
      await refresh();
      return true;
    },
    [refresh],
  );

  const redetect = useCallback(
    async (id: string) => {
      await fetch(`/api/robot/panoramas/${id}/detect`, { method: "POST" }).catch(() => null);
      setPanoramas((ps) => ps.map((p) => (p.id === id ? { ...p, detection: { ...p.detection, status: "pending" } } : p)));
    },
    [],
  );

  const remove = useCallback(async (id: string) => {
    await fetch(`/api/robot/panoramas/${id}`, { method: "DELETE" }).catch(() => null);
    setPanoramas((ps) => ps.filter((p) => p.id !== id));
  }, []);

  return { panoramas, loaded, save, redetect, remove, refresh };
}
