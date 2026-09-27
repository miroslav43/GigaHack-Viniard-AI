"use client";

import { useSyncExternalStore } from "react";

// The OS / browser "reduce motion" setting: every animation of the app (counters, charts, map) shows its final
// state at once when it is on (docs/STATISTICI.md).
const QUERY = "(prefers-reduced-motion: reduce)";

const media = () => (typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia(QUERY) : null);

const subscribe = (onChange: () => void) => {
  const mq = media();
  mq?.addEventListener("change", onChange);
  return () => mq?.removeEventListener("change", onChange);
};

const getSnapshot = () => media()?.matches ?? false;
// the server renders every animated value in its final state anyway
const getServerSnapshot = () => false;

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
