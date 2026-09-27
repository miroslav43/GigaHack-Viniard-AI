"use client";

// The robot page's settings: the boards' addresses on the local Wi-Fi and the motion parameters. Kept in this browser
// (localStorage, read through useSyncExternalStore so the server render and hydration agree); defaults from
// NEXT_PUBLIC_ROBOT_*_URL. Without storage (private window, blocked) they live in memory for the visit.
import { useCallback, useMemo, useSyncExternalStore } from "react";
import type { Device } from "@/lib/robot/commands";

export interface RobotSettings {
  urls: Record<Device, string>;
  /** steps per camera motor turn (200 for a 1.8° motor without microstepping) */
  stepsPerRev: number;
  /** degrees per press of a camera arrow */
  stepDeg: number;
  /** camera motor speed (pulses per second) */
  speedPps: number;
  /** duration of one drive command (ms) */
  driveMs: number;
}

export type SettingsPatch = Partial<Omit<RobotSettings, "urls">> & { urls?: Partial<Record<Device, string>> };

const KEY = "solemtrix.robot.settings";

export const DEFAULT_SETTINGS: RobotSettings = {
  urls: {
    cam: process.env.NEXT_PUBLIC_ROBOT_CAM_URL ?? "",
    motors: process.env.NEXT_PUBLIC_ROBOT_MOTORS_URL ?? "",
    drive: process.env.NEXT_PUBLIC_ROBOT_DRIVE_URL ?? "",
  },
  stepsPerRev: 200,
  stepDeg: 15,
  speedPps: 450,
  driveMs: 800,
};

// the store: the saved JSON (or the in-memory copy when storage is unavailable) and who listens to it
let memory = "";
const listeners = new Set<() => void>();

function read(): string {
  try {
    return window.localStorage.getItem(KEY) ?? memory;
  } catch {
    return memory;
  }
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

function parse(raw: string): RobotSettings {
  if (!raw) return DEFAULT_SETTINGS;
  try {
    const saved = JSON.parse(raw) as Partial<RobotSettings>;
    return { ...DEFAULT_SETTINGS, ...saved, urls: { ...DEFAULT_SETTINGS.urls, ...(saved.urls ?? {}) } };
  } catch {
    return DEFAULT_SETTINGS; // a broken value: the defaults
  }
}

export function useRobotSettings() {
  const raw = useSyncExternalStore(subscribe, read, () => "");
  const settings = useMemo(() => parse(raw), [raw]);
  // merged onto the latest saved value, not this render's: several quick edits all survive
  const update = useCallback(
    (patch: SettingsPatch) => {
      const latest = parse(read());
      const next = { ...latest, ...patch, urls: { ...latest.urls, ...(patch.urls ?? {}) } };
      memory = JSON.stringify(next);
      try {
        window.localStorage.setItem(KEY, memory);
      } catch {
        // not persisted; the in-memory copy serves the rest of the visit
      }
      listeners.forEach((l) => l());
    },
    [],
  );
  return { settings, update };
}
