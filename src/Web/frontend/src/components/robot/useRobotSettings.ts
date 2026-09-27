"use client";

// The robot page's settings: the boards' addresses on the local Wi-Fi and the motion parameters. Kept in this browser
// (localStorage, read through useSyncExternalStore so the server render and hydration agree); defaults from
// NEXT_PUBLIC_ROBOT_*_URL. Without storage (private window, blocked) they live in memory for the visit.
import { useCallback, useMemo, useSyncExternalStore } from "react";
import type { Device } from "@/lib/robot/commands";
import { DEFAULT_WHEEL_MOVES, wheelMovesOr, type WheelMoves } from "@/lib/robot/wheels";

export interface RobotSettings {
  urls: Record<Device, string>;
  /** camera pan (motor 2): steps per turn (200 for a 1.8° motor without microstepping) */
  stepsPerRev: number;
  /** camera pan speed (pulses per second) */
  speedPps: number;
  /** camera height speed: fast (the driver microsteps; 3000 steps/s is what lifts it, checked on the robot) */
  liftSpeedPps: number;
  /** the camera is mounted upside down: flip its picture */
  flipV: boolean;
  flipH: boolean;
  /** a camera axis turning the other way round */
  panInvert: boolean;
  liftInvert: boolean;
  /** obstacle guard: forward is blocked when the distance sensor reads less than this (cm; 0 = off) */
  safeStopCm: number;
  /** panorama step: how far the robot moves before its photos, and how long the wheels run to cover it (calibration) */
  recordStepCm: number;
  recordStepMs: number;
  /** wheel motor speed (PWM 0..255) */
  wheelSpeed: number;
  /** what each wheel motor 1..4 does in each drive move (the user sets it: which motor is where, how it turns) */
  wheelMoves: WheelMoves;
}


export type SettingsPatch = Partial<Omit<RobotSettings, "urls">> & { urls?: Partial<Record<Device, string>> };

const KEY = "solemtrix.robot.settings";
/** bumped when a default changes meaning: older saved values of those keys are dropped (2: the height speed that
 *  actually lifts the camera, 3000 steps/s, replaces the 150 saved before) */
const VERSION = 2;
const RESET_IN: Record<number, (keyof RobotSettings)[]> = { 2: ["liftSpeedPps"] };

export const DEFAULT_SETTINGS: RobotSettings = {
  urls: {
    cam: process.env.NEXT_PUBLIC_ROBOT_CAM_URL ?? "",
    motors: process.env.NEXT_PUBLIC_ROBOT_MOTORS_URL ?? "",
    drive: process.env.NEXT_PUBLIC_ROBOT_DRIVE_URL ?? "",
  },
  stepsPerRev: 200,
  speedPps: 450,
  liftSpeedPps: 3000,
  flipV: true,
  flipH: false,
  panInvert: false,
  liftInvert: false,
  safeStopCm: 20,
  recordStepCm: 50,
  recordStepMs: 1500,
  wheelSpeed: 150,
  wheelMoves: DEFAULT_WHEEL_MOVES,
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
    const { version = 1, ...saved } = JSON.parse(raw) as Partial<RobotSettings> & { version?: number };
    const stale = Object.entries(RESET_IN).flatMap(([v, keys]) => (version < Number(v) ? keys : []));
    const kept = Object.fromEntries(Object.entries(saved).filter(([k]) => !stale.includes(k as keyof RobotSettings)));
    return { ...DEFAULT_SETTINGS, ...kept, urls: { ...DEFAULT_SETTINGS.urls, ...(saved.urls ?? {}) }, wheelMoves: wheelMovesOr(saved.wheelMoves) };
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
      memory = JSON.stringify({ ...next, version: VERSION });
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
