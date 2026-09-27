"use client";

// Moving the robot: the camera (motor 2 turns it, motor 1 raises it) and the wheels, hold to move. While an arrow is
// held the camera gets short moves one after the other and the wheels run (`go`, renewed by `keep` every 300 ms so the
// server's watchdog lets them run); the release stops both. Also the timed moves the record mode builds on.
import { useCallback, useRef, useState } from "react";
import { applyInvert, type DriveMove } from "@/lib/robot/wheels";
import type { Arrow } from "./ArrowPad";
import { callRobot } from "./robotApi";
import type { RobotSettings } from "./useRobotSettings";

/** camera arrows → stepper: motor 2 turns the camera (left / right), motor 1 raises and lowers it (checked on the
 *  robot, 27.09.2026); dir 0 = right / up unless the axis is set as reversed */
const CAMERA_OF: Record<Arrow, { motor: 1 | 2; axis: "pan" | "height"; sign: 1 | -1 }> = {
  right: { motor: 2, axis: "pan", sign: 1 },
  left: { motor: 2, axis: "pan", sign: -1 },
  up: { motor: 1, axis: "height", sign: 1 },
  down: { motor: 1, axis: "height", sign: -1 },
};
export const DRIVE_OF: Record<Arrow, DriveMove> = { up: "forward", down: "back", left: "left", right: "right" };
/** one camera move while an arrow is held (s): short, so the camera stops soon after the release */
const CAMERA_CHUNK_S = 0.25;
/** renewal of the wheels' watchdog while a drive arrow is held (ms; the server stops them after 800 ms without) */
const KEEP_EVERY_MS = 300;

export type Holding = { kind: "camera" | "drive"; arrow: Arrow } | null;
/** where the camera is, from the zeroed position, in motor steps (degrees follow from the steps per turn, so a new
 *  calibration corrects the angle shown at once) */
type Steps = { panSteps: number; height: number };
export type Pose = { pan: number; panSteps: number; height: number };

export function useRobotMotion({
  motors,
  drive,
  settings,
  onError,
}: {
  motors: string | null;
  drive: string | null;
  settings: RobotSettings;
  onError: (e: unknown) => void;
}) {
  const [holding, setHolding] = useState<Holding>(null);
  const [steps, setSteps] = useState<Steps>({ panSteps: 0, height: 0 });
  // the same position, current inside a running sequence (the record mode turns the camera several times in a row)
  const poseRef = useRef<Steps>(steps);
  const setBoth = useCallback((next: Steps) => {
    poseRef.current = next;
    setSteps(next);
  }, []);
  // each hold gets a token; a release (or a new hold) moves it on, which ends the running camera loop
  const token = useRef(0);
  const driveHold = useRef<{ go: Promise<unknown>; keep: ReturnType<typeof setInterval> } | null>(null);

  const moveAxis = useCallback(
    async (a: Arrow, n: number) => {
      if (!motors) return;
      const m = CAMERA_OF[a];
      const pan = m.axis === "pan";
      const reversed = pan ? settings.panInvert : settings.liftInvert;
      const dir = (m.sign > 0) !== reversed ? 0 : 1;
      await callRobot("motors", motors, { cmd: "move", motor: m.motor, dir, speed: pan ? settings.speedPps : settings.liftSpeedPps, steps: n });
      const key = pan ? "panSteps" : "height";
      setBoth({ ...poseRef.current, [key]: poseRef.current[key] + m.sign * n });
    },
    [motors, settings, setBoth],
  );

  const holdCamera = useCallback(
    async (a: Arrow) => {
      if (!motors || token.current % 2 === 1) return; // odd token: a hold is running
      const mine = ++token.current;
      setHolding({ kind: "camera", arrow: a });
      const pan = CAMERA_OF[a].axis === "pan";
      const steps = Math.max(1, Math.round((pan ? settings.speedPps : settings.liftSpeedPps) * CAMERA_CHUNK_S));
      try {
        while (token.current === mine) await moveAxis(a, steps);
      } catch (e) {
        onError(e);
      } finally {
        if (token.current === mine) token.current++;
        setHolding(null);
      }
    },
    [motors, settings, moveAxis, onError],
  );

  const stopWheels = useCallback(async () => {
    if (!drive) return;
    try {
      await callRobot("drive", drive, { cmd: "stop" });
    } catch (e) {
      onError(e);
    }
  }, [drive, onError]);

  const holdDrive = useCallback(
    (a: Arrow) => {
      if (!drive || token.current % 2 === 1) return;
      token.current++;
      setHolding({ kind: "drive", arrow: a });
      const dirs = applyInvert(settings.wheelMoves[DRIVE_OF[a]], settings.wheelInvert).join(",");
      const go = callRobot("drive", drive, { cmd: "go", dirs, speed: settings.wheelSpeed }).catch(onError);
      const keep = setInterval(() => void callRobot("drive", drive, { cmd: "keep" }).catch(() => undefined), KEEP_EVERY_MS);
      driveHold.current = { go, keep };
    },
    [drive, settings, onError],
  );

  /** The held arrow was let go: the camera loop ends after its current move, the wheels stop. */
  const release = useCallback(async () => {
    if (token.current % 2 === 0) return;
    token.current++;
    const d = driveHold.current;
    driveHold.current = null;
    if (d) {
      clearInterval(d.keep);
      await d.go; // a stop sent before the start would be overtaken by it
      await stopWheels();
      setHolding(null);
    }
  }, [stopWheels]);

  /** Record mode: turn the camera to `deg` (degrees from the zeroed position). */
  const panTo = useCallback(
    async (deg: number) => {
      const delta = Math.round((deg / 360) * settings.stepsPerRev) - poseRef.current.panSteps;
      if (delta !== 0) await moveAxis(delta > 0 ? "right" : "left", Math.abs(delta));
    },
    [settings.stepsPerRev, moveAxis],
  );

  /** Record mode: drive straight ahead for `ms` (the server stops the wheels at the end). */
  const forwardFor = useCallback(
    async (ms: number) => {
      if (!drive) return;
      const dirs = applyInvert(settings.wheelMoves.forward, settings.wheelInvert).join(",");
      await callRobot("drive", drive, { cmd: "run", dirs, speed: settings.wheelSpeed, ms });
    },
    [drive, settings],
  );

  const degrees = (s: number) => (s * 360) / settings.stepsPerRev;
  const pose: Pose = { pan: degrees(steps.panSteps), panSteps: steps.panSteps, height: steps.height };
  /** the position right now (inside a sequence), with the angle in degrees */
  const poseNow = (): Pose => ({ pan: degrees(poseRef.current.panSteps), panSteps: poseRef.current.panSteps, height: poseRef.current.height });

  return { holding, pose, poseNow, zero: () => setBoth({ panSteps: 0, height: 0 }), holdCamera, holdDrive, release, stopWheels, panTo, forwardFor };
}
