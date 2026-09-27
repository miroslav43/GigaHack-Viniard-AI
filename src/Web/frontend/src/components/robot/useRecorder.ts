"use client";

// Record mode: the robot stops at a row of stations `recordStepCm` apart. At each one the camera turns to 0°, 90° and
// 180° (from the zeroed position) and takes a fresh photo at each, the three become one panorama (side by side), the
// camera turns back to 0° and the wheels run forward for the calibrated time. A stop ends it at once (wheels too).
import { useCallback, useRef, useState } from "react";
import { callRobot, RobotCallError } from "./robotApi";
import type { RobotSettings } from "./useRobotSettings";
import { orientJpeg } from "./orient";

/** camera angles of a station, degrees from the zeroed position */
export const RECORD_ANGLES = [0, 90, 180] as const;
/** after a camera turn, let it settle before the photo (ms) */
const SETTLE_MS = 400;
/** after the wheels stop, before the next station's photos (ms) */
const PAUSE_MS = 300;
const PANORAMA_QUALITY = 0.9;

export interface PanoramaFrame {
  deg: number;
  blob: Blob;
  url: string;
}

export interface Panorama {
  id: string;
  /** 1-based station number in its recording, and the recording it belongs to */
  station: number;
  recording: string;
  /** distance from the recording's start (cm) */
  atCm: number;
  takenAt: Date;
  frames: PanoramaFrame[];
  /** the frames side by side, one JPEG */
  strip: { blob: Blob; url: string };
}

export type RecordPhase = { station: number; of: number; step: "photo"; deg: number } | { station: number; of: number; step: "drive" };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The frames side by side, same height, in order. */
async function stitch(frames: PanoramaFrame[]): Promise<Blob> {
  const images = await Promise.all(frames.map((f) => createImageBitmap(f.blob)));
  const h = Math.min(...images.map((i) => i.height));
  const widths = images.map((i) => Math.round((i.width * h) / i.height));
  const canvas = document.createElement("canvas");
  canvas.width = widths.reduce((a, b) => a + b, 0);
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  images.reduce((x, img, k) => {
    ctx.drawImage(img, x, 0, widths[k], h);
    return x + widths[k];
  }, 0);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new RobotCallError("failed"))), "image/jpeg", PANORAMA_QUALITY));
}

export function useRecorder({
  cam,
  settings,
  panTo,
  forwardFor,
  stopWheels,
  onPanorama,
  onError,
}: {
  cam: string | null;
  settings: RobotSettings;
  panTo: (deg: number) => Promise<void>;
  forwardFor: (ms: number) => Promise<void>;
  stopWheels: () => Promise<void>;
  onPanorama: (p: Panorama) => void;
  onError: (e: unknown) => void;
}) {
  const [phase, setPhase] = useState<RecordPhase | null>(null);
  const running = useRef(false);

  const stop = useCallback(async () => {
    running.current = false;
    await stopWheels();
  }, [stopWheels]);

  const start = useCallback(async () => {
    if (!cam || running.current) return;
    running.current = true;
    const recording = new Date().toISOString();
    const of = settings.recordStations;
    try {
      for (let station = 1; station <= of && running.current; station++) {
        const frames: PanoramaFrame[] = [];
        for (const deg of RECORD_ANGLES) {
          if (!running.current) break;
          setPhase({ station, of, step: "photo", deg });
          await panTo(deg);
          await sleep(SETTLE_MS);
          const blob = await orientJpeg(await (await callRobot("cam", cam, { cmd: "capture" })).blob(), settings);
          frames.push({ deg, blob, url: URL.createObjectURL(blob) });
        }
        if (frames.length === RECORD_ANGLES.length) {
          const strip = await stitch(frames);
          onPanorama({
            id: crypto.randomUUID(),
            station,
            recording,
            atCm: (station - 1) * settings.recordStepCm,
            takenAt: new Date(),
            frames,
            strip: { blob: strip, url: URL.createObjectURL(strip) },
          });
        } else frames.forEach((f) => URL.revokeObjectURL(f.url));
        await panTo(0); // straight ahead again before driving (and no cable wound up)
        if (station < of && running.current) {
          setPhase({ station, of, step: "drive" });
          await forwardFor(settings.recordStepMs);
          await sleep(PAUSE_MS);
        }
      }
    } catch (e) {
      onError(e);
      await stopWheels();
    } finally {
      running.current = false;
      setPhase(null);
    }
  }, [cam, settings, panTo, forwardFor, stopWheels, onPanorama, onError]);

  return { phase, recording: phase !== null, start, stop };
}
