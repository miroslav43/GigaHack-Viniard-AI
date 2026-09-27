"use client";

// The live picture as a run of single frames (/capture through /api/robot, one after the other, ~2–4 per second).
// The ESP32-CAM serves one connection at a time: an MJPEG stream held open would block every other command (flash,
// photo) for as long as it lasts, while between two frames the camera is free. The last frame is also the photo.
import { useEffect, useRef, useState } from "react";
import { callRobot } from "./robotApi";

/** consecutive failed frames before the picture shows as unreachable */
const FAILS_BEFORE_DOWN = 3;
/** wait before trying again once the camera is down (ms) */
const RETRY_MS = 2000;

export interface CameraFrames {
  /** object URL of the frame on screen */
  url: string | null;
  /** that frame as a JPEG */
  blob: Blob | null;
  /** the camera stopped answering */
  down: boolean;
  fps: number;
}

export function useCameraFrames(cam: string | null, enabled: boolean): CameraFrames {
  const [frame, setFrame] = useState<CameraFrames>({ url: null, blob: null, down: false, fps: 0 });
  const shown = useRef<string | null>(null);

  useEffect(() => {
    if (!cam || !enabled) return;
    let alive = true;
    let fails = 0;
    const times: number[] = [];
    const next = async () => {
      while (alive) {
        try {
          const blob = await (await callRobot("cam", cam, { cmd: "capture" })).blob();
          if (!alive) return;
          if (!blob.type.startsWith("image/")) throw new Error("not an image");
          fails = 0;
          const now = performance.now();
          times.push(now);
          while (times.length && now - times[0] > 3000) times.shift();
          const url = URL.createObjectURL(blob);
          const previous = shown.current;
          shown.current = url;
          setFrame({ url, blob, down: false, fps: times.length / 3 });
          if (previous) URL.revokeObjectURL(previous);
        } catch {
          fails += 1;
          if (fails >= FAILS_BEFORE_DOWN && alive) {
            setFrame((f) => ({ ...f, down: true, fps: 0 }));
            await new Promise((r) => setTimeout(r, RETRY_MS));
          }
        }
      }
    };
    void next();
    return () => {
      alive = false;
    };
  }, [cam, enabled]);

  // the last frame's URL is released when the page goes away
  useEffect(
    () => () => {
      if (shown.current) URL.revokeObjectURL(shown.current);
    },
    [],
  );

  return frame;
}
