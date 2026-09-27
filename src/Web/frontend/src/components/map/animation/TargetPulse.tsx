"use client";

import { useEffect } from "react";
import { useMap } from "@vis.gl/react-maplibre";
import type { CircleLayerSpecification, ExpressionSpecification, FilterSpecification } from "maplibre-gl";
import { mapPalette } from "@/theme/mapPalette";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";

// A pulsing halo under the priority-1 inspection targets (docs/STATISTICI.md). The layer is declared in MapExplorer,
// inside the targets source and before "targets-circle" (so it draws under it), with TARGET_PULSE_PAINT: a constant
// object, so React never re-applies it over the animated values. <TargetPulse> animates it; "reduce motion" leaves the
// static halo of TARGET_PULSE_PAINT.
export const TARGET_PULSE_LAYER = "targets-pulse";
export const TARGET_PULSE_FILTER: FilterSpecification = ["==", ["get", "priority"], 1];

const PERIOD_MS = 1600;
// ~25 fps is smooth enough for a halo and keeps the style updates cheap
const FRAME_MS = 40;
const MAX_OPACITY = 0.55;
const STATIC = { scale: 2.2, opacity: 0.3 };
// the zoom stops of the "targets-circle" radius, scaled
const RADIUS_STOPS: [number, number][] = [
  [14, 2.5],
  [17, 5],
  [20, 8],
];
const radius = (k: number): ExpressionSpecification => [
  "interpolate",
  ["linear"],
  ["zoom"],
  ...RADIUS_STOPS.flatMap(([z, r]) => [z, r * k]),
];

export const TARGET_PULSE_PAINT: CircleLayerSpecification["paint"] = {
  "circle-color": mapPalette.target,
  "circle-radius": radius(STATIC.scale),
  "circle-opacity": STATIC.opacity,
  "circle-stroke-width": 0,
};

/** Animates the TARGET_PULSE_LAYER while `active` (the targets layer is on). Renders nothing; goes inside <Map>. */
export function TargetPulse({ active }: { active: boolean }) {
  const { current } = useMap();
  const reduced = useReducedMotion();

  useEffect(() => {
    const map = current?.getMap();
    if (!map || !active || reduced) return;
    const paint = (scale: number, opacity: number) => {
      try {
        if (!map.getLayer(TARGET_PULSE_LAYER)) return;
        map.setPaintProperty(TARGET_PULSE_LAYER, "circle-radius", radius(scale));
        map.setPaintProperty(TARGET_PULSE_LAYER, "circle-opacity", opacity);
      } catch {
        // the map is being removed
      }
    };
    const start = performance.now();
    let last = 0;
    let raf = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < FRAME_MS) return;
      last = now;
      const p = ((now - start) % PERIOD_MS) / PERIOD_MS;
      paint(1 + 2 * p, MAX_OPACITY * (1 - p));
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      paint(STATIC.scale, STATIC.opacity);
    };
  }, [current, active, reduced]);

  return null;
}
