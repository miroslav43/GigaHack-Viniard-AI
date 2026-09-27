"use client";

import { useEffect, useMemo } from "react";
import { useMap } from "@vis.gl/react-maplibre";
import type { GeoJSONSource } from "maplibre-gl";
import { easeInOutQuad } from "@/lib/motion/easing";
import { routeSlicer, type RouteLines } from "@/lib/motion/sliceLine";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";

export const ROUTE_DRAW_MS = 2000;

/**
 * Draws the line of a GeoJSON source progressively each time `active` turns on (the official route layer switched on),
 * by feeding the source the first part of the line at each frame; the full line is back at the end, when it turns off
 * and on unmount. With "reduce motion" the line is drawn at once. Renders nothing; goes inside <Map>.
 */
export function RouteDraw({
  sourceId,
  data,
  active,
  durationMs = ROUTE_DRAW_MS,
}: {
  sourceId: string;
  data: RouteLines;
  active: boolean;
  durationMs?: number;
}) {
  const { current } = useMap();
  const reduced = useReducedMotion();
  const slice = useMemo(() => routeSlicer(data), [data]);

  useEffect(() => {
    const map = current?.getMap();
    if (!map || !active || reduced) return;
    const source = () => {
      try {
        return map.getSource<GeoJSONSource>(sourceId);
      } catch {
        return undefined; // the map is being removed
      }
    };
    let raf = 0;
    let start: number | null = null;
    const tick = (now: number) => {
      const src = source();
      // the <Source> may reach the map a frame after this component
      if (src) {
        start ??= now;
        const t = Math.min(1, (now - start) / durationMs);
        src.setData(slice(easeInOutQuad(t)));
        if (t >= 1) return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      source()?.setData(data);
    };
  }, [current, active, reduced, slice, data, sourceId, durationMs]);

  return null;
}
