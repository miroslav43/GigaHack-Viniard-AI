"use client";

// State of the farm route tool, a 3-step guide: choose a farm → click the start (= finish) → the worker plans the
// tour through every target of the farm, inside the farm (src/lib/farmRoute, ADR-028).
import { useCallback, useEffect, useRef, useState } from "react";
import type { FeatureCollection, MultiLineString, Point } from "geojson";
import type { TargetProps } from "@/lib/types";
import type { LonLat } from "@/lib/utm";
import type { FarmRouteResult } from "@/lib/farmRoute/plan";
import type { FarmRouteRequest, FarmRouteResponse } from "@/lib/farmRoute/worker";
import type { FarmsFc, RoadsFc } from "../useFarmsRoads";

export type FarmRouteState =
  | { status: "idle" }
  | { status: "choosingFarm" }
  | { status: "picking"; farmId: string }
  | { status: "computing"; farmId: string; start: LonLat }
  | { status: "done"; farmId: string; start: LonLat; result: FarmRouteResult }
  | { status: "error"; farmId: string; start: LonLat; error: string };

export interface FarmRoute {
  state: FarmRouteState;
  /** the farm the tool works on, null before one is chosen */
  farmId: string | null;
  /** the tool can run: farms, roads, rows and targets are loaded */
  ready: boolean;
  /** step 1: choose a farm */
  open: () => void;
  /** step 2: click the start on the map */
  begin: (farmId: string) => void;
  /** step 3: plan the route from this start */
  pick: (start: LonLat) => void;
  clear: () => void;
}

const farmIdOf = (s: FarmRouteState) => ("farmId" in s ? s.farmId : null);

export function useFarmRoute({
  farms,
  roads,
  rows,
  targets,
}: {
  farms: FarmsFc | null;
  roads: RoadsFc | null;
  rows: FeatureCollection<MultiLineString> | null;
  targets: FeatureCollection<Point, TargetProps> | null;
}): FarmRoute {
  const [state, setState] = useState<FarmRouteState>({ status: "idle" });
  const worker = useRef<Worker | null>(null);
  const lastRequest = useRef(0);

  useEffect(() => () => worker.current?.terminate(), []);

  const getWorker = useCallback(() => {
    if (!worker.current) {
      const w = new Worker(new URL("../../../../lib/farmRoute/worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FarmRouteResponse>) => {
        const reply = e.data;
        if (reply.id !== lastRequest.current) return; // superseded by a newer start
        setState((s) =>
          s.status !== "computing"
            ? s
            : reply.ok
              ? { status: "done", farmId: s.farmId, start: s.start, result: reply.result }
              : { status: "error", farmId: s.farmId, start: s.start, error: reply.error },
        );
      };
      w.onerror = (e) => {
        setState((s) => (s.status === "computing" ? { status: "error", farmId: s.farmId, start: s.start, error: e.message } : s));
      };
      worker.current = w;
    }
    return worker.current;
  }, []);

  const cancelPending = () => {
    lastRequest.current++;
  };
  const open = useCallback(() => {
    cancelPending();
    setState({ status: "choosingFarm" });
  }, []);
  const begin = useCallback((farmId: string) => {
    cancelPending();
    setState({ status: "picking", farmId });
  }, []);
  const clear = useCallback(() => {
    cancelPending();
    setState({ status: "idle" });
  }, []);

  const pick = useCallback(
    (start: LonLat) => {
      const farmId = farmIdOf(state);
      if (!farmId || state.status === "computing") return;
      const farm = farms?.features.find((f) => f.properties.farm_id === farmId);
      if (!farm || !roads || !rows || !targets) return;
      const blocks = new Set(farm.properties.vineyard_ids);
      const request: FarmRouteRequest = {
        id: ++lastRequest.current,
        input: {
          start,
          farm: farm.geometry,
          roads: roads.features,
          rows: rows.features.filter((f) => blocks.has(String(f.properties?.vineyard_id))) as FarmRouteRequest["input"]["rows"],
          targets: targets.features.filter((f) => f.properties.vineyard_id != null && blocks.has(f.properties.vineyard_id)),
        },
      };
      setState({ status: "computing", farmId, start });
      try {
        getWorker().postMessage(request);
      } catch (err) {
        setState({ status: "error", farmId, start, error: err instanceof Error ? err.message : String(err) });
      }
    },
    [state, farms, roads, rows, targets, getWorker],
  );

  return { state, farmId: farmIdOf(state), ready: Boolean(farms && roads && rows && targets), open, begin, pick, clear };
}
