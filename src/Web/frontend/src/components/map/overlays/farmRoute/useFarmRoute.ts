"use client";

// State of the route tools (src/lib/farmRoute, ADR-028):
// - one farm, a 3-step guide: choose the farm → click the start (= finish) → the tour through its targets, inside it;
// - all farms, 2 steps: the start (a click, or the official START) → one tour through every target of every farm.
import { useCallback, useEffect, useRef, useState } from "react";
import type { FeatureCollection, MultiLineString, Point } from "geojson";
import type { TargetProps } from "@/lib/types";
import type { LonLat } from "@/lib/utm";
import type { FarmRouteResult, RowFeature } from "@/lib/farmRoute/plan";
import type { FarmRouteRequest, FarmRouteResponse } from "@/lib/farmRoute/worker";
import type { FarmsFc, RoadsFc } from "../useFarmsRoads";

export type RouteMode = "farm" | "all";

export type FarmRouteState =
  | { status: "idle" }
  | { status: "choosingFarm" }
  | { status: "picking"; mode: RouteMode; farmId: string | null }
  | { status: "computing"; mode: RouteMode; farmId: string | null; start: LonLat }
  | { status: "done"; mode: RouteMode; farmId: string | null; start: LonLat; result: FarmRouteResult }
  | { status: "error"; mode: RouteMode; farmId: string | null; start: LonLat; error: string };

export interface FarmRoute {
  state: FarmRouteState;
  mode: RouteMode | null;
  /** the farm the one-farm tool works on, null before one is chosen (and for all farms) */
  farmId: string | null;
  /** the tools can run: farms, roads, rows and targets are loaded */
  ready: boolean;
  /** one farm, step 1: choose a farm */
  open: () => void;
  /** one farm, step 2: click the start on the map */
  begin: (farmId: string) => void;
  /** all farms, step 1: the start */
  openAll: () => void;
  /** last step: plan the route from this start */
  pick: (start: LonLat) => void;
  clear: () => void;
}

const farmIdOf = (s: FarmRouteState) => ("farmId" in s ? s.farmId : null);
const modeOf = (s: FarmRouteState): RouteMode | null => ("mode" in s ? s.mode : s.status === "choosingFarm" ? "farm" : null);

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
              ? { ...s, status: "done", result: reply.result }
              : { ...s, status: "error", error: reply.error },
        );
      };
      w.onerror = (e) => {
        setState((s) => (s.status === "computing" ? { ...s, status: "error", error: e.message } : s));
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
    setState({ status: "picking", mode: "farm", farmId });
  }, []);
  const openAll = useCallback(() => {
    cancelPending();
    setState({ status: "picking", mode: "all", farmId: null });
  }, []);
  const clear = useCallback(() => {
    cancelPending();
    setState({ status: "idle" });
  }, []);

  const pick = useCallback(
    (start: LonLat) => {
      const mode = modeOf(state);
      const farmId = farmIdOf(state);
      if (!mode || state.status === "choosingFarm" || state.status === "computing") return;
      if (!farms || !roads || !rows || !targets) return;
      const featuresOf = (blocks: ReadonlySet<string>) => ({
        rows: rows.features.filter((f) => blocks.has(String(f.properties?.vineyard_id))) as RowFeature[],
        targets: targets.features.filter((f) => f.properties.vineyard_id != null && blocks.has(f.properties.vineyard_id)),
      });
      let request: FarmRouteRequest;
      if (mode === "farm") {
        const farm = farms.features.find((f) => f.properties.farm_id === farmId);
        if (!farm) return;
        request = { id: ++lastRequest.current, kind: "farm", input: { start, farm: farm.geometry, roads: roads.features, ...featuresOf(new Set(farm.properties.vineyard_ids)) } };
      } else {
        request = {
          id: ++lastRequest.current,
          kind: "all",
          input: {
            start,
            roads: roads.features,
            farms: farms.features.map((f) => ({ farmId: f.properties.farm_id, geometry: f.geometry, ...featuresOf(new Set(f.properties.vineyard_ids)) })),
          },
        };
      }
      setState({ status: "computing", mode, farmId, start });
      try {
        getWorker().postMessage(request);
      } catch (err) {
        setState({ status: "error", mode, farmId, start, error: err instanceof Error ? err.message : String(err) });
      }
    },
    [state, farms, roads, rows, targets, getWorker],
  );

  return { state, mode: modeOf(state), farmId: farmIdOf(state), ready: Boolean(farms && roads && rows && targets), open, begin, openAll, pick, clear };
}
