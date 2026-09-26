"use client";

// The optional farm and road layers of a survey (farms.geojson, roads.geojson; src/Web/CLAUDE.md §6.3). Unlike the
// tile / mask overlays they are ON by default: each file the survey ships is fetched once, on mount. A survey without
// them (the mock, older bundles) keeps the switches disabled and never requests anything.
import { useCallback, useEffect, useState } from "react";
import type { FeatureCollection, LineString, MultiLineString, MultiPolygon, Polygon } from "geojson";
import type { FarmProps, OverlayFiles, RoadProps, RoadsSummary } from "@/lib/types";

/** farms = outlines + labels, roads = public + field roads, internalRoads = a farm's internal roads (same file) */
export type FarmRoadKey = "farms" | "roads" | "internalRoads";
type FileKey = "farms" | "roads";

export type FarmsFc = FeatureCollection<Polygon | MultiPolygon, FarmProps>;
export type RoadsFc = FeatureCollection<LineString | MultiLineString, RoadProps>;

export interface FarmsRoads {
  available: Record<FarmRoadKey, boolean>;
  visible: Record<FarmRoadKey, boolean>;
  setVisible: (key: FarmRoadKey, on: boolean) => void;
  farms: FarmsFc | null;
  roads: RoadsFc | null;
  /** why a file failed to load (its switches are turned off); null = no failure */
  failed: Record<FileKey, string | null>;
}

const FILE_OF: Record<FarmRoadKey, FileKey> = { farms: "farms", roads: "roads", internalRoads: "roads" };
const NO_FAILURE: Record<FileKey, string | null> = { farms: null, roads: null };

const getJson = async <T,>(url: string): Promise<T> => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return (await r.json()) as T;
};
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function useFarmsRoads(dataBase: string, files: OverlayFiles | undefined, roadsSummary?: RoadsSummary): FarmsRoads {
  const hasFarms = Boolean(files?.farms), hasRoads = Boolean(files?.roads);
  // no internal road in the survey (summary.roads.internal_m = 0): nothing to switch on
  const available: Record<FarmRoadKey, boolean> = {
    farms: hasFarms,
    roads: hasRoads,
    internalRoads: hasRoads && (roadsSummary ? roadsSummary.internal_m > 0 : true),
  };
  const [visible, setVisibleState] = useState<Record<FarmRoadKey, boolean>>({ farms: true, roads: true, internalRoads: true });
  const [failed, setFailed] = useState(NO_FAILURE);
  const [farms, setFarms] = useState<FarmsFc | null>(null);
  const [roads, setRoads] = useState<RoadsFc | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async <T,>(file: FileKey, set: (fc: T) => void) => {
      try {
        const fc = await getJson<T>(`${dataBase}/${file}.geojson`);
        if (!cancelled) set(fc);
      } catch (e) {
        // the layer panel shows the failure next to the switch; the rest of the map is unaffected
        if (!cancelled) setFailed((s) => ({ ...s, [file]: message(e) }));
      }
    };
    if (hasFarms) void load<FarmsFc>("farms", setFarms);
    if (hasRoads) void load<RoadsFc>("roads", setRoads);
    return () => {
      cancelled = true;
    };
  }, [dataBase, hasFarms, hasRoads]);

  const setVisible = useCallback((key: FarmRoadKey, on: boolean) => setVisibleState((s) => ({ ...s, [key]: on })), []);

  const shown = (key: FarmRoadKey) => available[key] && visible[key] && failed[FILE_OF[key]] === null;
  return {
    available,
    visible: { farms: shown("farms"), roads: shown("roads"), internalRoads: shown("internalRoads") },
    setVisible,
    farms,
    roads,
    failed,
  };
}
