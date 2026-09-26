"use client";

// The two optional map layers of a survey (web bundle v3, src/Web/CLAUDE.md §6.3): the tile footprints and the
// pipeline vegetation masks. Both are off by default and fetched only the first time they are switched on;
// a survey without the files (the mock, older bundles) keeps them disabled and never requests anything.
import { useCallback, useRef, useState } from "react";
import type { FeatureCollection, Polygon } from "geojson";
import type { OverlayFiles, TileProps } from "@/lib/types";

export type OverlayKey = "tiles" | "vegMask";
type LonLat = [number, number];
/** [lon, lat] corners in MapLibre image-source order: top-left, top-right, bottom-right, bottom-left. */
export type MaskCorners = [LonLat, LonLat, LonLat, LonLat];
/** masks/index.json: tile id → the corners of its mask image. */
export type MaskIndex = Record<string, MaskCorners>;

export interface SurveyOverlays {
  dataBase: string;
  available: Record<OverlayKey, boolean>;
  visible: Record<OverlayKey, boolean>;
  setVisible: (key: OverlayKey, on: boolean) => void;
  tiles: FeatureCollection<Polygon, TileProps> | null;
  maskIndex: MaskIndex | null;
  /** why a switched-on layer failed to load (it is switched off again); null = no failure */
  failed: Record<OverlayKey, string | null>;
}

const OFF: Record<OverlayKey, boolean> = { tiles: false, vegMask: false };
const NO_FAILURE: Record<OverlayKey, string | null> = { tiles: null, vegMask: null };

const getJson = async <T,>(url: string): Promise<T> => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return (await r.json()) as T;
};

export function useSurveyOverlays(dataBase: string, files: OverlayFiles | undefined): SurveyOverlays {
  const available = { tiles: Boolean(files?.tiles), vegMask: Boolean(files?.masks) };
  const [visible, setVisibleState] = useState(OFF);
  const [failed, setFailed] = useState(NO_FAILURE);
  const [tiles, setTiles] = useState<SurveyOverlays["tiles"]>(null);
  const [maskIndex, setMaskIndex] = useState<MaskIndex | null>(null);
  // requests in flight or done: a quick off/on does not fetch twice
  const requested = useRef(new Set<OverlayKey>());

  const load = useCallback(
    async (key: OverlayKey) => {
      requested.current.add(key);
      try {
        if (key === "tiles") setTiles(await getJson(`${dataBase}/tiles.geojson`));
        else setMaskIndex(await getJson(`${dataBase}/masks/index.json`));
        setFailed((s) => ({ ...s, [key]: null }));
      } catch (e) {
        // the layer panel shows the failure (and its detail) next to the switch; the rest of the map is unaffected
        requested.current.delete(key);
        setFailed((s) => ({ ...s, [key]: e instanceof Error ? e.message : String(e) }));
        setVisibleState((s) => ({ ...s, [key]: false }));
      }
    },
    [dataBase],
  );

  const setVisible = useCallback(
    (key: OverlayKey, on: boolean) => {
      setVisibleState((s) => ({ ...s, [key]: on }));
      if (on && !requested.current.has(key)) void load(key);
    },
    [load],
  );

  return { dataBase, available, visible, setVisible, tiles, maskIndex, failed };
}
