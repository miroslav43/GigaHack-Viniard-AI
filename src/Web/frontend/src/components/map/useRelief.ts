"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { MapRef } from "@vis.gl/react-maplibre";
import type { HillshadeLayerSpecification, Map as MapLibreMap, RasterDEMSourceSpecification, TerrainSpecification } from "maplibre-gl";
import { mapPalette } from "@/theme/mapPalette";
import { TERRAIN_DIR, type TerrainMeta } from "@/lib/terrain";
import { ORTHO_ANCHOR } from "./mapStyle";

export type ReliefView = "top" | "oblique";
/** Where the heights come from: the canopy annotations, or none (an oblique view of the flat orthophoto). */
export type ReliefSource = "annotations" | "flat";
export interface ReliefSettings {
  view: ReliefView;
  source: ReliefSource;
  /** canopy height shown, metres */
  heightM: number;
}

export const RELIEF = { minHeightM: 0, maxHeightM: 2, stepM: 0.1, pitch: 60, easeMs: 900 } as const;

const TERRAIN_SOURCE = "relief-terrain";
// a second source on the same tiles: MapLibre warns (and renders worse) when one raster-dem feeds terrain and hillshade
const SHADE_SOURCE = "relief-shade";
const SHADE_LAYER = "relief-shade";

const demSource = (meta: TerrainMeta, url: string): RasterDEMSourceSpecification => ({
  type: "raster-dem",
  tiles: [url],
  tileSize: meta.tile_size,
  minzoom: meta.minzoom,
  maxzoom: meta.maxzoom,
  bounds: meta.bounds,
  encoding: meta.encoding,
});

// hillshade strength grows with the height shown; the draped orthophoto alone gives little depth cue
const shadeStrength = (heightM: number) => Math.min(0.8, 0.2 + 0.3 * heightM);

const shadeLayer: HillshadeLayerSpecification = {
  id: SHADE_LAYER,
  type: "hillshade",
  source: SHADE_SOURCE,
  layout: { visibility: "none" },
  paint: {
    // igor: shading grows with the slope and flat ground stays untouched, so the orthophoto keeps its colours
    "hillshade-method": "igor",
    "hillshade-shadow-color": mapPalette.relief.shadow,
    "hillshade-highlight-color": mapPalette.relief.highlight,
    "hillshade-accent-color": mapPalette.relief.accent,
    "hillshade-illumination-anchor": "map",
  },
};

/** Adds the relief sources and the hillshade once, the shade right above the orthophoto anchor (under every vector layer). */
function addRelief(map: MapLibreMap, meta: TerrainMeta, tilesUrl: string) {
  if (map.getSource(TERRAIN_SOURCE)) return;
  map.addSource(TERRAIN_SOURCE, demSource(meta, tilesUrl));
  map.addSource(SHADE_SOURCE, demSource(meta, tilesUrl));
  const order = map.getLayersOrder();
  const anchor = order.indexOf(ORTHO_ANCHOR);
  map.addLayer(shadeLayer, anchor >= 0 ? order[anchor + 1] : undefined);
}

const sameTerrain = (a: TerrainSpecification | null, b: TerrainSpecification | null) =>
  a?.source === b?.source && a?.exaggeration === b?.exaggeration;

/**
 * Oblique 3D view over a synthetic relief raised from the canopy annotations (MapLibre terrain + hillshade).
 * Nothing is added to the map until the first switch to the oblique view, so the 2D map stays exactly as it was.
 * `mapLoaded`: the map fired `load` (setTerrain needs a loaded style).
 */
export function useRelief(mapRef: RefObject<MapRef | null>, meta: TerrainMeta | null, dataBase: string, mapLoaded: boolean) {
  const [settings, setSettings] = useState<ReliefSettings>(() => ({ view: "top", source: "annotations", heightM: meta?.height_m ?? 1.1 }));
  const shownView = useRef<ReliefView>("top");

  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (!map || !meta || !mapLoaded) return;
    const oblique = settings.view === "oblique";
    const raised = oblique && settings.source === "annotations" && settings.heightM > 0;
    if (raised) addRelief(map, meta, `${window.location.origin}${dataBase}/${TERRAIN_DIR}/${meta.tiles}`);

    // exaggeration rescales the encoded canopy height (meta.height_m) to the height chosen on the slider
    const terrain: TerrainSpecification | null = raised ? { source: TERRAIN_SOURCE, exaggeration: settings.heightM / meta.height_m } : null;
    if (!sameTerrain(map.getTerrain(), terrain)) map.setTerrain(terrain);
    if (map.getLayer(SHADE_LAYER)) {
      map.setLayoutProperty(SHADE_LAYER, "visibility", raised ? "visible" : "none");
      map.setPaintProperty(SHADE_LAYER, "hillshade-exaggeration", shadeStrength(settings.heightM));
    }
    if (shownView.current !== settings.view) {
      shownView.current = settings.view;
      map.easeTo(oblique ? { pitch: RELIEF.pitch, duration: RELIEF.easeMs } : { pitch: 0, bearing: 0, duration: RELIEF.easeMs });
    }
  }, [mapRef, meta, dataBase, mapLoaded, settings]);

  const update = useCallback((patch: Partial<ReliefSettings>) => setSettings((s) => ({ ...s, ...patch })), []);
  return { available: meta != null, ready: meta != null && mapLoaded, settings, update };
}
