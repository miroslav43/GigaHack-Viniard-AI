"use client";

// Map layers of the live cadastre (src/lib/cadastre.ts): the AGCC WMS raster and the highlight of the queried parcel.
// CadastreAnchor is mounted from the first render right above the orthophoto and the tile overlays: the raster, which
// exists only while the switch is on (so nothing is requested from the external server otherwise), is inserted
// below it and therefore under every vector layer of ours.
import { Layer, Source } from "@vis.gl/react-maplibre";
import type { Feature, FeatureCollection } from "geojson";
import { mapPalette } from "@/theme/mapPalette";
import {
  CADASTRE_ATTRIBUTION, CADASTRE_MAX_ZOOM, CADASTRE_MIN_ZOOM, CADASTRE_TILE_PX, CADASTRE_TILE_URL, type CadastreParcel,
} from "@/lib/cadastre";

const ANCHOR = "cadastre-anchor";
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

export function CadastreAnchor() {
  return (
    <Source id="cadastre-anchor-src" type="geojson" data={EMPTY}>
      <Layer id={ANCHOR} type="line" paint={{ "line-opacity": 0 }} />
    </Source>
  );
}

export function CadastreRaster({ on }: { on: boolean }) {
  if (!on) return null;
  return (
    <Source
      id="cadastre"
      type="raster"
      tiles={[CADASTRE_TILE_URL]}
      tileSize={CADASTRE_TILE_PX}
      minzoom={CADASTRE_MIN_ZOOM}
      maxzoom={CADASTRE_MAX_ZOOM}
      attribution={CADASTRE_ATTRIBUTION}
    >
      <Layer id="cadastre-raster" type="raster" minzoom={CADASTRE_MIN_ZOOM} beforeId={ANCHOR} paint={{ "raster-opacity": 0.9, "raster-fade-duration": 150 }} />
    </Source>
  );
}

/** Outline of the parcel returned by the last click (empty otherwise); mounted from the start, above the vine layers. */
export function CadastreHighlight({ parcel }: { parcel: CadastreParcel | null }) {
  const data: FeatureCollection = parcel?.geometry
    ? { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: parcel.geometry } as Feature] }
    : EMPTY;
  return (
    <Source id="cadastre-parcel" type="geojson" data={data}>
      <Layer id="cadastre-parcel-fill" type="fill" paint={{ "fill-color": mapPalette.cadastre.highlight, "fill-opacity": 0.12 }} />
      <Layer id="cadastre-parcel-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 5, "line-opacity": 0.8 }} />
      <Layer id="cadastre-parcel-line" type="line" paint={{ "line-color": mapPalette.cadastre.highlight, "line-width": 2.5 }} />
    </Source>
  );
}
