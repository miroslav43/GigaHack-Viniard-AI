import type { StyleSpecification } from "maplibre-gl";
import { mapPalette } from "@/theme/mapPalette";

/** Base style: plain background + optional OSM raster for context outside the orthophoto. */
export const baseStyle: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": mapPalette.background } },
    { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.6, "raster-opacity": 0.85 } },
  ],
};

/** Invisible layer rendered first; orthophoto layers are inserted right below it, every vector layer lands above it. */
export const ORTHO_ANCHOR = "ortho-anchor";

export const ZOOM = { interrows: 16.5, rows: 15.5, canopies: 17.5, orthoDetail: 17 };
