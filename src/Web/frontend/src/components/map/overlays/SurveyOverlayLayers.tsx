"use client";

// MapLibre layers of the optional survey overlays. Render right after the orthophoto anchor so the stack is
//   orthophoto · ORTHO_ANCHOR · vegetation masks · MASK_ANCHOR · tile footprints · every other vector layer.
// The anchor and the tile layers are mounted once (hidden while off), so they keep that place; the mask images come
// and go with the view and are inserted before MASK_ANCHOR, like the orthophoto detail tiles before ORTHO_ANCHOR.
import { Layer, Source } from "@vis.gl/react-maplibre";
import type { ExpressionSpecification } from "maplibre-gl";
import type { FeatureCollection } from "geojson";
import { mapPalette } from "@/theme/mapPalette";
import type { MaskCorners, SurveyOverlays } from "./useSurveyOverlays";

export const TILE_FILL = "tiles-fill";
const MASK_ANCHOR = "veg-mask-anchor";
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

// review_status set (tile_review.csv) wins over the status: the tile waits for completion in Marcaj
const TO_COMPLETE: ExpressionSpecification = ["to-boolean", ["get", "review_status"]];
const IS_VINEYARD: ExpressionSpecification = ["==", ["get", "status"], "vineyard"];
const byClass = <T,>(toComplete: T, vineyard: T, noVineyard: T) =>
  ["case", TO_COMPLETE, toComplete, IS_VINEYARD, vineyard, noVineyard] as unknown as ExpressionSpecification;
const TILE_COLOR = byClass(mapPalette.tile.toComplete, mapPalette.tile.vineyard, mapPalette.tile.no_vineyard);
const TILE_FILL_OPACITY = byClass(0.24, 0.14, 0.05);
const TILE_LINE_WIDTH = byClass(3, 1.5, 0.75);

const vis = (on: boolean) => ({ visibility: on ? ("visible" as const) : ("none" as const) });

/** Orthophoto detail tiles in view (MapExplorer.refreshDetail): zoom ≥ ZOOM.orthoDetail, capped at MAX_DETAIL_TILES. */
interface DetailTile {
  id: string;
}

export function SurveyOverlayLayers({
  overlays,
  detailTiles,
  selectedTile,
}: {
  overlays: SurveyOverlays;
  detailTiles: DetailTile[];
  selectedTile: string | null;
}) {
  const { available, visible, tiles, maskIndex, dataBase } = overlays;
  if (!available.tiles && !available.vegMask) return null;
  // masks follow the orthophoto detail mechanism: only the tiles in view, close up, capped
  const masks: [string, MaskCorners][] =
    visible.vegMask && maskIndex ? detailTiles.flatMap((t) => (maskIndex[t.id] ? [[t.id, maskIndex[t.id]] as [string, MaskCorners]] : [])) : [];

  return (
    <>
      {available.vegMask && (
        <Source id="veg-mask-anchor-src" type="geojson" data={EMPTY}>
          <Layer id={MASK_ANCHOR} type="line" paint={{ "line-opacity": 0 }} />
        </Source>
      )}
      {masks.map(([id, corners]) => (
        <Source key={id} id={`veg-mask-${id}`} type="image" url={`${dataBase}/masks/${id}.png`} coordinates={corners}>
          <Layer id={`veg-mask-${id}`} type="raster" beforeId={MASK_ANCHOR} paint={{ "raster-fade-duration": 150 }} />
        </Source>
      ))}
      {available.tiles && (
        <Source id="tiles" type="geojson" data={tiles ?? EMPTY}>
          <Layer id={TILE_FILL} type="fill" paint={{ "fill-color": TILE_COLOR, "fill-opacity": TILE_FILL_OPACITY }} layout={vis(visible.tiles)} />
          <Layer id="tiles-line" type="line" paint={{ "line-color": TILE_COLOR, "line-width": TILE_LINE_WIDTH }} layout={vis(visible.tiles)} />
          <Layer
            id="tiles-selected"
            type="line"
            filter={["==", ["get", "tile"], selectedTile ?? ""]}
            paint={{ "line-color": mapPalette.selected, "line-width": 3.5 }}
            layout={vis(visible.tiles)}
          />
        </Source>
      )}
    </>
  );
}
