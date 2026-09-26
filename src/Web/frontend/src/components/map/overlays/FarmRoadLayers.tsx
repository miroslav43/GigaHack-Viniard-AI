"use client";

// MapLibre layers of the optional farm and road layers (farms.geojson, roads.geojson). Rendered right after the
// survey overlays, so the stack is: orthophoto · masks · tile footprints · ROADS · FARMS · blocks, inter-rows,
// canopies, rows … The sources are mounted from the first render (empty until loaded, hidden while off), so the
// layers keep that place. Farm labels are DOM markers (FarmLabels): the base style has no glyphs, and it stays offline.
import { Layer, Marker, Source } from "@vis.gl/react-maplibre";
import type { ExpressionSpecification, FilterSpecification } from "maplibre-gl";
import type { FeatureCollection } from "geojson";
import Box from "@mui/material/Box";
import { mapPalette } from "@/theme/mapPalette";
import type { RoadClass } from "@/lib/types";
import type { FarmsRoads } from "./useFarmsRoads";

export const ROADS_HIT = "roads-hit";
export const FARMS_FILL = "farms-fill";
/** farm labels from this zoom on (medium and closer: the whole survey on a desktop screen) */
export const FARM_LABEL_ZOOM = 14.5;
const OSM_ATTRIBUTION = "© OpenStreetMap contributors";
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

const vis = (on: boolean) => ({ visibility: on ? ("visible" as const) : ("none" as const) });
const isClass = (classes: RoadClass[]): FilterSpecification => ["in", ["get", "road_class"], ["literal", classes]];
const NETWORK = isClass(["public", "field"]);
const INTERNAL = isClass(["internal"]);
const byZoom = (...stops: [number, number | ExpressionSpecification][]) =>
  ["interpolate", ["linear"], ["zoom"], ...stops.flat()] as unknown as ExpressionSpecification;
const publicOr = (pub: number, other: number) => ["match", ["get", "road_class"], "public", pub, other] as unknown as ExpressionSpecification;
// public roads are drawn wider than field roads; every width grows with the zoom
const NETWORK_WIDTH = byZoom([13, publicOr(1.4, 0.7)], [16, publicOr(3.4, 1.7)], [19, publicOr(9, 5)]);
const NETWORK_CASING = byZoom([13, publicOr(2.6, 1.6)], [16, publicOr(5.4, 3.2)], [19, publicOr(12, 7.5)]);
const INTERNAL_WIDTH = byZoom([13, 0.8], [16, 2], [19, 4.5]);
const INTERNAL_CASING = byZoom([13, 1.8], [16, 3.4], [19, 6.5]);
const FARM_WIDTH = byZoom([13, 1.2], [16, 2], [19, 3]);

export function FarmRoadLayers({ data, selectedRoad, selectedFarm }: { data: FarmsRoads; selectedRoad: string | null; selectedFarm: string | null }) {
  const { available, visible, roads, farms } = data;
  const hitClasses: RoadClass[] = [...(visible.roads ? (["public", "field"] as const) : []), ...(visible.internalRoads ? (["internal"] as const) : [])];
  const round = { "line-join": "round", "line-cap": "round" } as const;
  return (
    <>
      {available.roads && (
        <Source id="roads" type="geojson" data={roads ?? EMPTY} attribution={OSM_ATTRIBUTION}>
          <Layer id="roads-casing" type="line" filter={NETWORK} paint={{ "line-color": mapPalette.road.casing, "line-width": NETWORK_CASING, "line-opacity": 0.65 }}
            layout={{ ...vis(visible.roads), ...round, "line-sort-key": publicOr(2, 1) }} />
          <Layer id="roads-line" type="line" filter={NETWORK} paint={{ "line-color": mapPalette.road.network, "line-width": NETWORK_WIDTH }}
            layout={{ ...vis(visible.roads), ...round, "line-sort-key": publicOr(2, 1) }} />
          <Layer id="roads-internal-casing" type="line" filter={INTERNAL}
            paint={{ "line-color": mapPalette.road.internalCasing, "line-width": INTERNAL_CASING, "line-opacity": 0.35 }} layout={{ ...vis(visible.internalRoads), ...round }} />
          <Layer id="roads-internal-line" type="line" filter={INTERNAL}
            paint={{ "line-color": mapPalette.road.internal, "line-width": INTERNAL_WIDTH, "line-dasharray": [2, 1.2] }} layout={vis(visible.internalRoads)} />
          <Layer id={ROADS_HIT} type="line" filter={isClass(hitClasses)} paint={{ "line-color": mapPalette.road.casing, "line-width": 14, "line-opacity": 0 }}
            layout={vis(hitClasses.length > 0)} />
          <Layer id="roads-selected" type="line" filter={["==", ["get", "road_id"], selectedRoad ?? ""]}
            paint={{ "line-color": mapPalette.selected, "line-width": 5 }} layout={{ ...vis(visible.roads || visible.internalRoads), ...round }} />
        </Source>
      )}
      {available.farms && (
        <Source id="farms" type="geojson" data={farms ?? EMPTY}>
          <Layer id={FARMS_FILL} type="fill" paint={{ "fill-color": mapPalette.farm.fill, "fill-opacity": 0.06 }} layout={vis(visible.farms)} />
          <Layer id="farms-casing" type="line" paint={{ "line-color": mapPalette.farm.halo, "line-width": byZoom([13, 2.4], [16, 3.6], [19, 5]), "line-opacity": 0.55 }}
            layout={{ ...vis(visible.farms), "line-join": "round" }} />
          <Layer id="farms-line" type="line" paint={{ "line-color": mapPalette.farm.line, "line-width": FARM_WIDTH }} layout={{ ...vis(visible.farms), "line-join": "round" }} />
          <Layer id="farms-selected" type="line" filter={["==", ["get", "farm_id"], selectedFarm ?? ""]}
            paint={{ "line-color": mapPalette.selected, "line-width": 3.5 }} layout={vis(visible.farms)} />
        </Source>
      )}
    </>
  );
}

/** "F01" at each farm's label point, from FARM_LABEL_ZOOM on; a click selects the farm. */
export function FarmLabels({ data, zoom, onPick }: { data: FarmsRoads; zoom: number; onPick: (farmId: string) => void }) {
  if (!data.visible.farms || !data.farms || zoom < FARM_LABEL_ZOOM) return null;
  const halo = mapPalette.farm.halo;
  return (
    <>
      {data.farms.features.flatMap(({ properties: p }) =>
        p.label_point
          ? [
              <Marker key={p.farm_id} longitude={p.label_point[0]} latitude={p.label_point[1]} anchor="center"
                onClick={(e) => { e.originalEvent.stopPropagation(); onPick(p.farm_id); }}>
                <Box
                  component="button"
                  type="button"
                  aria-label={p.farm_id}
                  data-farm-label={p.farm_id}
                  sx={{
                    all: "unset", cursor: "pointer", fontSize: 13, fontWeight: 800, letterSpacing: 0.3, lineHeight: 1,
                    color: mapPalette.farm.label,
                    textShadow: `0 0 2px ${halo}, 0 0 2px ${halo}, 0 0 3px ${halo}, 0 0 4px ${halo}`,
                  }}
                >
                  {p.farm_id}
                </Box>
              </Marker>,
            ]
          : [],
      )}
    </>
  );
}
