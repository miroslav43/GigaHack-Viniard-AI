"use client";

// The planned farm route: a violet line with direction arrows, numbered stops and the start (= finish) marker.
// Stop numbers are DOM markers (the base style has no glyphs): all of them on a small farm, only those in view
// close up on a large one.
import { useMemo } from "react";
import { Layer, Marker, Source } from "@vis.gl/react-maplibre";
import type { Feature, FeatureCollection, Point } from "geojson";
import Box from "@mui/material/Box";
import { mapPalette } from "@/theme/mapPalette";
import type { TargetProps } from "@/lib/types";
import type { LonLat } from "@/lib/utm";
import { ROUTE_ARROW } from "../../arrowImage";
import { intersects, type BBox } from "../../geo";
import type { FarmRouteState } from "./useFarmRoute";

/** farms with at most this many stops show every number; bigger ones only close up and in view */
const ALL_LABELS_MAX = 40;
const LABEL_ZOOM = 18;
const MAX_LABELS = 120;
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

export interface FarmRouteStop {
  id: string;
  n: number;
  at: LonLat;
}

/** The planned stops with their positions, in visiting order. */
export function useFarmRouteStops(state: FarmRouteState, targets: FeatureCollection<Point, TargetProps> | null): FarmRouteStop[] {
  return useMemo(() => {
    if (state.status !== "done" || !targets) return [];
    const at = new Map(targets.features.map((f) => [f.properties.target_id, f.geometry.coordinates as LonLat]));
    return state.result.order.flatMap((id, i) => {
      const p = at.get(id);
      return p ? [{ id, n: i + 1, at: p }] : [];
    });
  }, [state, targets]);
}

export function FarmRouteLayers({
  state,
  stops,
  arrowReady,
  zoom,
  view,
}: {
  state: FarmRouteState;
  stops: FarmRouteStop[];
  arrowReady: boolean;
  zoom: number;
  view: BBox | null;
}) {
  const line = useMemo<FeatureCollection>(
    () =>
      state.status === "done"
        ? { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: state.result.line } }] }
        : EMPTY,
    [state],
  );
  const points = useMemo<FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: stops.map((s): Feature<Point> => ({ type: "Feature", properties: { n: s.n }, geometry: { type: "Point", coordinates: s.at } })),
    }),
    [stops],
  );
  const labelled =
    stops.length <= ALL_LABELS_MAX
      ? stops
      : zoom < LABEL_ZOOM || !view
        ? []
        : stops.filter(({ at: [lon, lat] }) => intersects([lon, lat, lon, lat], view)).slice(0, MAX_LABELS);
  const start = state.status === "computing" || state.status === "done" || state.status === "error" ? state.start : null;
  const round = { "line-join": "round", "line-cap": "round" } as const;

  return (
    <>
      <Source id="farm-route" type="geojson" data={line}>
        <Layer id="farm-route-halo" type="line" paint={{ "line-color": mapPalette.farmRoute.casing, "line-width": 7, "line-opacity": 0.85 }} layout={round} />
        <Layer id="farm-route-line" type="line" paint={{ "line-color": mapPalette.farmRoute.line, "line-width": 4 }} layout={round} />
        {arrowReady && (
          <Layer
            id="farm-route-arrows"
            type="symbol"
            layout={{
              "symbol-placement": "line",
              "symbol-spacing": 90,
              "icon-image": ROUTE_ARROW,
              "icon-rotation-alignment": "map",
              "icon-allow-overlap": true,
              "icon-ignore-placement": true,
            }}
          />
        )}
      </Source>
      <Source id="farm-route-stops" type="geojson" data={points}>
        <Layer
          id="farm-route-stops"
          type="circle"
          paint={{
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 14, 3, 17, 5.5, 20, 9],
            "circle-color": mapPalette.farmRoute.line,
            "circle-stroke-color": mapPalette.farmRoute.casing,
            "circle-stroke-width": 1.5,
          }}
        />
      </Source>
      {labelled.map((s) => (
        <Marker key={s.id} longitude={s.at[0]} latitude={s.at[1]}>
          <Box
            title={`${s.n}. ${s.id}`}
            data-farm-route-stop={s.id}
            sx={{
              minWidth: 22, height: 22, px: 0.5, borderRadius: 11, display: "grid", placeItems: "center",
              bgcolor: mapPalette.farmRoute.line, color: "common.white", fontSize: 11, fontWeight: 700,
              border: 2, borderColor: "common.white", boxShadow: 2, pointerEvents: "none",
            }}
          >
            {s.n}
          </Box>
        </Marker>
      ))}
      {start && (
        <Marker longitude={start[0]} latitude={start[1]} anchor="center">
          <Box
            data-testid="farm-route-start"
            sx={{
              px: 1.5, py: 0.5, borderRadius: 1, bgcolor: mapPalette.farmRoute.line, color: "common.white", fontSize: 11,
              fontWeight: 700, border: 2, borderColor: "common.white", boxShadow: 2, pointerEvents: "none",
            }}
          >
            S/F
          </Box>
        </Marker>
      )}
    </>
  );
}
