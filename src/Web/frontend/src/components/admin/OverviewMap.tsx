"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapRef } from "@vis.gl/react-maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { FeatureCollection, Geometry } from "geojson";
import Box from "@mui/material/Box";
import { mapPalette } from "@/theme/mapPalette";
import { baseStyle } from "@/components/map/mapStyle";
import { bboxOf, bboxOfCollection } from "@/components/map/geo";

if (typeof window !== "undefined") setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

export interface OverviewUat {
  key: string;
  name: string;
  active: boolean;
  hasSurvey: boolean;
  geofence: Geometry;
}

/** All municipalities (filled by status) and all survey footprints, labelled, fitted to everything. */
export function OverviewMap({ uats, footprints }: { uats: OverviewUat[]; footprints: Geometry[] }) {
  const mapRef = useRef<MapRef>(null);

  const uatFc = useMemo<FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: uats.map((u) => ({
        type: "Feature",
        properties: { key: u.key, status: !u.active ? "inactive" : u.hasSurvey ? "surveyed" : "enrolled" },
        geometry: u.geofence,
      })),
    }),
    [uats],
  );
  const surveyFc = useMemo<FeatureCollection>(
    () => ({ type: "FeatureCollection", features: footprints.map((g) => ({ type: "Feature", properties: {}, geometry: g })) }),
    [footprints],
  );
  const labels = useMemo(
    () =>
      uats.map((u) => {
        const b = bboxOf([u.geofence])!;
        return { key: u.key, name: u.name, lon: (b[0] + b[2]) / 2, lat: (b[1] + b[3]) / 2 };
      }),
    [uats],
  );

  // fit once the map is ready (a ref is not available on the first render frame)
  const fit = useCallback(() => {
    const all: FeatureCollection = { type: "FeatureCollection", features: [...uatFc.features, ...surveyFc.features] };
    const b = all.features.length ? bboxOfCollection(all) : null;
    if (b) mapRef.current?.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 48, duration: 0 });
  }, [uatFc, surveyFc]);

  // the card can change width after the map is created (grid / fonts): keep the canvas in sync and re-fit
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      mapRef.current?.resize();
      fit();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fit]);

  return (
    <Box ref={boxRef} sx={{ width: "100%", height: "100%" }}>
    <Map
      ref={mapRef}
      initialViewState={{ longitude: 28.7, latitude: 47.05, zoom: 9 }}
      mapStyle={baseStyle}
      onLoad={() => {
        mapRef.current?.resize();
        fit();
      }}
      attributionControl={{ compact: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      <ScaleControl position="bottom-left" unit="metric" />
      <Source id="uats" type="geojson" data={uatFc}>
        <Layer
          id="uats-fill"
          type="fill"
          paint={{
            "fill-color": ["match", ["get", "status"], "surveyed", mapPalette.geofence, "inactive", mapPalette.passage, mapPalette.block],
            "fill-opacity": ["match", ["get", "status"], "surveyed", 0.22, 0.28],
          }}
        />
        <Layer id="uats-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 3.5, "line-opacity": 0.8 }} />
        <Layer
          id="uats-line"
          type="line"
          paint={{
            "line-color": ["match", ["get", "status"], "inactive", mapPalette.passage, mapPalette.geofence],
            "line-width": 1.8,
            "line-dasharray": [3, 1.5],
          }}
        />
      </Source>
      <Source id="surveys" type="geojson" data={surveyFc}>
        <Layer id="surveys-fill" type="fill" paint={{ "fill-color": mapPalette.canopyFill, "fill-opacity": 0.55 }} />
        <Layer id="surveys-line" type="line" paint={{ "line-color": mapPalette.canopyLine, "line-width": 1.5 }} />
      </Source>
      {labels.map((l) => (
        <Marker key={l.key} longitude={l.lon} latitude={l.lat} anchor="center">
          <Box
            sx={{
              px: 1.5,
              py: 0.5,
              borderRadius: 1,
              bgcolor: "background.paper",
              boxShadow: 2,
              fontSize: 12,
              fontWeight: 700,
              color: "text.primary",
              whiteSpace: "nowrap",
              pointerEvents: "none",
            }}
          >
            {l.name}
          </Box>
        </Marker>
      ))}
    </Map>
    </Box>
  );
}
