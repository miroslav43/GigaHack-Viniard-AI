"use client";

import { useEffect, useMemo, useRef } from "react";
import Map, { Layer, NavigationControl, ScaleControl, Source, type MapRef } from "@vis.gl/react-maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { FeatureCollection, Geometry } from "geojson";
import { mapPalette } from "@/theme/mapPalette";
import { baseStyle } from "./mapStyle";
import { bboxOfCollection, maskOutside } from "./geo";

if (typeof window !== "undefined") setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

/** One or more boundaries on the base map; with `mask`, everything outside the first is greyed out. */
export function BoundaryMap({
  geofence,
  others = [],
  mask: withMask = true,
}: {
  geofence: Geometry | null;
  others?: Geometry[];
  mask?: boolean;
}) {
  const mapRef = useRef<MapRef>(null);
  const fc = useMemo<FeatureCollection | null>(
    () => (geofence ? { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: geofence }] } : null),
    [geofence],
  );
  const othersFc = useMemo<FeatureCollection>(
    () => ({ type: "FeatureCollection", features: others.map((g) => ({ type: "Feature", properties: {}, geometry: g })) }),
    [others],
  );
  const mask = useMemo(() => (fc && withMask ? maskOutside(fc) : null), [fc, withMask]);

  useEffect(() => {
    const all: FeatureCollection = { type: "FeatureCollection", features: [...(fc?.features ?? []), ...othersFc.features] };
    const b = all.features.length ? bboxOfCollection(all) : null;
    if (!b) return;
    const frame = requestAnimationFrame(() => mapRef.current?.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 40, duration: 300 }));
    return () => cancelAnimationFrame(frame);
  }, [fc, othersFc]);

  return (
    <Map
      ref={mapRef}
      initialViewState={{ longitude: 28.68, latitude: 47.11, zoom: 10.5 }}
      mapStyle={baseStyle}
      attributionControl={{ compact: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      <ScaleControl position="bottom-left" unit="metric" />
      {others.length > 0 && (
        <Source id="others" type="geojson" data={othersFc}>
          <Layer id="others-fill" type="fill" paint={{ "fill-color": mapPalette.block, "fill-opacity": 0.25 }} />
          <Layer id="others-line" type="line" paint={{ "line-color": mapPalette.geofence, "line-width": 1, "line-opacity": 0.6 }} />
        </Source>
      )}
      {mask && (
        <Source id="mask" type="geojson" data={mask}>
          <Layer id="mask" type="fill" paint={{ "fill-color": mapPalette.geofenceMask, "fill-opacity": 0.45 }} />
        </Source>
      )}
      {fc && (
        <Source id="geofence" type="geojson" data={fc}>
          <Layer id="geofence-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 4, "line-opacity": 0.8 }} />
          <Layer id="geofence-line" type="line" paint={{ "line-color": mapPalette.geofence, "line-width": 2, "line-dasharray": [4, 2] }} />
        </Source>
      )}
    </Map>
  );
}
