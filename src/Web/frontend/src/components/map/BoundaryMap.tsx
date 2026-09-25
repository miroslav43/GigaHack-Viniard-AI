"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Map, { Layer, NavigationControl, ScaleControl, Source, type MapRef } from "@vis.gl/react-maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { FeatureCollection } from "geojson";
import { mapPalette } from "@/theme/mapPalette";
import { baseStyle } from "./mapStyle";
import { bboxOfCollection, maskOutside } from "./geo";

if (typeof window !== "undefined") setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

/** A municipality boundary on the base map, everything outside greyed out. */
export function BoundaryMap({ geofenceUrl }: { geofenceUrl: string }) {
  const mapRef = useRef<MapRef>(null);
  const [geofence, setGeofence] = useState<FeatureCollection | null>(null);

  useEffect(() => {
    fetch(geofenceUrl)
      .then((r) => r.json() as Promise<FeatureCollection>)
      .then(setGeofence);
  }, [geofenceUrl]);

  const mask = useMemo(() => (geofence ? maskOutside(geofence) : null), [geofence]);

  useEffect(() => {
    const b = geofence && bboxOfCollection(geofence);
    if (!b) return;
    const frame = requestAnimationFrame(() => mapRef.current?.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 40, duration: 0 }));
    return () => cancelAnimationFrame(frame);
  }, [geofence]);

  return (
    <Map
      ref={mapRef}
      initialViewState={{ longitude: 28.68, latitude: 47.11, zoom: 11.5 }}
      mapStyle={baseStyle}
      attributionControl={{ compact: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      <ScaleControl position="bottom-left" unit="metric" />
      {mask && (
        <Source id="mask" type="geojson" data={mask}>
          <Layer id="mask" type="fill" paint={{ "fill-color": mapPalette.geofenceMask, "fill-opacity": 0.45 }} />
        </Source>
      )}
      {geofence && (
        <Source id="geofence" type="geojson" data={geofence}>
          <Layer id="geofence-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 4, "line-opacity": 0.8 }} />
          <Layer id="geofence-line" type="line" paint={{ "line-color": mapPalette.geofence, "line-width": 2, "line-dasharray": [4, 2] }} />
        </Source>
      )}
    </Map>
  );
}
