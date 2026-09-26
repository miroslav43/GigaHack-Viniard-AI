"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import Map, {
  Layer,
  Marker,
  NavigationControl,
  ScaleControl,
  Source,
  type MapLayerMouseEvent,
  type MapRef,
} from "@vis.gl/react-maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Feature, FeatureCollection, Geometry, LineString, MultiLineString, Point, Polygon } from "geojson";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import { mapPalette } from "@/theme/mapPalette";
import type { RowRecord, SurveySummary, TargetProps, WasteProps } from "@/lib/types";
import { baseStyle, ORTHO_ANCHOR, ZOOM } from "./mapStyle";
import { bboxOf, bboxOfCollection, intersects, maskOutside, type BBox } from "./geo";
import { LayerPanel, DEFAULT_VISIBILITY, type LayerKey } from "./LayerPanel";
import { AttributePanel, type Selection } from "./AttributePanel";
import { KpiStrip } from "./KpiStrip";
import { MeasureTool } from "./MeasureTool";
import { ReliefControl } from "./ReliefControl";
import { useRelief } from "./useRelief";
import { ROUTE_ARROW, routeArrowImage } from "./arrowImage";
import type { LonLat } from "@/lib/utm";
import type { TerrainMeta } from "@/lib/terrain";

interface TileIndex {
  overview: { url: string; corners: [number, number][] };
  tiles: { id: string; url: string; corners: [number, number][] }[];
}

const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };
const INTERACTIVE = ["targets-circle", "waste-fill", "rows-hit", "canopies-fill", "interrows-fill", "blocks-fill"];
const MAX_DETAIL_TILES = 48;
// every target is a circle on the canvas; the numbered DOM markers are only for the route stops in view, close up
const TARGET_LABEL_ZOOM = 18;
const MAX_TARGET_LABELS = 120;

// see scripts/copy-maplibre-worker.mjs
if (typeof window !== "undefined") setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

const vis = (on: boolean) => ({ visibility: on ? ("visible" as const) : ("none" as const) });

export function MapExplorer({
  summary,
  rows,
  dataBase,
  geofence: geofenceGeometry,
  terrain = null,
}: {
  summary: SurveySummary;
  rows: RowRecord[];
  dataBase: string;
  geofence: Geometry;
  /** synthetic canopy relief for the oblique 3D view (null: not generated, the view stays 2D) */
  terrain?: TerrainMeta | null;
}) {
  const t = useTranslations();
  const theme = useTheme();
  const compact = useMediaQuery(theme.breakpoints.down("lg"));
  const params = useSearchParams();
  const mapRef = useRef<MapRef>(null);

  const [tileIndex, setTileIndex] = useState<TileIndex | null>(null);
  const [rowsFc, setRowsFc] = useState<FeatureCollection<MultiLineString> | null>(null);
  const [blocksFc, setBlocksFc] = useState<FeatureCollection<Polygon> | null>(null);
  const [targets, setTargets] = useState<FeatureCollection<Point, TargetProps> | null>(null);
  const [wasteFc, setWasteFc] = useState<FeatureCollection<Polygon, WasteProps> | null>(null);
  const [route, setRoute] = useState<FeatureCollection<LineString> | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const geofence = useMemo<FeatureCollection>(
    () => ({ type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: geofenceGeometry }] }),
    [geofenceGeometry],
  );
  const [start, setStart] = useState<[number, number] | null>(null);
  const [studyBbox, setStudyBbox] = useState<BBox | null>(null);
  const [detailTiles, setDetailTiles] = useState<TileIndex["tiles"]>([]);
  const [targetLabels, setTargetLabels] = useState<Feature<Point, TargetProps>[]>([]);
  const [visible, setVisible] = useState<Record<LayerKey, boolean>>(DEFAULT_VISIBILITY);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [cursor, setCursor] = useState<string>("grab");
  // null = follow the screen size (closed on phones/tablets) until the user toggles it. Reading `compact` only in the
  // initial state was wrong: useMediaQuery is false on the first render, so the panel stayed open over the map on phones
  const [panelChoice, setPanelChoice] = useState<boolean | null>(null);
  const panelOpen = panelChoice ?? !compact;
  const [measuring, setMeasuring] = useState(false);
  const [measurePts, setMeasurePts] = useState<LonLat[]>([]);
  const [arrowReady, setArrowReady] = useState(false);
  // arrowReady is set in onLoad: the style is loaded, so the relief can add its sources and set the terrain
  const relief = useRelief(mapRef, terrain, dataBase, arrowReady);

  useEffect(() => {
    // `absent`: an optional layer a bundle may not ship (no waste.geojson before the waste model runs) — used only on 404
    const get = <T,>(url: string, absent?: T) =>
      fetch(url).then((r) => {
        if (r.status === 404 && absent) return absent;
        if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
        return r.json() as Promise<T>;
      });
    Promise.all([
      get<TileIndex>("/data/tiles.json"),
      get<FeatureCollection<MultiLineString>>(`${dataBase}/rows.geojson`),
      get<FeatureCollection<Polygon>>(`${dataBase}/blocks.geojson`),
      get<FeatureCollection<Point, TargetProps>>(`${dataBase}/targets.geojson`),
      get<FeatureCollection<LineString>>(`${dataBase}/route.geojson`),
      get<FeatureCollection<Point>>("/data/ref/start.geojson"),
      get<FeatureCollection>("/data/ref/study_area.geojson"),
      get<FeatureCollection<Polygon, WasteProps>>(`${dataBase}/waste.geojson`, { type: "FeatureCollection", features: [] }),
    ])
      .then(([ti, rw, bl, tg, rt, st, sa, ws]) => {
        setTileIndex(ti);
        setRowsFc(rw);
        setBlocksFc(bl);
        setTargets(tg);
        setRoute(rt);
        setWasteFc(ws);
        setStart(st.features[0].geometry.coordinates as [number, number]);
        setStudyBbox(bboxOfCollection(sa));
      })
      // a missing or broken required file: show what failed instead of "loading" forever
      .catch((e: unknown) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, [dataBase]);

  const mask = useMemo(() => (geofence ? maskOutside(geofence) : null), [geofence]);
  const tileBoxes = useMemo(
    () => tileIndex?.tiles.map((t) => ({ ...t, bbox: bboxOf([{ type: "Polygon", coordinates: [[...t.corners, t.corners[0]]] }])! })) ?? [],
    [tileIndex],
  );

  const fit = useCallback((b: BBox | null, maxZoom = 20) => {
    if (b) mapRef.current?.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 60, maxZoom, duration: 800 });
  }, []);

  const routeStops = useMemo(() => targets?.features.filter((f) => f.properties.route_order != null) ?? [], [targets]);

  const refreshDetail = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const zoom = map.getZoom();
    const b = map.getBounds();
    const view: BBox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    setDetailTiles(zoom < ZOOM.orthoDetail ? [] : tileBoxes.filter((t) => intersects(t.bbox, view)).slice(0, MAX_DETAIL_TILES));
    setTargetLabels(
      zoom < TARGET_LABEL_ZOOM
        ? []
        : routeStops.filter(({ geometry: { coordinates: [lon, lat] } }) => intersects([lon, lat, lon, lat], view)).slice(0, MAX_TARGET_LABELS),
    );
  }, [tileBoxes, routeStops]);

  const selectRow = useCallback(
    (rowId: string, zoom = true) => {
      const f = rowsFc?.features.find((x) => x.properties?.row_id === rowId);
      if (!f) return;
      setSelection({ layer: "rows", props: f.properties ?? {} });
      if (zoom) fit(bboxOf([f]), 19.5);
    },
    [rowsFc, fit],
  );
  const selectBlock = useCallback(
    (id: string) => {
      const f = blocksFc?.features.find((x) => x.properties?.vineyard_id === id);
      if (!f) return;
      setSelection({ layer: "blocks", props: f.properties ?? {} });
      fit(bboxOf([f]), 19);
    },
    [blocksFc, fit],
  );

  const selectTarget = useCallback(
    (id: string) => {
      const f = targets?.features.find((x) => x.properties.target_id === id);
      if (!f) return;
      setSelection({ layer: "targets", props: f.properties as unknown as Record<string, unknown> });
      const [lon, lat] = f.geometry.coordinates;
      mapRef.current?.flyTo({ center: [lon, lat], zoom: 20, duration: 800 });
    },
    [targets],
  );

  // deep links: /harta?bloc=V01, /harta?rand=V01-R05, /harta?tinta=T003 (from a field task)
  const deepLinked = useRef(false);
  useEffect(() => {
    if (deepLinked.current || !rowsFc || !blocksFc || !studyBbox || !targets) return;
    let frame = 0;
    const apply = () => {
      // react-maplibre imports maplibre-gl lazily, so the data can arrive before the map exists: wait for it
      if (!mapRef.current) {
        frame = requestAnimationFrame(apply);
        return;
      }
      deepLinked.current = true;
      const rand = params.get("rand"), bloc = params.get("bloc"), tinta = params.get("tinta");
      if (tinta) selectTarget(tinta);
      else if (rand) selectRow(rand);
      else if (bloc) selectBlock(bloc);
      else fit(studyBbox, 17);
    };
    frame = requestAnimationFrame(apply);
    return () => cancelAnimationFrame(frame);
  }, [rowsFc, blocksFc, studyBbox, targets, params, selectRow, selectBlock, selectTarget, fit]);

  const onLoad = () => {
    const map = mapRef.current?.getMap();
    if (map && !map.hasImage(ROUTE_ARROW)) map.addImage(ROUTE_ARROW, routeArrowImage(), { pixelRatio: 2 });
    setArrowReady(true);
    refreshDetail();
  };

  const measureFc = useMemo<FeatureCollection>(() => {
    const features: FeatureCollection["features"] = measurePts.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
    if (measurePts.length >= 2)
      features.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: measuring ? measurePts : [...measurePts, ...(measurePts.length >= 3 ? [measurePts[0]] : [])] } });
    if (!measuring && measurePts.length >= 3)
      features.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...measurePts, measurePts[0]]] } });
    return { type: "FeatureCollection", features };
  }, [measurePts, measuring]);

  const onClick = (e: MapLayerMouseEvent) => {
    if (measuring) {
      // the second click of a double-click is ignored; onDblClick finishes the shape
      if (e.originalEvent.detail > 1) return;
      setMeasurePts((pts) => [...pts, [e.lngLat.lng, e.lngLat.lat]]);
      return;
    }
    const f = e.features?.[0];
    if (!f) return setSelection(null);
    const layer = f.layer.id.split("-")[0] as Selection["layer"];
    const props = f.properties ?? {};
    // MapLibre serialises nested objects to strings
    if (typeof props.tile_structures === "string") props.tile_structures = JSON.parse(props.tile_structures);
    setSelection({ layer, props });
  };

  const selectedFilter = (key: string, layer: Selection["layer"]) =>
    ["==", ["get", key], selection?.layer === layer ? String(selection.props[key]) : "__none__"] as const;

  const overviewCoords = tileIndex?.overview.corners as [[number, number], [number, number], [number, number], [number, number]] | undefined;

  return (
    <Box sx={{ position: "relative", height: { xs: "calc(100dvh - 64px - 56px)", md: "100dvh" }, display: "flex", flexDirection: "column" }}>
      <KpiStrip summary={summary} />
      <Box sx={{ position: "relative", flex: 1, minHeight: 0 }}>
        <Map
          ref={mapRef}
          initialViewState={{ longitude: 28.707, latitude: 47.125, zoom: 15 }}
          mapStyle={baseStyle}
          interactiveLayerIds={INTERACTIVE}
          onClick={onClick}
          onDblClick={() => measuring && setMeasuring(false)}
          doubleClickZoom={!measuring}
          onMouseEnter={() => setCursor("pointer")}
          onMouseLeave={() => setCursor("grab")}
          cursor={measuring ? "crosshair" : cursor}
          onMoveEnd={refreshDetail}
          onLoad={onLoad}
          maxZoom={22}
          attributionControl={{ compact: true, customAttribution: "Ortofoto Sireț3 © 3DATA COLLECT / OpenAerialMap, CC BY 4.0" }}
          style={{ position: "absolute", inset: 0 }}
        >
          <NavigationControl position="bottom-right" showCompass={false} />
          <ScaleControl position="bottom-left" unit="metric" />

          {/* ---- anchor: always the first layer, so the orthophoto (beforeId) stays under every vector layer ---- */}
          <Source id="ortho-anchor-src" type="geojson" data={EMPTY}>
            <Layer id={ORTHO_ANCHOR} type="line" paint={{ "line-opacity": 0 }} />
          </Source>

          {/* ---- vectors ---- */}
          {mask && (
            <Source id="geofence-mask-src" type="geojson" data={mask}>
              <Layer id="geofence-mask" type="fill" paint={{ "fill-color": mapPalette.geofenceMask, "fill-opacity": 0.45 }} layout={vis(visible.geofence)} />
            </Source>
          )}
          {geofence && (
            <Source id="geofence" type="geojson" data={geofence}>
              <Layer id="geofence-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 4, "line-opacity": 0.8 }} layout={vis(visible.geofence)} />
              <Layer id="geofence-line" type="line" paint={{ "line-color": mapPalette.geofence, "line-width": 2, "line-dasharray": [4, 2] }} layout={vis(visible.geofence)} />
            </Source>
          )}
          <Source id="passages" type="geojson" data="/data/ref/passages.geojson">
            <Layer id="passages-fill" type="fill" paint={{ "fill-color": mapPalette.passage, "fill-opacity": 0.3 }} layout={vis(visible.reference)} />
          </Source>
          <Source id="forbidden" type="geojson" data="/data/ref/forbidden.geojson">
            <Layer id="forbidden-fill" type="fill" paint={{ "fill-color": mapPalette.forbidden, "fill-opacity": 0.15 }} layout={vis(visible.reference)} />
            <Layer id="forbidden-line" type="line" paint={{ "line-color": mapPalette.forbidden, "line-width": 1 }} layout={vis(visible.reference)} />
          </Source>
          <Source id="study-area" type="geojson" data="/data/ref/study_area.geojson">
            <Layer id="study-area-line" type="line" paint={{ "line-color": mapPalette.studyArea, "line-width": 1, "line-opacity": 0.6, "line-dasharray": [2, 2] }} />
          </Source>
          {blocksFc && (
            <Source id="blocks" type="geojson" data={blocksFc}>
              <Layer id="blocks-fill" type="fill" maxzoom={ZOOM.rows} paint={{ "fill-color": mapPalette.block, "fill-opacity": 0.35 }} layout={vis(visible.blocks)} />
              <Layer id="blocks-line" type="line" paint={{ "line-color": mapPalette.geofence, "line-width": 1.5, "line-opacity": 0.8 }} layout={vis(visible.blocks)} />
            </Source>
          )}
          <Source id="interrows" type="geojson" data={`${dataBase}/interrows.geojson`}>
            <Layer
              id="interrows-fill"
              type="fill"
              minzoom={ZOOM.interrows}
              paint={{
                "fill-color": ["match", ["get", "interrow_cover"], "bare_soil", mapPalette.interrow.bare_soil, "mixed", mapPalette.interrow.mixed, "vegetation", mapPalette.interrow.vegetation, mapPalette.interrow.unassessable],
                "fill-opacity": 0.28,
              }}
              layout={vis(visible.interrows)}
            />
            <Layer id="interrows-selected" type="line" filter={selectedFilter("interrow_id", "interrows") as never} paint={{ "line-color": mapPalette.selected, "line-width": 2.5 }} />
          </Source>
          <Source id="canopies" type="geojson" data={`${dataBase}/canopies.geojson`}>
            <Layer id="canopies-fill" type="fill" minzoom={ZOOM.canopies} paint={{ "fill-color": mapPalette.canopyFill, "fill-opacity": 0.4 }} layout={vis(visible.canopies)} />
            <Layer id="canopies-line" type="line" minzoom={ZOOM.canopies} paint={{ "line-color": mapPalette.canopyLine, "line-width": 1 }} layout={vis(visible.canopies)} />
            <Layer id="canopies-selected" type="line" filter={selectedFilter("canopy_id", "canopies") as never} paint={{ "line-color": mapPalette.selected, "line-width": 2.5 }} />
          </Source>
          {rowsFc && (
            <Source id="rows" type="geojson" data={rowsFc}>
              <Layer id="rows-casing" type="line" minzoom={ZOOM.rows} paint={{ "line-color": mapPalette.casing, "line-width": 3.5, "line-opacity": 0.7 }} layout={vis(visible.rows)} />
              <Layer
                id="rows-line"
                type="line"
                minzoom={ZOOM.rows}
                paint={{
                  "line-color": ["match", ["get", "row_structure"], "disrupted", mapPalette.row.disrupted, "unassessable", mapPalette.row.unassessable, mapPalette.row.regular],
                  "line-width": ["match", ["get", "row_structure"], "disrupted", 3, 2],
                }}
                layout={{ ...vis(visible.rows), "line-cap": "round" }}
              />
              <Layer id="rows-hit" type="line" minzoom={ZOOM.rows} paint={{ "line-color": mapPalette.casing, "line-width": 14, "line-opacity": 0 }} layout={vis(visible.rows)} />
              <Layer id="rows-selected" type="line" filter={selectedFilter("row_id", "rows") as never} paint={{ "line-color": mapPalette.selected, "line-width": 5 }} />
            </Source>
          )}
          {wasteFc && (
            <Source id="waste" type="geojson" data={wasteFc}>
              <Layer id="waste-fill" type="fill" paint={{ "fill-color": mapPalette.waste, "fill-opacity": 0.25 }} layout={vis(visible.waste)} />
              <Layer id="waste-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 4, "line-opacity": 0.8 }} layout={vis(visible.waste)} />
              <Layer id="waste-line" type="line" paint={{ "line-color": mapPalette.waste, "line-width": 2 }} layout={vis(visible.waste)} />
              <Layer id="waste-selected" type="line" filter={selectedFilter("waste_id", "waste") as never} paint={{ "line-color": mapPalette.selected, "line-width": 3 }} />
            </Source>
          )}
          {route && (
            <Source id="route" type="geojson" data={route}>
              <Layer id="route-halo" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 7, "line-opacity": 0.85 }} layout={{ ...vis(visible.route), "line-join": "round", "line-cap": "round" }} />
              <Layer id="route-line" type="line" paint={{ "line-color": mapPalette.route, "line-width": 4 }} layout={{ ...vis(visible.route), "line-join": "round", "line-cap": "round" }} />
              {arrowReady && (
                <Layer
                  id="route-arrows"
                  type="symbol"
                  layout={{
                    ...vis(visible.route),
                    "symbol-placement": "line",
                    "symbol-spacing": 90,
                    "icon-image": ROUTE_ARROW,
                    "icon-size": 1,
                    "icon-rotation-alignment": "map",
                    "icon-allow-overlap": true,
                    "icon-ignore-placement": true,
                  }}
                />
              )}
            </Source>
          )}
          {targets && (
            <Source id="targets" type="geojson" data={targets}>
              <Layer
                id="targets-circle"
                type="circle"
                paint={{
                  "circle-radius": ["interpolate", ["linear"], ["zoom"], 14, 2.5, 17, 5, 20, 8],
                  // route_order null (or absent) = the route does not visit it
                  "circle-color": ["case", ["==", ["typeof", ["get", "route_order"]], "number"], mapPalette.target, mapPalette.targetOffRoute],
                  "circle-stroke-color": mapPalette.casing,
                  "circle-stroke-width": 1.5,
                }}
                layout={vis(visible.route)}
              />
              <Layer
                id="targets-selected"
                type="circle"
                filter={selectedFilter("target_id", "targets") as never}
                paint={{ "circle-radius": 11, "circle-opacity": 0, "circle-stroke-color": mapPalette.selected, "circle-stroke-width": 3 }}
              />
            </Source>
          )}

          <Source id="measure" type="geojson" data={measureFc}>
            <Layer id="measure-fill" type="fill" filter={["==", ["geometry-type"], "Polygon"]} paint={{ "fill-color": mapPalette.selected, "fill-opacity": 0.2 }} />
            <Layer id="measure-casing" type="line" filter={["==", ["geometry-type"], "LineString"]} paint={{ "line-color": mapPalette.casing, "line-width": 5 }} />
            <Layer id="measure-line" type="line" filter={["==", ["geometry-type"], "LineString"]} paint={{ "line-color": mapPalette.selected, "line-width": 2.5, "line-dasharray": [2, 1] }} />
            <Layer id="measure-points" type="circle" filter={["==", ["geometry-type"], "Point"]} paint={{ "circle-radius": 4.5, "circle-color": mapPalette.selected, "circle-stroke-color": mapPalette.casing, "circle-stroke-width": 2 }} />
          </Source>

          {/* ---- orthophoto: overview mosaic + full-resolution tiles in view ---- */}
          {overviewCoords && visible.ortho && (
            <Source id="ortho-overview" type="image" url={tileIndex!.overview.url} coordinates={overviewCoords}>
              <Layer id="ortho-overview" type="raster" beforeId={ORTHO_ANCHOR} paint={{ "raster-fade-duration": 0 }} />
            </Source>
          )}
          {visible.ortho &&
            detailTiles.map((t) => (
              <Source key={t.id} id={`ortho-${t.id}`} type="image" url={t.url} coordinates={t.corners as never}>
                <Layer id={`ortho-${t.id}`} type="raster" beforeId={ORTHO_ANCHOR} paint={{ "raster-fade-duration": 150 }} />
              </Source>
            ))}

          {/* ---- markers: numbered route stops in view (refreshDetail) ---- */}
          {visible.route &&
            targetLabels.map((f) => {
              const [lon, lat] = f.geometry.coordinates;
              const p = f.properties;
              const active = selection?.layer === "targets" && selection.props.target_id === p.target_id;
              return (
                <Marker key={p.target_id} longitude={lon} latitude={lat} onClick={(e) => { e.originalEvent.stopPropagation(); setSelection({ layer: "targets", props: p as unknown as Record<string, unknown> }); }}>
                  <Box
                    title={`${p.route_order}. ${p.target_id}`}
                    sx={{
                      width: 24, height: 24, borderRadius: "50%", display: "grid", placeItems: "center", cursor: "pointer",
                      bgcolor: active ? "warning.main" : "info.main", color: "common.white", fontSize: 11, fontWeight: 700,
                      border: 2, borderColor: "common.white", boxShadow: 2,
                    }}
                  >
                    {p.route_order}
                  </Box>
                </Marker>
              );
            })}
          {start && (
            <Marker longitude={start[0]} latitude={start[1]} anchor="center">
              <Box
                title="START / FINISH"
                sx={{ px: 1.5, py: 0.5, borderRadius: 1, bgcolor: "grey.900", color: "common.white", fontSize: 11, fontWeight: 700, border: 2, borderColor: "common.white", boxShadow: 2 }}
              >
                S/F
              </Box>
            </Marker>
          )}
        </Map>

        <LayerPanel
          open={panelOpen}
          onToggleOpen={() => setPanelChoice(!panelOpen)}
          visible={visible}
          onChange={(k, v) => setVisible((s) => ({ ...s, [k]: v }))}
          rows={rows}
          blocks={summary.blocks.map((b) => b.vineyard_id)}
          onPickRow={selectRow}
          onPickBlock={selectBlock}
          onFitAll={() => fit(studyBbox, 17)}
        />
        <MeasureTool
          active={measuring}
          points={measurePts}
          onToggle={() => {
            setSelection(null);
            if (!measuring) setMeasurePts([]);
            setMeasuring((v) => !v);
          }}
          onClear={() => {
            setMeasurePts([]);
            setMeasuring(false);
          }}
        />
        <ReliefControl available={relief.available} ready={relief.ready} settings={relief.settings} onChange={relief.update} />
        {selection && (
          <AttributePanel
            selection={selection}
            summary={summary}
            onClose={() => setSelection(null)}
            onZoomRow={(id) => selectRow(id)}
          />
        )}
        {!rowsFc && (
          <Box sx={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", pointerEvents: "none", px: 4 }}>
            {loadError ? (
              <Alert severity="error" sx={{ maxWidth: 560 }}>
                {t("common.mapLoadError", { detail: loadError })}
              </Alert>
            ) : (
              <Typography color="text.secondary">{t("common.loadingMap")}</Typography>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
