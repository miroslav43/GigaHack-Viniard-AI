"use client";

// The analysed tile on a map: the uploaded image, the pipeline's vegetation mask and its objects (EPSG:4326 GeoJSON
// from the job folder), with the same colours as /harta (src/theme/mapPalette.ts), layer switches and click → attributes.
import { useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent } from "@vis.gl/react-maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { FeatureCollection } from "geojson";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import Close from "@mui/icons-material/Close";
import { mapPalette } from "@/theme/mapPalette";
import type { JobResult } from "@/lib/analiza/contract";
import { baseStyle } from "@/components/map/mapStyle";
import { bboxOf, type BBox } from "@/components/map/geo";
import { ResultAttributes, type AnalysisSelection } from "./ResultAttributes";

// see scripts/copy-maplibre-worker.mjs
if (typeof window !== "undefined") setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

type LayerKey = "preview" | "vegMask" | "interrows" | "canopies" | "rows" | "waste";
const LAYER_KEYS: LayerKey[] = ["preview", "vegMask", "interrows", "canopies", "rows", "waste"];
type ObjectLayer = "canopies" | "rows" | "interrows" | "waste";
const OBJECT_LAYERS: ObjectLayer[] = ["canopies", "rows", "interrows", "waste"];
/** feature id key of each object layer (selection highlight) */
const ID_KEY: Record<ObjectLayer, string> = { canopies: "canopy_id", rows: "row_id", interrows: "interrow_id", waste: "waste_id" };
// top first: the first feature under a click is the one picked
const INTERACTIVE = ["waste-fill", "rows-hit", "canopies-fill", "interrows-fill"];
const MAX_WASTE_LABELS = 60;
/** wide screens: keep the tile clear of the layer card on the left (~260 px) */
const WIDE_PX = 900;
const fitPadding = () =>
  typeof window !== "undefined" && window.innerWidth >= WIDE_PX ? { top: 24, bottom: 24, right: 24, left: 290 } : 24;

const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };
const vis = (on: boolean) => ({ visibility: on ? ("visible" as const) : ("none" as const) });

function Swatch({ color, line }: { color: string; line?: boolean }) {
  return (
    <Box
      component="span"
      sx={{ display: "inline-block", width: 16, flexShrink: 0, ...(line ? { borderTop: `3px solid ${color}` } : { height: 10, bgcolor: color, opacity: 0.8, borderRadius: 0.5 }) }}
    />
  );
}

export function ResultMap({ job, result }: { job: string; result: JobResult }) {
  const t = useTranslations("analiza.map");
  const tc = useTranslations();
  const base = `/api/analiza/${encodeURIComponent(job)}`;
  const [data, setData] = useState<Record<ObjectLayer, FeatureCollection> | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [visible, setVisible] = useState<Record<LayerKey, boolean>>({ preview: true, vegMask: true, interrows: true, canopies: true, rows: true, waste: true });
  const [selection, setSelection] = useState<AnalysisSelection | null>(null);
  const [cursor, setCursor] = useState("grab");

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      OBJECT_LAYERS.map((k) =>
        fetch(`${base}/${k}.geojson`, { cache: "no-store" }).then((r) => {
          if (!r.ok) throw new Error(`${k}.geojson: HTTP ${r.status}`);
          return r.json() as Promise<FeatureCollection>;
        }),
      ),
    )
      .then((fcs) => !cancelled && setData(Object.fromEntries(OBJECT_LAYERS.map((k, i) => [k, fcs[i]])) as Record<ObjectLayer, FeatureCollection>))
      .catch((e: unknown) => !cancelled && setLoadError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [base]);

  const corners = result.corners;
  const bounds = useMemo<BBox>(() => {
    const lons = corners.map((c) => c[0]);
    const lats = corners.map((c) => c[1]);
    return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
  }, [corners]);

  // "nevalidat" label at the centre of each waste box (DOM markers: the base style has no glyphs)
  const wasteLabels = useMemo(
    () =>
      (data?.waste.features ?? []).slice(0, MAX_WASTE_LABELS).flatMap((f, i) => {
        const b = bboxOf([f]);
        return b ? [{ key: String(f.properties?.waste_id ?? i), lon: (b[0] + b[2]) / 2, lat: b[3] }] : [];
      }),
    [data],
  );

  const onClick = (e: MapLayerMouseEvent) => {
    const f = e.features?.[0];
    if (!f) return setSelection(null);
    setSelection({ layer: f.layer.id.split("-")[0] as ObjectLayer, props: (f.properties ?? {}) as Record<string, unknown> });
  };
  const selected = (layer: ObjectLayer) =>
    ["==", ["to-string", ["get", ID_KEY[layer]]], selection?.layer === layer ? String(selection.props[ID_KEY[layer]]) : "__none__"] as const;

  const legend: { label: string; color: string; line?: boolean }[] = [
    { label: t("legend.vegMask"), color: mapPalette.vegMask },
    { label: t("legend.canopy"), color: mapPalette.canopyFill },
    { label: t("legend.row", { structure: tc("structure.regular") }), color: mapPalette.row.regular, line: true },
    { label: t("legend.row", { structure: tc("structure.disrupted") }), color: mapPalette.row.disrupted, line: true },
    { label: t("legend.row", { structure: tc("structure.unassessable") }), color: mapPalette.row.unassessable, line: true },
    ...(Object.keys(mapPalette.interrow) as (keyof typeof mapPalette.interrow)[]).map((k) => ({
      label: t("legend.interrow", { cover: tc(`cover.${k}`) }),
      color: mapPalette.interrow[k],
    })),
    { label: t("legend.waste"), color: mapPalette.waste },
  ];

  return (
    <Box sx={{ position: "relative", height: { xs: "65vh", md: "72vh" }, minHeight: 420, borderRadius: 2, overflow: "hidden", border: 1, borderColor: "divider" }}>
      <Map
        initialViewState={{ bounds, fitBoundsOptions: { padding: fitPadding() } }}
        mapStyle={baseStyle}
        interactiveLayerIds={INTERACTIVE}
        onClick={onClick}
        onMouseEnter={() => setCursor("pointer")}
        onMouseLeave={() => setCursor("grab")}
        cursor={cursor}
        maxZoom={23}
        attributionControl={{ compact: true, customAttribution: "Sireț3 © 3DATA COLLECT / OpenAerialMap, CC BY 4.0" }}
        style={{ position: "absolute", inset: 0 }}
      >
        <NavigationControl position="bottom-right" showCompass={false} />
        <ScaleControl position="bottom-left" unit="metric" />

        {/* images stay mounted (visibility only), so the stack order never changes */}
        <Source id="preview" type="image" url={`${base}/preview.jpg`} coordinates={corners}>
          <Layer id="preview-raster" type="raster" paint={{ "raster-fade-duration": 0 }} layout={vis(visible.preview)} />
        </Source>
        <Source id="veg-mask" type="image" url={`${base}/veg_mask.png`} coordinates={corners}>
          <Layer id="veg-mask-raster" type="raster" paint={{ "raster-fade-duration": 0 }} layout={vis(visible.vegMask)} />
        </Source>

        <Source id="interrows" type="geojson" data={data?.interrows ?? EMPTY}>
          <Layer
            id="interrows-fill"
            type="fill"
            paint={{
              "fill-color": ["match", ["get", "interrow_cover"], "bare_soil", mapPalette.interrow.bare_soil, "mixed", mapPalette.interrow.mixed, "vegetation", mapPalette.interrow.vegetation, mapPalette.interrow.unassessable],
              "fill-opacity": 0.28,
            }}
            layout={vis(visible.interrows)}
          />
          <Layer id="interrows-selected" type="line" filter={selected("interrows") as never} paint={{ "line-color": mapPalette.selected, "line-width": 2.5 }} />
        </Source>
        <Source id="canopies" type="geojson" data={data?.canopies ?? EMPTY}>
          <Layer id="canopies-fill" type="fill" paint={{ "fill-color": mapPalette.canopyFill, "fill-opacity": 0.4 }} layout={vis(visible.canopies)} />
          <Layer id="canopies-line" type="line" paint={{ "line-color": mapPalette.canopyLine, "line-width": 1 }} layout={vis(visible.canopies)} />
          <Layer id="canopies-selected" type="line" filter={selected("canopies") as never} paint={{ "line-color": mapPalette.selected, "line-width": 2.5 }} />
        </Source>
        <Source id="rows" type="geojson" data={data?.rows ?? EMPTY}>
          <Layer id="rows-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 3.5, "line-opacity": 0.7 }} layout={vis(visible.rows)} />
          <Layer
            id="rows-line"
            type="line"
            paint={{
              "line-color": ["match", ["get", "row_structure"], "disrupted", mapPalette.row.disrupted, "unassessable", mapPalette.row.unassessable, mapPalette.row.regular],
              "line-width": ["match", ["get", "row_structure"], "disrupted", 3, 2],
            }}
            layout={{ ...vis(visible.rows), "line-cap": "round" }}
          />
          <Layer id="rows-hit" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 14, "line-opacity": 0 }} layout={vis(visible.rows)} />
          <Layer id="rows-selected" type="line" filter={selected("rows") as never} paint={{ "line-color": mapPalette.selected, "line-width": 5 }} />
        </Source>
        <Source id="waste" type="geojson" data={data?.waste ?? EMPTY}>
          <Layer id="waste-fill" type="fill" paint={{ "fill-color": mapPalette.waste, "fill-opacity": 0.25 }} layout={vis(visible.waste)} />
          <Layer id="waste-casing" type="line" paint={{ "line-color": mapPalette.casing, "line-width": 4, "line-opacity": 0.8 }} layout={vis(visible.waste)} />
          <Layer id="waste-line" type="line" paint={{ "line-color": mapPalette.waste, "line-width": 2 }} layout={vis(visible.waste)} />
          <Layer id="waste-selected" type="line" filter={selected("waste") as never} paint={{ "line-color": mapPalette.selected, "line-width": 3 }} />
        </Source>

        {visible.waste &&
          wasteLabels.map((w) => (
            <Marker key={w.key} longitude={w.lon} latitude={w.lat} anchor="bottom">
              <Box sx={{ px: 1, py: 0.25, mb: 0.5, borderRadius: 1, bgcolor: "error.main", color: "common.white", fontSize: 10, fontWeight: 700, boxShadow: 1, pointerEvents: "none" }}>
                {t("unvalidated")}
              </Box>
            </Marker>
          ))}
      </Map>

      {/* layer switches + legend */}
      <Card sx={{ position: "absolute", top: 12, left: 12, p: 3, maxWidth: 260, maxHeight: "calc(100% - 24px)", overflow: "auto", boxShadow: 3 }}>
        <Typography variant="overline" color="text.secondary">
          {t("layers")}
        </Typography>
        <Box sx={{ display: "flex", flexDirection: "column" }}>
          {LAYER_KEYS.map((k) => (
            <FormControlLabel
              key={k}
              sx={{ mr: 0 }}
              control={<Checkbox size="small" checked={visible[k]} onChange={(e) => setVisible((s) => ({ ...s, [k]: e.target.checked }))} />}
              label={<Typography variant="body2">{t(`layer.${k}`)}</Typography>}
            />
          ))}
        </Box>
        <Typography variant="overline" color="text.secondary" sx={{ display: "block", mt: 2 }}>
          {t("legendTitle")}
        </Typography>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {legend.map((l) => (
            <Box key={l.label} sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Swatch color={l.color} line={l.line} />
              <Typography variant="caption">{l.label}</Typography>
            </Box>
          ))}
        </Box>
      </Card>

      {selection && (
        <Card sx={{ position: "absolute", top: 12, right: 12, width: 300, maxWidth: "calc(100% - 24px)", p: 3, boxShadow: 3 }}>
          <Box sx={{ display: "flex", justifyContent: "flex-end", mb: -2 }}>
            <IconButton size="small" aria-label={tc("common.close")} onClick={() => setSelection(null)}>
              <Close fontSize="small" />
            </IconButton>
          </Box>
          <ResultAttributes selection={selection} />
        </Card>
      )}

      {loadError && (
        <Alert severity="error" sx={{ position: "absolute", bottom: 40, left: 12, right: 12 }}>
          {tc("common.mapLoadError", { detail: loadError })}
        </Alert>
      )}
    </Box>
  );
}
