"use client";

import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Checkbox from "@mui/material/Checkbox";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import FormControlLabel from "@mui/material/FormControlLabel";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import ChevronLeft from "@mui/icons-material/ChevronLeft";
import ZoomOutMap from "@mui/icons-material/ZoomOutMap";
import SearchOutlined from "@mui/icons-material/SearchOutlined";
import { mapPalette } from "@/theme/mapPalette";
import { useTranslations } from "next-intl";
import { useFormat } from "@/lib/useFormat";
import type { RowRecord } from "@/lib/types";

export type LayerKey = "ortho" | "geofence" | "blocks" | "canopies" | "rows" | "interrows" | "route" | "reference";

export const DEFAULT_VISIBILITY: Record<LayerKey, boolean> = {
  ortho: true,
  geofence: true,
  blocks: true,
  canopies: true,
  rows: true,
  interrows: true,
  route: true,
  reference: false,
};

const LAYERS: { key: LayerKey; label: string; minZoom?: number }[] = [
  { key: "ortho", label: "layerOrtho" },
  { key: "geofence", label: "layerGeofence" },
  { key: "blocks", label: "layerBlocks" },
  { key: "canopies", label: "layerCanopies", minZoom: 17.5 },
  { key: "rows", label: "layerRows" },
  { key: "interrows", label: "layerInterrows", minZoom: 16.5 },
  { key: "route", label: "layerRoute" },
  { key: "reference", label: "layerReference" },
];

type SearchOption = { kind: "row" | "block"; id: string; group: string };

function Swatch({ color, line, dashed }: { color: string; line?: boolean; dashed?: boolean }) {
  return (
    <Box
      sx={{
        width: 18,
        height: line ? 0 : 12,
        borderRadius: line ? 0 : 0.5,
        flexShrink: 0,
        ...(line
          ? { borderTop: `3px ${dashed ? "dotted" : "solid"} ${color}` }
          : { bgcolor: color, opacity: 0.8 }),
      }}
    />
  );
}

function LegendItem({ label, ...swatch }: { label: string; color: string; line?: boolean; dashed?: boolean }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
      <Swatch {...swatch} />
      <Typography variant="caption">{label}</Typography>
    </Box>
  );
}

export function LayerPanel({
  open,
  onToggleOpen,
  visible,
  onChange,
  rows,
  blocks,
  onPickRow,
  onPickBlock,
  onFitAll,
}: {
  open: boolean;
  onToggleOpen: () => void;
  visible: Record<LayerKey, boolean>;
  onChange: (k: LayerKey, v: boolean) => void;
  rows: RowRecord[];
  blocks: string[];
  onPickRow: (id: string) => void;
  onPickBlock: (id: string) => void;
  onFitAll: () => void;
}) {
  const t = useTranslations("map");
  const tc = useTranslations();
  const f = useFormat();
  if (!open) {
    return (
      <Tooltip title={t("layersAndSearch")} placement="right">
        <Paper sx={{ position: "absolute", top: 16, left: 16, zIndex: 2, boxShadow: 3 }}>
          <IconButton onClick={onToggleOpen} aria-label={t("openLayers")}>
            <LayersOutlined />
          </IconButton>
        </Paper>
      </Tooltip>
    );
  }

  const options: SearchOption[] = [
    ...blocks.map((id) => ({ kind: "block" as const, id, group: t("groupBlocks") })),
    ...rows.map((r) => ({ kind: "row" as const, id: r.row_id, group: t("groupRows") })),
  ];

  return (
    <Paper
      sx={{
        position: "absolute",
        top: 16,
        left: 16,
        bottom: 16,
        width: 280,
        zIndex: 2,
        boxShadow: 3,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <Box sx={{ px: 4, pt: 3, pb: 2, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Typography variant="subtitle1">{t("layers")}</Typography>
        <IconButton size="small" onClick={onToggleOpen} aria-label={t("closePanel")}>
          <ChevronLeft />
        </IconButton>
      </Box>
      <Box sx={{ px: 4, pb: 3 }}>
        <Autocomplete
          size="small"
          options={options}
          groupBy={(o) => o.group}
          getOptionLabel={(o) => o.id}
          onChange={(_, o) => o && (o.kind === "row" ? onPickRow(o.id) : onPickBlock(o.id))}
          renderInput={(p) => (
            <TextField
              {...p}
              placeholder={t("searchPlaceholder")}
              slotProps={{
                ...p.slotProps,
                input: { ...p.slotProps.input, startAdornment: <SearchOutlined fontSize="small" sx={{ color: "text.secondary", mr: 1 }} /> },
              }}
            />
          )}
        />
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mt: 2 }}>
          {blocks.map((b) => (
            <Chip key={b} size="small" label={b} onClick={() => onPickBlock(b)} />
          ))}
          <Button size="small" startIcon={<ZoomOutMap fontSize="small" />} onClick={onFitAll} sx={{ ml: "auto" }}>
            {t("wholeArea")}
          </Button>
        </Box>
      </Box>
      <Divider />
      <Box sx={{ overflow: "auto", flex: 1, px: 4, py: 2 }}>
        {LAYERS.map((l) => (
          <FormControlLabel
            key={l.key}
            sx={{ display: "flex", mr: 0, my: -0.5 }}
            control={<Checkbox size="small" checked={visible[l.key]} onChange={(e) => onChange(l.key, e.target.checked)} />}
            label={
              <Box>
                <Typography variant="body2">{t(l.label)}</Typography>
                {l.minZoom && (
                  <Typography variant="caption" color="text.secondary">
                    {t("fromZoom", { zoom: f.num(l.minZoom, 1) })}
                  </Typography>
                )}
              </Box>
            }
          />
        ))}
        <Divider sx={{ my: 3 }} />
        <Typography variant="overline" color="text.secondary">
          {t("legend")}
        </Typography>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, mt: 1 }}>
          <LegendItem label={t("legendCanopy")} color={mapPalette.canopyFill} />
          <LegendItem label={t("legendRow", { structure: tc("structure.regular") })} color={mapPalette.row.regular} line />
          <LegendItem label={t("legendRowDisrupted")} color={mapPalette.row.disrupted} line />
          <LegendItem label={t("legendRow", { structure: tc("structure.unassessable") })} color={mapPalette.row.unassessable} line dashed />
          {(Object.keys(mapPalette.interrow) as (keyof typeof mapPalette.interrow)[]).map((k) => (
            <LegendItem key={k} label={t("legendInterrow", { cover: tc(`cover.${k}`) })} color={mapPalette.interrow[k]} />
          ))}
          <LegendItem label={t("legendRoute")} color={mapPalette.route} line />
          <LegendItem label={t("legendTarget")} color={mapPalette.target} />
          <LegendItem label={t("legendGeofence")} color={mapPalette.geofence} line dashed />
        </Box>
      </Box>
    </Paper>
  );
}
