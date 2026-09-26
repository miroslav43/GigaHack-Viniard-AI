"use client";

import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Close from "@mui/icons-material/Close";
import { mapPalette, type InterrowCover, type RowStructure } from "@/theme/mapPalette";
import { useFormat } from "@/lib/useFormat";
import type { SurveySummary } from "@/lib/types";

export type Selection = {
  layer: "rows" | "canopies" | "interrows" | "waste" | "targets" | "blocks";
  props: Record<string, unknown>;
};

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", gap: 3, py: 1.25 }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" component="div" sx={{ fontWeight: 600, textAlign: "right" }}>
        {children}
      </Typography>
    </Box>
  );
}

function ColorChip({ label, color }: { label: string; color: string }) {
  return <Chip size="small" label={label} sx={{ bgcolor: color, color: "grey.900", fontWeight: 600 }} />;
}

export function AttributePanel({
  selection,
  summary,
  onClose,
  onZoomRow,
}: {
  selection: Selection;
  summary: SurveySummary;
  onClose: () => void;
  onZoomRow: (rowId: string) => void;
}) {
  const t = useTranslations("map.fields");
  const tc = useTranslations();
  const f = useFormat();
  const p = selection.props;
  const s = (k: string) => (p[k] == null ? "—" : String(p[k]));
  const n = (k: string) => Number(p[k] ?? 0);
  const rowChip = p.row_id ? <Chip size="small" label={s("row_id")} onClick={() => onZoomRow(String(p.row_id))} clickable /> : "—";

  let title = s("vineyard_id");
  let body: ReactNode = null;

  switch (selection.layer) {
    case "rows": {
      const st = s("row_structure") as RowStructure;
      title = s("row_id");
      const tiles = (p.tile_structures ?? {}) as Record<string, RowStructure>;
      body = (
        <>
          <Field label="row_id">{s("row_id")}</Field>
          <Field label="vineyard_id">{s("vineyard_id")}</Field>
          <Field label={t("structure")}>
            <ColorChip label={tc(`structure.${st}`)} color={mapPalette.row[st] ?? mapPalette.row.regular} />
          </Field>
          <Field label={t("length")}>{f.m(n("length_m"), 2)}</Field>
          <Field label={t("plants")}>{f.int(n("plant_count"))}</Field>
          <Field label={t("maxGap")}>{f.m(n("max_gap_m"), 1)}</Field>
          <Divider sx={{ my: 2 }} />
          <Typography variant="caption" color="text.secondary">
            {t("perTile")}
          </Typography>
          {Object.entries(tiles).map(([tile, v]) => (
            <Field key={tile} label={tile.replace("siret3_", "")}>
              {tc(`structure.${v}`)}
            </Field>
          ))}
        </>
      );
      break;
    }
    case "canopies":
      title = t("canopy");
      body = (
        <>
          <Field label="ID">{s("canopy_id").replace("siret3_", "")}</Field>
          <Field label="vineyard_id">{s("vineyard_id")}</Field>
          <Field label={t("row")}>{rowChip}</Field>
          <Field label={t("area")}>{f.m2(n("area_m2"), 2)}</Field>
        </>
      );
      break;
    case "interrows": {
      const cv = s("interrow_cover") as InterrowCover;
      title = t("interrow");
      body = (
        <>
          <Field label="ID">{s("interrow_id").replace("siret3_", "")}</Field>
          <Field label="vineyard_id">{s("vineyard_id")}</Field>
          <Field label={t("cover")}>
            <ColorChip label={tc(`cover.${cv}`)} color={mapPalette.interrow[cv] ?? mapPalette.interrow.unassessable} />
          </Field>
          <Field label={t("area")}>{f.area(n("area_m2"))}</Field>
          {p.interrow_total_m2 != null && <Field label={t("interrowTotal")}>{f.area(n("interrow_total_m2"))}</Field>}
        </>
      );
      break;
    }
    case "waste":
      title = t("waste", { id: s("waste_id") });
      body = (
        <>
          <Field label="waste_id">{s("waste_id")}</Field>
          <Field label="vineyard_id">{s("vineyard_id")}</Field>
          {p.category != null && <Field label={t("category")}>{s("category")}</Field>}
          {p.confidence != null && <Field label={t("confidence")}>{f.pct(n("confidence"))}</Field>}
        </>
      );
      break;
    case "targets": {
      const kind = s("kind"), skip = s("skip_reason");
      title = t("target", { id: s("target_id") });
      body = (
        <>
          <Field label={t("routeOrder")}>
            {p.route_order != null ? s("route_order") : <Chip size="small" variant="outlined" label={t("offRoute")} />}
          </Field>
          {p.route_order == null && p.skip_reason != null && (
            <Field label={t("skipReason")}>{tc.has(`skipReason.${skip}`) ? tc(`skipReason.${skip}`) : skip}</Field>
          )}
          <Field label={t("type")}>{tc.has(`targetType.${s("type")}`) ? tc(`targetType.${s("type")}`) : s("type")}</Field>
          {p.kind != null && <Field label={t("kind")}>{tc.has(`targetKind.${kind}`) ? tc(`targetKind.${kind}`) : kind}</Field>}
          <Field label="vineyard_id">{s("vineyard_id")}</Field>
          <Field label="row_id">{rowChip}</Field>
          {p.gap_length_m != null && <Field label={t("gapLength")}>{f.m(n("gap_length_m"), 1)}</Field>}
          {p.note != null && (
            <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 1 }}>
              {s("note")}
            </Typography>
          )}
        </>
      );
      break;
    }
    case "blocks": {
      const b = summary.blocks.find((x) => x.vineyard_id === p.vineyard_id);
      title = t("block", { id: s("vineyard_id") });
      body = b && (
        <>
          <Field label={t("rows")}>{f.int(b.row_count)}</Field>
          <Field label={t("rowLength")}>{f.m(b.row_length_m)}</Field>
          <Field label={t("canopyCount")}>{f.int(b.canopy_count)}</Field>
          <Field label={t("canopyArea")}>{f.area(b.canopy_area_m2)}</Field>
          <Field label={t("interrowArea")}>{f.area(b.interrow_area_m2)}</Field>
          <Field label={t("disrupted")}>{f.int(b.disrupted_rows)}</Field>
        </>
      );
      break;
    }
  }

  return (
    <Paper
      sx={{
        position: "absolute",
        top: 16,
        right: 16,
        width: { xs: "calc(100% - 32px)", sm: 320 },
        maxHeight: "calc(100% - 32px)",
        overflow: "auto",
        zIndex: 3,
        boxShadow: 3,
      }}
    >
      <Box sx={{ px: 4, pt: 3, pb: 2, display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <Box>
          <Typography variant="overline" color="text.secondary">
            {tc(`layer.${selection.layer}`)}
          </Typography>
          <Typography variant="h3">{title}</Typography>
        </Box>
        <IconButton size="small" onClick={onClose} aria-label={tc("common.close")}>
          <Close />
        </IconButton>
      </Box>
      <Divider />
      <Box sx={{ px: 4, py: 2 }}>{body}</Box>
      <Divider />
      <Typography variant="caption" color="text.secondary" component="p" sx={{ px: 4, py: 2 }}>
        {tc("common.measuredNote")}
      </Typography>
    </Paper>
  );
}
