"use client";

// Attribute-panel body of a survey tile (tiles.geojson feature, src/Web/CLAUDE.md §6.3).
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { mapPalette } from "@/theme/mapPalette";
import { useFormat } from "@/lib/useFormat";
import { ColorChip, Field } from "../AttributePanel";

/** MapLibre hands feature properties back as plain values; null ones may come back missing. */
const numOrNull = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const strOrNull = (v: unknown) => (typeof v === "string" && v !== "" ? v : null);

export const tileTitle = (props: Record<string, unknown>) => String(props.tile ?? "—").replace("siret3_", "");

export function TileAttributes({ props }: { props: Record<string, unknown> }) {
  const t = useTranslations("map.tile");
  const f = useFormat();
  const vineyard = props.status === "vineyard";
  const reviewStatus = strOrNull(props.review_status);
  const note = strOrNull(props.review_note);
  const veg = numOrNull(props.veg_frac), nodata = numOrNull(props.nodata_frac), priority = numOrNull(props.review_priority);
  const count = (k: string) => f.int(numOrNull(props[k]) ?? 0);
  return (
    <>
      <Field label={t("id")}>{String(props.tile ?? "—")}</Field>
      <Field label={t("status")}>
        <ColorChip
          label={t(vineyard ? "vineyard" : "noVineyard")}
          color={vineyard ? mapPalette.tile.vineyard : mapPalette.tile.no_vineyard}
        />
      </Field>
      {reviewStatus && (
        <Field label={t("review")}>
          <ColorChip label={t(`reviewStatus.${reviewStatus}`)} color={mapPalette.tile.toComplete} />
        </Field>
      )}
      {priority != null && <Field label={t("priority")}>{f.int(priority)}</Field>}
      <Field label={t("rows")}>{count("n_rows")}</Field>
      <Field label={t("canopies")}>{count("n_canopies")}</Field>
      <Field label={t("interrows")}>{count("n_interrows")}</Field>
      <Field label={t("waste")}>{count("n_waste")}</Field>
      <Field label={t("vegetation")}>{veg == null ? "—" : f.pct(veg)}</Field>
      <Field label={t("nodata")}>{nodata == null ? "—" : f.pct(nodata)}</Field>
      {note && (
        <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 1 }}>
          {t("note", { note })}
        </Typography>
      )}
    </>
  );
}
