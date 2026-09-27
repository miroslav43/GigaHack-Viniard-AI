"use client";

// Attributes of the object clicked on the analysis map (properties of the job's GeoJSON, contract in the spec).
import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import { useFormat } from "@/lib/useFormat";

export interface AnalysisSelection {
  layer: "canopies" | "rows" | "interrows" | "waste";
  props: Record<string, unknown>;
}

const ID_KEY: Record<AnalysisSelection["layer"], string> = { canopies: "canopy_id", rows: "row_id", interrows: "interrow_id", waste: "waste_id" };

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", gap: 3, py: 1, borderBottom: 1, borderColor: "divider" }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" sx={{ fontWeight: 600, textAlign: "right", wordBreak: "break-word" }}>
        {children}
      </Typography>
    </Box>
  );
}

export function ResultAttributes({ selection }: { selection: AnalysisSelection }) {
  const t = useTranslations("analiza.attr");
  const tc = useTranslations();
  const f = useFormat();
  const { layer, props } = selection;
  const text = (k: string) => (props[k] == null || props[k] === "" ? "—" : String(props[k]));
  const num = (k: string) => (typeof props[k] === "number" ? (props[k] as number) : null);
  const area = num("area_m2");
  const enumOf = (group: "structure" | "cover", k: string) => {
    const v = text(k);
    return tc.has(`${group}.${v}`) ? tc(`${group}.${v}`) : v;
  };

  return (
    <Box>
      <Typography variant="h3" component="h2" sx={{ mb: 2, pr: 6 }}>
        {t(`title.${layer}`, { id: text(ID_KEY[layer]) })}
      </Typography>
      {layer === "waste" && (
        <Box sx={{ mb: 2 }}>
          <Chip size="small" color="error" label={tc("analiza.map.unvalidated")} />
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
            {t("wasteNote")}
          </Typography>
        </Box>
      )}
      {layer !== "waste" && <Field label="vineyard_id">{text("vineyard_id")}</Field>}
      {layer === "canopies" && <Field label="row_id">{text("row_id")}</Field>}
      {(layer === "rows" || layer === "interrows") && <Field label="piece_id">{text("piece_id")}</Field>}
      {layer === "rows" && (
        <>
          <Field label={t("structure")}>{enumOf("structure", "row_structure")}</Field>
          {num("length_m") != null && <Field label={t("length")}>{f.m(num("length_m")!, 2)}</Field>}
          {num("max_gap_m") != null && <Field label={t("maxGap")}>{f.m(num("max_gap_m")!, 2)}</Field>}
        </>
      )}
      {layer === "interrows" && <Field label={t("cover")}>{enumOf("cover", "interrow_cover")}</Field>}
      {layer === "waste" && (
        <>
          {num("rank_score") != null && <Field label={t("rankScore")}>{f.num(num("rank_score")!, 2)}</Field>}
          {num("probe_p") != null && <Field label={t("probe")}>{f.num(num("probe_p")!, 2)}</Field>}
          <Field label={t("category")}>{text("category")}</Field>
        </>
      )}
      {area != null && <Field label={t("area")}>{f.area(area)}</Field>}
    </Box>
  );
}
