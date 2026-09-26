"use client";

// Layer-panel pieces and attribute-panel body of the live cadastre layer: the switch (off by default, external server,
// visible from CADASTRE_MIN_ZOOM), its legend entry, and the parcel returned by a click (plain text only: the
// server's HTML `description` is never rendered).
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import CircularProgress from "@mui/material/CircularProgress";
import FormControlLabel from "@mui/material/FormControlLabel";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import { useTranslations } from "next-intl";
import { mapPalette } from "@/theme/mapPalette";
import { useFormat } from "@/lib/useFormat";
import { CADASTRE_MIN_ZOOM } from "@/lib/cadastre";
import { Field } from "../AttributePanel";
import type { Cadastre, CadastreQuery } from "./useCadastre";

export function CadastreToggle({ cadastre }: { cadastre: Cadastre }) {
  const t = useTranslations("map.cadastre");
  const f = useFormat();
  return (
    <FormControlLabel
      sx={{ display: "flex", mr: 0, my: -0.5 }}
      control={
        <Checkbox
          size="small"
          checked={cadastre.on}
          onChange={(e) => cadastre.setOn(e.target.checked)}
          slotProps={{ input: { "aria-describedby": "cadastre-hint" } }}
        />
      }
      label={
        <Box id="cadastre-hint">
          <Typography variant="body2">{t("layer")}</Typography>
          <Typography variant="caption" color="text.secondary" component="div">
            {t("hint", { zoom: f.num(CADASTRE_MIN_ZOOM, 0) })}
          </Typography>
        </Box>
      }
    />
  );
}

export function CadastreLegend({ cadastre }: { cadastre: Cadastre }) {
  const t = useTranslations("map.cadastre");
  if (!cadastre.on) return null;
  return (
    <>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Box sx={{ width: 18, height: 12, flexShrink: 0, borderRadius: 0.5, border: `1.5px solid ${mapPalette.cadastre.parcel}` }} />
        <Typography variant="caption">{t("legend")}</Typography>
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Box sx={{ width: 18, height: 12, flexShrink: 0, borderRadius: 0.5, border: `2.5px solid ${mapPalette.cadastre.highlight}` }} />
        <Typography variant="caption">{t("legendPicked")}</Typography>
      </Box>
    </>
  );
}

export function CadastreAttributes({ query }: { query: CadastreQuery }) {
  const t = useTranslations("map.cadastre");
  switch (query.status) {
    case "idle":
    case "loading":
      return (
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, py: 2 }} role="status">
          <CircularProgress size={18} />
          <Typography variant="body2">{t("loading")}</Typography>
        </Box>
      );
    case "none":
      return (
        <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
          {t("none")}
        </Typography>
      );
    case "error":
      return (
        <Alert severity="warning" sx={{ my: 2 }}>
          {t(query.detail === "timeout" ? "timeout" : "error")}
        </Alert>
      );
    case "found": {
      const p = query.parcel;
      return (
        <Box data-testid="cadastre-parcel">
          <Field label={t("number")}>{p.codcadastral ?? "—"}</Field>
          {p.cod_parcel && <Field label={t("parcel")}>{p.cod_parcel}</Field>}
          <Field label={t("landuse")}>{p.landuse ?? "—"}</Field>
          <Field label={t("area")}>{p.aria ?? "—"}</Field>
          <Field label={t("property")}>{p.typeproperty ?? "—"}</Field>
        </Box>
      );
    }
  }
}
