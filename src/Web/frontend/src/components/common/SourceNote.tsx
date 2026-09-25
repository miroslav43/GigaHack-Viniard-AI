import Alert from "@mui/material/Alert";
import Typography from "@mui/material/Typography";
import { getLocale, getTranslations } from "next-intl/server";
import type { SurveySummary } from "@/lib/types";
import { makeFormat } from "@/lib/format";

/** "Sireț3 · flight 20.05.2025 · measured in EPSG:32635" — the source every figure refers to. */
export async function sourceLine(s: SurveySummary) {
  const t = await getTranslations("common");
  const f = makeFormat(await getLocale());
  return t("source", { date: f.date(s.survey.captured_at), crs: s.survey.crs });
}

export async function MockBanner({ summary }: { summary: SurveySummary }) {
  if (!summary.survey.mock) return null;
  const t = await getTranslations("common");
  return (
    <Alert severity="info" variant="outlined" sx={{ mb: 5, bgcolor: "background.paper" }}>
      <Typography variant="body2">
        <strong>{t("mockTitle")}</strong>{" "}
        {t("mockBody", { annotated: summary.survey.tiles_annotated, total: summary.survey.tiles_total })}
      </Typography>
    </Alert>
  );
}
