"use client";

import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import type { SurveySummary } from "@/lib/types";
import { useFormat } from "@/lib/useFormat";

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ display: "flex", alignItems: "baseline", gap: 1.5, whiteSpace: "nowrap" }}>
      <Typography sx={{ fontWeight: 700, fontSize: 15 }}>{value}</Typography>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}

export function KpiStrip({ summary }: { summary: SurveySummary }) {
  const t = useTranslations();
  const f = useFormat();
  const tt = summary.totals;
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        px: 5,
        py: 2.5,
        bgcolor: "background.paper",
        borderBottom: 1,
        borderColor: "divider",
        overflowX: "auto",
        flexShrink: 0,
      }}
    >
      {tt.farm_count != null && <Kpi value={f.int(tt.farm_count)} label={t("map.kpiFarms")} />}
      <Kpi value={f.int(tt.block_count)} label={t("map.kpiBlocks")} />
      <Kpi value={f.int(tt.row_count)} label={t("map.kpiRows")} />
      <Kpi value={f.length(tt.row_length_m)} label={t("map.kpiRowLength")} />
      <Kpi value={f.ha(tt.canopy_area_m2)} label={t("map.kpiCanopy")} />
      <Kpi value={f.ha(tt.interrow_area_m2)} label={t("map.kpiInterrow")} />
      <Kpi value={`${f.length(summary.route.length_m)} · ${f.min(summary.route.duration_min)}`} label={t("map.kpiRoute")} />
      {summary.survey.mock && <Chip size="small" color="info" variant="outlined" label={t("common.mockChip")} sx={{ ml: "auto" }} />}
    </Box>
  );
}
