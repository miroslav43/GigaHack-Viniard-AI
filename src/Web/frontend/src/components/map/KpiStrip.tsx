"use client";

import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import type { SurveySummary } from "@/lib/types";
import { CountUp } from "@/components/common/CountUp";

function Kpi({ label, value }: { label: string; value: ReactNode }) {
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
      {tt.farm_count != null && <Kpi value={<CountUp value={tt.farm_count} />} label={t("map.kpiFarms")} />}
      <Kpi value={<CountUp value={tt.block_count} />} label={t("map.kpiBlocks")} />
      <Kpi value={<CountUp value={tt.row_count} />} label={t("map.kpiRows")} />
      <Kpi value={<CountUp value={tt.row_length_m} format="length" />} label={t("map.kpiRowLength")} />
      <Kpi value={<CountUp value={tt.canopy_area_m2} format="ha" />} label={t("map.kpiCanopy")} />
      <Kpi value={<CountUp value={tt.interrow_area_m2} format="ha" />} label={t("map.kpiInterrow")} />
      <Kpi
        value={
          <>
            <CountUp value={summary.route.length_m} format="length" /> · <CountUp value={summary.route.duration_min} format="min" />
          </>
        }
        label={t("map.kpiRoute")} />
      {summary.survey.mock && <Chip size="small" color="info" variant="outlined" label={t("common.mockChip")} sx={{ ml: "auto" }} />}
    </Box>
  );
}
