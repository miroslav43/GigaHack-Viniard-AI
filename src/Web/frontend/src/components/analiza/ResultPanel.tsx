"use client";

// Figures of one live tile analysis (result.json): counts, areas, row length, time per pipeline stage.
import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";
import { orderedTimings, type JobResult } from "@/lib/analiza/contract";
import { useFormat } from "@/lib/useFormat";
import { useStageName } from "./JobProgress";

function Row({ label, value, strong, code }: { label: string; value: ReactNode; strong?: boolean; code?: boolean }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", gap: 3, py: 1.25, borderBottom: 1, borderColor: "divider" }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography
        variant="body2"
        sx={{ fontWeight: strong ? 700 : 600, textAlign: "right", minWidth: 0, ...(code ? { fontFamily: "monospace", fontSize: 12, wordBreak: "break-all" } : { wordBreak: "break-word" }) }}
      >
        {value}
      </Typography>
    </Box>
  );
}

export function ResultPanel({ result }: { result: JobResult }) {
  const t = useTranslations("analiza.result");
  const f = useFormat();
  const stageName = useStageName();
  const seconds = (v: number) => t("seconds", { value: f.num(v, 1) });
  const timings = orderedTimings(result.timings_s);

  return (
    <Card variant="outlined" sx={{ p: 5, display: "flex", flexDirection: "column", gap: 4 }}>
      <Box>
        <Typography variant="overline" color="text.secondary">
          {t("tile")}
        </Typography>
        <Typography variant="h2" component="p" sx={{ wordBreak: "break-all" }}>
          {result.tile_id}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {t("modelOnly")}
        </Typography>
      </Box>
      <Box>
        <Row label={t("canopies")} value={f.int(result.counts.canopies)} />
        <Row label={t("rows")} value={f.int(result.counts.rows)} />
        <Row label={t("interrows")} value={f.int(result.counts.interrows)} />
        <Row label={t("waste")} value={f.int(result.counts.waste)} />
        <Row label={t("canopyArea")} value={f.area(result.canopy_area_m2)} />
        <Row label={t("interrowArea")} value={f.area(result.interrow_area_m2)} />
        <Row label={t("rowLength")} value={f.m(result.row_length_m, 2)} />
      </Box>
      <Box>
        <Typography variant="overline" color="text.secondary">
          {t("timings")}
        </Typography>
        {timings.map(([stage, s]) => (
          <Row key={stage} label={stageName(stage)} value={seconds(s)} />
        ))}
        <Row label={t("total")} value={seconds(result.total_s)} strong />
      </Box>
      <Box>
        <Row label={t("model")} value={result.model_version || "—"} code />
        <Row label={t("verifyLevel")} value={result.waste_verify_level || "—"} />
      </Box>
    </Card>
  );
}
