"use client";

import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import LinearProgress from "@mui/material/LinearProgress";
import Typography from "@mui/material/Typography";
import { progressOf, type JobStatus } from "@/lib/analiza/contract";
import { useFormat } from "@/lib/useFormat";

/** Stage name of the pipeline, translated when known (analiza.stages.*), else as the pipeline names it. */
export function useStageName() {
  const t = useTranslations("analiza.stages");
  return (stage: string) => (t.has(stage) ? t(stage) : stage);
}

export function JobProgress({ uploading, status, elapsedS }: { uploading: string | null; status: JobStatus | null; elapsedS: number }) {
  const t = useTranslations("analiza.progress");
  const f = useFormat();
  const stageName = useStageName();
  const running = status?.state === "running" && status.stage && status.n_stages > 0;

  const label = uploading
    ? t("uploading", { name: uploading })
    : running
      ? t("stage", { index: Math.max(1, Math.min(status.stage_index, status.n_stages)), total: status.n_stages, stage: stageName(status.stage!) })
      : t("queued");

  return (
    <Card variant="outlined" sx={{ p: 5 }} role="status" aria-live="polite">
      <Box sx={{ display: "flex", justifyContent: "space-between", gap: 4, mb: 3, flexWrap: "wrap" }}>
        <Typography variant="body1" sx={{ fontWeight: 600 }}>
          {label}
        </Typography>
        {!uploading && (
          <Typography variant="body2" color="text.secondary">
            {t("elapsed", { value: f.num(elapsedS, 0) })}
          </Typography>
        )}
      </Box>
      <LinearProgress variant={running ? "determinate" : "indeterminate"} value={status ? progressOf(status) * 100 : 0} />
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 3 }}>
        {t("note")}
      </Typography>
    </Card>
  );
}
