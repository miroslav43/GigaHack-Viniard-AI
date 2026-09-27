"use client";

// /analiza: upload one Sireț3 tile, follow the real pipeline running on it, then show its masks and objects.
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Refresh from "@mui/icons-material/Refresh";
import { JobProgress } from "./JobProgress";
import { ResultMap } from "./ResultMap";
import { ResultPanel } from "./ResultPanel";
import { UploadDrop } from "./UploadDrop";
import { useAnalizaJob } from "./useAnalizaJob";

export function AnalizaWorkbench() {
  const t = useTranslations("analiza");
  const { phase, elapsedS, upload, follow, reset } = useAnalizaJob();

  if (phase.kind === "uploading" || phase.kind === "running") {
    return <JobProgress uploading={phase.kind === "uploading" ? phase.name : null} status={phase.kind === "running" ? phase.status : null} elapsedS={elapsedS} />;
  }

  if (phase.kind === "done") {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
          <Button variant="outlined" startIcon={<Refresh />} onClick={reset}>
            {t("upload.another")}
          </Button>
        </Box>
        <Box sx={{ display: "grid", gap: 4, gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1fr) 380px" }, alignItems: "start" }}>
          <ResultMap job={phase.job} result={phase.result} />
          <ResultPanel result={phase.result} />
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {phase.kind === "error" && (
        <Alert
          severity={phase.code === "busy" ? "warning" : "error"}
          action={
            phase.busyJob ? (
              <Button color="inherit" size="small" onClick={() => follow(phase.busyJob!)}>
                {t("errors.follow")}
              </Button>
            ) : undefined
          }
        >
          {t(`errors.${phase.code}`)}
          {phase.message && (
            <Box component="span" sx={{ display: "block", mt: 1, fontSize: 12, opacity: 0.8 }}>
              {phase.message}
            </Box>
          )}
        </Alert>
      )}
      <UploadDrop disabled={false} onFile={(file) => void upload(file)} />
    </Box>
  );
}
