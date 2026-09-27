"use client";

// The record mode's card: what it will do, start / stop, and where it is.
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import LinearProgress from "@mui/material/LinearProgress";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import FiberManualRecord from "@mui/icons-material/FiberManualRecord";
import StopCircleOutlined from "@mui/icons-material/StopCircleOutlined";
import { useTranslations } from "next-intl";
import type { RecordPhase } from "./useRecorder";
import type { RobotSettings } from "./useRobotSettings";

export function RecordCard({
  settings,
  ready,
  phase,
  onStart,
  onStop,
}: {
  settings: RobotSettings;
  /** camera, camera motors and wheels all set */
  ready: boolean;
  phase: RecordPhase | null;
  onStart: () => void;
  onStop: () => void;
}) {
  const t = useTranslations("robot.record");
  const stations = settings.recordStations;
  return (
    <Paper sx={{ p: 4 }} data-testid="robot-record">
      <Typography variant="h3" sx={{ mb: 1 }}>
        {t("title")}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {t("hint", { stations, cm: settings.recordStepCm, total: (stations - 1) * settings.recordStepCm })}
      </Typography>
      {phase ? (
        <>
          <Alert severity="info" icon={false} sx={{ mb: 1 }} data-testid="robot-record-phase">
            {phase.step === "photo"
              ? t("phasePhoto", { station: phase.station, of: phase.of, deg: phase.deg })
              : t("phaseDrive", { station: phase.station, of: phase.of, cm: settings.recordStepCm })}
          </Alert>
          <LinearProgress variant="determinate" value={((phase.station - (phase.step === "drive" ? 0 : 1)) / phase.of) * 100} sx={{ mb: 2 }} />
          <Button variant="contained" color="error" startIcon={<StopCircleOutlined />} onClick={onStop} data-testid="robot-record-stop">
            {t("stop")}
          </Button>
        </>
      ) : (
        <Box sx={{ display: "grid", gap: 1 }}>
          <Button variant="contained" color="error" startIcon={<FiberManualRecord />} onClick={onStart} disabled={!ready} data-testid="robot-record-start">
            {t("start")}
          </Button>
          {!ready && (
            <Typography variant="caption" color="text.secondary">
              {t("needsBoards")}
            </Typography>
          )}
        </Box>
      )}
    </Paper>
  );
}
