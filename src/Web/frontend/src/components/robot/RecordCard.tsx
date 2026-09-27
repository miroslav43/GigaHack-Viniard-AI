"use client";

// The panorama step's card: one Start = forward, three photos (0° / 90° / 180°), camera back to 0°.
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import LinearProgress from "@mui/material/LinearProgress";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import PlayArrowRounded from "@mui/icons-material/PlayArrowRounded";
import StopCircleOutlined from "@mui/icons-material/StopCircleOutlined";
import { useTranslations } from "next-intl";
import { RECORD_ANGLES, type RecordPhase } from "./useRecorder";
import type { RobotSettings } from "./useRobotSettings";

/** progress through a step: driving, the three photos, the camera back */
const progressOf = (p: RecordPhase) => {
  const steps = RECORD_ANGLES.length + 2;
  const at = p.step === "photo" ? 1 + RECORD_ANGLES.indexOf(p.deg as (typeof RECORD_ANGLES)[number]) : p.step === "drive" ? 0 : steps - 1;
  return ((at + 0.5) / steps) * 100;
};

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
  return (
    <Paper sx={{ p: 4 }} data-testid="robot-record">
      <Typography variant="h3" sx={{ mb: 1 }}>
        {t("title")}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {t("hint", { cm: settings.recordStepCm })}
      </Typography>
      {phase ? (
        <>
          <Alert severity="info" icon={false} sx={{ mb: 1 }} data-testid="robot-record-phase">
            {phase.step === "photo"
              ? t("phasePhoto", { station: phase.station, deg: phase.deg })
              : phase.step === "drive"
                ? t("phaseDrive", { station: phase.station, cm: settings.recordStepCm })
                : t("phaseBack", { station: phase.station })}
          </Alert>
          <LinearProgress variant="determinate" value={progressOf(phase)} sx={{ mb: 2 }} />
          <Button variant="contained" color="error" startIcon={<StopCircleOutlined />} onClick={onStop} data-testid="robot-record-stop">
            {t("stop")}
          </Button>
        </>
      ) : (
        <Box sx={{ display: "grid", gap: 1 }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrowRounded />}
            onClick={onStart}
            disabled={!ready}
            data-testid="robot-record-start"
            sx={{ py: 1.5, fontSize: 18 }}
          >
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
