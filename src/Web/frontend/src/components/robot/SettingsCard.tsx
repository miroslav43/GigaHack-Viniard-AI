"use client";

// The boards' addresses and the motion parameters (saved in this browser).
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { LIMITS, parseBoardUrl, type Device } from "@/lib/robot/commands";
import type { RobotSettings, SettingsPatch } from "./useRobotSettings";

const DEVICES: Device[] = ["cam", "motors", "drive"];

export function SettingsCard({ settings, onChange }: { settings: RobotSettings; onChange: (patch: SettingsPatch) => void }) {
  const t = useTranslations("robot.settings");
  const num = (key: "stepsPerRev" | "stepDeg" | "speedPps" | "driveMs", min: number, max: number) => (
    <TextField
      size="small"
      type="number"
      label={t(key)}
      value={settings[key]}
      slotProps={{ htmlInput: { min, max } }}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v) && v >= min && v <= max) onChange({ [key]: v });
      }}
    />
  );
  return (
    <Paper sx={{ p: 4 }}>
      <Typography variant="h3" sx={{ mb: 1 }}>
        {t("title")}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {t("hint")}
      </Typography>
      <Box sx={{ display: "grid", gap: 2 }}>
        {DEVICES.map((d) => {
          const value = settings.urls[d];
          const invalid = value.trim() !== "" && parseBoardUrl(value) === null;
          return (
            <TextField
              key={d}
              size="small"
              label={t(`url.${d}`)}
              placeholder="192.168.1.50"
              value={value}
              error={invalid}
              helperText={invalid ? t("badAddress") : undefined}
              onChange={(e) => onChange({ urls: { [d]: e.target.value } })}
              slotProps={{ htmlInput: { "data-testid": `robot-url-${d}` } }}
            />
          );
        })}
        <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
          {num("stepDeg", 1, 180)}
          {num("stepsPerRev", 1, 100000)}
          {num("speedPps", LIMITS.speedPps[0], LIMITS.speedPps[1])}
          {num("driveMs", LIMITS.driveMs[0], LIMITS.driveMs[1])}
        </Box>
      </Box>
    </Paper>
  );
}
