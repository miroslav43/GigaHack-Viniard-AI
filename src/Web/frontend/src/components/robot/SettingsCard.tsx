"use client";

// The boards' addresses and the motion parameters (saved in this browser).
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { LIMITS, parseBoardUrl, type Device } from "@/lib/robot/commands";
import type { RobotSettings, SettingsPatch, WheelSide } from "./useRobotSettings";

const DEVICES: Device[] = ["cam", "motors", "drive"];

export function SettingsCard({ settings, onChange }: { settings: RobotSettings; onChange: (patch: SettingsPatch) => void }) {
  const t = useTranslations("robot.settings");
  const num = (key: "stepsPerRev" | "stepDeg" | "speedPps" | "driveMs" | "wheelSpeed", min: number, max: number) => (
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
          {num("wheelSpeed", LIMITS.wheelSpeed[0], LIMITS.wheelSpeed[1])}
        </Box>
        {/* which wheel motor is on which side, and which are wired the other way round (calibrate with "Înainte") */}
        <Typography variant="subtitle2">{t("wheels")}</Typography>
        <Box sx={{ display: "grid", gridTemplateColumns: "auto 1fr auto", alignItems: "center", gap: 1.5 }}>
          {settings.wheelSide.map((side, i) => (
            <Box key={i} sx={{ display: "contents" }}>
              <Typography variant="body2">{t("wheelMotor", { n: i + 1 })}</Typography>
              <TextField
                select
                size="small"
                value={side}
                onChange={(e) => onChange({ wheelSide: settings.wheelSide.map((s, k) => (k === i ? (e.target.value as WheelSide) : s)) })}
              >
                <MenuItem value="left">{t("sideLeft")}</MenuItem>
                <MenuItem value="right">{t("sideRight")}</MenuItem>
              </TextField>
              <FormControlLabel
                sx={{ mr: 0 }}
                control={
                  <Checkbox
                    size="small"
                    checked={settings.wheelInvert[i]}
                    onChange={(e) => onChange({ wheelInvert: settings.wheelInvert.map((v, k) => (k === i ? e.target.checked : v)) })}
                  />
                }
                label={t("invert")}
              />
            </Box>
          ))}
        </Box>
      </Box>
    </Paper>
  );
}
