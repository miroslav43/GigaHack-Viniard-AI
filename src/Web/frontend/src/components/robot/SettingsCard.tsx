"use client";

// The robot's settings, edited as a draft and applied with "Salvează" (kept in this browser): the boards' addresses,
// speeds, the picture's orientation, the record mode's calibration and what each wheel motor does in each move.
// Numbers are typed freely and checked on save (a field refusing "3" on the way to "3000" would be unusable).
import { useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import SaveOutlined from "@mui/icons-material/SaveOutlined";
import { useTranslations } from "next-intl";
import { LIMITS, parseBoardUrl, type Device, type WheelDir } from "@/lib/robot/commands";
import { DRIVE_MOVES, validMove, type DriveMove, type WheelMoves } from "@/lib/robot/wheels";
import type { RobotSettings, SettingsPatch } from "./useRobotSettings";

const DEVICES: Device[] = ["cam", "motors", "drive"];
type NumKey = "stepsPerRev" | "speedPps" | "liftSpeedPps" | "wheelSpeed" | "recordStepCm" | "recordStepMs";
const RANGES: Record<NumKey, readonly [number, number]> = {
  stepsPerRev: [1, 100000],
  speedPps: LIMITS.speedPps,
  liftSpeedPps: LIMITS.speedPps,
  wheelSpeed: LIMITS.wheelSpeed,
  recordStepCm: [5, 500],
  recordStepMs: LIMITS.driveMs,
};
const NUM_KEYS = Object.keys(RANGES) as NumKey[];
type FlagKey = "flipV" | "flipH" | "panInvert" | "liftInvert";

interface Draft {
  urls: Record<Device, string>;
  nums: Record<NumKey, string>;
  flags: Record<FlagKey, boolean>;
  wheelMoves: WheelMoves;
}

const draftOf = (s: RobotSettings): Draft => ({
  urls: { ...s.urls },
  nums: Object.fromEntries(NUM_KEYS.map((k) => [k, String(s[k])])) as Record<NumKey, string>,
  flags: { flipV: s.flipV, flipH: s.flipH, panInvert: s.panInvert, liftInvert: s.liftInvert },
  wheelMoves: s.wheelMoves,
});

const numberIn = (raw: string, [lo, hi]: readonly [number, number]) => {
  const v = Number(raw);
  return raw.trim() !== "" && Number.isInteger(v) && v >= lo && v <= hi ? v : null;
};

export function SettingsCard({ settings, onChange }: { settings: RobotSettings; onChange: (patch: SettingsPatch) => void }) {
  const t = useTranslations("robot.settings");
  const [draft, setDraft] = useState<Draft>(() => draftOf(settings));
  const [saved, setSaved] = useState(false);
  // the saved settings changed under us (read from storage after mount): start the draft over from them
  const [basis, setBasis] = useState(settings);
  if (basis !== settings) {
    setBasis(settings);
    setDraft(draftOf(settings));
  }

  const edit = (next: Draft) => {
    setDraft(next);
    setSaved(false);
  };
  const badNum = (k: NumKey) => numberIn(draft.nums[k], RANGES[k]) === null;
  const badUrl = (d: Device) => draft.urls[d].trim() !== "" && parseBoardUrl(draft.urls[d]) === null;
  const badMove = (m: DriveMove) => !validMove(draft.wheelMoves[m]);
  const invalid = NUM_KEYS.some(badNum) || DEVICES.some(badUrl) || DRIVE_MOVES.some(badMove);
  const dirty = JSON.stringify(draft) !== JSON.stringify(draftOf(settings));

  const save = () => {
    if (invalid) return;
    onChange({
      urls: draft.urls,
      ...(Object.fromEntries(NUM_KEYS.map((k) => [k, numberIn(draft.nums[k], RANGES[k])!])) as Record<NumKey, number>),
      ...draft.flags,
      wheelMoves: draft.wheelMoves,
    });
    setSaved(true);
  };

  const num = (k: NumKey) => (
    <TextField
      size="small"
      label={t(k)}
      value={draft.nums[k]}
      error={badNum(k)}
      helperText={badNum(k) ? t("range", { min: RANGES[k][0], max: RANGES[k][1] }) : undefined}
      onChange={(e) => edit({ ...draft, nums: { ...draft.nums, [k]: e.target.value } })}
      slotProps={{ htmlInput: { inputMode: "numeric", "data-testid": `robot-set-${k}` } }}
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
        {DEVICES.map((d) => (
          <TextField
            key={d}
            size="small"
            label={t(`url.${d}`)}
            placeholder="192.168.1.50"
            value={draft.urls[d]}
            error={badUrl(d)}
            helperText={badUrl(d) ? t("badAddress") : undefined}
            onChange={(e) => edit({ ...draft, urls: { ...draft.urls, [d]: e.target.value } })}
            slotProps={{ htmlInput: { "data-testid": `robot-url-${d}` } }}
          />
        ))}

        <Typography variant="subtitle2">{t("camera")}</Typography>
        <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
          {num("speedPps")}
          {num("stepsPerRev")}
          {num("liftSpeedPps")}
        </Box>
        <Box sx={{ display: "flex", columnGap: 2, flexWrap: "wrap" }}>
          {(["flipV", "flipH", "panInvert", "liftInvert"] as const).map((k) => (
            <FormControlLabel
              key={k}
              control={<Checkbox size="small" checked={draft.flags[k]} onChange={(e) => edit({ ...draft, flags: { ...draft.flags, [k]: e.target.checked } })} />}
              label={t(k)}
            />
          ))}
        </Box>

        <Typography variant="subtitle2">{t("wheels")}</Typography>
        {num("wheelSpeed")}
        <Typography variant="caption" color="text.secondary">
          {t("wheelsHint")}
        </Typography>
        {/* what each motor does in each move: rows = moves, columns = motors 1–4 */}
        <Box sx={{ display: "grid", gridTemplateColumns: "auto repeat(4, 1fr)", alignItems: "center", gap: 1 }} data-testid="robot-wheel-moves">
          <span />
          {[1, 2, 3, 4].map((n) => (
            <Typography key={n} variant="caption" sx={{ textAlign: "center", fontWeight: 600 }}>
              {t("wheelMotor", { n })}
            </Typography>
          ))}
          {DRIVE_MOVES.map((m) => (
            <Box key={m} sx={{ display: "contents" }}>
              <Typography variant="body2" color={badMove(m) ? "error" : undefined}>
                {t(`move.${m}`)}
              </Typography>
              {draft.wheelMoves[m].map((d, i) => (
                <TextField
                  key={i}
                  select
                  size="small"
                  value={String(d)}
                  onChange={(e) => {
                    const dirs = draft.wheelMoves[m].map((v, k) => (k === i ? (Number(e.target.value) as WheelDir) : v));
                    edit({ ...draft, wheelMoves: { ...draft.wheelMoves, [m]: dirs } });
                  }}
                  slotProps={{ htmlInput: { "aria-label": `${t(`move.${m}`)} · ${t("wheelMotor", { n: i + 1 })}` } }}
                >
                  <MenuItem value="1">{t("dirForward")}</MenuItem>
                  <MenuItem value="-1">{t("dirBack")}</MenuItem>
                  <MenuItem value="0">{t("dirStill")}</MenuItem>
                </TextField>
              ))}
            </Box>
          ))}
        </Box>
        {DRIVE_MOVES.some(badMove) && (
          <Typography variant="caption" color="error">
            {t("moveNeedsWheel")}
          </Typography>
        )}

        <Typography variant="subtitle2">{t("record")}</Typography>
        <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
          {num("recordStepCm")}
          {num("recordStepMs")}
        </Box>
        <Typography variant="caption" color="text.secondary">
          {t("recordHint", { cm: draft.nums.recordStepCm })}
        </Typography>

        <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap", pt: 1 }}>
          <Button variant="contained" startIcon={<SaveOutlined />} onClick={save} disabled={invalid || !dirty} data-testid="robot-settings-save">
            {t("save")}
          </Button>
          <Button onClick={() => edit(draftOf(settings))} disabled={!dirty}>
            {t("discard")}
          </Button>
        </Box>
        {dirty && !invalid && (
          <Alert severity="warning" icon={false}>
            {t("unsaved")}
          </Alert>
        )}
        {saved && !dirty && (
          <Alert severity="success" data-testid="robot-settings-saved">
            {t("savedNote")}
          </Alert>
        )}
      </Box>
    </Paper>
  );
}
