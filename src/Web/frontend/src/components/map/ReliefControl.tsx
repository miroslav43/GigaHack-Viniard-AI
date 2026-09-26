"use client";

import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import FormControlLabel from "@mui/material/FormControlLabel";
import Paper from "@mui/material/Paper";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import Slider from "@mui/material/Slider";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import ViewInArOutlined from "@mui/icons-material/ViewInArOutlined";
import MapOutlined from "@mui/icons-material/MapOutlined";
import { useFormat } from "@/lib/useFormat";
import { RELIEF, type ReliefSettings, type ReliefSource, type ReliefView } from "./useRelief";

/** "Oblique view / From above (2D)" toggle + the relief settings card; the map applies them (useRelief). */
export function ReliefControl({
  available,
  ready,
  settings,
  onChange,
}: {
  /** a relief was generated for this survey (terrain.json) */
  available: boolean;
  /** available and the map has loaded */
  ready: boolean;
  settings: ReliefSettings;
  onChange: (patch: Partial<ReliefSettings>) => void;
}) {
  const t = useTranslations("map.relief");
  const f = useFormat();
  const oblique = settings.view === "oblique";

  const toggle = (
    <ToggleButtonGroup
      size="small"
      exclusive
      value={settings.view}
      disabled={!ready}
      onChange={(_, v: ReliefView | null) => v && onChange({ view: v })}
      aria-label={t("view")}
      sx={{ bgcolor: "background.paper" }}
    >
      <ToggleButton value="oblique" sx={{ gap: 1, px: 2 }}>
        <ViewInArOutlined fontSize="small" />
        {t("oblique")}
      </ToggleButton>
      <ToggleButton value="top" sx={{ gap: 1, px: 2 }}>
        <MapOutlined fontSize="small" />
        {t("top")}
      </ToggleButton>
    </ToggleButtonGroup>
  );

  return (
    <Box sx={{ position: "absolute", top: 16, right: 16, zIndex: 2, display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
      <Paper sx={{ boxShadow: 3 }}>
        {available ? (
          toggle
        ) : (
          // a disabled button fires no events: the tooltip needs a wrapper
          <Tooltip title={t("unavailable")} placement="left">
            <span>{toggle}</span>
          </Tooltip>
        )}
      </Paper>
      {oblique && (
        <Paper sx={{ p: 3, width: 272, boxShadow: 3 }} role="group" aria-label={t("oblique")}>
          <Typography variant="subtitle2">{t("heightSource")}</Typography>
          <RadioGroup row value={settings.source} onChange={(_, v) => onChange({ source: v as ReliefSource })}>
            <FormControlLabel value="annotations" control={<Radio size="small" />} label={t("sourceAnnotations")} />
            <FormControlLabel value="flat" control={<Radio size="small" />} label={t("sourceFlat")} />
          </RadioGroup>
          <Box sx={{ display: "flex", justifyContent: "space-between", mt: 1 }}>
            <Typography variant="body2" color="text.secondary" id="relief-height-label">
              {t("height")}
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {f.m(settings.source === "flat" ? 0 : settings.heightM, 1)}
            </Typography>
          </Box>
          <Slider
            size="small"
            min={RELIEF.minHeightM}
            max={RELIEF.maxHeightM}
            step={RELIEF.stepM}
            value={settings.heightM}
            disabled={settings.source === "flat"}
            onChange={(_, v) => onChange({ heightM: v as number })}
            getAriaValueText={(v) => f.m(v, 1)}
            aria-labelledby="relief-height-label"
            sx={{ mx: 1, width: "calc(100% - 8px)" }}
          />
          <Typography variant="caption" color="text.secondary" component="p">
            {t("note")}
          </Typography>
          <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1 }}>
            {t("hint")}
          </Typography>
        </Paper>
      )}
    </Box>
  );
}
