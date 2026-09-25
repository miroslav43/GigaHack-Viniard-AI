"use client";

import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import StraightenOutlined from "@mui/icons-material/StraightenOutlined";
import { useFormat } from "@/lib/useFormat";
import { areaM2, lengthM, type LonLat } from "@/lib/utm";

/** Toggle button + result card; the map owns the points and draws them. */
export function MeasureTool({
  active,
  points,
  onToggle,
  onClear,
}: {
  active: boolean;
  points: LonLat[];
  onToggle: () => void;
  onClear: () => void;
}) {
  const t = useTranslations("map");
  const f = useFormat();
  const length = lengthM(points);
  const area = areaM2(points);
  const hasResult = points.length >= 2;

  return (
    <Box sx={{ position: "absolute", right: 16, bottom: 120, zIndex: 2, display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
      {(active || hasResult) && (
        <Paper sx={{ p: 3, width: 240, boxShadow: 3 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            {t("measure")}
          </Typography>
          {active && points.length < 2 && (
            <Typography variant="caption" color="text.secondary" component="p">
              {t("measureHint")}
            </Typography>
          )}
          {hasResult && (
            <>
              <Box sx={{ display: "flex", justifyContent: "space-between", py: 0.5 }}>
                <Typography variant="body2" color="text.secondary">
                  {t("measureDistance")}
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {f.length(length)}
                </Typography>
              </Box>
              {area > 0 && (
                <Box sx={{ display: "flex", justifyContent: "space-between", py: 0.5 }}>
                  <Typography variant="body2" color="text.secondary">
                    {t("measureArea")}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600, textAlign: "right" }}>
                    {f.m2(area)}
                    <br />
                    {f.ha(area)}
                  </Typography>
                </Box>
              )}
              <Button size="small" onClick={onClear} sx={{ mt: 1 }}>
                {t("measureClear")}
              </Button>
            </>
          )}
        </Paper>
      )}
      <Tooltip title={t("measure")} placement="left">
        <Paper sx={{ boxShadow: 3 }}>
          <IconButton onClick={onToggle} color={active ? "primary" : "default"} aria-pressed={active} aria-label={t("measure")}>
            <StraightenOutlined />
          </IconButton>
        </Paper>
      </Tooltip>
    </Box>
  );
}
