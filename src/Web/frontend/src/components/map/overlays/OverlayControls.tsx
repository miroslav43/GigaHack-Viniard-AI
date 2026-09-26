"use client";

// Layer-panel pieces of the optional survey overlays (tile footprints, vegetation masks): the two switches (off by
// default, disabled with a hint when the survey does not ship the files) and their legend entries.
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { mapPalette } from "@/theme/mapPalette";
import { color } from "@/theme/tokens";
import { useFormat } from "@/lib/useFormat";
import { ZOOM } from "../mapStyle";
import type { OverlayKey, SurveyOverlays } from "./useSurveyOverlays";

const TOGGLES: { key: OverlayKey; label: string; minZoom?: number }[] = [
  { key: "tiles", label: "layerTiles" },
  { key: "vegMask", label: "layerVegMask", minZoom: ZOOM.orthoDetail },
];

function Hint({ children, error = false }: { children: string; error?: boolean }) {
  return (
    <Typography variant="caption" color={error ? "error" : "text.secondary"} component="div">
      {children}
    </Typography>
  );
}

export function OverlayToggles({ overlays }: { overlays: SurveyOverlays }) {
  const t = useTranslations("map");
  const f = useFormat();
  return (
    <>
      {TOGGLES.map((l) => {
        const available = overlays.available[l.key];
        const failed = overlays.failed[l.key];
        return (
          <FormControlLabel
            key={l.key}
            sx={{ display: "flex", mr: 0, my: -0.5 }}
            disabled={!available}
            control={
              <Checkbox
                size="small"
                checked={available && overlays.visible[l.key]}
                onChange={(e) => overlays.setVisible(l.key, e.target.checked)}
                slotProps={{ input: { "aria-describedby": `overlay-hint-${l.key}` } }}
              />
            }
            label={
              <Box id={`overlay-hint-${l.key}`}>
                <Typography variant="body2">{t(l.label)}</Typography>
                {!available && <Hint>{t("overlayUnavailable")}</Hint>}
                {available && l.minZoom && <Hint>{t("fromZoom", { zoom: f.num(l.minZoom, 1) })}</Hint>}
                {failed && <Hint error>{t("overlayLoadError", { detail: failed })}</Hint>}
              </Box>
            }
          />
        );
      })}
    </>
  );
}

function Square({ fill, stroke, opacity }: { fill: string; stroke?: string; opacity: number }) {
  return (
    <Box sx={{ width: 18, height: 12, flexShrink: 0, borderRadius: 0.5, border: stroke ? `2px solid ${stroke}` : "none", position: "relative", overflow: "hidden" }}>
      <Box sx={{ position: "absolute", inset: 0, bgcolor: fill, opacity }} />
    </Box>
  );
}

function Entry({ label, fill, outline = true, opacity = 0.6 }: { label: string; fill: string; outline?: boolean; opacity?: number }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
      <Square fill={fill} stroke={outline ? fill : undefined} opacity={opacity} />
      <Typography variant="caption">{label}</Typography>
    </Box>
  );
}

/** Legend entries of the overlays that are switched on (nothing otherwise). */
export function OverlayLegend({ overlays }: { overlays: SurveyOverlays }) {
  const t = useTranslations("map");
  const { visible, available } = overlays;
  return (
    <>
      {available.tiles && visible.tiles && (
        <>
          <Entry label={t("legendTileVineyard")} fill={mapPalette.tile.vineyard} />
          <Entry label={t("legendTileNoVineyard")} fill={mapPalette.tile.no_vineyard} />
          <Entry label={t("legendTileToComplete")} fill={mapPalette.tile.toComplete} />
        </>
      )}
      {available.vegMask && visible.vegMask && (
        <Entry label={t("legendVegMask")} fill={mapPalette.vegMask} outline={false} opacity={color.map.vegMaskAlpha} />
      )}
    </>
  );
}
