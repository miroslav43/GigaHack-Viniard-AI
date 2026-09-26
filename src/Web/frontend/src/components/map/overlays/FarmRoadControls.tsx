"use client";

// Layer-panel pieces of the farm and road layers: three switches (on by default, disabled with a hint when the
// survey does not ship the file) and the legend entries of the ones that are shown.
import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { mapPalette } from "@/theme/mapPalette";
import type { FarmRoadKey, FarmsRoads } from "./useFarmsRoads";

const TOGGLES: { key: FarmRoadKey; label: string; note?: string; file: "farms" | "roads" }[] = [
  { key: "farms", label: "layerFarms", file: "farms" },
  { key: "roads", label: "layerRoads", note: "roadsNote", file: "roads" },
  { key: "internalRoads", label: "layerRoadsInternal", note: "roadsInternalNote", file: "roads" },
];

function Hint({ children, error = false }: { children: string; error?: boolean }) {
  return (
    <Typography variant="caption" color={error ? "error" : "text.secondary"} component="div">
      {children}
    </Typography>
  );
}

export function FarmRoadToggles({ data }: { data: FarmsRoads }) {
  const t = useTranslations("map");
  return (
    <>
      {TOGGLES.map((l) => {
        const available = data.available[l.key];
        const failed = data.failed[l.file];
        return (
          <FormControlLabel
            key={l.key}
            sx={{ display: "flex", mr: 0, my: -0.5 }}
            disabled={!available || failed !== null}
            control={
              <Checkbox
                size="small"
                checked={data.visible[l.key]}
                onChange={(e) => data.setVisible(l.key, e.target.checked)}
                slotProps={{ input: { "aria-describedby": `farm-road-hint-${l.key}` } }}
              />
            }
            label={
              <Box id={`farm-road-hint-${l.key}`}>
                <Typography variant="body2">{t(l.label)}</Typography>
                {!available && <Hint>{t("overlayUnavailable")}</Hint>}
                {available && l.note && <Hint>{t(l.note)}</Hint>}
                {failed && <Hint error>{t("overlayLoadError", { detail: failed })}</Hint>}
              </Box>
            }
          />
        );
      })}
    </>
  );
}

/** A road swatch; `backdrop` = a dark strip under a light line (the internal roads' casing). */
function Line({ color, width, dashed = false, backdrop }: { color: string; width: number; dashed?: boolean; backdrop?: string }) {
  return (
    <Box sx={{ width: 18, flexShrink: 0, py: "1px", bgcolor: backdrop, borderRadius: 0.5 }}>
      <Box sx={{ borderTop: `${width}px ${dashed ? "dashed" : "solid"} ${color}` }} />
    </Box>
  );
}

function Entry({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
      {children}
      <Typography variant="caption">{label}</Typography>
    </Box>
  );
}

/** Legend entries of the farm and road layers that are shown (nothing otherwise). */
export function FarmRoadLegend({ data }: { data: FarmsRoads }) {
  const t = useTranslations("map");
  const { visible } = data;
  const { road, farm } = mapPalette;
  return (
    <>
      {visible.farms && (
        <Entry label={t("legendFarm")}>
          <Box sx={{ width: 18, height: 12, flexShrink: 0, borderRadius: 0.5, border: `2px solid ${farm.line}`, position: "relative", overflow: "hidden" }}>
            <Box sx={{ position: "absolute", inset: 0, bgcolor: farm.fill, opacity: 0.12 }} />
          </Box>
        </Entry>
      )}
      {visible.roads && (
        <>
          <Entry label={t("legendRoadPublic")}>
            <Line color={road.network} width={4} />
          </Entry>
          <Entry label={t("legendRoadField")}>
            <Line color={road.network} width={2} />
          </Entry>
        </>
      )}
      {visible.internalRoads && (
        <Entry label={t("legendRoadInternal")}>
          <Line color={road.internal} width={3} dashed backdrop={road.internalCasing} />
        </Entry>
      )}
    </>
  );
}
