"use client";

// Shortcut in a farm's panel: open the farm route tool on this farm, straight at step 2 (click the start).
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import Typography from "@mui/material/Typography";
import AltRouteOutlined from "@mui/icons-material/AltRouteOutlined";
import { useTranslations } from "next-intl";

export function FarmRouteSection({ targetCount, ready, onStart }: { targetCount: number; ready: boolean; onStart: () => void }) {
  const t = useTranslations("map.farmRoute");
  return (
    <Box data-testid="farm-route">
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" sx={{ mb: 1 }}>
        {t("title")}
      </Typography>
      {targetCount === 0 ? (
        <Typography variant="body2" color="text.secondary">
          {t("noTargets")}
        </Typography>
      ) : (
        <Button variant="contained" size="small" startIcon={<AltRouteOutlined />} disabled={!ready} onClick={onStart} data-testid="farm-route-begin">
          {t("compute", { n: targetCount })}
        </Button>
      )}
    </Box>
  );
}
