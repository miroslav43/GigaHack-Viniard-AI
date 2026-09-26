"use client";

// Farm panel section of the farm route tool: start the pick, show progress, then length, duration and the GPX.
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import LinearProgress from "@mui/material/LinearProgress";
import Typography from "@mui/material/Typography";
import AltRouteOutlined from "@mui/icons-material/AltRouteOutlined";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import { useTranslations } from "next-intl";
import { useFormat } from "@/lib/useFormat";
import { farmRouteGpx } from "@/lib/farmRoute/gpx";
import { Field } from "../../AttributePanel";
import type { FarmRoute } from "./useFarmRoute";
import type { FarmRouteStop } from "./FarmRouteLayers";

/** straight walks shorter than this (start to the nearest path) are not worth a note (m) */
const OFF_NETWORK_NOTE_M = 25;

function downloadGpx(farmId: string, route: FarmRoute, stops: FarmRouteStop[]) {
  if (route.state.status !== "done") return;
  const gpx = farmRouteGpx(`Solemtrix ${farmId}`, route.state.result.line, stops.map((s) => ({ id: s.id, at: s.at })));
  const url = URL.createObjectURL(new Blob([gpx], { type: "application/gpx+xml" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `traseu_${farmId}.gpx`;
  a.click();
  URL.revokeObjectURL(url);
}

export function FarmRouteSection({
  farmId,
  targetCount,
  route,
  stops,
}: {
  farmId: string;
  targetCount: number;
  route: FarmRoute;
  stops: FarmRouteStop[];
}) {
  const t = useTranslations("map.farmRoute");
  const f = useFormat();
  const { state } = route;
  const mine = state.status !== "idle" && state.farmId === farmId;

  let body;
  if (targetCount === 0) {
    body = <Typography variant="body2" color="text.secondary">{t("noTargets")}</Typography>;
  } else if (!mine) {
    body = (
      <Button variant="outlined" size="small" startIcon={<AltRouteOutlined />} disabled={!route.ready} onClick={() => route.begin(farmId)} data-testid="farm-route-begin">
        {t("compute", { n: targetCount })}
      </Button>
    );
  } else if (state.status === "picking") {
    body = (
      <>
        <Alert severity="info" sx={{ mb: 1 }} data-testid="farm-route-pick">
          {t("pickHint")}
        </Alert>
        <Button size="small" onClick={route.clear}>
          {t("cancel")}
        </Button>
      </>
    );
  } else if (state.status === "computing") {
    body = (
      <>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {t("computing", { n: targetCount })}
        </Typography>
        <LinearProgress />
      </>
    );
  } else if (state.status === "error") {
    body = (
      <>
        <Alert severity="error" sx={{ mb: 1 }}>
          {t("error", { detail: state.error })}
        </Alert>
        <Button size="small" onClick={() => route.begin(farmId)}>
          {t("newStart")}
        </Button>
      </>
    );
  } else if (state.status === "done") {
    const r = state.result;
    body = (
      <Box data-testid="farm-route-result">
        <Field label={t("stops")}>{f.int(r.order.length)}</Field>
        <Field label={t("length")}>{f.length(r.lengthM)}</Field>
        <Field label={t("duration")}>{f.min(r.durationMin)}</Field>
        {r.offNetworkM > OFF_NETWORK_NOTE_M && (
          <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 0.5 }}>
            {t("offNetwork", { length: f.length(r.offNetworkM) })}
          </Typography>
        )}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, pt: 1.5 }}>
          <Button size="small" variant="outlined" onClick={() => route.begin(farmId)}>
            {t("newStart")}
          </Button>
          <Button size="small" variant="outlined" startIcon={<FileDownloadOutlined />} onClick={() => downloadGpx(farmId, route, stops)}>
            GPX
          </Button>
          <Button size="small" onClick={route.clear}>
            {t("clear")}
          </Button>
        </Box>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 1 }}>
          {t("note")}
        </Typography>
      </Box>
    );
  }

  return (
    <Box data-testid="farm-route">
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" sx={{ mb: 1 }}>
        {t("title")}
      </Typography>
      {body}
    </Box>
  );
}

/** While the start is being picked on a small screen, the farm panel steps aside for this banner. */
export function FarmRoutePickBanner({ onCancel }: { onCancel: () => void }) {
  const t = useTranslations("map.farmRoute");
  return (
    <Alert
      severity="info"
      data-testid="farm-route-pick"
      action={
        <Button color="inherit" size="small" onClick={onCancel}>
          {t("cancel")}
        </Button>
      }
      sx={{ position: "absolute", top: 16, left: 16, right: 16, zIndex: 3, boxShadow: 3 }}
    >
      {t("pickHint")}
    </Alert>
  );
}
