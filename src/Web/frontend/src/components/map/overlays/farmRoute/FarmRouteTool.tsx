"use client";

// The farm route tool on the map: a labelled button (bottom right, above the ruler) that opens a 3-step guide —
// 1 choose the farm (on the map or from the list), 2 click the start, 3 the route with length, duration and GPX.
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import LinearProgress from "@mui/material/LinearProgress";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Stepper from "@mui/material/Stepper";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import AltRouteOutlined from "@mui/icons-material/AltRouteOutlined";
import Close from "@mui/icons-material/Close";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import { useTranslations } from "next-intl";
import { useFormat } from "@/lib/useFormat";
import { farmRouteGpx } from "@/lib/farmRoute/gpx";
import type { FarmSummary } from "@/lib/types";
import { Field } from "../../AttributePanel";
import type { FarmRoute } from "./useFarmRoute";
import type { FarmRouteStop } from "./FarmRouteLayers";

/** straight walks shorter than this are not worth a note (m) */
const NOTE_MIN_M = 5;

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

const stepOf = (status: FarmRoute["state"]["status"]) => (status === "choosingFarm" ? 0 : status === "picking" ? 1 : 2);

function Body({ route, farms, stops, onChooseFarm }: { route: FarmRoute; farms: FarmSummary[]; stops: FarmRouteStop[]; onChooseFarm: (id: string) => void }) {
  const t = useTranslations("map.farmRoute");
  const f = useFormat();
  const { state } = route;
  const withTargets = farms.filter((x) => x.target_count > 0);
  const farm = farms.find((x) => x.farm_id === route.farmId);
  const farmLine = farm && (
    <Typography variant="subtitle2" sx={{ mb: 1 }}>
      {t("farmLine", { id: farm.farm_id, n: farm.target_count })}
    </Typography>
  );
  const otherFarm = (
    <Button size="small" onClick={route.open}>
      {t("otherFarm")}
    </Button>
  );

  switch (state.status) {
    case "choosingFarm":
      return (
        <>
          <Typography variant="body2" sx={{ mb: 2 }}>
            {t("chooseFarmHint")}
          </Typography>
          <TextField
            select
            fullWidth
            size="small"
            label={t("farmSelect")}
            value=""
            onChange={(e) => onChooseFarm(e.target.value)}
            slotProps={{ htmlInput: { "data-testid": "farm-route-select" } }}
          >
            {withTargets.map((x) => (
              <MenuItem key={x.farm_id} value={x.farm_id}>
                {t("farmLine", { id: x.farm_id, n: x.target_count })}
              </MenuItem>
            ))}
          </TextField>
        </>
      );
    case "picking":
      return (
        <>
          {farmLine}
          <Alert severity="info" icon={false} sx={{ mb: 1 }} data-testid="farm-route-pick">
            {t("pickHint")}
          </Alert>
          {otherFarm}
        </>
      );
    case "computing":
      return (
        <>
          {farmLine}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t("computing", { n: farm?.target_count ?? 0 })}
          </Typography>
          <LinearProgress />
        </>
      );
    case "error":
      return (
        <>
          {farmLine}
          <Alert severity="error" sx={{ mb: 1 }}>
            {t("error", { detail: state.error })}
          </Alert>
          <Button size="small" onClick={() => route.begin(state.farmId)}>
            {t("newStart")}
          </Button>
          {otherFarm}
        </>
      );
    case "done": {
      const r = state.result;
      return (
        <Box data-testid="farm-route-result">
          {farmLine}
          <Field label={t("stops")}>{f.int(r.order.length)}</Field>
          <Field label={t("length")}>{f.length(r.lengthM)}</Field>
          <Field label={t("duration")}>{f.min(r.durationMin)}</Field>
          {r.startMovedM > NOTE_MIN_M && (
            <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 0.5 }}>
              {t("startMoved", { length: f.length(r.startMovedM) })}
            </Typography>
          )}
          {r.offNetworkM > NOTE_MIN_M && (
            <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 0.5 }}>
              {t("offNetwork", { length: f.length(r.offNetworkM) })}
            </Typography>
          )}
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, pt: 1.5 }}>
            <Button size="small" variant="outlined" onClick={() => route.begin(state.farmId)}>
              {t("newStart")}
            </Button>
            <Button size="small" variant="outlined" startIcon={<FileDownloadOutlined />} onClick={() => downloadGpx(state.farmId, route, stops)}>
              GPX
            </Button>
            {otherFarm}
          </Box>
          <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 1 }}>
            {t("note")}
          </Typography>
        </Box>
      );
    }
    default:
      return null;
  }
}

export function FarmRouteTool({
  route,
  farms,
  stops,
  onChooseFarm,
}: {
  route: FarmRoute;
  /** summary.json farms (with their target counts) */
  farms: FarmSummary[];
  stops: FarmRouteStop[];
  /** a farm chosen from the list: start step 2 and zoom to it */
  onChooseFarm: (farmId: string) => void;
}) {
  const t = useTranslations("map.farmRoute");
  const active = route.state.status !== "idle";
  // desktop: the card sits above the button (bottom right); phone: at the top, height capped, so the map stays clickable
  return (
    <>
      {active && (
        <Paper
          data-testid="farm-route-card"
          sx={{
            position: "absolute", zIndex: 3, p: 3, boxShadow: 3, overflow: "auto",
            top: { xs: 16, md: "auto" }, left: { xs: 16, md: "auto" }, right: 16, bottom: { xs: "auto", md: 236 },
            width: { md: 320 }, maxHeight: { xs: "45%", md: "calc(100% - 268px)" },
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
            <Typography variant="subtitle1">{t("title")}</Typography>
            <IconButton size="small" onClick={route.clear} aria-label={t("close")}>
              <Close fontSize="small" />
            </IconButton>
          </Box>
          <Stepper activeStep={stepOf(route.state.status)} alternativeLabel sx={{ mb: 2 }}>
            {(["step1", "step2", "step3"] as const).map((k) => (
              <Step key={k}>
                <StepLabel>{t(k)}</StepLabel>
              </Step>
            ))}
          </Stepper>
          <Body route={route} farms={farms} stops={stops} onChooseFarm={onChooseFarm} />
        </Paper>
      )}
      <Paper sx={{ position: "absolute", right: 16, bottom: 176, zIndex: 2, boxShadow: 3 }}>
        <Button
          onClick={active ? route.clear : route.open}
          disabled={!route.ready}
          variant={active ? "contained" : "text"}
          startIcon={<AltRouteOutlined />}
          aria-pressed={active}
          data-testid="farm-route-tool"
          sx={{ px: 2, py: 1, whiteSpace: "nowrap" }}
        >
          {t("tool")}
        </Button>
      </Paper>
    </>
  );
}
