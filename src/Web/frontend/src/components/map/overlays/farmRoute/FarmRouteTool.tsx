"use client";

// The route tools on the map, two labelled buttons (bottom right, above the ruler):
// - "Traseu fermă": 1 choose the farm (on the map or from the list), 2 click the start, 3 the route;
// - "Toate fermele": 1 the start (a click, or the official START), 2 one route through every target of every farm.
// Both end with length, duration, GPX and how much shorter the route is than sweeping every inter-row.
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
import HubOutlined from "@mui/icons-material/HubOutlined";
import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import { useFormat } from "@/lib/useFormat";
import { farmRouteGpx } from "@/lib/farmRoute/gpx";
import type { FarmRouteResult } from "@/lib/farmRoute/plan";
import type { FarmSummary } from "@/lib/types";
import type { LonLat } from "@/lib/utm";
import { Field } from "../../AttributePanel";
import type { FarmRoute } from "./useFarmRoute";
import type { FarmRouteStop } from "./FarmRouteLayers";

/** straight walks shorter than this are not worth a note (m) */
const NOTE_MIN_M = 5;

function downloadGpx(name: string, result: FarmRouteResult, stops: FarmRouteStop[]) {
  const gpx = farmRouteGpx(`Solemtrix ${name}`, result.line, stops.map((s) => ({ id: s.id, at: s.at })));
  const url = URL.createObjectURL(new Blob([gpx], { type: "application/gpx+xml" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `traseu_${name}.gpx`;
  a.click();
  URL.revokeObjectURL(url);
}

const STEPS = { farm: ["step1", "step2", "step3"], all: ["step2", "step3"] } as const;

function activeStep(route: FarmRoute): number {
  const s = route.state.status;
  if (route.mode === "all") return s === "picking" ? 0 : 1;
  return s === "choosingFarm" ? 0 : s === "picking" ? 1 : 2;
}

/** Length, duration and the saving against the full inter-row sweep. */
function ResultFigures({ r }: { r: FarmRouteResult }) {
  const t = useTranslations("map.farmRoute");
  const f = useFormat();
  const saving = r.baselineM > 0 ? 1 - r.lengthM / r.baselineM : null;
  return (
    <>
      <Field label={t("stops")}>{f.int(r.order.length)}</Field>
      {r.farmOrder && <Field label={t("farmsVisited")}>{f.int(r.farmOrder.length)}</Field>}
      <Field label={t("length")}>{f.length(r.lengthM)}</Field>
      <Field label={t("duration")}>{f.min(r.durationMin)}</Field>
      {saving != null && (
        <Alert severity={saving > 0 ? "success" : "info"} icon={false} sx={{ my: 1 }} data-testid="farm-route-saving">
          <Typography variant="subtitle2" component="p">
            {saving > 0 ? t("saving", { pct: f.pct(saving) }) : t("noSaving")}
          </Typography>
          <Typography variant="caption" component="p">
            {t("savingDetail", { baseline: f.length(r.baselineM), minutes: f.min(Math.max(0, (r.baselineM - r.lengthM) / (4000 / 60))) })}
          </Typography>
        </Alert>
      )}
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
    </>
  );
}

function Body({
  route,
  farms,
  stops,
  officialStart,
  onChooseFarm,
}: {
  route: FarmRoute;
  farms: FarmSummary[];
  stops: FarmRouteStop[];
  officialStart: LonLat | null;
  onChooseFarm: (id: string) => void;
}) {
  const t = useTranslations("map.farmRoute");
  const { state } = route;
  const all = route.mode === "all";
  const withTargets = farms.filter((x) => x.target_count > 0);
  const totalTargets = withTargets.reduce((s, x) => s + x.target_count, 0);
  const farm = farms.find((x) => x.farm_id === route.farmId);
  const heading = all ? (
    <Typography variant="subtitle2" sx={{ mb: 1 }}>
      {t("allLine", { farms: withTargets.length, n: totalTargets })}
    </Typography>
  ) : (
    farm && (
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        {t("farmLine", { id: farm.farm_id, n: farm.target_count })}
      </Typography>
    )
  );
  const otherFarm = !all && (
    <Button size="small" onClick={route.open}>
      {t("otherFarm")}
    </Button>
  );
  const again = () => (all ? route.openAll() : route.farmId && route.begin(route.farmId));

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
          {heading}
          <Alert severity="info" icon={false} sx={{ mb: 1 }} data-testid="farm-route-pick">
            {all ? t("pickHintAll") : t("pickHint")}
          </Alert>
          {all && officialStart && (
            <Button size="small" variant="outlined" onClick={() => route.pick(officialStart)} data-testid="farm-route-official-start" sx={{ mr: 1 }}>
              {t("officialStart")}
            </Button>
          )}
          {otherFarm}
        </>
      );
    case "computing":
      return (
        <>
          {heading}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {all ? t("computingAll", { n: totalTargets }) : t("computing", { n: farm?.target_count ?? 0 })}
          </Typography>
          <LinearProgress />
        </>
      );
    case "error":
      return (
        <>
          {heading}
          <Alert severity="error" sx={{ mb: 1 }}>
            {t("error", { detail: state.error })}
          </Alert>
          <Button size="small" onClick={again}>
            {t("newStart")}
          </Button>
          {otherFarm}
        </>
      );
    case "done":
      return (
        <Box data-testid="farm-route-result">
          {heading}
          <ResultFigures r={state.result} />
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, pt: 1.5 }}>
            <Button size="small" variant="outlined" onClick={again}>
              {t("newStart")}
            </Button>
            <Button
              size="small"
              variant="outlined"
              startIcon={<FileDownloadOutlined />}
              onClick={() => downloadGpx(all ? "toate_fermele" : (state.farmId ?? "ferma"), state.result, stops)}
            >
              GPX
            </Button>
            {otherFarm}
          </Box>
          <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 1 }}>
            {all ? t("noteAll") : t("note")}
          </Typography>
        </Box>
      );
    default:
      return null;
  }
}

function ToolButton({ label, icon, active, disabled, bottom, onClick, testId }: {
  label: string; icon: ReactNode; active: boolean; disabled: boolean; bottom: number; onClick: () => void; testId: string;
}) {
  return (
    <Paper sx={{ position: "absolute", right: 16, bottom, zIndex: 2, boxShadow: 3 }}>
      <Button onClick={onClick} disabled={disabled} variant={active ? "contained" : "text"} startIcon={icon} aria-pressed={active} data-testid={testId} sx={{ px: 2, py: 1, whiteSpace: "nowrap" }}>
        {label}
      </Button>
    </Paper>
  );
}

export function FarmRouteTool({
  route,
  farms,
  stops,
  officialStart,
  onChooseFarm,
}: {
  route: FarmRoute;
  /** summary.json farms (with their target counts) */
  farms: FarmSummary[];
  stops: FarmRouteStop[];
  /** the organisers' START (data/ref/start.geojson), offered as the start of the route through all farms */
  officialStart: LonLat | null;
  /** a farm chosen from the list: start step 2 and zoom to it */
  onChooseFarm: (farmId: string) => void;
}) {
  const t = useTranslations("map.farmRoute");
  const active = route.state.status !== "idle";
  const all = route.mode === "all";
  // desktop: the card sits above the buttons (bottom right); phone: at the top, height capped, so the map stays clickable
  return (
    <>
      {active && (
        <Paper
          data-testid="farm-route-card"
          sx={{
            position: "absolute", zIndex: 3, p: 3, boxShadow: 3, overflow: "auto",
            top: { xs: 16, md: "auto" }, left: { xs: 16, md: "auto" }, right: 16, bottom: { xs: "auto", md: 292 },
            width: { md: 340 }, maxHeight: { xs: "45%", md: "calc(100% - 324px)" },
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
            <Typography variant="subtitle1">{all ? t("titleAll") : t("title")}</Typography>
            <IconButton size="small" onClick={route.clear} aria-label={t("close")}>
              <Close fontSize="small" />
            </IconButton>
          </Box>
          <Stepper activeStep={activeStep(route)} alternativeLabel sx={{ mb: 2 }}>
            {STEPS[all ? "all" : "farm"].map((k) => (
              <Step key={k}>
                <StepLabel>{t(k)}</StepLabel>
              </Step>
            ))}
          </Stepper>
          <Body route={route} farms={farms} stops={stops} officialStart={officialStart} onChooseFarm={onChooseFarm} />
        </Paper>
      )}
      <ToolButton
        label={t("toolAll")}
        icon={<HubOutlined />}
        active={active && all}
        disabled={!route.ready}
        bottom={232}
        onClick={active && all ? route.clear : route.openAll}
        testId="farm-route-all"
      />
      <ToolButton
        label={t("tool")}
        icon={<AltRouteOutlined />}
        active={active && !all}
        disabled={!route.ready}
        bottom={176}
        onClick={active && !all ? route.clear : route.open}
        testId="farm-route-tool"
      />
    </>
  );
}
