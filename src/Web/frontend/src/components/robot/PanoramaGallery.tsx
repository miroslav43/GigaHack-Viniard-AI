"use client";

// The panoramas saved on the laptop: the three photos (0° / 90° / 180°) with the grape and leaf boxes Gemini found
// drawn over them, the counts, and downloads (the panorama strip, the photos, the detections as JSON).
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import RefreshOutlined from "@mui/icons-material/RefreshOutlined";
import { useTranslations } from "next-intl";
import { countByLabel, DETECTION_LABELS, isDetectionLabel, type DetectionBox, type DetectionLabel } from "@/lib/robot/detections";
import { robotPalette } from "@/theme/robotPalette";
import type { PanoramaMeta } from "@/lib/robot/panoramaStore";

const ANGLES = [0, 90, 180] as const;
const pad = (n: number) => String(n).padStart(2, "0");
const stamp = (iso: string) => {
  const d = new Date(iso);
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
};
const baseName = (p: PanoramaMeta) => `panorama_${stamp(p.takenAt)}_statia${pad(p.station)}_${p.atCm}cm`;
const fileUrl = (p: PanoramaMeta, file: string) => `/api/robot/panoramas/${p.id}/${file}`;

function save(href: string, name: string) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.click();
}

function saveJson(p: PanoramaMeta) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(p, null, 2)], { type: "application/json" }));
  save(url, `${baseName(p)}_detectii.json`);
  URL.revokeObjectURL(url);
}

/** One photo with its boxes (an SVG over the image, in fractions of it). */
function Frame({ src, alt, boxes, colors }: { src: string; alt: string; boxes: DetectionBox[]; colors: Readonly<Record<DetectionLabel, string>> }) {
  return (
    <Box sx={{ position: "relative", lineHeight: 0 }}>
      {/* eslint-disable-next-line @next/next/no-img-element -- a photo served by our API */}
      <img src={src} alt={alt} style={{ width: "100%", display: "block" }} />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} data-testid="robot-boxes">
        {boxes.map((b, i) => (
          <rect
            key={i}
            x={b.box[0]}
            y={b.box[1]}
            width={b.box[2] - b.box[0]}
            height={b.box[3] - b.box[1]}
            fill="none"
            stroke={colors[b.label]}
            strokeWidth={b.label === "waste" ? 3.5 : 2.5}
            vectorEffect="non-scaling-stroke"
            data-label={b.label}
          >
            <title>{`${b.label}${b.score === null ? "" : ` ${Math.round(b.score * 100)}%`}`}</title>
          </rect>
        ))}
      </svg>
    </Box>
  );
}

export function PanoramaGallery({ panoramas, onRedetect, onRemove }: { panoramas: PanoramaMeta[]; onRedetect: (id: string) => void; onRemove: (id: string) => void }) {
  const t = useTranslations("robot.panoramas");
  const colors = robotPalette;
  if (panoramas.length === 0) return null;
  return (
    <Paper sx={{ p: 4 }} data-testid="robot-panoramas">
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1, gap: 2, flexWrap: "wrap" }}>
        <Typography variant="h3">{t("title", { n: panoramas.length })}</Typography>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "center" }}>
          {DETECTION_LABELS.map((l) => (
            <Box key={l} sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
              <Box sx={{ width: 14, height: 10, border: l === "waste" ? 3 : 2.5, borderColor: colors[l], borderRadius: 0.5 }} />
              <Typography variant="caption">{t(`label.${l}`)}</Typography>
            </Box>
          ))}
        </Box>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {t("saved")}
      </Typography>
      <Box sx={{ display: "grid", gap: 3 }}>
        {panoramas.map((p) => {
          const d = p.detection;
          // only the labels detected now (older saves may hold "object" boxes)
          const shown = (boxes: DetectionBox[] | undefined) => (boxes ?? []).filter((b) => isDetectionLabel(b.label));
          const all = Object.values(d.frames).flatMap(shown);
          const counts = countByLabel(all);
          return (
            <Box key={p.id} sx={{ border: 1, borderColor: "divider", borderRadius: 2, overflow: "hidden" }} data-testid="robot-panorama">
              <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)" }}>
                {ANGLES.map((deg) => (
                  <Frame key={deg} src={fileUrl(p, `${deg}.jpg`)} alt={`${baseName(p)} ${deg}°`} boxes={shown(d.frames[String(deg)])} colors={colors} />
                ))}
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1.5, py: 1, flexWrap: "wrap" }}>
                <Typography variant="caption" color="text.secondary">
                  {t("caption", { station: p.station, cm: p.atCm, time: new Date(p.takenAt).toLocaleString() })}
                </Typography>
                <Box sx={{ flex: 1 }} />
                <Box data-testid="robot-detection" data-status={d.status} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {(d.status === "pending" || d.status === "running") && (
                    <>
                      <CircularProgress size={14} />
                      <Typography variant="caption">{t("detecting")}</Typography>
                    </>
                  )}
                  {d.status === "done" && (
                    <>
                      {DETECTION_LABELS.map((l) => (
                        <Chip key={l} size="small" variant="outlined" label={t(`count.${l}`, { n: counts[l] })} sx={{ borderColor: colors[l], borderWidth: 2 }} data-label={l} />
                      ))}
                    </>
                  )}
                  {d.status === "off" && <Typography variant="caption" color="text.secondary">{t("detectOff")}</Typography>}
                </Box>
                {(d.status === "done" || d.status === "error") && (
                  <Tooltip title={t("redetect")}>
                    <IconButton size="small" onClick={() => onRedetect(p.id)} aria-label={t("redetect")}>
                      <RefreshOutlined fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
              {d.status === "error" && (
                <Alert severity="error" sx={{ mx: 1.5, mb: 1 }}>
                  {t("detectError", { detail: d.error ?? "?" })}
                </Alert>
              )}
              <Box sx={{ display: "flex", gap: 1, px: 1.5, pb: 1, flexWrap: "wrap" }}>
                <Button size="small" startIcon={<FileDownloadOutlined />} onClick={() => save(fileUrl(p, "strip.jpg"), `${baseName(p)}.jpg`)}>
                  {t("download")}
                </Button>
                <Button size="small" onClick={() => ANGLES.forEach((deg) => save(fileUrl(p, `${deg}.jpg`), `${baseName(p)}_${deg}grade.jpg`))}>
                  {t("downloadFrames")}
                </Button>
                {d.status === "done" && (
                  <Button size="small" onClick={() => saveJson(p)}>
                    {t("downloadJson")}
                  </Button>
                )}
                <Box sx={{ flex: 1 }} />
                <Tooltip title={t("remove")}>
                  <IconButton size="small" onClick={() => onRemove(p.id)} aria-label={t("remove")}>
                    <DeleteOutlined fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          );
        })}
      </Box>
    </Paper>
  );
}
