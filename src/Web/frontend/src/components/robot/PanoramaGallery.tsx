"use client";

// Panoramas of the record mode: one per station (the 0° / 90° / 180° photos side by side), downloadable as the
// panorama or as its three photos; all of them at once.
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import { useTranslations } from "next-intl";
import type { Panorama } from "./useRecorder";

const pad = (n: number) => String(n).padStart(2, "0");
const stamp = (d: Date) => `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
export const panoramaFileName = (p: Panorama, deg?: number) =>
  `panorama_${stamp(p.takenAt)}_statia${pad(p.station)}_${p.atCm}cm${deg === undefined ? "" : `_${deg}grade`}.jpg`;

function save(url: string, name: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

export function PanoramaGallery({ panoramas, onRemove }: { panoramas: Panorama[]; onRemove: (id: string) => void }) {
  const t = useTranslations("robot.panoramas");
  if (panoramas.length === 0) return null;
  return (
    <Paper sx={{ p: 4 }} data-testid="robot-panoramas">
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2, gap: 2, flexWrap: "wrap" }}>
        <Typography variant="h3">{t("title", { n: panoramas.length })}</Typography>
        <Button size="small" variant="outlined" startIcon={<FileDownloadOutlined />} onClick={() => panoramas.forEach((p) => save(p.strip.url, panoramaFileName(p)))}>
          {t("downloadAll")}
        </Button>
      </Box>
      <Box sx={{ display: "grid", gap: 2 }}>
        {panoramas.map((p) => (
          <Box key={p.id} sx={{ border: 1, borderColor: "divider", borderRadius: 2, overflow: "hidden" }} data-testid="robot-panorama">
            {/* eslint-disable-next-line @next/next/no-img-element -- a local object URL */}
            <img src={p.strip.url} alt={panoramaFileName(p)} style={{ width: "100%", display: "block" }} />
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1.5, py: 1, flexWrap: "wrap" }}>
              <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
                {t("caption", { station: p.station, cm: p.atCm, time: p.takenAt.toLocaleTimeString() })}
              </Typography>
              <Button size="small" startIcon={<FileDownloadOutlined />} onClick={() => save(p.strip.url, panoramaFileName(p))}>
                {t("download")}
              </Button>
              <Button size="small" onClick={() => p.frames.forEach((f) => save(f.url, panoramaFileName(p, f.deg)))}>
                {t("downloadFrames")}
              </Button>
              <Tooltip title={t("remove")}>
                <IconButton size="small" onClick={() => onRemove(p.id)} aria-label={t("remove")}>
                  <DeleteOutlined fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        ))}
      </Box>
    </Paper>
  );
}
