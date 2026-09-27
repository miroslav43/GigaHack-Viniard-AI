"use client";

// Photos taken this session: thumbnails with the camera angles, each downloadable as a JPEG, or all at once.
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import { useTranslations } from "next-intl";

export interface Photo {
  id: string;
  /** object URL of the JPEG */
  url: string;
  takenAt: Date;
  /** estimated camera position when it was taken, from the zeroed position (the street view mode will use them):
   *  pan in degrees, height in motor steps */
  panDeg: number;
  heightSteps: number;
}

const pad = (n: number) => String(n).padStart(2, "0");
export const photoFileName = (p: Photo) => {
  const d = p.takenAt;
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return `robot_${stamp}_pan${Math.round(p.panDeg)}_h${p.heightSteps}.jpg`;
};

function download(p: Photo) {
  const a = document.createElement("a");
  a.href = p.url;
  a.download = photoFileName(p);
  a.click();
}

export function PhotoGallery({ photos, onRemove }: { photos: Photo[]; onRemove: (id: string) => void }) {
  const t = useTranslations("robot.gallery");
  return (
    <Paper sx={{ p: 4 }} data-testid="robot-gallery">
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2, gap: 2, flexWrap: "wrap" }}>
        <Typography variant="h3">{t("title", { n: photos.length })}</Typography>
        {photos.length > 0 && (
          <Button size="small" variant="outlined" startIcon={<FileDownloadOutlined />} onClick={() => photos.forEach(download)}>
            {t("downloadAll")}
          </Button>
        )}
      </Box>
      {photos.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          {t("empty")}
        </Typography>
      ) : (
        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 2 }}>
          {photos.map((p) => (
            <Box key={p.id} sx={{ border: 1, borderColor: "divider", borderRadius: 2, overflow: "hidden" }} data-testid="robot-photo">
              {/* eslint-disable-next-line @next/next/no-img-element -- a local object URL, not an optimisable asset */}
              <img src={p.url} alt={photoFileName(p)} style={{ width: "100%", display: "block", aspectRatio: "4 / 3", objectFit: "cover" }} />
              <Box sx={{ display: "flex", alignItems: "center", px: 1.5, py: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
                  {p.takenAt.toLocaleTimeString()} · {t("angles", { pan: Math.round(p.panDeg), height: p.heightSteps })}
                </Typography>
                <Tooltip title={t("download")}>
                  <IconButton size="small" onClick={() => download(p)} aria-label={t("download")}>
                    <FileDownloadOutlined fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("remove")}>
                  <IconButton size="small" onClick={() => onRemove(p.id)} aria-label={t("remove")}>
                    <DeleteOutlined fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          ))}
        </Box>
      )}
    </Paper>
  );
}
