"use client";

import { useRef, useState, type DragEvent } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import UploadFileOutlined from "@mui/icons-material/UploadFileOutlined";

/** Drag & drop zone (or file picker) for one GeoTIFF tile. */
export function UploadDrop({ disabled, onFile }: { disabled: boolean; onFile: (file: File) => void }) {
  const t = useTranslations("analiza.upload");
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && !disabled) onFile(file);
  };

  return (
    <Box
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      sx={{
        border: 2,
        borderStyle: "dashed",
        borderColor: over ? "primary.main" : "divider",
        bgcolor: over ? "action.hover" : "background.paper",
        borderRadius: 2,
        px: 6,
        py: 8,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        gap: 3,
        transition: "border-color 120ms, background-color 120ms",
      }}
    >
      <UploadFileOutlined color="primary" sx={{ fontSize: 44 }} />
      <Typography variant="h3" component="p">
        {t("title")}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: "64ch" }}>
        {t("hint")}
      </Typography>
      <Button variant="contained" disabled={disabled} onClick={() => input.current?.click()}>
        {t("choose")}
      </Button>
      <input
        ref={input}
        type="file"
        accept=".tif,.tiff,image/tiff"
        hidden
        aria-label={t("choose")}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) onFile(file);
        }}
      />
    </Box>
  );
}
