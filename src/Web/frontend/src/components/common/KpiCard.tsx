import type { ReactNode } from "react";
import Card from "@mui/material/Card";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export function KpiCard({
  label,
  value,
  hint,
  icon,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  tone?: "default" | "primary" | "warning";
}) {
  const accent = tone === "primary" ? "primary.main" : tone === "warning" ? "warning.main" : "text.primary";
  return (
    <Card sx={{ p: 5, height: "100%", display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, color: "text.secondary" }}>
        {icon}
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          {label}
        </Typography>
      </Box>
      <Typography sx={{ fontSize: 26, fontWeight: 700, lineHeight: 1.2, color: accent, letterSpacing: -0.3 }}>{value}</Typography>
      {hint && (
        <Typography variant="caption" color="text.secondary">
          {hint}
        </Typography>
      )}
    </Card>
  );
}

export function KpiGrid({ children, min = 200 }: { children: ReactNode; min?: number }) {
  return (
    <Box sx={{ display: "grid", gridTemplateColumns: `repeat(auto-fill, minmax(${min}px, 1fr))`, gap: 4 }}>{children}</Box>
  );
}
