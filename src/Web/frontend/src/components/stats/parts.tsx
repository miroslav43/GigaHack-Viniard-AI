"use client";

import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";
import { Reveal } from "./Reveal";

/** One lettered section of the page: overline, title, the one-sentence takeaway, then its cards. */
export function StatSection({ id, letter, title, lead, children }: { id: string; letter: string; title: string; lead: ReactNode; children: ReactNode }) {
  return (
    <Box component="section" id={id} aria-labelledby={`${id}-title`} sx={{ mt: 10, scrollMarginTop: 24 }}>
      <Reveal>
        <Typography variant="overline" color="primary" sx={{ fontWeight: 700, letterSpacing: 1.5 }}>
          {letter}
        </Typography>
        <Typography id={`${id}-title`} variant="h2" component="h2" sx={{ mb: 1 }}>
          {title}
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 5, maxWidth: 760 }}>
          {lead}
        </Typography>
      </Reveal>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))", xl: "repeat(3, minmax(0, 1fr))" }, gap: 4, alignItems: "start" }}>
        {children}
      </Box>
    </Box>
  );
}

/** A card holding one chart; the chart mounts (and animates) when the card comes on screen. */
export function ChartCard({
  title,
  subtitle,
  children,
  wide = false,
  minHeight = 260,
  delayMs = 0,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  children: ReactNode;
  wide?: boolean;
  minHeight?: number;
  delayMs?: number;
  footer?: ReactNode;
}) {
  return (
    <Card sx={{ p: 5, display: "flex", flexDirection: "column", gap: 3, gridColumn: wide ? { md: "span 2" } : undefined, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle1" component="h3" sx={{ fontWeight: 600 }}>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </Box>
      <Reveal mount minHeight={minHeight} delayMs={delayMs}>
        {children}
      </Reveal>
      {footer && (
        <Typography variant="caption" color="text.secondary">
          {footer}
        </Typography>
      )}
    </Card>
  );
}

/** A small headline figure inside a section (the value is usually a CountUp). */
export function StatTile({ label, value, hint, delayMs = 0 }: { label: string; value: ReactNode; hint?: ReactNode; delayMs?: number }) {
  return (
    <Card sx={{ p: 5, display: "flex", flexDirection: "column", gap: 1 }}>
      <Reveal delayMs={delayMs}>
        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
          {label}
        </Typography>
        <Typography sx={{ fontSize: 30, fontWeight: 700, lineHeight: 1.2, letterSpacing: -0.4, my: 1 }}>{value}</Typography>
        {hint && (
          <Typography variant="caption" color="text.secondary">
            {hint}
          </Typography>
        )}
      </Reveal>
    </Card>
  );
}
