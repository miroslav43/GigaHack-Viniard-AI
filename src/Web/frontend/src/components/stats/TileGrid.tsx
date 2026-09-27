"use client";

import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { useTheme } from "@mui/material/styles";
import type { TilesStats } from "@/lib/types";
import { useInView } from "@/lib/motion/useInView";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";

// The survey tiles on their grid (row / column of the tile id): vineyard tiles on a one-hue ramp by vegetation share,
// tiles without vines faint, tiles to complete in Marcaj outlined. They appear in a diagonal sweep, like the flight.
const CELL = 0.88;
const PAD = (1 - CELL) / 2;
const SWEEP_MS = 1600;
const FADE_MS = 380;
const NO_VINEYARD_OPACITY = 0.35;
/** the lightest ramp step stays visible on the white card */
const RAMP_FLOOR = 0.15;
const MAX_HEIGHT_PX = 420;

const hexRgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** Colour at t ∈ [0, 1] between two #rrggbb colours (linear in sRGB, enough for a light → dark ramp of one hue). */
const mix = (from: string, to: string, t: number) => {
  const a = hexRgb(from);
  const b = hexRgb(to);
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
};

export const rampColor = (vegFrac: number) =>
  mix(chartPalette.sequential.from, chartPalette.sequential.to, RAMP_FLOOR + (1 - RAMP_FLOOR) * Math.max(0, Math.min(1, vegFrac)));

export interface TileGridLabels {
  aria: string;
  tileTitle: (tile: TilesStats["items"][number]) => string;
  legendVeg: string;
  legendNone: string;
  legendToComplete: string;
  pct: (v: number) => string;
}

export function TileGrid({ tiles, labels }: { tiles: TilesStats; labels: TileGridLabels }) {
  const theme = useTheme();
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>();
  const shown = inView || reduced;
  const minR = Math.min(...tiles.items.map((t) => t.r));
  const minC = Math.min(...tiles.items.map((t) => t.c));
  const width = tiles.cols - minC;
  const height = tiles.rows - minR;
  const stepMs = SWEEP_MS / Math.max(1, width + height);

  return (
    <Box ref={ref}>
      <Box sx={{ display: "flex", justifyContent: "center" }}>
        <svg
          viewBox={`${minC} ${minR} ${width} ${height}`}
          role="img"
          aria-label={labels.aria}
          style={{ width: "100%", maxHeight: MAX_HEIGHT_PX, aspectRatio: `${width} / ${height}`, display: "block" }}
        >
          {tiles.items.map((t) => {
            const vineyard = t.status === "vineyard";
            return (
              <rect
                key={t.tile}
                x={t.c + PAD}
                y={t.r + PAD}
                width={CELL}
                height={CELL}
                rx={0.12}
                fill={vineyard ? rampColor(t.veg_frac) : chartPalette.tile.no_vineyard}
                fillOpacity={vineyard ? 1 : NO_VINEYARD_OPACITY}
                stroke={t.to_complete ? chartPalette.tile.toComplete : "none"}
                strokeWidth={t.to_complete ? 0.14 : 0}
                style={{
                  opacity: shown ? 1 : 0,
                  transition: reduced ? "none" : `opacity ${FADE_MS}ms ease-out ${Math.round((t.r - minR + t.c - minC) * stepMs)}ms`,
                }}
              >
                <title>{labels.tileTitle(t)}</title>
              </rect>
            );
          })}
        </svg>
      </Box>
      <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 4, mt: 3 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Typography variant="caption" color="text.secondary">
            {labels.legendVeg}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {labels.pct(0)}
          </Typography>
          <Box
            aria-hidden
            sx={{ width: 96, height: 10, borderRadius: 1, backgroundImage: `linear-gradient(90deg, ${rampColor(0)}, ${rampColor(1)})` }}
          />
          <Typography variant="caption" color="text.secondary">
            {labels.pct(1)}
          </Typography>
        </Box>
        <Legend swatch={<Box sx={{ width: 12, height: 12, borderRadius: 0.5, bgcolor: chartPalette.tile.no_vineyard, opacity: NO_VINEYARD_OPACITY }} />} label={labels.legendNone} />
        <Legend
          swatch={<Box sx={{ width: 12, height: 12, borderRadius: 0.5, border: 2, borderColor: chartPalette.tile.toComplete, bgcolor: theme.palette.background.paper }} />}
          label={labels.legendToComplete}
        />
      </Box>
    </Box>
  );
}

function Legend({ swatch, label }: { swatch: ReactNode; label: string }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
      {swatch}
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}
