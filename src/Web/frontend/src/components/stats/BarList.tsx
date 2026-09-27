"use client";

import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { useInView } from "@/lib/motion/useInView";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";

const GROW_MS = 900;
const STAGGER_MS = 45;
const BAR_PX = 10;

export interface BarItem {
  key: string;
  label: ReactNode;
  value: number;
  /** text shown at the end of the row (defaults to nothing) */
  valueText?: string;
  color?: string;
}

/**
 * Horizontal bars with long labels (skip reasons, rankings): each bar grows from 0 when the list comes on screen,
 * one after the other. Values and labels stay in text colours; the bar carries the colour.
 */
export function BarList({ items, max, ariaLabel }: { items: BarItem[]; max?: number; ariaLabel: string }) {
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLUListElement>();
  const top = max ?? Math.max(1, ...items.map((i) => i.value));
  const grown = inView || reduced;
  return (
    <Box component="ul" ref={ref} aria-label={ariaLabel} sx={{ listStyle: "none", m: 0, p: 0, display: "flex", flexDirection: "column", gap: 2.5 }}>
      {items.map((it, i) => (
        <Box component="li" key={it.key}>
          <Box sx={{ display: "flex", justifyContent: "space-between", gap: 3, mb: 0.75 }}>
            <Typography variant="body2" component="span" sx={{ minWidth: 0 }}>
              {it.label}
            </Typography>
            {it.valueText && (
              <Typography variant="body2" component="span" color="text.secondary" sx={{ fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>
                {it.valueText}
              </Typography>
            )}
          </Box>
          <Box sx={{ height: BAR_PX, borderRadius: 1, bgcolor: chartPalette.track, overflow: "hidden" }}>
            <Box
              sx={{
                height: "100%",
                borderRadius: 1,
                bgcolor: it.color ?? chartPalette.primary,
                width: grown ? `${Math.max(0, Math.min(100, (it.value / top) * 100))}%` : 0,
                transition: reduced ? "none" : `width ${GROW_MS}ms cubic-bezier(.2,.8,.2,1) ${i * STAGGER_MS}ms`,
              }}
            />
          </Box>
        </Box>
      ))}
    </Box>
  );
}
