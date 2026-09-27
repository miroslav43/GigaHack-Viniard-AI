"use client";

import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import { CountUp } from "@/components/common/CountUp";
import { useFormat } from "@/lib/useFormat";
import { useInView } from "@/lib/motion/useInView";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";

const GROW_MS = 1100;
/** the optimised bar starts once the baseline has grown */
const ROUTE_DELAY_MS = 900;
const BADGE_DELAY_MS = ROUTE_DELAY_MS + GROW_MS;
const BAR_PX = 22;

/**
 * Route without optimisation vs the optimised route, as two bars on one scale: the baseline grows first, then the
 * route grows to its (much shorter) length and the saving badge pops in. Reduced motion: final state straight away.
 */
export function RouteCompare({
  baselineM,
  routeM,
  savedShare,
  baselineLabel,
  routeLabel,
}: {
  baselineM: number;
  routeM: number;
  savedShare: number;
  baselineLabel: string;
  routeLabel: string;
}) {
  const f = useFormat();
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>();
  const shown = inView || reduced;
  const routePct = baselineM > 0 ? Math.min(100, (routeM / baselineM) * 100) : 100;
  const bars = [
    { key: "baseline", label: baselineLabel, value: baselineM, pct: 100, color: chartPalette.muted, delay: 0 },
    { key: "route", label: routeLabel, value: routeM, pct: routePct, color: chartPalette.primary, delay: ROUTE_DELAY_MS },
  ];
  return (
    <Box ref={ref} sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {bars.map((b) => (
        <Box key={b.key}>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 3, mb: 1 }}>
            <Typography variant="body2">{b.label}</Typography>
            <Typography sx={{ fontWeight: 700, fontSize: 20 }}>
              <CountUp value={b.value} format="length" durationMs={GROW_MS} />
            </Typography>
          </Box>
          <Box sx={{ height: BAR_PX, borderRadius: 1, bgcolor: chartPalette.track, overflow: "hidden" }}>
            <Box
              sx={{
                height: "100%",
                borderRadius: 1,
                bgcolor: b.color,
                width: shown ? `${b.pct}%` : 0,
                transition: reduced ? "none" : `width ${GROW_MS}ms cubic-bezier(.2,.8,.2,1) ${b.delay}ms`,
              }}
            />
          </Box>
        </Box>
      ))}
      <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
        <Chip
          color="primary"
          label={`−${f.pct(savedShare)}`}
          sx={{
            fontWeight: 700,
            fontSize: 16,
            opacity: shown ? 1 : 0,
            transform: shown ? "scale(1)" : "scale(0.6)",
            transition: reduced ? "none" : `opacity 300ms ease-out ${BADGE_DELAY_MS}ms, transform 400ms cubic-bezier(.34,1.56,.64,1) ${BADGE_DELAY_MS}ms`,
          }}
        />
      </Box>
    </Box>
  );
}
