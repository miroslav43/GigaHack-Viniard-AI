"use client";

import Box from "@mui/material/Box";
import { useTheme } from "@mui/material/styles";
import { useInView } from "@/lib/motion/useInView";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";

// Orientation rose of the rows (stats.rows.orientation): row length per bin of the row axis on [0°, 180°), drawn on
// the full circle (a row axis has no direction, so each bin shows twice, opposite). Petal area ∝ length.
const R = 100;
const LABEL_R = R + 12;
const VIEW = LABEL_R + 10;
const RINGS = [0.25, 0.5, 0.75, 1];
const GAP_DEG = 0.8;
const MIN_OPACITY = 0.35;
const GROW_MS = 700;
const STAGGER_MS = 28;
const ROTATE_IN_DEG = -25;
const ROTATE_MS = 1100;
const DEG = Math.PI / 180;

const point = (bearingDeg: number, r: number) => [r * Math.sin(bearingDeg * DEG), -r * Math.cos(bearingDeg * DEG)] as const;

/** SVG path of a wedge from the centre, clockwise from bearing a0 to a1 (degrees from north). */
const wedge = (a0: number, a1: number, r: number) => {
  const [x0, y0] = point(a0, r);
  const [x1, y1] = point(a1, r);
  return `M0 0L${x0.toFixed(2)} ${y0.toFixed(2)}A${r.toFixed(2)} ${r.toFixed(2)} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}Z`;
};

export function OrientationRose({
  binDeg,
  lengthM,
  compass,
  ariaLabel,
  petalLabel,
}: {
  binDeg: number;
  lengthM: number[];
  /** N, E, S, W in the current language */
  compass: [string, string, string, string];
  ariaLabel: string;
  petalLabel: (fromDeg: number, toDeg: number, lengthM: number) => string;
}) {
  const theme = useTheme();
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>();
  const grown = inView || reduced;
  const max = Math.max(0, ...lengthM);
  const petals = lengthM.flatMap((len, i) =>
    [0, 180].map((offset) => ({ key: `${i}-${offset}`, i, from: i * binDeg, to: (i + 1) * binDeg, offset, len })),
  );

  return (
    <Box ref={ref} sx={{ width: "100%", maxWidth: 340, mx: "auto" }}>
      <svg viewBox={`${-VIEW} ${-VIEW} ${2 * VIEW} ${2 * VIEW}`} role="img" aria-label={ariaLabel} style={{ width: "100%", height: "auto", display: "block" }}>
        {RINGS.map((k) => (
          <circle key={k} r={R * Math.sqrt(k)} fill="none" stroke={theme.palette.divider} strokeWidth={0.8} />
        ))}
        {[0, 45, 90, 135].map((a) => {
          const [x0, y0] = point(a, R);
          const [x1, y1] = point(a + 180, R);
          return <line key={a} x1={x0} y1={y0} x2={x1} y2={y1} stroke={theme.palette.divider} strokeWidth={0.6} />;
        })}
        <g
          style={{
            transformBox: "view-box",
            transformOrigin: "0 0",
            transform: grown ? "rotate(0deg)" : `rotate(${ROTATE_IN_DEG}deg)`,
            transition: reduced ? "none" : `transform ${ROTATE_MS}ms cubic-bezier(.2,.8,.2,1)`,
          }}
        >
          {petals
            .filter((p) => p.len > 0 && max > 0)
            .map((p) => {
              const share = p.len / max;
              return (
                <path
                  key={p.key}
                  d={wedge(p.from + p.offset + GAP_DEG / 2, p.to + p.offset - GAP_DEG / 2, R * Math.sqrt(share))}
                  fill={chartPalette.primary}
                  fillOpacity={MIN_OPACITY + (1 - MIN_OPACITY) * share}
                  stroke={theme.palette.background.paper}
                  strokeWidth={0.6}
                  style={{
                    transformBox: "view-box",
                    transformOrigin: "0 0",
                    transform: grown ? "scale(1)" : "scale(0)",
                    transition: reduced ? "none" : `transform ${GROW_MS}ms cubic-bezier(.2,.8,.2,1) ${p.i * STAGGER_MS}ms`,
                  }}
                >
                  <title>{petalLabel(p.from, p.to, p.len)}</title>
                </path>
              );
            })}
        </g>
        {compass.map((label, k) => {
          const [x, y] = point(k * 90, LABEL_R);
          return (
            <text
              key={label}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={11}
              fontWeight={600}
              fill={theme.palette.text.secondary}
            >
              {label}
            </text>
          );
        })}
      </svg>
    </Box>
  );
}
