"use client";

import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { keyframes } from "@mui/material/styles";
import { CountUp, type CountFormat } from "@/components/common/CountUp";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";

const drift = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export interface HeroFigure {
  key: string;
  value: number;
  format: CountFormat;
  digits?: number;
  prefix?: string;
}

/** The opening band: a slowly drifting brand gradient and the headline figures counting up. */
export function StatsHero({ title, subtitle, figures }: { title: string; subtitle: string; figures: HeroFigure[] }) {
  const t = useTranslations("stats.hero");
  const reduced = useReducedMotion();
  return (
    <Box
      sx={{
        borderRadius: 3,
        px: { xs: 5, md: 8 },
        py: { xs: 6, md: 8 },
        color: chartPalette.heroText,
        backgroundImage: `linear-gradient(120deg, ${chartPalette.heroFrom}, ${chartPalette.heroTo}, ${chartPalette.heroFrom})`,
        backgroundSize: "220% 220%",
        animation: reduced ? "none" : `${drift} 14s ease-in-out infinite`,
      }}
    >
      <Typography variant="h1" component="h1" sx={{ color: "inherit" }}>
        {title}
      </Typography>
      <Typography variant="body1" sx={{ mt: 1, opacity: 0.8, maxWidth: 820 }}>
        {subtitle}
      </Typography>
      <Box
        component="dl"
        sx={{ m: 0, mt: 6, display: "grid", gridTemplateColumns: { xs: "repeat(2, 1fr)", sm: "repeat(3, 1fr)", lg: "repeat(6, 1fr)" }, gap: { xs: 4, md: 5 } }}
      >
        {figures.map((fig) => (
          <Box key={fig.key}>
            <Typography component="dd" sx={{ m: 0, fontSize: { xs: 28, md: 34 }, fontWeight: 700, lineHeight: 1.1, letterSpacing: -0.5 }}>
              <CountUp value={fig.value} format={fig.format} digits={fig.digits} prefix={fig.prefix} durationMs={1600} />
            </Typography>
            <Typography component="dt" variant="body2" sx={{ mt: 1, opacity: 0.8 }}>
              {t(fig.key)}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
