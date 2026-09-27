"use client";

import { useTranslations } from "next-intl";
import { BarChart } from "@mui/x-charts/BarChart";
import { CountUp } from "@/components/common/CountUp";
import type { RoadClass } from "@/lib/types";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { useFormat } from "@/lib/useFormat";
import { chartPalette } from "@/theme/chartPalette";
import { ChartCard, StatSection, StatTile } from "../parts";
import type { SectionProps } from "../types";

const CLASSES: RoadClass[] = ["public", "field", "internal"];
// OSM `surface` values folded into three groups; anything not listed counts as unpaved, "unknown" = no tag in OSM
const PAVED = new Set(["paved", "asphalt", "concrete", "cobblestone", "sett", "paving_stones", "metal", "concrete:plates", "unhewn_cobblestone"]);
const UNKNOWN = "unknown";
type SurfaceGroup = "paved" | "unpaved" | "unknown";
const GROUPS: { key: SurfaceGroup; color: string }[] = [
  { key: "paved", color: chartPalette.categorical[0] },
  { key: "unpaved", color: chartPalette.categorical[1] },
  { key: "unknown", color: chartPalette.unknown },
];

const groupOf = (surface: string): SurfaceGroup => (surface === UNKNOWN ? "unknown" : PAVED.has(surface) ? "paved" : "unpaved");

/** metres per surface group of one road class */
const grouped = (bySurface: Record<string, number>): Record<SurfaceGroup, number> =>
  Object.entries(bySurface).reduce((acc, [s, m]) => ({ ...acc, [groupOf(s)]: acc[groupOf(s)] + m }), { paved: 0, unpaved: 0, unknown: 0 });

export function RoadsSection({ letter, stats }: SectionProps) {
  const t = useTranslations("stats.roads");
  const f = useFormat();
  const reduced = useReducedMotion();
  if (!stats.roads) return null;
  const byClass = stats.roads.by_class;
  const present = CLASSES.filter((c) => byClass[c]);
  const groups = present.map((c) => grouped(byClass[c]?.by_surface ?? {}));
  const km = (c: RoadClass) => byClass[c]?.total_m ?? 0;
  const total = present.reduce((a, c) => a + km(c), 0);
  const unknown = groups.reduce((a, g) => a + g.unknown, 0);

  return (
    <StatSection
      id="drumuri"
      letter={letter}
      title={t("title")}
      lead={t("lead", { public: f.length(km("public")), field: f.length(km("field")), internal: f.length(km("internal")), unknown: f.pct(total > 0 ? unknown / total : 0) })}
    >
      {present.map((c, i) => (
        <StatTile key={c} label={t(`class.${c}`)} value={<CountUp value={km(c)} format="length" />} hint={t(`classHint.${c}`)} delayMs={i * 80} />
      ))}
      <ChartCard wide title={t("bySurface")} subtitle={t("bySurfaceHint")} minHeight={240} footer={t("footer")}>
        <BarChart
          height={220}
          layout="horizontal"
          skipAnimation={reduced}
          borderRadius={4}
          yAxis={[{ scaleType: "band", data: present.map((c) => t(`class.${c}`)), width: 110 }]}
          xAxis={[{ valueFormatter: (v: number) => f.num(v / 1000, 0), label: f.units.km }]}
          series={GROUPS.map((g) => ({
            data: groups.map((gr) => gr[g.key]),
            label: t(`surface.${g.key}`),
            stack: "surface",
            color: g.color,
            valueFormatter: (v: number | null) => (v == null ? null : f.length(v)),
          }))}
        />
      </ChartCard>
    </StatSection>
  );
}
