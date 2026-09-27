"use client";

import { useTranslations } from "next-intl";
import { Gauge, gaugeClasses } from "@mui/x-charts/Gauge";
import { PieChart } from "@mui/x-charts/PieChart";
import { CountUp } from "@/components/common/CountUp";
import { useFormat } from "@/lib/useFormat";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";
import { BarList } from "../BarList";
import { ChartCard, StatSection, StatTile } from "../parts";
import { RouteCompare } from "../RouteCompare";
import { DONUT_LEGEND, type SectionProps } from "../types";

const MIN_PER_H = 60;
const M_PER_KM = 1000;
const PCT = 100;
const GAUGE_HEIGHT = 200;
const DONUT_HEIGHT = 220;

/** Section E: the optimised route against the baseline, the 2 % outside rule, and which targets the route visits. */
export function RouteSection({ letter, stats }: SectionProps) {
  const t = useTranslations("stats.route");
  const tc = useTranslations();
  const f = useFormat();
  const reduced = useReducedMotion();
  const r = stats.route;
  const tg = stats.targets;
  const speedKmh = r.duration_min > 0 ? r.length_m / M_PER_KM / (r.duration_min / MIN_PER_H) : 0;
  const hasSaving = r.baseline_length_m != null && r.saved_share != null && r.saved_share > 0;
  const savedHours = r.saved_min != null ? r.saved_min / MIN_PER_H : null;

  const lead = hasSaving
    ? t("lead", {
        route: f.length(r.length_m),
        baseline: f.length(r.baseline_length_m ?? 0),
        pct: f.pct(r.saved_share ?? 0),
        hours: f.num(savedHours ?? 0, 1),
      })
    : t("leadNoBaseline", { route: f.length(r.length_m), min: f.min(r.duration_min) });

  const skip = Object.entries(tg.skip_reasons ?? {}).sort((a, b) => b[1] - a[1]);
  const outsideOk = r.outside_share != null && r.outside_share <= r.outside_limit;

  return (
    <StatSection id="ruta" letter={letter} title={t("title")} lead={lead}>
      {hasSaving && (
        <ChartCard title={t("compareTitle")} subtitle={t("compareSubtitle")} wide minHeight={150}>
          <RouteCompare
            baselineM={r.baseline_length_m ?? 0}
            routeM={r.length_m}
            savedShare={r.saved_share ?? 0}
            baselineLabel={t("baseline")}
            routeLabel={t("route")}
          />
        </ChartCard>
      )}

      {r.outside_share != null && (
        <ChartCard
          title={t("outsideTitle")}
          subtitle={t("outsideSubtitle")}
          minHeight={GAUGE_HEIGHT}
          footer={t("outsideNote", { limit: `${f.num(r.outside_limit * PCT, 0)} %` })}
        >
          <Gauge
            height={GAUGE_HEIGHT}
            value={r.outside_share * PCT}
            valueMin={0}
            valueMax={r.outside_limit * PCT}
            startAngle={-110}
            endAngle={110}
            innerRadius="72%"
            cornerRadius={4}
            skipAnimation={reduced}
            text={() => t("outsideText", { value: `${f.num((r.outside_share ?? 0) * PCT, 2)} %`, limit: `${f.num(r.outside_limit * PCT, 0)} %` })}
            sx={{
              [`& .${gaugeClasses.valueArc}`]: { fill: outsideOk ? chartPalette.primary : chartPalette.limit },
              [`& .${gaugeClasses.referenceArc}`]: { fill: chartPalette.track },
              [`& .${gaugeClasses.valueText}`]: { fontSize: 18, fontWeight: 700 },
            }}
          />
        </ChartCard>
      )}

      <ChartCard title={t("coverageTitle")} subtitle={t("coverageSubtitle", { on: f.int(tg.on_route), total: f.int(tg.total) })} minHeight={DONUT_HEIGHT}>
        <PieChart
          height={DONUT_HEIGHT}
          skipAnimation={reduced}
          slotProps={DONUT_LEGEND}
          series={[
            {
              data: [
                { id: "on", value: tg.on_route, label: t("onRoute"), color: chartPalette.primary },
                { id: "off", value: tg.off_route, label: t("offRoute"), color: chartPalette.muted },
              ],
              innerRadius: 55,
              outerRadius: 95,
              paddingAngle: 2,
              cornerRadius: 4,
              valueFormatter: (item) => f.int(item.value),
              highlightScope: { fade: "global", highlight: "item" },
            },
          ]}
        />
      </ChartCard>

      {skip.length > 0 && (
        <ChartCard title={t("skipTitle")} subtitle={t("skipSubtitle")} minHeight={200}>
          <BarList
            ariaLabel={t("skipTitle")}
            items={skip.map(([key, n]) => ({
              key,
              label: tc.has(`skipReason.${key}`) ? tc(`skipReason.${key}`) : key,
              value: n,
              valueText: f.int(n),
            }))}
          />
        </ChartCard>
      )}

      <StatTile
        label={t("duration")}
        value={<CountUp value={r.duration_min} format="min" />}
        hint={speedKmh > 0 ? t("durationHint", { speed: f.num(speedKmh, 0) }) : undefined}
      />
      {savedHours != null && savedHours > 0 && (
        <StatTile
          label={t("saved")}
          value={<CountUp value={savedHours} format="num" digits={1} suffix={` ${t("hoursUnit")}`} />}
          hint={t("savedHint", { min: f.min(r.saved_min ?? 0) })}
          delayMs={120}
        />
      )}
    </StatSection>
  );
}
