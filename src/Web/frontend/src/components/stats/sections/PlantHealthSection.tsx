"use client";

import { useTranslations } from "next-intl";
import { BarChart } from "@mui/x-charts/BarChart";
import { PieChart } from "@mui/x-charts/PieChart";
import { CountUp } from "@/components/common/CountUp";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { useFormat } from "@/lib/useFormat";
import { chartPalette } from "@/theme/chartPalette";
import { HEALTH_ATTENTION, HealthRanking } from "../HealthRanking";
import { ChartCard, StatSection, StatTile } from "../parts";
import { binLabels, type SectionProps, DONUT_LEGEND } from "../types";

// Fixed colour per target kind (colour follows the entity, never its rank); older bundles only have the type.
const KIND_ORDER = ["missing_plant", "row_gap", "row_end_short", "missing_row", "waste"];
const TYPE_ORDER = ["missing", "gap", "waste"];
const OTHER = "other";
const PRIORITIES = ["1", "2", "3"];
// a bar gets its value label only when it is tall enough to hold it (share of the tallest bar)
const LABEL_MIN_SHARE = 0.12;
const M2_PER_HA = 1e4;
const CHART_H = 240;

interface Slice {
  id: string;
  value: number;
  label: string;
  color: string;
}

/** Known keys in their fixed order and colour; anything else folds into one muted "other" slice. */
function slices(counts: Record<string, number>, order: string[], label: (k: string) => string): Slice[] {
  const known = order.filter((k) => (counts[k] ?? 0) > 0).map((k) => ({ id: k, value: counts[k], label: label(k), color: chartPalette.categorical[order.indexOf(k)] }));
  const rest = Object.entries(counts).filter(([k]) => !order.includes(k)).reduce((a, [, v]) => a + v, 0);
  return rest > 0 ? [...known, { id: OTHER, value: rest, label: label(OTHER), color: chartPalette.muted }] : known;
}

export function PlantHealthSection({ letter, stats, totals }: SectionProps) {
  const t = useTranslations("stats.health");
  const tc = useTranslations();
  const f = useFormat();
  const reduced = useReducedMotion();
  const tg = stats.targets;
  const byKind = tg.by_kind != null;
  const kindLabel = (k: string) => (byKind ? (tc.has(`targetKind.${k}`) ? tc(`targetKind.${k}`) : k) : tc.has(`targetType.${k}`) ? tc(`targetType.${k}`) : k);
  const pie = slices(tg.by_kind ?? tg.by_type, byKind ? KIND_ORDER : TYPE_ORDER, kindLabel);

  const missing = tg.by_kind?.missing_plant ?? tg.by_type.missing ?? 0;
  const gaps = tg.by_kind?.row_gap ?? tg.by_type.gap ?? 0;
  const plantedHa = (totals.canopy_area_m2 + totals.interrow_area_m2) / M2_PER_HA;
  const disruptedShare = totals.row_count > 0 ? totals.disrupted_rows / totals.row_count : 0;
  const gapHist = stats.rows.gap_hist;
  const h = stats.health;

  return (
    <StatSection
      id="stare"
      letter={letter}
      title={t("title")}
      lead={t("lead", { missing: f.int(missing), gaps: f.int(gaps), disrupted: f.pct(disruptedShare), gapTotal: f.length(tg.gap_total_m) })}
    >
      <StatTile label={t("gapTotal")} value={<CountUp value={tg.gap_total_m} format="length" />} hint={t("gapTotalHint")} />
      <StatTile
        label={t("missingPerHa")}
        value={<CountUp value={plantedHa > 0 ? missing / plantedHa : 0} format="num" digits={0} />}
        hint={t("missingPerHaHint", { area: f.num(plantedHa, 2) })}
        delayMs={80}
      />
      <StatTile
        label={t("disrupted")}
        value={<CountUp value={disruptedShare} format="pct" />}
        hint={t("disruptedHint", { n: f.int(totals.disrupted_rows), total: f.int(totals.row_count) })}
        delayMs={160}
      />
      <StatTile label={t("gapMedian")} value={<CountUp value={stats.rows.gap_median_m} format="m" digits={1} />} hint={t("gapMedianHint", { max: f.m(stats.rows.gap_max_m, 1) })} />

      <ChartCard title={t("byKind")} subtitle={t("byKindHint", { n: f.int(tg.total) })}>
        <PieChart
          height={CHART_H}
          skipAnimation={reduced}
          slotProps={DONUT_LEGEND}
          series={[{ data: pie, innerRadius: 55, outerRadius: 95, paddingAngle: 1.5, cornerRadius: 4, valueFormatter: (item) => f.int(item.value) }]}
        />
      </ChartCard>

      {tg.by_priority && (
        <ChartCard title={t("byPriority")} subtitle={t("byPriorityHint")} delayMs={80}>
          <BarChart
            height={CHART_H}
            skipAnimation={reduced}
            hideLegend
            borderRadius={4}
            xAxis={[{ scaleType: "band", data: PRIORITIES.map((p) => t("priority", { p })) }]}
            yAxis={[{ valueFormatter: (v: number) => f.int(v) }]}
            series={[{ data: PRIORITIES.map((p) => tg.by_priority?.[p] ?? 0), barLabel: (item) => ((item.value ?? 0) >= LABEL_MIN_SHARE * Math.max(...PRIORITIES.map((p) => tg.by_priority?.[p] ?? 0)) ? f.int(item.value ?? 0) : null), color: chartPalette.primary, label: t("byPriority"), valueFormatter: (v) => (v == null ? null : f.int(v)) }]}
          />
        </ChartCard>
      )}

      <ChartCard title={t("gapHist")} subtitle={t("gapHistHint", { n: f.int(stats.rows.count) })} delayMs={160}>
        <BarChart
          height={CHART_H}
          skipAnimation={reduced}
          hideLegend
          borderRadius={4}
          xAxis={[{ scaleType: "band", data: binLabels(gapHist.edges, (v) => f.num(v, 0), f.units.m) }]}
          yAxis={[{ valueFormatter: (v: number) => f.int(v) }]}
          series={[{ data: gapHist.counts, color: chartPalette.primary, label: t("rows"), valueFormatter: (v) => (v == null ? null : t("rowsN", { n: f.int(v) })) }]}
        />
      </ChartCard>

      <ChartCard
        wide
        title={t("ranking")}
        subtitle={t("rankingHint", { threshold: f.int(HEALTH_ATTENTION) })}
        minHeight={320}
        footer={t("formula", {
          wd: f.num(h.weights.disrupted, 1),
          wm: f.num(h.weights.missing, 1),
          wg: f.num(h.weights.gap, 1),
          m: f.num(h.norm.missing_per_100m, 2),
          g: f.num(h.norm.gap_m_per_100m, 1),
          q: f.int(h.norm_quantile * 100),
          min: f.m(h.min_row_length_m, 0),
        })}
      >
        <HealthRanking blocks={h.blocks} />
      </ChartCard>
    </StatSection>
  );
}
