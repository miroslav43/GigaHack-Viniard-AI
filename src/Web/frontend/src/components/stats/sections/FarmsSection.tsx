"use client";

import { useTranslations } from "next-intl";
import { BarChart } from "@mui/x-charts/BarChart";
import { CountUp } from "@/components/common/CountUp";
import { useFormat } from "@/lib/useFormat";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";
import { BarList } from "../BarList";
import { ChartCard, StatSection, StatTile } from "../parts";
import type { SectionProps } from "../types";

const M2_PER_HA = 1e4;
const PCT = 100;
const TOP_DENSITY = 10;
const BAR_HEIGHT = 280;

/** Section C: how the vineyard area splits into farms (largest first), problem density per farm, cadastral parcels. */
export function FarmsSection({ letter, stats, uatAreaHa }: SectionProps) {
  const t = useTranslations("stats.farms");
  const f = useFormat();
  const reduced = useReducedMotion();
  const farms = stats.farms;
  if (!farms || farms.items.length === 0) return null;

  const items = farms.items;
  const lead =
    items.length >= 2
      ? t("lead", { pct: f.pct(items[1].cumulative_share), n: f.int(items.length) })
      : t("leadOne", { area: f.ha(farms.total_area_m2) });
  const blocks = items.reduce((a, it) => a + it.n_blocks, 0);
  const dense = [...items].sort((a, b) => b.targets_per_ha - a.targets_per_ha).slice(0, TOP_DENSITY);
  const fragmented = items.reduce<(typeof items)[number] | null>(
    (best, it) => (it.n_parcels != null && (best?.n_parcels == null || it.n_parcels > best.n_parcels) ? it : best),
    null,
  );

  return (
    <StatSection id="ferme" letter={letter} title={t("title")} lead={lead}>
      <ChartCard title={t("areaTitle")} subtitle={t("areaSubtitle")} wide minHeight={BAR_HEIGHT}>
        <BarChart
          height={BAR_HEIGHT}
          skipAnimation={reduced}
          hideLegend
          borderRadius={4}
          xAxis={[{ scaleType: "band", data: items.map((it) => it.farm_id) }]}
          yAxis={[{ valueFormatter: (v: number) => f.num(v, 1), label: f.units.ha }]}
          series={[
            {
              data: items.map((it) => it.area_m2 / M2_PER_HA),
              label: t("areaSeries"),
              color: chartPalette.primary,
              valueFormatter: (v) => (v == null ? "" : f.ha(v * M2_PER_HA)),
            },
          ]}
          grid={{ horizontal: true }}
        />
      </ChartCard>

      <ChartCard title={t("densityTitle")} subtitle={t("densitySubtitle", { n: f.int(dense.length) })} minHeight={260}>
        <BarList
          ariaLabel={t("densityTitle")}
          items={dense.map((it) => ({
            key: it.farm_id,
            label: t("farmLabel", { id: it.farm_id, targets: f.int(it.target_count) }),
            value: it.targets_per_ha,
            valueText: t("perHa", { value: f.num(it.targets_per_ha, 1) }),
          }))}
        />
      </ChartCard>

      <StatTile label={t("count")} value={<CountUp value={farms.count} />} hint={t("countHint", { blocks: f.int(blocks) })} />
      <StatTile label={t("totalArea")} value={<CountUp value={farms.total_area_m2} format="ha" />} hint={t("totalAreaHint")} delayMs={80} />
      <StatTile
        label={t("communeShare")}
        value={<CountUp value={farms.commune_share * PCT} format="num" digits={2} suffix=" %" />}
        hint={t("communeShareHint", { area: f.ha(farms.total_area_m2), commune: `${f.num(uatAreaHa, 0)} ${f.units.ha}` })}
        delayMs={160}
      />
      {farms.parcels_total != null && (
        <StatTile
          label={t("parcels")}
          value={<CountUp value={farms.parcels_total} />}
          hint={fragmented?.n_parcels != null ? t("parcelsHint", { farm: fragmented.farm_id, n: f.int(fragmented.n_parcels) }) : undefined}
          delayMs={240}
        />
      )}
    </StatSection>
  );
}
