"use client";

import { useTranslations } from "next-intl";
import { BarChart } from "@mui/x-charts/BarChart";
import { PieChart } from "@mui/x-charts/PieChart";
import { Gauge, gaugeClasses } from "@mui/x-charts/Gauge";
import { CountUp } from "@/components/common/CountUp";
import { useFormat } from "@/lib/useFormat";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";
import { chartPalette } from "@/theme/chartPalette";
import type { InterrowCover } from "@/theme/mapPalette";
import { OrientationRose } from "../OrientationRose";
import { ChartCard, StatSection, StatTile } from "../parts";
import { binLabels, type SectionProps, DONUT_LEGEND } from "../types";

const COMPASS_POINTS = 16;
/** interrow covers below this share of the interrow area are left out of the donut (e.g. 0.5 m² "unassessable") */
const MIN_COVER_SHARE = 0.001;
const CHART_HEIGHT = 260;

/** 16-point compass name of a bearing, from the comma-separated list in the messages (N, NNE, …). */
const compassName = (points: string[], deg: number) => points[Math.round((((deg % 360) + 360) % 360) / (360 / COMPASS_POINTS)) % COMPASS_POINTS];

export function StructureSection({ letter, stats, totals }: SectionProps) {
  const t = useTranslations("stats.structure");
  const tc = useTranslations();
  const f = useFormat();
  const reduced = useReducedMotion();
  const { orientation, length_hist: lengthHist } = stats.rows;
  const s = stats.structure;

  const points = t("compassPoints").split(",");
  const from = orientation.dominant_deg - orientation.window_deg / 2;
  const to = orientation.dominant_deg + orientation.window_deg / 2;
  // name the axis from its northern end (e.g. 145° → "NW–SE")
  const northEnd = orientation.dominant_deg >= 90 ? orientation.dominant_deg + 180 : orientation.dominant_deg;
  const axis = `${compassName(points, northEnd)}–${compassName(points, northEnd + 180)}`;

  const covers = (Object.entries(s.interrow_area_by_cover) as [InterrowCover, number][]).filter(
    ([, area]) => s.interrow_area_total_m2 > 0 && area / s.interrow_area_total_m2 >= MIN_COVER_SHARE,
  );

  return (
    <StatSection
      id="structura"
      letter={letter}
      title={t("title")}
      lead={t("lead", {
        share: f.pct(orientation.dominant_share),
        from: f.int(from),
        to: f.int(to),
        axis,
        width: f.m(s.interrow_width_m, 2),
        cover: f.pct(s.canopy_cover_share),
      })}
    >
      <ChartCard title={t("roseTitle")} subtitle={t("roseSubtitle", { bin: orientation.bin_deg })} minHeight={300} footer={t("roseFooter")}>
        <OrientationRose
          binDeg={orientation.bin_deg}
          lengthM={orientation.length_m}
          compass={t("compass").split(",") as [string, string, string, string]}
          ariaLabel={t("roseAria", { share: f.pct(orientation.dominant_share), axis })}
          petalLabel={(a, b, len) => t("petal", { from: f.int(a), to: f.int(b), length: f.length(len) })}
        />
      </ChartCard>

      <ChartCard title={t("lengthTitle")} subtitle={t("lengthSubtitle")} minHeight={CHART_HEIGHT} delayMs={80}>
        <BarChart
          height={CHART_HEIGHT}
          skipAnimation={reduced}
          hideLegend
          borderRadius={4}
          grid={{ horizontal: true }}
          xAxis={[{ scaleType: "band", data: binLabels(lengthHist.edges, (v) => f.int(v), f.units.m), tickLabelStyle: { fontSize: 10 } }]}
          series={[{ data: lengthHist.counts, color: chartPalette.primary, label: t("rowsSeries"), valueFormatter: (v) => t("rowsCount", { n: f.int(v ?? 0) }) }]}
        />
      </ChartCard>

      <ChartCard title={t("coverTitle")} subtitle={t("coverSubtitle")} minHeight={220} delayMs={160}>
        <Gauge
          height={220}
          value={s.canopy_cover_share * 100}
          valueMin={0}
          valueMax={100}
          startAngle={-110}
          endAngle={110}
          innerRadius="74%"
          cornerRadius={4}
          skipAnimation={reduced}
          text={f.pct(s.canopy_cover_share)}
          aria-label={t("coverAria", { cover: f.pct(s.canopy_cover_share) })}
          sx={{
            [`& .${gaugeClasses.valueText}`]: { fontSize: 32, fontWeight: 700 },
            [`& .${gaugeClasses.valueArc}`]: { fill: chartPalette.primary },
            [`& .${gaugeClasses.referenceArc}`]: { fill: chartPalette.track },
          }}
        />
      </ChartCard>

      <ChartCard title={t("interrowTitle")} subtitle={t("interrowSubtitle", { area: f.ha(s.interrow_area_total_m2) })} minHeight={CHART_HEIGHT}>
        <PieChart
          height={CHART_HEIGHT}
          skipAnimation={reduced}
          slotProps={DONUT_LEGEND}
          series={[
            {
              data: covers.map(([k, area]) => ({ id: k, value: area, label: tc(`cover.${k}`), color: chartPalette.interrow[k] })),
              innerRadius: 56,
              outerRadius: 100,
              paddingAngle: 1.5,
              cornerRadius: 4,
              highlightScope: { fade: "global", highlight: "item" },
              valueFormatter: (item) => `${f.ha(item.value)} · ${f.pct(item.value / (s.interrow_area_total_m2 || 1))}`,
            },
          ]}
        />
      </ChartCard>

      <StatTile
        label={t("widthLabel")}
        value={<CountUp value={s.interrow_width_m} format="m" digits={2} />}
        hint={t("widthHint")}
      />
      <StatTile
        label={t("canopyLabel")}
        value={<CountUp value={s.canopy_mean_m2} format="m2" digits={2} />}
        hint={t("canopyHint", { n: f.int(totals.canopy_count) })}
        delayMs={60}
      />
      <StatTile
        label={t("perCanopyLabel")}
        value={<CountUp value={s.row_m_per_canopy} format="m" digits={1} />}
        hint={t("perCanopyHint")}
        delayMs={120}
      />
      <StatTile
        label={t("medianLabel")}
        value={<CountUp value={stats.rows.length_median_m} format="m" digits={1} />}
        hint={t("medianHint", { max: f.length(stats.rows.length_max_m) })}
        delayMs={180}
      />
    </StatSection>
  );
}
