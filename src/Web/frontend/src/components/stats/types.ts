import type { SurveyStats, SurveySummary } from "@/lib/types";

/** What every section of /statistici receives from the page (server → client, kept small). */
export interface SectionProps {
  /** section letter shown in the overline (A, B, …), set by the page from the order of the sections */
  letter: string;
  stats: SurveyStats;
  totals: SurveySummary["totals"];
  survey: SurveySummary["survey"];
  uatAreaHa: number;
}

/** Bin labels of a histogram over fixed edges: "0–10 m", …, "≥ 300 m" (the last bin is open-ended). */
export const binLabels = (edges: number[], num: (v: number) => string, unit: string): string[] =>
  edges.map((e, i) => (i === edges.length - 1 ? `≥ ${num(e)} ${unit}` : `${num(e)}–${num(edges[i + 1])} ${unit}`));

/** Donut legends sit under the chart, so the donut keeps its full width on a phone. */
export const DONUT_LEGEND = {
  legend: { direction: "horizontal", position: { vertical: "bottom", horizontal: "center" } },
} as const;
