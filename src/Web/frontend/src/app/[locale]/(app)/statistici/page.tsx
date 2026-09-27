import type { Metadata } from "next";
import type { ComponentType } from "react";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import { Page } from "@/components/common/PageHeader";
import { MockBanner, sourceLine } from "@/components/common/SourceNote";
import { NoSurvey } from "@/components/common/NoSurvey";
import { StatsHero, type HeroFigure } from "@/components/stats/StatsHero";
import type { SectionProps } from "@/components/stats/types";
import { PlantHealthSection } from "@/components/stats/sections/PlantHealthSection";
import { StructureSection } from "@/components/stats/sections/StructureSection";
import { FarmsSection } from "@/components/stats/sections/FarmsSection";
import { RoadsSection } from "@/components/stats/sections/RoadsSection";
import { RouteSection } from "@/components/stats/sections/RouteSection";
import { FlightSection } from "@/components/stats/sections/FlightSection";
import { getViewer } from "@/lib/viewer";
import { SURVEY_ID, getStats, getSummary } from "@/lib/data";
import type { SurveyStats, SurveySummary } from "@/lib/types";

export async function generateMetadata({ params }: PageProps<"/[locale]/statistici">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("stats.title")} · Solemtrix` };
}

// Sections in page order; a section whose data the survey lacks (farms, roads on the mock) is left out,
// and the letters follow what is shown.
const SECTIONS: { id: string; nav: string; Component: ComponentType<SectionProps>; shown: (s: SurveyStats) => boolean }[] = [
  { id: "stare", nav: "health", Component: PlantHealthSection, shown: () => true },
  { id: "structura", nav: "structure", Component: StructureSection, shown: () => true },
  { id: "ferme", nav: "farms", Component: FarmsSection, shown: (s) => s.farms != null },
  { id: "drumuri", nav: "roads", Component: RoadsSection, shown: (s) => s.roads != null },
  { id: "ruta", nav: "route", Component: RouteSection, shown: () => true },
  { id: "zbor", nav: "flight", Component: FlightSection, shown: () => true },
];

const heroFigures = (summary: SurveySummary, stats: SurveyStats): HeroFigure[] => {
  const tt = summary.totals;
  return [
    { key: "canopies", value: tt.canopy_count, format: "int" },
    { key: "rows", value: tt.row_count, format: "int" },
    { key: "rowLength", value: tt.row_length_m, format: "length" },
    { key: "targets", value: stats.targets.total, format: "int" },
    tt.farm_count != null
      ? { key: "farms", value: tt.farm_count, format: "int" }
      : { key: "blocks", value: tt.block_count, format: "int" },
    stats.route.saved_share != null
      ? { key: "routeSaving", value: stats.route.saved_share, format: "pct", prefix: "−" }
      : { key: "route", value: stats.route.length_m, format: "length" },
  ];
};

export default async function StatsPage({ params }: PageProps<"/[locale]/statistici">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  if (!viewer.uat?.surveys.includes(SURVEY_ID)) return <NoSurvey viewer={viewer} />;
  const [summary, stats, t] = await Promise.all([getSummary(), getStats(), getTranslations("stats")]);
  const source = await sourceLine(summary);

  if (!stats) {
    return (
      <Page>
        <Alert severity="info" variant="outlined">
          {t("missing", { command: `pnpm data:stats --survey ${SURVEY_ID}` })}
        </Alert>
      </Page>
    );
  }

  const shown = SECTIONS.filter((s) => s.shown(stats));
  const common = { stats, totals: summary.totals, survey: summary.survey, uatAreaHa: summary.uat.area_ha };
  return (
    <Page>
      <MockBanner summary={summary} />
      <StatsHero title={t("title")} subtitle={t("subtitle", { source })} figures={heroFigures(summary, stats)} />
      <Box component="nav" aria-label={t("jump")} sx={{ display: "flex", flexWrap: "wrap", gap: 2, mt: 5 }}>
        {shown.map((s, i) => (
          <Chip key={s.id} component="a" href={`#${s.id}`} clickable variant="outlined" label={`${String.fromCharCode(65 + i)} · ${t(`nav.${s.nav}`)}`} />
        ))}
      </Box>
      {shown.map(({ id, Component }, i) => (
        <Component key={id} letter={String.fromCharCode(65 + i)} {...common} />
      ))}
    </Page>
  );
}
