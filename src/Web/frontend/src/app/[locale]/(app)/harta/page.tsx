import { Suspense } from "react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { MapExplorer } from "@/components/map/MapExplorer";
import { NoSurvey } from "@/components/common/NoSurvey";
import { getViewer } from "@/lib/viewer";
import { dataUrl, getRows, getSummary } from "@/lib/data";

export async function generateMetadata({ params }: PageProps<"/[locale]/harta">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("map.title")} · Solemtrix` };
}

export default async function MapPage({ params }: PageProps<"/[locale]/harta">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  if (!viewer.uat.surveys.length) return <NoSurvey viewer={viewer} fullHeight />;
  const [summary, rows] = await Promise.all([getSummary(), getRows()]);
  return (
    <Suspense>
      <MapExplorer summary={summary} rows={rows} dataBase={dataUrl("").replace(/\/$/, "")} geofenceUrl={viewer.uat.geofenceUrl} />
    </Suspense>
  );
}
