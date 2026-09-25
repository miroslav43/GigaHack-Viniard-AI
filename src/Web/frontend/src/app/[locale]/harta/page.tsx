import { Suspense } from "react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { MapExplorer } from "@/components/map/MapExplorer";
import { dataUrl, getRows, getSummary } from "@/lib/data";

export async function generateMetadata({ params }: PageProps<"/[locale]/harta">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("map.title")} · Solemtrix` };
}

export default async function MapPage({ params }: PageProps<"/[locale]/harta">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const [summary, rows] = await Promise.all([getSummary(), getRows()]);
  return (
    <Suspense>
      <MapExplorer summary={summary} rows={rows} dataBase={dataUrl("").replace(/\/$/, "")} />
    </Suspense>
  );
}
