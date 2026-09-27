import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { AnalizaWorkbench } from "@/components/analiza/AnalizaWorkbench";

export async function generateMetadata({ params }: PageProps<"/[locale]/analiza">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("analiza.title")} · Solemtrix` };
}

/** Live tile analysis: the real pipeline (`vineyard demo tile`, src/AI) on one uploaded Sireț3 tile. */
export default async function AnalysisPage({ params }: PageProps<"/[locale]/analiza">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("analiza");
  return (
    <Page>
      <PageHeader title={t("title")} subtitle={t("subtitle")} />
      <AnalizaWorkbench />
    </Page>
  );
}
