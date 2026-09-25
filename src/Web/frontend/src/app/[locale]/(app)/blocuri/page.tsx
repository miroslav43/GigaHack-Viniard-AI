import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Button from "@mui/material/Button";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { MockBanner, sourceLine } from "@/components/common/SourceNote";
import { RowsTable } from "@/components/tables/RowsTable";
import { NoSurvey } from "@/components/common/NoSurvey";
import { getViewer } from "@/lib/viewer";
import { dataUrl, getRows, getSummary } from "@/lib/data";

export async function generateMetadata({ params }: PageProps<"/[locale]/blocuri">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("blocks.title")} · Solemtrix` };
}

export default async function BlocksPage({ params }: PageProps<"/[locale]/blocuri">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  if (!viewer.uat.surveys.length) return <NoSurvey viewer={viewer} />;
  const [summary, rows, t] = await Promise.all([getSummary(), getRows(), getTranslations("blocks")]);
  return (
    <Page>
      <PageHeader
        title={t("title")}
        subtitle={t("subtitle", { source: await sourceLine(summary) })}
        action={
          <Button variant="contained" startIcon={<FileDownloadOutlined />} href={dataUrl("measurements.csv")} download>
            {t("export")}
          </Button>
        }
      />
      <MockBanner summary={summary} />
      <RowsTable rows={rows} blocks={summary.blocks} />
    </Page>
  );
}
