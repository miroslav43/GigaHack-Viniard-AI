import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";
import OpenInNew from "@mui/icons-material/OpenInNew";
import { LinkChip } from "@/components/common/links";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { DIAGRAMS, pickDiagram } from "@/lib/diagrams";

export async function generateMetadata({ params }: PageProps<"/[locale]/arhitectura">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("architecture.title")} · Solemtrix` };
}

export default async function ArchitecturePage({ params, searchParams }: PageProps<"/[locale]/arhitectura">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const current = pickDiagram((await searchParams).d);
  const t = await getTranslations("architecture");

  return (
    <Page>
      <PageHeader
        title={t("title")}
        subtitle={t("subtitle")}
        action={
          <Button variant="outlined" startIcon={<OpenInNew />} href={current.file} target="_blank" rel="noopener">
            {t("openNew")}
          </Button>
        }
      />
      <Box component="nav" aria-label={t("title")} sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 4 }}>
        {DIAGRAMS.map((d, i) => (
          <LinkChip
            key={d.slug}
            href={`/arhitectura?d=${d.slug}`}
            label={`${i + 1}. ${t(`d.${d.slug}.title`)}`}
            color={d.slug === current.slug ? "primary" : "default"}
            variant={d.slug === current.slug ? "filled" : "outlined"}
          />
        ))}
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, maxWidth: "80ch" }}>
        {t(`d.${current.slug}.caption`)}
      </Typography>
      <Card variant="outlined" sx={{ overflow: "hidden" }}>
        <Box
          component="iframe"
          key={current.slug}
          src={current.file}
          title={t(`d.${current.slug}.title`)}
          sx={{ display: "block", width: "100%", height: { xs: "75vh", md: "calc(100vh - 260px)" }, minHeight: 560, border: 0 }}
        />
      </Card>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 2 }}>
        {t("note")}
      </Typography>
    </Page>
  );
}
