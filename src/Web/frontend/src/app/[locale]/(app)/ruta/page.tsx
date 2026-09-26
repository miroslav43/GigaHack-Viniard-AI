import type { Metadata } from "next";
import { getLocale, getTranslations, setRequestLocale } from "next-intl/server";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import MapOutlined from "@mui/icons-material/MapOutlined";
import { LinkButton, LinkChip } from "@/components/common/links";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { KpiCard, KpiGrid } from "@/components/common/KpiCard";
import { MockBanner, sourceLine } from "@/components/common/SourceNote";
import { NoSurvey } from "@/components/common/NoSurvey";
import { getViewer } from "@/lib/viewer";
import { SURVEY_ID, dataUrl, getSummary, getTargets } from "@/lib/data";
import { makeFormat } from "@/lib/format";

export async function generateMetadata({ params }: PageProps<"/[locale]/ruta">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("route.title")} · Solemtrix` };
}

export default async function RoutePage({ params }: PageProps<"/[locale]/ruta">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  if (!viewer.uat?.surveys.includes(SURVEY_ID)) return <NoSurvey viewer={viewer} />;
  const [s, targets, t, tc] = await Promise.all([getSummary(), getTargets(), getTranslations("route"), getTranslations()]);
  const f = makeFormat(await getLocale());
  const r = s.route;
  const saved = r.baseline_length_m - r.length_m;
  // only the stops the route visits, in walking order (route_order null = left out by the route planner)
  const ordered = targets.features
    .filter((x) => x.properties.route_order != null)
    .sort((a, b) => (a.properties.route_order ?? 0) - (b.properties.route_order ?? 0));

  return (
    <Page>
      <PageHeader
        title={t("title")}
        subtitle={t("subtitle", { source: await sourceLine(s) })}
        action={
          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            <Button variant="outlined" startIcon={<FileDownloadOutlined />} href={dataUrl("route.gpx")} download>
              GPX
            </Button>
            <Button variant="outlined" startIcon={<FileDownloadOutlined />} href={dataUrl("route_EPSG32635.geojson")} download="route.geojson">
              GeoJSON (EPSG:32635)
            </Button>
            <LinkButton variant="contained" startIcon={<MapOutlined />} href="/harta">
              {tc("common.seeOnMap")}
            </LinkButton>
          </Box>
        }
      />
      <MockBanner summary={s} />
      <KpiGrid>
        <KpiCard label={t("length")} value={f.length(r.length_m)} tone="primary" hint={t("lengthHint")} />
        <KpiCard label={t("duration")} value={f.min(r.duration_min)} hint={t("durationHint", { speed: r.speed_kmh })} />
        <KpiCard label={t("targets")} value={f.int(targets.features.length)} hint={t("targetsHint")} />
        {saved > 0 && (
          <KpiCard
            label={tc("dashboard.savings")}
            value={`−${f.length(saved)}`}
            hint={tc("dashboard.savingsHint", { pct: f.pct(saved / r.baseline_length_m), min: f.min((saved / 1000 / r.speed_kmh) * 60) })}
          />
        )}
      </KpiGrid>

      <Card sx={{ mt: 6 }}>
        <Box sx={{ px: 5, pt: 5, pb: 2, display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 3, flexWrap: "wrap" }}>
          <Typography variant="h3">{t("order")}</Typography>
          <Typography variant="body2" color="text.secondary">
            {t("onRoute", { n: f.int(ordered.length), total: f.int(targets.features.length) })}
          </Typography>
        </Box>
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>#</TableCell>
                <TableCell>{t("colTarget")}</TableCell>
                <TableCell>{t("colType")}</TableCell>
                <TableCell>{t("colBlock")}</TableCell>
                <TableCell>{t("colRow")}</TableCell>
                <TableCell align="right">{t("colGap")}</TableCell>
                <TableCell>{t("colStatus")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {ordered.map(({ properties: p }) => (
                <TableRow key={p.target_id} hover>
                  <TableCell>{p.route_order}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{p.target_id}</TableCell>
                  <TableCell>{tc.has(`targetType.${p.type}`) ? tc(`targetType.${p.type}`) : p.type}</TableCell>
                  <TableCell>{p.vineyard_id}</TableCell>
                  <TableCell>{p.row_id ? <LinkChip size="small" label={p.row_id} href={`/harta?rand=${p.row_id}`} /> : "—"}</TableCell>
                  <TableCell align="right">{p.gap_length_m != null ? f.m(p.gap_length_m) : "—"}</TableCell>
                  <TableCell>
                    <Chip size="small" variant="outlined" label={t("statusOpen")} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      </Card>
    </Page>
  );
}
