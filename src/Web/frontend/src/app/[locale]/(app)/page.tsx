import { getLocale, getTranslations, setRequestLocale } from "next-intl/server";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import LinearProgress from "@mui/material/LinearProgress";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import MapOutlined from "@mui/icons-material/MapOutlined";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import { LinkButton, LinkChip } from "@/components/common/links";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { KpiCard, KpiGrid } from "@/components/common/KpiCard";
import { MockBanner, sourceLine } from "@/components/common/SourceNote";
import { NoSurvey } from "@/components/common/NoSurvey";
import { getViewer } from "@/lib/viewer";
import { dataUrl, getSummary } from "@/lib/data";
import { makeFormat, type Format } from "@/lib/format";
import { mapPalette, type InterrowCover, type RowStructure } from "@/theme/mapPalette";

export default async function DashboardPage({ params }: PageProps<"/[locale]">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  if (!viewer.uat.surveys.length) return <NoSurvey viewer={viewer} />;
  const [s, t, tc] = await Promise.all([getSummary(), getTranslations("dashboard"), getTranslations()]);
  const f = makeFormat(await getLocale());
  const tt = s.totals;
  const coverage = s.survey.surveyed_area_ha / s.uat.area_ha;
  const saved = s.route.baseline_length_m - s.route.length_m;

  return (
    <Page>
      <PageHeader
        title={tc(`uat.${viewer.uat.key}.name`)}
        subtitle={t("subtitle", { source: await sourceLine(s) })}
        action={
          <Box sx={{ display: "flex", gap: 2 }}>
            <Button variant="outlined" startIcon={<FileDownloadOutlined />} href={dataUrl("measurements.csv")} download>
              measurements.csv
            </Button>
            <LinkButton variant="contained" startIcon={<MapOutlined />} href="/harta">
              {tc("common.openMap")}
            </LinkButton>
          </Box>
        }
      />
      <MockBanner summary={s} />

      <Section title={t("sectionCommune")} first />
      <KpiGrid>
        <KpiCard label={t("communeArea")} value={`${f.num(s.uat.area_ha, 0)} ${f.units.ha}`} hint={t("communeAreaHint", { id: s.uat.osm_relation_id })} />
        <KpiCard
          label={t("surveyedArea")}
          value={`${f.num(s.survey.surveyed_area_ha, 1)} ${f.units.ha}`}
          hint={t("surveyedAreaHint", { pct: f.pct(coverage), tiles: s.survey.tiles_total })}
        />
        <KpiCard label={t("gsd")} value={`${f.num(s.survey.gsd_m * 100, 1)} ${f.units.cmpx}`} hint={s.survey.source} />
      </KpiGrid>

      <Section title={t("sectionPlantings")} />
      <KpiGrid>
        <KpiCard label={t("blocks")} value={f.int(tt.block_count)} tone="primary" hint={t("blocksHint")} />
        <KpiCard label={t("rows")} value={f.int(tt.row_count)} tone="primary" hint={t("rowsHint")} />
        <KpiCard label={t("rowLength")} value={f.length(tt.row_length_m)} hint={t("rowLengthHint")} />
        <KpiCard label={t("canopyArea")} value={f.ha(tt.canopy_area_m2)} hint={t("canopyAreaHint", { area: f.area(tt.canopy_area_m2), plants: f.int(tt.canopy_count) })} />
        <KpiCard label={t("interrowArea")} value={f.ha(tt.interrow_area_m2)} hint={f.area(tt.interrow_area_m2)} />
        <KpiCard
          label={t("disrupted")}
          value={f.pct(tt.row_count ? tt.disrupted_rows / tt.row_count : 0)}
          tone="warning"
          hint={t("disruptedHint", { n: tt.disrupted_rows })}
        />
      </KpiGrid>

      <Section title={t("sectionInspection")} />
      <KpiGrid>
        <KpiCard label={t("targets")} value={f.int(tt.target_count)} hint={t("targetsHint", { waste: tt.waste_count })} />
        <KpiCard label={t("route")} value={f.length(s.route.length_m)} tone="primary" hint={t("routeHint", { min: f.min(s.route.duration_min), speed: s.route.speed_kmh })} />
        {saved > 0 && (
          <KpiCard
            label={t("savings")}
            value={`−${f.length(saved)}`}
            hint={t("savingsHint", { pct: f.pct(saved / s.route.baseline_length_m), min: f.min((saved / 1000 / s.route.speed_kmh) * 60) })}
          />
        )}
      </KpiGrid>

      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "2fr 1fr" }, gap: 4, mt: 7 }}>
        <Card>
          <Box sx={{ px: 5, pt: 5, pb: 2, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Typography variant="h3">{t("blocksTable")}</Typography>
            <LinkButton size="small" href="/blocuri">
              {t("allRows")}
            </LinkButton>
          </Box>
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>{t("colBlock")}</TableCell>
                  <TableCell align="right">{t("colRows")}</TableCell>
                  <TableCell align="right">{t("colLength")}</TableCell>
                  <TableCell align="right">{t("colCanopy")}</TableCell>
                  <TableCell align="right">{t("colInterrow")}</TableCell>
                  <TableCell align="right">{t("colDisrupted")}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {s.blocks.map((b) => (
                  <TableRow key={b.vineyard_id} hover>
                    <TableCell>
                      <LinkChip size="small" label={b.vineyard_id} href={`/harta?bloc=${b.vineyard_id}`} />
                    </TableCell>
                    <TableCell align="right">{b.row_count}</TableCell>
                    <TableCell align="right">{f.length(b.row_length_m)}</TableCell>
                    <TableCell align="right">{f.m2(b.canopy_area_m2)}</TableCell>
                    <TableCell align="right">{f.m2(b.interrow_area_m2)}</TableCell>
                    <TableCell align="right">{b.disrupted_rows}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </Card>

        <Card sx={{ p: 5, display: "flex", flexDirection: "column", gap: 5 }}>
          <Distribution
            f={f}
            title={t("structureDist")}
            items={(Object.keys(mapPalette.row) as RowStructure[]).map((k) => ({
              label: tc(`structure.${k}`),
              value: s.structure_counts[k] ?? 0,
              color: mapPalette.row[k],
            }))}
          />
          <Distribution
            f={f}
            title={t("coverDist")}
            items={(Object.keys(mapPalette.interrow) as InterrowCover[]).map((k) => ({
              label: tc(`cover.${k}`),
              value: s.cover_counts[k] ?? 0,
              color: mapPalette.interrow[k],
            }))}
          />
        </Card>
      </Box>
    </Page>
  );
}

function Section({ title, first = false }: { title: string; first?: boolean }) {
  return (
    <Typography variant="overline" color="text.secondary" component="h2" sx={{ display: "block", mt: first ? 0 : 7, mb: 2 }}>
      {title}
    </Typography>
  );
}

function Distribution({ f, title, items }: { f: Format; title: string; items: { label: string; value: number; color: string }[] }) {
  const total = items.reduce((a, i) => a + i.value, 0) || 1;
  return (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: 3 }}>
        {title}
      </Typography>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
        {items.map((i) => (
          <Box key={i.label}>
            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
              <Typography variant="body2">{i.label}</Typography>
              <Typography variant="body2" color="text.secondary">
                {i.value} · {f.pct(i.value / total)}
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={(i.value / total) * 100}
              sx={{ height: 8, borderRadius: 1, bgcolor: "grey.100", "& .MuiLinearProgress-bar": { bgcolor: i.color, borderRadius: 1 } }}
            />
          </Box>
        ))}
      </Box>
    </Box>
  );
}
