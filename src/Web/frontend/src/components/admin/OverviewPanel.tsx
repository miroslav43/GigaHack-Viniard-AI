"use client";

import { useLocale, useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import type { Geometry } from "geojson";
import { KpiCard, KpiGrid } from "@/components/common/KpiCard";
import { Link } from "@/i18n/routing";
import { useFormat } from "@/lib/useFormat";
import { mapPalette } from "@/theme/mapPalette";
import { OverviewMap } from "./OverviewMap";
import type { Role } from "./types";

export interface OverviewData {
  uats: {
    key: string;
    name: string;
    district: string | null;
    country: string;
    active: boolean;
    area_ha: number;
    geofence: Geometry;
    surveys: { id: string; overlap_ha: number }[];
    users: number | null;
  }[];
  surveys: { id: string; name: string; captured_at: string | null; area_ha: number; footprint: Geometry }[];
  roles: Partial<Record<Role, number>> | null;
  unassigned: number | null;
  totalUsers: number | null;
  activity: { id: number; at: string; actor_email: string | null; action: string; entity: string; entity_id: string | null; details: Record<string, unknown> }[];
}

function Swatch({ color, dashed }: { color: string; dashed?: boolean }) {
  return <Box sx={{ width: 14, height: 14, borderRadius: 0.5, bgcolor: color, opacity: 0.6, border: dashed ? "1.5px dashed" : 0, borderColor: color }} />;
}

export function OverviewPanel({ data }: { data: OverviewData }) {
  const t = useTranslations("superAdmin");
  const tr = useTranslations("auth.roles");
  const f = useFormat();
  const locale = useLocale();
  // small shares (e.g. 0,45 %) would round to 0 % — keep one decimal below 10 %
  const pctSmall = (v: number) => (v > 0 && v < 0.1 ? `${f.num(v * 100, 1)} %` : f.pct(v));
  const active = data.uats.filter((u) => u.active);
  const totalHa = active.reduce((s, u) => s + u.area_ha, 0);
  const coveredHa = active.reduce((s, u) => s + u.surveys.reduce((a, x) => a + x.overlap_ha, 0), 0);
  const withSurvey = active.filter((u) => u.surveys.length > 0).length;
  const surveyHa = data.surveys.reduce((s, x) => s + x.area_ha, 0);
  const dt = (iso: string) => new Intl.DateTimeFormat(locale, { dateStyle: "short", timeStyle: "short" }).format(new Date(iso));

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <KpiGrid min={190}>
        <KpiCard label={t("overview.uats")} value={f.int(active.length)} tone="primary" hint={t("overview.uatsHint", { total: data.uats.length, surveyed: withSurvey })} />
        <KpiCard label={t("overview.area")} value={`${f.num(totalHa, 0)} ${f.units.ha}`} hint={t("overview.areaHint")} />
        <KpiCard label={t("overview.surveys")} value={f.int(data.surveys.length)} hint={`${f.num(surveyHa, 1)} ${f.units.ha}`} />
        <KpiCard label={t("overview.coverage")} value={pctSmall(totalHa ? coveredHa / totalHa : 0)} tone="warning" hint={t("overview.coverageHint", { ha: f.num(coveredHa, 1) })} />
        <KpiCard
          label={t("overview.accounts")}
          value={data.totalUsers === null ? "—" : f.int(data.totalUsers)}
          hint={data.totalUsers === null ? t("overview.accountsNoKey") : t("overview.accountsHint", { n: data.unassigned ?? 0 })}
        />
      </KpiGrid>

      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "2fr 1fr" }, gap: 5 }}>
        <Card sx={{ overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
            <Typography variant="h3" sx={{ flex: 1 }}>
              {t("overview.map")}
            </Typography>
            {[
              [mapPalette.geofence, t("overview.legendSurveyed")],
              [mapPalette.block, t("overview.legendEnrolled")],
              [mapPalette.passage, t("overview.legendInactive")],
              [mapPalette.canopyFill, t("overview.legendSurvey")],
            ].map(([c, l]) => (
              <Box key={l} sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                <Swatch color={c} />
                <Typography variant="caption">{l}</Typography>
              </Box>
            ))}
          </Box>
          <Box sx={{ flex: 1, minHeight: 520, borderTop: 1, borderColor: "divider" }}>
            <OverviewMap
              uats={data.uats.map((u) => ({ key: u.key, name: u.name, active: u.active, hasSurvey: u.surveys.length > 0, geofence: u.geofence }))}
              footprints={data.surveys.map((s) => s.footprint)}
            />
          </Box>
        </Card>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <Card sx={{ p: 5 }}>
            <Typography variant="h3" sx={{ mb: 3 }}>
              {t("overview.roles")}
            </Typography>
            {data.roles === null ? (
              <Typography variant="body2" color="text.secondary">
                {t("overview.accountsNoKey")}
              </Typography>
            ) : (
              (["platform_admin", "uat_admin", "inspector", "viewer"] as Role[]).map((r) => (
                <Box key={r} sx={{ display: "flex", justifyContent: "space-between", py: 1 }}>
                  <Typography variant="body2">{tr(r)}</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {data.roles?.[r] ?? 0}
                  </Typography>
                </Box>
              ))
            )}
          </Card>
          <Card sx={{ p: 5, flex: 1 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 2 }}>
              <Typography variant="h3">{t("overview.activity")}</Typography>
              <Typography component={Link} href="/super-admin?tab=audit" variant="body2" color="primary" sx={{ textDecoration: "none" }}>
                {t("overview.allActivity")}
              </Typography>
            </Box>
            {data.activity.length === 0 && (
              <Typography variant="body2" color="text.secondary">
                {t("audit.empty")}
              </Typography>
            )}
            {data.activity.map((a) => (
              <Box key={a.id} sx={{ py: 1.25, borderTop: 1, borderColor: "divider" }}>
                <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
                  <Chip size="small" variant="outlined" label={a.action} />
                  <Typography variant="body2" noWrap sx={{ minWidth: 0 }}>
                    {(a.details?.email as string | undefined) ?? (a.details?.name as string | undefined) ?? a.entity_id ?? ""}
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {dt(a.at)} · {a.actor_email ?? "—"}
                </Typography>
              </Box>
            ))}
          </Card>
        </Box>
      </Box>

      <Card>
        <Box sx={{ px: 5, py: 4, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <Typography variant="h3">{t("overview.byUat")}</Typography>
          <Typography component={Link} href="/super-admin?tab=uat" variant="body2" color="primary" sx={{ textDecoration: "none" }}>
            {t("overview.manage")}
          </Typography>
        </Box>
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{t("uat.colName")}</TableCell>
                <TableCell>{t("uat.colDistrict")}</TableCell>
                <TableCell align="right">{t("uat.colArea")}</TableCell>
                <TableCell>{t("uat.colSurveys")}</TableCell>
                <TableCell align="right">{t("overview.coverage")}</TableCell>
                <TableCell align="right">{t("uat.colUsers")}</TableCell>
                <TableCell>{t("uat.colActive")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.uats.map((u) => {
                const covered = u.surveys.reduce((a, x) => a + x.overlap_ha, 0);
                return (
                  <TableRow key={u.key} hover>
                    <TableCell sx={{ fontWeight: 600 }}>{u.name}</TableCell>
                    <TableCell>{u.district ?? "—"}</TableCell>
                    <TableCell align="right">{`${f.num(u.area_ha, 0)} ${f.units.ha}`}</TableCell>
                    <TableCell>{u.surveys.length ? u.surveys.map((s) => s.id).join(", ") : "—"}</TableCell>
                    <TableCell align="right">{u.surveys.length ? `${f.num(covered, 1)} ${f.units.ha} · ${pctSmall(covered / u.area_ha)}` : "—"}</TableCell>
                    <TableCell align="right">{u.users ?? "—"}</TableCell>
                    <TableCell>
                      <Chip size="small" variant="outlined" color={u.active ? "success" : "default"} label={u.active ? t("common.yes") : t("common.no")} />
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </Box>
      </Card>
    </Box>
  );
}
