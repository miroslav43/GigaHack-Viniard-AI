"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import Link from "@mui/material/Link";
import Switch from "@mui/material/Switch";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import AddLocationAltOutlined from "@mui/icons-material/AddLocationAltOutlined";
import DeleteOutline from "@mui/icons-material/DeleteOutlined";
import PersonAddOutlined from "@mui/icons-material/PersonAddOutlined";
import { Link as NextIntlLink } from "@/i18n/routing";
import { BoundaryMap } from "@/components/map/BoundaryMap";
import { useFormat } from "@/lib/useFormat";
import { deleteUat, setUatActive } from "@/app/[locale]/(admin)/super-admin/actions";
import { ConfirmDialog, useAdminAction } from "./common";
import { EnrollUatDialog } from "./EnrollUatDialog";
import type { AdminUat } from "./types";

export function UatPanel({ uats }: { uats: AdminUat[] }) {
  const t = useTranslations("superAdmin");
  const f = useFormat();
  const { run, pending, snackbar } = useAdminAction();
  const [enrolling, setEnrolling] = useState(false);
  const [toDelete, setToDelete] = useState<AdminUat | null>(null);
  const geometries = useMemo(() => uats.map((u) => u.geofence), [uats]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <Card>
        <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 3, flexWrap: "wrap" }}>
          <Typography variant="h3">{t("uat.count", { n: uats.length })}</Typography>
          <Button variant="contained" startIcon={<AddLocationAltOutlined />} onClick={() => setEnrolling(true)}>
            {t("uat.enroll")}
          </Button>
        </Box>
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{t("uat.colName")}</TableCell>
                <TableCell>{t("uat.colKey")}</TableCell>
                <TableCell>{t("uat.colDistrict")}</TableCell>
                <TableCell>{t("uat.colCountry")}</TableCell>
                <TableCell>{t("uat.colOsm")}</TableCell>
                <TableCell align="right">{t("uat.colArea")}</TableCell>
                <TableCell>{t("uat.colSurveys")}</TableCell>
                <TableCell align="right">{t("uat.colUsers")}</TableCell>
                <TableCell>{t("uat.colActive")}</TableCell>
                <TableCell>{t("uat.colCreated")}</TableCell>
                <TableCell align="right">{t("common.actions")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {uats.length === 0 && (
                <TableRow>
                  <TableCell colSpan={11}>
                    <Typography color="text.secondary">{t("uat.empty")}</Typography>
                  </TableCell>
                </TableRow>
              )}
              {uats.map((u) => (
                <TableRow key={u.key} hover>
                  <TableCell sx={{ fontWeight: 600 }}>{u.name}</TableCell>
                  <TableCell>
                    <code>{u.key}</code>
                  </TableCell>
                  <TableCell>{u.district ?? t("common.none")}</TableCell>
                  <TableCell>{u.country}</TableCell>
                  <TableCell>
                    {u.osm_relation_id ? (
                      <Link href={`https://www.openstreetmap.org/relation/${u.osm_relation_id}`} target="_blank" rel="noreferrer">
                        {u.osm_relation_id}
                      </Link>
                    ) : (
                      t("common.none")
                    )}
                  </TableCell>
                  <TableCell align="right">{`${f.num(u.area_ha, 0)} ${f.units.ha}`}</TableCell>
                  <TableCell>
                    {u.surveys.length === 0
                      ? t("common.none")
                      : u.surveys.map((s) => <Chip key={s.id} size="small" label={`${s.id} · ${f.num(s.overlap_ha, 1)} ${f.units.ha}`} sx={{ mr: 1 }} />)}
                  </TableCell>
                  <TableCell align="right">{u.users ?? t("common.none")}</TableCell>
                  <TableCell>
                    <Switch size="small" checked={u.active} disabled={pending} onChange={(e) => run(() => setUatActive(u.key, e.target.checked))} />
                  </TableCell>
                  <TableCell>{f.date(u.created_at)}</TableCell>
                  <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                    <Tooltip title={t("uat.addAdmin")}>
                      <IconButton size="small" component={NextIntlLink} href={`/super-admin?tab=users&new=uat_admin&uat=${u.key}`}>
                        <PersonAddOutlined fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={t("common.delete")}>
                      <IconButton size="small" color="error" onClick={() => setToDelete(u)} disabled={pending}>
                        <DeleteOutline fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      </Card>

      <Card sx={{ overflow: "hidden" }}>
        <Typography variant="h3" sx={{ px: 5, py: 4 }}>
          {t("uat.mapTitle")}
        </Typography>
        <Box sx={{ height: 420, borderTop: 1, borderColor: "divider" }}>
          <BoundaryMap geofence={null} others={geometries} mask={false} />
        </Box>
      </Card>

      <EnrollUatDialog open={enrolling} onClose={() => setEnrolling(false)} />
      <ConfirmDialog
        open={toDelete !== null}
        text={toDelete ? t("uat.deleteConfirm", { name: toDelete.name }) : ""}
        onConfirm={() => toDelete && run(() => deleteUat(toDelete.key))}
        onClose={() => setToDelete(null)}
      />
      {snackbar}
    </Box>
  );
}
