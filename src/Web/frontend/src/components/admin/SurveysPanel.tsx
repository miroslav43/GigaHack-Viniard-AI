"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutline from "@mui/icons-material/DeleteOutlined";
import CloudUploadOutlined from "@mui/icons-material/CloudUploadOutlined";
import { useFormat } from "@/lib/useFormat";
import { deleteSurvey, registerSurvey } from "@/app/[locale]/(app)/super-admin/actions";
import { ConfirmDialog, useAdminAction } from "./common";
import type { AdminSurvey, LocalBundle } from "./types";

export function SurveysPanel({ surveys, bundles }: { surveys: AdminSurvey[]; bundles: LocalBundle[] }) {
  const t = useTranslations("superAdmin");
  const f = useFormat();
  const { run, pending, snackbar } = useAdminAction();
  const [toDelete, setToDelete] = useState<string | null>(null);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <Card>
        <Typography variant="h3" sx={{ px: 5, py: 4 }}>
          {t("surveys.count", { n: surveys.length })}
        </Typography>
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{t("surveys.colId")}</TableCell>
                <TableCell>{t("surveys.colName")}</TableCell>
                <TableCell>{t("surveys.colDate")}</TableCell>
                <TableCell align="right">{t("surveys.colArea")}</TableCell>
                <TableCell>{t("surveys.colUats")}</TableCell>
                <TableCell>{t("surveys.colData")}</TableCell>
                <TableCell align="right">{t("common.actions")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {surveys.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7}>
                    <Typography color="text.secondary">{t("surveys.empty")}</Typography>
                  </TableCell>
                </TableRow>
              )}
              {surveys.map((s) => (
                <TableRow key={s.id} hover>
                  <TableCell>
                    <code>{s.id}</code>
                  </TableCell>
                  <TableCell>{s.name}</TableCell>
                  <TableCell>{s.captured_at ? f.date(s.captured_at) : t("common.none")}</TableCell>
                  <TableCell align="right">{`${f.num(s.area_ha, 1)} ${f.units.ha}`}</TableCell>
                  <TableCell>
                    {s.uats.length === 0
                      ? t("common.none")
                      : s.uats.map((u) => <Chip key={u.key} size="small" label={`${u.name} · ${f.num(u.overlap_ha, 1)} ${f.units.ha}`} sx={{ mr: 1 }} />)}
                  </TableCell>
                  <TableCell>
                    <code>{s.data_path}</code>
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title={t("common.delete")}>
                      <IconButton size="small" color="error" onClick={() => setToDelete(s.id)} disabled={pending}>
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

      <Card sx={{ p: 5 }}>
        <Typography variant="h3">{t("surveys.local")}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 3 }}>
          {t("surveys.localHint")}
        </Typography>
        {bundles.map((b) => (
          <Box key={b.id} sx={{ display: "flex", alignItems: "center", gap: 3, py: 1.5, borderTop: 1, borderColor: "divider" }}>
            <code>{b.id}</code>
            <Typography variant="body2" sx={{ flex: 1 }}>
              {b.name}
            </Typography>
            <Chip size="small" variant="outlined" color={b.registered ? "success" : "default"} label={b.registered ? t("surveys.registered") : t("surveys.notRegistered")} />
            <Button size="small" startIcon={<CloudUploadOutlined />} disabled={pending} onClick={() => run(() => registerSurvey(b.id))}>
              {b.registered ? t("surveys.refresh") : t("surveys.register")}
            </Button>
          </Box>
        ))}
      </Card>

      <ConfirmDialog
        open={toDelete !== null}
        text={toDelete ? t("surveys.deleteConfirm", { id: toDelete }) : ""}
        onConfirm={() => toDelete && run(() => deleteSurvey(toDelete))}
        onClose={() => setToDelete(null)}
      />
      {snackbar}
    </Box>
  );
}
