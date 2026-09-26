"use client";

import { useLocale, useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import MenuItem from "@mui/material/MenuItem";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { setLeadStatus } from "@/app/[locale]/(admin)/super-admin/actions";
import { useAdminAction } from "./common";
import { LEAD_STATUSES, type LeadRow } from "./types";

/** Requests from the presentation site ("request a pilot"), newest first, with their follow-up status. */
export function LeadsPanel({ rows }: { rows: LeadRow[] }) {
  const t = useTranslations("superAdmin.leads");
  const tt = useTranslations("landing.contact.types");
  const locale = useLocale();
  const { run, pending, snackbar } = useAdminAction();
  const dt = new Intl.DateTimeFormat(locale, { dateStyle: "short", timeStyle: "short" });
  const fresh = rows.filter((r) => r.status === "new").length;

  return (
    <Card>
      <Typography variant="h3" sx={{ px: 5, py: 4 }}>
        {t("count", { n: rows.length, fresh })}
      </Typography>
      <Box sx={{ overflowX: "auto" }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>{t("colAt")}</TableCell>
              <TableCell>{t("colContact")}</TableCell>
              <TableCell>{t("colInstitution")}</TableCell>
              <TableCell>{t("colMessage")}</TableCell>
              <TableCell>{t("colStatus")}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={5}>
                  <Typography color="text.secondary">{t("empty")}</Typography>
                </TableCell>
              </TableRow>
            )}
            {rows.map((r) => (
              <TableRow key={r.id} hover sx={{ verticalAlign: "top", bgcolor: r.status === "new" ? "action.hover" : undefined }}>
                <TableCell sx={{ whiteSpace: "nowrap" }}>{dt.format(new Date(r.created_at))}</TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {r.name}
                  </Typography>
                  {r.position && (
                    <Typography variant="caption" color="text.secondary" component="p">
                      {r.position}
                    </Typography>
                  )}
                  <Typography variant="body2" component="a" href={`mailto:${r.email}`} sx={{ color: "primary.main", display: "block" }}>
                    {r.email}
                  </Typography>
                  {r.phone && (
                    <Typography variant="body2" component="a" href={`tel:${r.phone}`} sx={{ color: "text.secondary", display: "block" }}>
                      {r.phone}
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Typography variant="body2">{r.institution}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {tt.has(r.institution_type) ? tt(r.institution_type) : r.institution_type} · {r.locale.toUpperCase()}
                  </Typography>
                </TableCell>
                <TableCell sx={{ maxWidth: 420 }}>
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {r.message ?? "—"}
                  </Typography>
                </TableCell>
                <TableCell>
                  <TextField
                    select
                    size="small"
                    value={r.status}
                    disabled={pending}
                    onChange={(e) => run(() => setLeadStatus(r.id, e.target.value))}
                    aria-label={t("colStatus")}
                    sx={{ minWidth: 150 }}
                  >
                    {LEAD_STATUSES.map((s) => (
                      <MenuItem key={s} value={s}>
                        {t(`status.${s}`)}
                      </MenuItem>
                    ))}
                  </TextField>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
      {snackbar}
    </Card>
  );
}
