import { getLocale, getTranslations } from "next-intl/server";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";

export interface AuditRow {
  id: number;
  at: string;
  actor_email: string | null;
  action: string;
  entity: string;
  entity_id: string | null;
  details: Record<string, unknown>;
}

export async function AuditPanel({ rows }: { rows: AuditRow[] }) {
  const t = await getTranslations("superAdmin.audit");
  const locale = await getLocale();
  const dt = new Intl.DateTimeFormat(locale, { dateStyle: "short", timeStyle: "medium" });
  return (
    <Card>
      <Typography variant="h3" sx={{ px: 5, py: 4 }}>
        {t("count", { n: rows.length })}
      </Typography>
      <Box sx={{ overflowX: "auto" }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>{t("colAt")}</TableCell>
              <TableCell>{t("colActor")}</TableCell>
              <TableCell>{t("colAction")}</TableCell>
              <TableCell>{t("colEntity")}</TableCell>
              <TableCell>{t("colDetails")}</TableCell>
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
              <TableRow key={r.id} hover>
                <TableCell sx={{ whiteSpace: "nowrap" }}>{dt.format(new Date(r.at))}</TableCell>
                <TableCell>{r.actor_email ?? "—"}</TableCell>
                <TableCell>
                  <Chip size="small" variant="outlined" label={r.action} />
                </TableCell>
                <TableCell>
                  {r.entity}
                  {r.entity_id && (
                    <Typography component="span" variant="body2" color="text.secondary">
                      {` · ${r.entity_id}`}
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Box component="code" sx={{ fontSize: 12, color: "text.secondary" }}>
                    {Object.keys(r.details ?? {}).length ? JSON.stringify(r.details) : ""}
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    </Card>
  );
}
