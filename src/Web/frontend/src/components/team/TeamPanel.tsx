"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import PersonAddOutlined from "@mui/icons-material/PersonAddOutlined";
import EditOutlined from "@mui/icons-material/EditOutlined";
import KeyOutlined from "@mui/icons-material/KeyOutlined";
import BlockOutlined from "@mui/icons-material/BlockOutlined";
import CheckCircleOutlined from "@mui/icons-material/CheckCircleOutlined";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import ForwardToInboxOutlined from "@mui/icons-material/ForwardToInboxOutlined";
import { KpiCard, KpiGrid } from "@/components/common/KpiCard";
import { PasswordField, generatePassword } from "@/components/common/PasswordField";
import { ConfirmDialog, useAdminAction } from "@/components/admin/common";
import { useFormat } from "@/lib/useFormat";
import { deleteMember, inviteMember, resendInvite, resetMemberPassword, setMemberBanned, updateMember } from "@/app/[locale]/(app)/echipa/actions";

export interface TeamRow {
  id: string;
  email: string;
  name: string | null;
  role: string | null;
  banned: boolean;
  invited: boolean;
  last_sign_in_at: string | null;
  isMe: boolean;
  tasks: { open: number; in_progress: number; done: number };
}

const ROLES = ["inspector", "viewer"] as const;

export function TeamPanel({ uatName, rows, doneWeek, openTotal }: { uatName: string; rows: TeamRow[]; doneWeek: number; openTotal: number }) {
  const t = useTranslations("team");
  const tr = useTranslations("auth.roles");
  const tc = useTranslations("superAdmin.common");
  const f = useFormat();
  const { run, pending, snackbar } = useAdminAction();
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<TeamRow | null>(null);
  const [resetting, setResetting] = useState<TeamRow | null>(null);
  const [deleting, setDeleting] = useState<TeamRow | null>(null);
  const [draft, setDraft] = useState({ email: "", fullName: "", password: "", role: "inspector" as string });

  const team = rows.filter((r) => !r.isMe);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <KpiGrid min={180}>
        <KpiCard label={t("kpiMembers")} value={f.int(team.length)} tone="primary" />
        <KpiCard label={t("kpiInspectors")} value={f.int(team.filter((r) => r.role === "inspector").length)} />
        <KpiCard label={t("kpiOpen")} value={f.int(openTotal)} tone="warning" />
        <KpiCard label={t("kpiDoneWeek")} value={f.int(doneWeek)} />
      </KpiGrid>

      <Card>
        <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 3, flexWrap: "wrap" }}>
          <Typography variant="h3">{uatName}</Typography>
          <Button
            variant="contained"
            startIcon={<PersonAddOutlined />}
            onClick={() => {
              setDraft({ email: "", fullName: "", password: "", role: "inspector" });
              setCreating(true);
            }}
          >
            {t("add")}
          </Button>
        </Box>
        {team.length === 0 && (
          <Alert severity="info" sx={{ mx: 5, mb: 4 }}>
            {t("empty")}
          </Alert>
        )}
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{t("colMember")}</TableCell>
                <TableCell>{t("colRole")}</TableCell>
                <TableCell>{t("colStatus")}</TableCell>
                <TableCell align="right">{t("colOpen")}</TableCell>
                <TableCell align="right">{t("colInProgress")}</TableCell>
                <TableCell align="right">{t("colDone")}</TableCell>
                <TableCell>{t("colLastSignIn")}</TableCell>
                <TableCell align="right">{t("colActions")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.id} hover>
                  <TableCell>
                    <Typography variant="body2" component="div" sx={{ fontWeight: 600 }}>
                      {r.name ?? r.email}
                      {r.isMe && <Chip size="small" label={t("you")} sx={{ ml: 1 }} />}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {r.email}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip size="small" variant="outlined" color={r.role === "uat_admin" ? "primary" : "default"} label={r.role ? tr(r.role) : "—"} />
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      variant="outlined"
                      color={r.banned ? "error" : r.invited ? "warning" : "success"}
                      label={r.banned ? t("disabled") : r.invited ? t("invited") : t("active")}
                    />
                  </TableCell>
                  <TableCell align="right">{r.tasks.open}</TableCell>
                  <TableCell align="right">{r.tasks.in_progress}</TableCell>
                  <TableCell align="right">{r.tasks.done}</TableCell>
                  <TableCell>{r.last_sign_in_at ? f.date(r.last_sign_in_at) : "—"}</TableCell>
                  <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                    {!r.isMe && r.role !== "uat_admin" && (
                      <>
                        {r.invited && (
                          <Tooltip title={t("resendInvite")}>
                            <IconButton
                              size="small"
                              disabled={pending}
                              onClick={() => run(() => resendInvite(r.id), undefined, (email) => t("inviteSent", { email }))}
                            >
                              <ForwardToInboxOutlined fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        <Tooltip title={t("edit")}>
                          <IconButton
                            size="small"
                            disabled={pending}
                            onClick={() => {
                              setDraft({ email: r.email, fullName: r.name ?? "", password: "", role: r.role ?? "inspector" });
                              setEditing(r);
                            }}
                          >
                            <EditOutlined fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title={t("resetPassword")}>
                          <IconButton
                            size="small"
                            disabled={pending}
                            onClick={() => {
                              setDraft((d) => ({ ...d, password: generatePassword() }));
                              setResetting(r);
                            }}
                          >
                            <KeyOutlined fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title={r.banned ? t("enable") : t("disable")}>
                          <IconButton size="small" disabled={pending} onClick={() => run(() => setMemberBanned(r.id, !r.banned))}>
                            {r.banned ? <CheckCircleOutlined fontSize="small" /> : <BlockOutlined fontSize="small" />}
                          </IconButton>
                        </Tooltip>
                        <Tooltip title={t("delete")}>
                          <IconButton size="small" color="error" disabled={pending} onClick={() => setDeleting(r)}>
                            <DeleteOutlined fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      </Card>

      {/* create */}
      <Dialog open={creating} onClose={() => setCreating(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{t("dialogCreate")}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <TextField label={t("fieldEmail")} type="email" value={draft.email} onChange={(e) => setDraft({ ...draft, email: e.target.value })} fullWidth autoFocus />
          <TextField label={t("fieldName")} value={draft.fullName} onChange={(e) => setDraft({ ...draft, fullName: e.target.value })} fullWidth />
          <TextField select label={t("fieldRole")} value={draft.role} onChange={(e) => setDraft({ ...draft, role: e.target.value })} fullWidth>
            {ROLES.map((r) => (
              <MenuItem key={r} value={r}>
                {tr(r)}
              </MenuItem>
            ))}
          </TextField>
          <Typography variant="caption" color="text.secondary">
            {t("roleHelp", { uat: uatName })} {t("inviteNote")}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreating(false)}>{tc("cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending || !draft.email}
            onClick={() =>
              run(
                () => inviteMember({ email: draft.email, fullName: draft.fullName, role: draft.role }),
                () => setCreating(false),
                (email) => t("inviteSent", { email }),
              )
            }
          >
            {t("sendInvite")}
          </Button>
        </DialogActions>
      </Dialog>

      {/* edit */}
      <Dialog open={editing !== null} onClose={() => setEditing(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{editing && t("dialogEdit", { email: editing.email })}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <TextField label={t("fieldName")} value={draft.fullName} onChange={(e) => setDraft({ ...draft, fullName: e.target.value })} fullWidth />
          <TextField select label={t("fieldRole")} value={draft.role} onChange={(e) => setDraft({ ...draft, role: e.target.value })} fullWidth>
            {ROLES.map((r) => (
              <MenuItem key={r} value={r}>
                {tr(r)}
              </MenuItem>
            ))}
          </TextField>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditing(null)}>{tc("cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending}
            onClick={() => editing && run(() => updateMember({ id: editing.id, fullName: draft.fullName, role: draft.role }), () => setEditing(null))}
          >
            {tc("save")}
          </Button>
        </DialogActions>
      </Dialog>

      {/* reset password */}
      <Dialog open={resetting !== null} onClose={() => setResetting(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{resetting && t("dialogReset", { email: resetting.email })}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <PasswordField value={draft.password} onChange={(password) => setDraft({ ...draft, password })} label={t("fieldPassword")} generateLabel={t("generate")} />
          <Typography variant="caption" color="text.secondary">
            {t("passwordNote")}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetting(null)}>{tc("cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending || draft.password.length < 10}
            onClick={() => resetting && run(() => resetMemberPassword(resetting.id, draft.password), () => setResetting(null))}
          >
            {tc("save")}
          </Button>
        </DialogActions>
      </Dialog>

      <ConfirmDialog
        open={deleting !== null}
        text={deleting ? t("deleteConfirm", { email: deleting.email }) : ""}
        onConfirm={() => deleting && run(() => deleteMember(deleting.id))}
        onClose={() => setDeleting(null)}
      />
      {snackbar}
    </Box>
  );
}
