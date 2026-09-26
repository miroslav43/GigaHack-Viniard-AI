"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
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
import ManageAccountsOutlined from "@mui/icons-material/ManageAccountsOutlined";
import KeyOutlined from "@mui/icons-material/KeyOutlined";
import BlockOutlined from "@mui/icons-material/BlockOutlined";
import CheckCircleOutline from "@mui/icons-material/CheckCircleOutlined";
import DeleteOutline from "@mui/icons-material/DeleteOutlined";
import ForwardToInboxOutlined from "@mui/icons-material/ForwardToInboxOutlined";
import { useFormat } from "@/lib/useFormat";
import { deleteUser, inviteUser, sendUserPasswordLink, setUserBanned, updateUserAccess } from "@/app/[locale]/(admin)/super-admin/actions";
import { ConfirmDialog, useAdminAction } from "./common";
import { ROLES, type AdminUser, type Role } from "./types";

type UatOption = { key: string; name: string };

function AccessFields({
  uats,
  uat,
  role,
  onChange,
}: {
  uats: UatOption[];
  uat: string | null;
  role: Role;
  onChange: (next: { uat: string | null; role: Role }) => void;
}) {
  const t = useTranslations("superAdmin.users");
  const tr = useTranslations("auth.roles");
  return (
    <>
      <TextField select label={t("fieldUat")} value={uat ?? ""} onChange={(e) => onChange({ uat: e.target.value || null, role })} fullWidth>
        <MenuItem value="">
          <em>{t("noUat")}</em>
        </MenuItem>
        {uats.map((u) => (
          <MenuItem key={u.key} value={u.key}>
            {u.name} ({u.key})
          </MenuItem>
        ))}
      </TextField>
      <TextField select label={t("fieldRole")} value={role} onChange={(e) => onChange({ uat, role: e.target.value as Role })} fullWidth>
        {ROLES.map((r) => (
          <MenuItem key={r} value={r}>
            {tr(r)} ({r})
          </MenuItem>
        ))}
      </TextField>
      <Typography variant="caption" color="text.secondary">
        {t("roleHelp")}
      </Typography>
    </>
  );
}

export function UsersPanel({
  users,
  uats,
  adminApi,
  meId,
  preset,
}: {
  users: AdminUser[];
  uats: UatOption[];
  adminApi: boolean;
  meId: string | null;
  /** opened from a municipality row: create an account for that UAT with this role */
  preset?: { uat: string; role: Role } | null;
}) {
  const t = useTranslations("superAdmin");
  const tr = useTranslations("auth.roles");
  const f = useFormat();
  const { run, pending, snackbar } = useAdminAction();
  const linkSent = ({ email, kind }: { email: string; kind: "invite" | "reset" }) =>
    t(kind === "invite" ? "users.inviteSent" : "users.resetSent", { email });
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<AdminUser | null>(null);
  const [resetting, setResetting] = useState<AdminUser | null>(null);
  const [deleting, setDeleting] = useState<AdminUser | null>(null);
  const [draft, setDraft] = useState({ email: "", fullName: "", uat: null as string | null, role: "uat_admin" as Role });
  // opened from a municipality row (?new=uat_admin&uat=…): open the create dialog prefilled, once
  const presetOpened = useRef(false);
  useEffect(() => {
    if (!preset || presetOpened.current || !adminApi) return;
    // mark as opened only when it really opens: in dev (StrictMode) the first effect run is cleaned up at once
    const frame = requestAnimationFrame(() => {
      presetOpened.current = true;
      setDraft({ email: "", fullName: "", uat: preset.uat, role: preset.role });
      setCreating(true);
    });
    return () => cancelAnimationFrame(frame);
  }, [preset, adminApi]);
  const uatName = (key: string | null) => uats.find((u) => u.key === key)?.name ?? key;

  if (!adminApi) {
    return (
      <Alert severity="warning" variant="outlined" sx={{ bgcolor: "background.paper" }}>
        <AlertTitle>{t("users.secretTitle")}</AlertTitle>
        <Typography variant="body2" sx={{ mb: 2 }}>
          {t("users.secretBody")}
        </Typography>
        <Typography variant="body2" component="div">
          {t("users.secretSteps")}
        </Typography>
      </Alert>
    );
  }

  const openCreate = (p?: { uat: string; role: Role } | null) => {
    setDraft({ email: "", fullName: "", uat: p?.uat ?? uats[0]?.key ?? null, role: p?.role ?? "uat_admin" });
    setCreating(true);
  };

  const openEdit = (u: AdminUser) => {
    setDraft({ email: u.email, fullName: u.name ?? "", uat: u.uat, role: u.role ?? "viewer" });
    setEditing(u);
  };

  return (
    <Card>
      <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 3, flexWrap: "wrap" }}>
        <Typography variant="h3">{t("users.count", { n: users.length })}</Typography>
        <Button variant="contained" startIcon={<PersonAddOutlined />} onClick={() => openCreate()}>
          {t("users.create")}
        </Button>
      </Box>
      <Box sx={{ overflowX: "auto" }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>{t("users.colEmail")}</TableCell>
              <TableCell>{t("users.colName")}</TableCell>
              <TableCell>{t("users.colUat")}</TableCell>
              <TableCell>{t("users.colRole")}</TableCell>
              <TableCell>{t("users.colStatus")}</TableCell>
              <TableCell>{t("users.colLastSignIn")}</TableCell>
              <TableCell>{t("users.colCreated")}</TableCell>
              <TableCell align="right">{t("common.actions")}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users.map((u) => (
              <TableRow key={u.id} hover>
                <TableCell sx={{ fontWeight: 600 }}>
                  {u.email}
                  {u.id === meId && <Chip size="small" label={t("users.you")} sx={{ ml: 1 }} />}
                </TableCell>
                <TableCell>{u.name ?? t("common.none")}</TableCell>
                <TableCell>{u.uat ? uatName(u.uat) : u.role === "platform_admin" ? t("users.platformScope") : <em>{t("users.unassigned")}</em>}</TableCell>
                <TableCell>
                  {u.role ? <Chip size="small" variant="outlined" color={u.role === "platform_admin" ? "primary" : "default"} label={tr(u.role)} /> : t("common.none")}
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    color={u.banned ? "error" : u.invited ? "warning" : "success"}
                    variant="outlined"
                    label={u.banned ? t("users.disabled") : u.invited ? t("users.invited") : t("users.active")}
                  />
                </TableCell>
                <TableCell>{u.last_sign_in_at ? f.date(u.last_sign_in_at) : t("common.none")}</TableCell>
                <TableCell>{f.date(u.created_at)}</TableCell>
                <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                  <Tooltip title={t("users.edit")}>
                    <IconButton size="small" onClick={() => openEdit(u)} disabled={pending}>
                      <ManageAccountsOutlined fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {u.invited ? (
                    <Tooltip title={t("users.resendInvite")}>
                      <IconButton size="small" onClick={() => run(() => sendUserPasswordLink(u.id), undefined, linkSent)} disabled={pending}>
                        <ForwardToInboxOutlined fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  ) : (
                    <Tooltip title={t("users.resetPassword")}>
                      <IconButton size="small" onClick={() => setResetting(u)} disabled={pending}>
                        <KeyOutlined fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                  <Tooltip title={u.banned ? t("users.enable") : t("users.disable")}>
                    <span>
                      <IconButton size="small" onClick={() => run(() => setUserBanned(u.id, !u.banned))} disabled={pending || u.id === meId}>
                        {u.banned ? <CheckCircleOutline fontSize="small" /> : <BlockOutlined fontSize="small" />}
                      </IconButton>
                    </span>
                  </Tooltip>
                  <Tooltip title={t("common.delete")}>
                    <span>
                      <IconButton size="small" color="error" onClick={() => setDeleting(u)} disabled={pending || u.id === meId}>
                        <DeleteOutline fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>

      {/* create */}
      <Dialog open={creating} onClose={() => setCreating(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{t("users.dialogCreate")}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <TextField label={t("users.fieldEmail")} type="email" value={draft.email} onChange={(e) => setDraft({ ...draft, email: e.target.value })} fullWidth autoFocus />
          <TextField label={t("users.fieldName")} value={draft.fullName} onChange={(e) => setDraft({ ...draft, fullName: e.target.value })} fullWidth />
          <AccessFields uats={uats} uat={draft.uat} role={draft.role} onChange={(a) => setDraft({ ...draft, ...a })} />
          <Typography variant="caption" color="text.secondary">
            {t("users.inviteNote")}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreating(false)}>{t("common.cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending || !draft.email}
            onClick={() => run(() => inviteUser(draft), () => setCreating(false), (email) => t("users.inviteSent", { email }))}
          >
            {t("users.sendInvite")}
          </Button>
        </DialogActions>
      </Dialog>

      {/* edit access */}
      <Dialog open={editing !== null} onClose={() => setEditing(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{editing && t("users.dialogEdit", { email: editing.email })}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <TextField label={t("users.fieldName")} value={draft.fullName} onChange={(e) => setDraft({ ...draft, fullName: e.target.value })} fullWidth />
          <AccessFields uats={uats} uat={draft.uat} role={draft.role} onChange={(a) => setDraft({ ...draft, ...a })} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditing(null)}>{t("common.cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending}
            onClick={() =>
              editing &&
              run(() => updateUserAccess({ id: editing.id, fullName: draft.fullName, uat: draft.uat, role: draft.role }), () => setEditing(null))
            }
          >
            {t("common.save")}
          </Button>
        </DialogActions>
      </Dialog>

      {/* password reset: a link by email, the user chooses the new password */}
      <ConfirmDialog
        open={resetting !== null}
        danger={false}
        text={resetting ? t("users.resetConfirm", { email: resetting.email }) : ""}
        onConfirm={() => resetting && run(() => sendUserPasswordLink(resetting.id), undefined, linkSent)}
        onClose={() => setResetting(null)}
      />
      <ConfirmDialog
        open={deleting !== null}
        text={deleting ? t("users.deleteConfirm", { email: deleting.email }) : ""}
        onConfirm={() => deleting && run(() => deleteUser(deleting.id))}
        onClose={() => setDeleting(null)}
      />
      {snackbar}
    </Card>
  );
}
