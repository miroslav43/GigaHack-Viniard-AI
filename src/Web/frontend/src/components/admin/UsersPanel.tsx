"use client";

import { useState } from "react";
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
import InputAdornment from "@mui/material/InputAdornment";
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
import AutorenewOutlined from "@mui/icons-material/AutorenewOutlined";
import { useFormat } from "@/lib/useFormat";
import {
  createUser,
  deleteUser,
  resetUserPassword,
  setUserBanned,
  updateUserAccess,
} from "@/app/[locale]/(app)/super-admin/actions";
import { ConfirmDialog, useAdminAction } from "./common";
import { ROLES, type AdminUser, type Role } from "./types";

type UatOption = { key: string; name: string };

/** 16 random characters from an unambiguous alphabet (crypto RNG). */
function generatePassword() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789-_";
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => alphabet[b % alphabet.length]).join("");
}

function PasswordField({ value, onChange, label, generateLabel }: { value: string; onChange: (v: string) => void; label: string; generateLabel: string }) {
  return (
    <TextField
      label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      fullWidth
      slotProps={{
        input: {
          sx: { fontFamily: "monospace" },
          endAdornment: (
            <InputAdornment position="end">
              <Tooltip title={generateLabel}>
                <IconButton edge="end" onClick={() => onChange(generatePassword())} aria-label={generateLabel}>
                  <AutorenewOutlined />
                </IconButton>
              </Tooltip>
            </InputAdornment>
          ),
        },
      }}
    />
  );
}

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

export function UsersPanel({ users, uats, adminApi, meId }: { users: AdminUser[]; uats: UatOption[]; adminApi: boolean; meId: string | null }) {
  const t = useTranslations("superAdmin");
  const tr = useTranslations("auth.roles");
  const f = useFormat();
  const { run, pending, snackbar } = useAdminAction();
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<AdminUser | null>(null);
  const [resetting, setResetting] = useState<AdminUser | null>(null);
  const [deleting, setDeleting] = useState<AdminUser | null>(null);
  const [draft, setDraft] = useState({ email: "", fullName: "", password: "", uat: null as string | null, role: "uat_admin" as Role });
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

  const openCreate = () => {
    setDraft({ email: "", fullName: "", password: generatePassword(), uat: uats[0]?.key ?? null, role: "uat_admin" });
    setCreating(true);
  };
  const openEdit = (u: AdminUser) => {
    setDraft({ email: u.email, fullName: u.name ?? "", password: "", uat: u.uat, role: u.role ?? "viewer" });
    setEditing(u);
  };
  const openReset = (u: AdminUser) => {
    setDraft((d) => ({ ...d, password: generatePassword() }));
    setResetting(u);
  };

  return (
    <Card>
      <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 3, flexWrap: "wrap" }}>
        <Typography variant="h3">{t("users.count", { n: users.length })}</Typography>
        <Button variant="contained" startIcon={<PersonAddOutlined />} onClick={openCreate}>
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
                <TableCell>{u.uat ? uatName(u.uat) : <em>{t("users.unassigned")}</em>}</TableCell>
                <TableCell>
                  {u.role ? <Chip size="small" variant="outlined" color={u.role === "platform_admin" ? "primary" : "default"} label={tr(u.role)} /> : t("common.none")}
                </TableCell>
                <TableCell>
                  <Chip size="small" color={u.banned ? "error" : "success"} variant="outlined" label={u.banned ? t("users.disabled") : t("users.active")} />
                </TableCell>
                <TableCell>{u.last_sign_in_at ? f.date(u.last_sign_in_at) : t("common.none")}</TableCell>
                <TableCell>{f.date(u.created_at)}</TableCell>
                <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                  <Tooltip title={t("users.edit")}>
                    <IconButton size="small" onClick={() => openEdit(u)} disabled={pending}>
                      <ManageAccountsOutlined fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={t("users.resetPassword")}>
                    <IconButton size="small" onClick={() => openReset(u)} disabled={pending}>
                      <KeyOutlined fontSize="small" />
                    </IconButton>
                  </Tooltip>
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
          <PasswordField value={draft.password} onChange={(password) => setDraft({ ...draft, password })} label={t("users.fieldPassword")} generateLabel={t("users.generate")} />
          <AccessFields uats={uats} uat={draft.uat} role={draft.role} onChange={(a) => setDraft({ ...draft, ...a })} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreating(false)}>{t("common.cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending || !draft.email || draft.password.length < 10}
            onClick={() => run(() => createUser(draft), () => setCreating(false))}
          >
            {t("common.save")}
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

      {/* reset password */}
      <Dialog open={resetting !== null} onClose={() => setResetting(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{resetting && t("users.dialogReset", { email: resetting.email })}</DialogTitle>
        <DialogContent dividers>
          <PasswordField value={draft.password} onChange={(password) => setDraft({ ...draft, password })} label={t("users.fieldPassword")} generateLabel={t("users.generate")} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetting(null)}>{t("common.cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending || draft.password.length < 10}
            onClick={() => resetting && run(() => resetUserPassword(resetting.id, draft.password), () => setResetting(null))}
          >
            {t("common.save")}
          </Button>
        </DialogActions>
      </Dialog>

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
