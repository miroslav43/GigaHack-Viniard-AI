"use client";

import { useState, useTransition, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Snackbar from "@mui/material/Snackbar";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Typography from "@mui/material/Typography";
import { Link, useRouter } from "@/i18n/routing";
import { ADMIN_TABS, type AdminTab } from "./types";


export function AdminTabs({ current }: { current: AdminTab }) {
  const t = useTranslations("superAdmin.tabs");
  return (
    <Tabs value={current} variant="scrollable" allowScrollButtonsMobile sx={{ mb: 5, borderBottom: 1, borderColor: "divider" }}>
      {ADMIN_TABS.map((k) => (
        <Tab key={k} value={k} label={t(k)} component={Link} href={`/super-admin?tab=${k}`} />
      ))}
    </Tabs>
  );
}

type Result<T> = { ok: true; data: T } | { ok: false; error: string };

/** Runs a server action, translates known error codes, refreshes the page on success and shows a toast. */
export function useAdminAction() {
  const t = useTranslations("superAdmin");
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [toast, setToast] = useState<{ kind: "success" | "error"; text: string } | null>(null);

  const message = (code: string) => (t.has(`errors.${code}`) ? t(`errors.${code}`) : t("common.error", { message: code }));

  function run<T>(action: () => Promise<Result<T>>, onOk?: (data: T) => void) {
    startTransition(async () => {
      const res = await action();
      if (res.ok) {
        setToast({ kind: "success", text: t("common.saved") });
        onOk?.(res.data);
        router.refresh();
      } else {
        setToast({ kind: "error", text: message(res.error) });
      }
    });
  }

  const snackbar = (
    <Snackbar open={toast !== null} autoHideDuration={5000} onClose={() => setToast(null)} anchorOrigin={{ vertical: "bottom", horizontal: "center" }}>
      {toast ? (
        <Alert severity={toast.kind} variant="filled" onClose={() => setToast(null)}>
          {toast.text}
        </Alert>
      ) : undefined}
    </Snackbar>
  );

  return { run, pending, snackbar, message };
}

export function ConfirmDialog({
  open,
  text,
  danger = true,
  onConfirm,
  onClose,
}: {
  open: boolean;
  text: ReactNode;
  danger?: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  const t = useTranslations("superAdmin.common");
  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>{t("confirm")}</DialogTitle>
      <DialogContent>
        <Typography>{text}</Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t("cancel")}</Button>
        <Button
          variant="contained"
          color={danger ? "error" : "primary"}
          onClick={() => {
            onConfirm();
            onClose();
          }}
        >
          {t("confirm")}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

/** "Sireți" → "sireti": lowercase ASCII key for URLs. */
export const slugify = (s: string) =>
  s
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
