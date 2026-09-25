"use client";

import { useTranslations } from "next-intl";
import Avatar from "@mui/material/Avatar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import Logout from "@mui/icons-material/Logout";
import Login from "@mui/icons-material/Login";
import { useRouter } from "@/i18n/routing";
import { createClient } from "@/lib/supabase/client";
import { AUTH_ENABLED, DEMO_COOKIE } from "@/lib/supabase/config";
import type { Viewer } from "@/lib/viewer";

const initials = (v: Viewer) =>
  (v.name ?? v.email ?? "?")
    .split(/[\s@.]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]!.toUpperCase())
    .join("");

export function UserCard({ viewer, collapsed }: { viewer: Viewer; collapsed: boolean }) {
  const t = useTranslations("auth");
  const router = useRouter();

  const signOut = async () => {
    if (viewer.kind === "user") await createClient().auth.signOut();
    document.cookie = `${DEMO_COOKIE}=; Max-Age=0; path=/; SameSite=Lax`;
    router.replace("/login");
    router.refresh();
  };

  if (viewer.kind === "demo") {
    if (!AUTH_ENABLED) return null;
    return collapsed ? (
      <Tooltip title={t("exitDemo")} placement="right">
        <IconButton onClick={signOut} aria-label={t("exitDemo")}>
          <Login />
        </IconButton>
      </Tooltip>
    ) : (
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}>
        <Chip size="small" color="info" variant="outlined" label={t("demoMode")} />
        <Button size="small" startIcon={<Login />} onClick={signOut}>
          {t("signInCta")}
        </Button>
      </Box>
    );
  }

  const avatar = <Avatar sx={{ bgcolor: "primary.main", width: 40, height: 40, fontSize: 15, fontWeight: 600 }}>{initials(viewer)}</Avatar>;
  if (collapsed)
    return (
      <Tooltip title={`${viewer.email ?? ""} · ${t("signOut")}`} placement="right">
        <IconButton onClick={signOut} aria-label={t("signOut")} sx={{ p: 0 }}>
          {avatar}
        </IconButton>
      </Tooltip>
    );

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
      {avatar}
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap>
          {viewer.name ?? viewer.email}
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" noWrap>
          {viewer.email}
        </Typography>
        {viewer.role && (
          <Typography variant="caption" color="primary" component="p" noWrap>
            {t(`roles.${viewer.role}`)}
          </Typography>
        )}
      </Box>
      <Tooltip title={t("signOut")}>
        <IconButton size="small" onClick={signOut} aria-label={t("signOut")}>
          <Logout fontSize="small" />
        </IconButton>
      </Tooltip>
    </Box>
  );
}
