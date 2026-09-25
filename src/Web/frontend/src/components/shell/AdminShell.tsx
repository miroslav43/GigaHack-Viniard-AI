"use client";

import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import AdminPanelSettingsOutlined from "@mui/icons-material/AdminPanelSettingsOutlined";
import { LanguageSwitch } from "./AppShell";
import { Logo } from "./Logo";
import { UserCard } from "./UserCard";
import type { ShellViewer } from "./types";

/** Platform-administration shell: no municipality sidebar — the platform admin is not a town hall user. */
export function AdminShell({ children, viewer }: { children: ReactNode; viewer: ShellViewer }) {
  const t = useTranslations("superAdmin");
  return (
    <Box sx={{ minHeight: "100dvh", bgcolor: "background.default", display: "flex", flexDirection: "column" }}>
      <Box
        component="header"
        sx={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          bgcolor: "background.paper",
          borderBottom: 1,
          borderColor: "divider",
          px: { xs: 4, md: 8 },
          py: 3,
          display: "flex",
          alignItems: "center",
          gap: 4,
        }}
      >
        <Logo />
        <Chip icon={<AdminPanelSettingsOutlined />} label={t("nav")} color="primary" variant="outlined" size="small" sx={{ fontWeight: 600 }} />
        <Box sx={{ flex: 1 }} />
        <LanguageSwitch dense />
        <Box sx={{ width: { xs: "auto", md: 300 } }}>
          <UserCard viewer={viewer} collapsed={false} />
        </Box>
      </Box>
      <Box component="main" sx={{ flex: 1, minWidth: 0 }}>
        {children}
      </Box>
    </Box>
  );
}
