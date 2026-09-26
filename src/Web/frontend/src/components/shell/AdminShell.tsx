"use client";

import { Suspense, useState, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Tooltip from "@mui/material/Tooltip";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import AdminPanelSettingsOutlined from "@mui/icons-material/AdminPanelSettingsOutlined";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import AccountBalanceOutlined from "@mui/icons-material/AccountBalanceOutlined";
import GroupOutlined from "@mui/icons-material/GroupOutlined";
import SatelliteAltOutlined from "@mui/icons-material/SatelliteAltOutlined";
import MonitorHeartOutlined from "@mui/icons-material/MonitorHeartOutlined";
import HistoryOutlined from "@mui/icons-material/HistoryOutlined";
import ContactMailOutlined from "@mui/icons-material/ContactMailOutlined";
import MenuOpen from "@mui/icons-material/MenuOpen";
import Menu from "@mui/icons-material/Menu";
import { Link } from "@/i18n/routing";
import { ADMIN_TABS, type AdminTab } from "@/components/admin/types";
import { AssistantChat } from "@/components/assistant/AssistantChat";
import { LanguageSwitch } from "./AppShell";
import { Logo } from "./Logo";
import { UserCard } from "./UserCard";
import type { ShellViewer } from "./types";

const ICONS: Record<AdminTab, ReactNode> = {
  overview: <DashboardOutlined />,
  uat: <AccountBalanceOutlined />,
  users: <GroupOutlined />,
  surveys: <SatelliteAltOutlined />,
  leads: <ContactMailOutlined />,
  system: <MonitorHeartOutlined />,
  audit: <HistoryOutlined />,
};

const WIDTH_OPEN = 248;
const WIDTH_CLOSED = 76;

function useCurrentTab(): AdminTab {
  const raw = useSearchParams().get("tab");
  return ADMIN_TABS.includes(raw as AdminTab) ? (raw as AdminTab) : "overview";
}

function SideNav({ collapsed }: { collapsed: boolean }) {
  const t = useTranslations("superAdmin.tabs");
  const current = useCurrentTab();
  return (
    <List sx={{ px: 4, py: 3, display: "flex", flexDirection: "column", gap: 1 }}>
      {ADMIN_TABS.map((k) => (
        <Tooltip key={k} title={collapsed ? t(k) : ""} placement="right">
          <ListItemButton
            component={Link}
            href={`/super-admin?tab=${k}`}
            selected={current === k}
            sx={{ py: 2.5, px: collapsed ? 3.5 : 4, justifyContent: collapsed ? "center" : "flex-start" }}
          >
            <ListItemIcon sx={{ minWidth: collapsed ? 0 : 40 }}>{ICONS[k]}</ListItemIcon>
            {!collapsed && <ListItemText primary={t(k)} slotProps={{ primary: { sx: { fontWeight: 500 } } }} />}
          </ListItemButton>
        </Tooltip>
      ))}
    </List>
  );
}

/** Horizontal, scrollable section bar for small screens. */
function MobileNav() {
  const t = useTranslations("superAdmin.tabs");
  const current = useCurrentTab();
  return (
    <Box sx={{ display: "flex", gap: 2, overflowX: "auto", px: 4, py: 2, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider" }}>
      {ADMIN_TABS.map((k) => (
        <Chip
          key={k}
          icon={ICONS[k] as React.ReactElement}
          label={t(k)}
          component={Link}
          href={`/super-admin?tab=${k}`}
          clickable
          color={current === k ? "primary" : "default"}
          variant={current === k ? "filled" : "outlined"}
        />
      ))}
    </Box>
  );
}

/** Platform-administration shell: its own section menu, no municipality identity. */
export function AdminShell({ children, viewer }: { children: ReactNode; viewer: ShellViewer }) {
  const t = useTranslations("superAdmin");
  const ts = useTranslations("shell");
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const [open, setOpen] = useState(true);
  const collapsed = !open;

  if (isMobile) {
    return (
      <Box sx={{ minHeight: "100dvh", display: "flex", flexDirection: "column", bgcolor: "background.default" }}>
        <Box sx={{ px: 4, py: 3, bgcolor: "background.paper", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}>
          <Logo />
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <LanguageSwitch dense />
            <UserCard viewer={viewer} collapsed />
          </Box>
        </Box>
        <Suspense>
          <MobileNav />
        </Suspense>
        <Box component="main" sx={{ flex: 1, minWidth: 0 }}>
          {children}
        </Box>
        <AssistantChat role={viewer.role} demo={false} bottomNav={false} />
      </Box>
    );
  }

  return (
    <Box sx={{ display: "flex", height: "100dvh", bgcolor: "background.default" }}>
      <Box
        component="nav"
        sx={{
          width: collapsed ? WIDTH_CLOSED : WIDTH_OPEN,
          flexShrink: 0,
          bgcolor: "background.paper",
          borderRight: 1,
          borderColor: "divider",
          display: "flex",
          flexDirection: "column",
          transition: "width .18s ease",
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            px: collapsed ? 0 : 5,
            pt: 6,
            pb: 5,
            display: "flex",
            flexDirection: collapsed ? "column" : "row",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 2,
          }}
        >
          <Logo collapsed={collapsed} />
          <IconButton size="small" onClick={() => setOpen((v) => !v)} aria-label={open ? ts("collapse") : ts("expand")}>
            {open ? <MenuOpen /> : <Menu />}
          </IconButton>
        </Box>
        <Divider sx={{ mx: 5 }} />
        <Box sx={{ px: collapsed ? 0 : 5, pt: 4, display: "flex", justifyContent: collapsed ? "center" : "flex-start" }}>
          {collapsed ? (
            <Tooltip title={t("nav")} placement="right">
              <AdminPanelSettingsOutlined color="primary" />
            </Tooltip>
          ) : (
            <Chip icon={<AdminPanelSettingsOutlined />} label={t("nav")} color="primary" variant="outlined" size="small" sx={{ fontWeight: 600 }} />
          )}
        </Box>
        <Suspense>
          <SideNav collapsed={collapsed} />
        </Suspense>
        <Box sx={{ flex: 1 }} />
        <Box sx={{ px: collapsed ? 2 : 5, pb: 5, display: "flex", flexDirection: "column", alignItems: collapsed ? "center" : "stretch" }}>
          {!collapsed && <LanguageSwitch />}
          <Divider sx={{ my: 4 }} />
          <UserCard viewer={viewer} collapsed={collapsed} />
        </Box>
      </Box>
      <Box component="main" sx={{ flex: 1, minWidth: 0, overflow: "auto" }}>
        {children}
      </Box>
      <AssistantChat role={viewer.role} demo={false} />
    </Box>
  );
}
