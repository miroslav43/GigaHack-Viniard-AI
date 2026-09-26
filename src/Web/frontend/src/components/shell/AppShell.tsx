"use client";

import { useState, type ReactNode } from "react";
import { useLocale, useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import ButtonBase from "@mui/material/ButtonBase";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import BottomNavigation from "@mui/material/BottomNavigation";
import BottomNavigationAction from "@mui/material/BottomNavigationAction";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import MapOutlined from "@mui/icons-material/MapOutlined";
import TableRowsOutlined from "@mui/icons-material/TableRowsOutlined";
import RouteOutlined from "@mui/icons-material/RouteOutlined";
import MenuOpen from "@mui/icons-material/MenuOpen";
import Menu from "@mui/icons-material/Menu";
import PlaceOutlined from "@mui/icons-material/PlaceOutlined";
import TaskAltOutlined from "@mui/icons-material/TaskAltOutlined";
import GroupsOutlined from "@mui/icons-material/GroupsOutlined";
import AccountTreeOutlined from "@mui/icons-material/AccountTreeOutlined";
import { Link, routing, usePathname, useRouter, type Locale } from "@/i18n/routing";
import { Logo } from "./Logo";
import { UserCard } from "./UserCard";
import type { ShellViewer } from "./types";

const BASE_NAV = [
  { href: "/", label: "nav.dashboard", short: "nav.dashboardShort", icon: <DashboardOutlined /> },
  { href: "/harta", label: "nav.map", short: "nav.mapShort", icon: <MapOutlined /> },
  { href: "/blocuri", label: "nav.blocks", short: "nav.blocksShort", icon: <TableRowsOutlined /> },
  { href: "/ruta", label: "nav.route", short: "nav.routeShort", icon: <RouteOutlined /> },
  { href: "/arhitectura", label: "nav.architecture", short: "nav.architectureShort", icon: <AccountTreeOutlined /> },
];


const TASKS_NAV = { href: "/sarcini", label: "nav.tasks", short: "nav.tasksShort", icon: <TaskAltOutlined /> };
const TEAM_NAV = { href: "/echipa", label: "nav.team", short: "nav.teamShort", icon: <GroupsOutlined /> };

const WIDTH_OPEN = 248;
const WIDTH_CLOSED = 76;

export function LanguageSwitch({ dense = false }: { dense?: boolean }) {
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const t = useTranslations("shell");
  const change = (l: Locale) => {
    // keep deep-link query (?rand=…) when switching language
    const query = Object.fromEntries(new URLSearchParams(window.location.search));
    router.replace({ pathname, query }, { locale: l });
  };
  return (
    <Box role="group" aria-label={t("language")} sx={{ display: "flex", gap: 1 }}>
      {routing.locales.map((l) => (
        <ButtonBase
          key={l}
          onClick={() => change(l)}
          aria-pressed={l === locale}
          sx={{
            px: dense ? 2 : 2.5,
            py: 1,
            borderRadius: 1,
            fontSize: 13,
            fontWeight: 600,
            bgcolor: l === locale ? "primary.main" : "transparent",
            color: l === locale ? "primary.contrastText" : "text.secondary",
            "&:hover": { bgcolor: l === locale ? "primary.dark" : "action.hover" },
          }}
        >
          {l.toUpperCase()}
        </ButtonBase>
      ))}
    </Box>
  );
}

export function AppShell({ children, viewer }: { children: ReactNode; viewer: ShellViewer }) {
  const t = useTranslations();
  const pathname = usePathname();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const isMap = pathname.startsWith("/harta");
  const [open, setOpen] = useState(!isMap);
  const collapsed = !open;
  const active = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));
  // /super-admin is intentionally not in the menu: reachable only by URL (and only for platform admins)
  const NAV = [
    ...BASE_NAV,
    ...(viewer.kind === "user" && viewer.uat ? [TASKS_NAV] : []),
    ...(viewer.kind === "user" && viewer.role === "uat_admin" ? [TEAM_NAV] : []),
  ];

  if (isMobile) {
    return (
      <Box sx={{ minHeight: "100dvh", display: "flex", flexDirection: "column", bgcolor: "background.default" }}>
        <Box
          sx={{
            px: 4,
            py: 3,
            bgcolor: "background.paper",
            borderBottom: 1,
            borderColor: "divider",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Logo />
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <LanguageSwitch dense />
            <UserCard viewer={viewer} collapsed />
          </Box>
        </Box>
        <Box component="main" sx={{ flex: 1, minHeight: 0, pb: 16 }}>
          {children}
        </Box>
        <BottomNavigation
          showLabels
          value={NAV.findIndex((n) => active(n.href))}
          sx={{ position: "fixed", bottom: 0, left: 0, right: 0, borderTop: 1, borderColor: "divider", zIndex: 10 }}
        >
          {NAV.map((n) => (
            <BottomNavigationAction key={n.href} component={Link} href={n.href} label={t(n.short)} icon={n.icon} />
          ))}
        </BottomNavigation>
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
          <IconButton size="small" onClick={() => setOpen((v) => !v)} aria-label={open ? t("shell.collapse") : t("shell.expand")}>
            {open ? <MenuOpen /> : <Menu />}
          </IconButton>
        </Box>
        <Divider sx={{ mx: 5 }} />
        {!collapsed && (
          <Box sx={{ px: 5, pt: 5, pb: 2 }}>
            <Typography variant="overline" color="text.secondary">
              {t("shell.uat")}
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2, mt: 1 }}>
              <PlaceOutlined fontSize="small" color="primary" />
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {viewer.uat ? t("uat.townHall", { name: viewer.uat.name }) : t("uat.unassigned")}
                </Typography>
                {viewer.uat && (
                  <Typography variant="caption" color="text.secondary">
                    {[viewer.uat.district, t(`country.${viewer.uat.country}`)].filter(Boolean).join(" · ")}
                  </Typography>
                )}
              </Box>
            </Box>
          </Box>
        )}
        <List sx={{ px: 4, py: 3, display: "flex", flexDirection: "column", gap: 1 }}>
          {NAV.map((n) => (
            <Tooltip key={n.href} title={collapsed ? t(n.label) : ""} placement="right">
              <ListItemButton
                component={Link}
                href={n.href}
                selected={active(n.href)}
                sx={{ py: 2.5, px: collapsed ? 3.5 : 4, justifyContent: collapsed ? "center" : "flex-start" }}
              >
                <ListItemIcon sx={{ minWidth: collapsed ? 0 : 40 }}>{n.icon}</ListItemIcon>
                {!collapsed && <ListItemText primary={t(n.label)} slotProps={{ primary: { sx: { fontWeight: 500 } } }} />}
              </ListItemButton>
            </Tooltip>
          ))}
        </List>
        <Box sx={{ flex: 1 }} />
        <Box sx={{ px: collapsed ? 2 : 5, pb: 5, display: "flex", flexDirection: "column", alignItems: collapsed ? "center" : "stretch" }}>
          {!collapsed && (
            <>
              <Divider sx={{ mb: 4 }} />
              <Typography variant="caption" color="text.secondary" component="p" sx={{ lineHeight: 1.5, mb: 3 }}>
                {t("shell.attribution")}
              </Typography>
            </>
          )}
          {collapsed ? null : <LanguageSwitch />}
          <Divider sx={{ my: 4 }} />
          <UserCard viewer={viewer} collapsed={collapsed} />
        </Box>
      </Box>
      <Box component="main" sx={{ flex: 1, minWidth: 0, overflow: "auto" }}>
        {children}
      </Box>
    </Box>
  );
}
