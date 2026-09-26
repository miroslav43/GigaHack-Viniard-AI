"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import MenuIcon from "@mui/icons-material/Menu";
import { Link } from "@/i18n/routing";
import { LanguageSwitch } from "@/components/shell/AppShell";
import { Logo } from "@/components/shell/Logo";

export const SECTIONS = ["problem", "how", "audiences", "security", "engagement", "faq"] as const;

/** Sticky header of the presentation site: in-page anchors, language, sign in, the pilot call to action. */
export function LandingHeader() {
  const t = useTranslations("landing.nav");
  const [open, setOpen] = useState(false);
  return (
    <Box
      component="header"
      sx={{
        position: "sticky",
        top: 0,
        zIndex: 20,
        bgcolor: "background.paper",
        borderBottom: 1,
        borderColor: "divider",
      }}
    >
      <Box sx={{ maxWidth: 1200, mx: "auto", px: { xs: 4, md: 6 }, height: 68, display: "flex", alignItems: "center", gap: 4 }}>
        <Link href="/prezentare" aria-label="Solemtrix" style={{ textDecoration: "none" }}>
          <Logo />
        </Link>
        <Box component="nav" aria-label={t("menu")} sx={{ display: { xs: "none", lg: "flex" }, gap: 1, ml: 4 }}>
          {SECTIONS.map((s) => (
            <Button key={s} href={`#${s}`} color="inherit" size="small" sx={{ color: "text.secondary", fontWeight: 500 }}>
              {t(s)}
            </Button>
          ))}
        </Box>
        <Box sx={{ flex: 1 }} />
        <Box sx={{ display: { xs: "none", sm: "block" } }}>
          <LanguageSwitch dense />
        </Box>
        <Button component={Link} href="/login" variant="text" sx={{ display: { xs: "none", md: "inline-flex" } }}>
          {t("signIn")}
        </Button>
        <Button href="#contact" variant="contained" sx={{ display: { xs: "none", sm: "inline-flex" } }}>
          {t("cta")}
        </Button>
        <IconButton onClick={() => setOpen(true)} aria-label={t("menu")} sx={{ display: { lg: "none" } }}>
          <MenuIcon />
        </IconButton>
      </Box>
      <Drawer anchor="right" open={open} onClose={() => setOpen(false)}>
        <Box sx={{ width: 280, p: 4, display: "flex", flexDirection: "column", gap: 4 }} role="presentation">
          <LanguageSwitch dense />
          <List onClick={() => setOpen(false)}>
            {SECTIONS.map((s) => (
              <ListItemButton key={s} component="a" href={`#${s}`}>
                <ListItemText primary={t(s)} />
              </ListItemButton>
            ))}
          </List>
          <Button component={Link} href="/login" variant="outlined" onClick={() => setOpen(false)}>
            {t("signIn")}
          </Button>
          <Button href="#contact" variant="contained" onClick={() => setOpen(false)}>
            {t("cta")}
          </Button>
        </Box>
      </Drawer>
    </Box>
  );
}
