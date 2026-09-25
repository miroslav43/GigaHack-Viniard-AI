"use client";

import { createTheme } from "@mui/material/styles";
import { color, elevation, font, radius, spacingUnit } from "./tokens";

export const theme = createTheme({
  cssVariables: true,
  spacing: spacingUnit,
  shape: { borderRadius: radius.lg },
  palette: {
    primary: { main: color.primary.main, light: color.primary.light, dark: color.primary.dark, contrastText: color.primary.contrast },
    secondary: { main: color.secondary.main, light: color.secondary.light, dark: color.secondary.dark },
    success: { main: color.success.main, dark: color.success.dark, light: color.success[50] },
    warning: { main: color.warning.main, dark: color.warning.dark, light: color.warning[50] },
    error: { main: color.error.main, dark: color.error.dark, light: color.error.light },
    info: { main: color.info.main, dark: color.info.dark, light: color.info[50] },
    background: { default: color.background.default, paper: color.background.paper },
    text: { primary: color.text.primary, secondary: color.text.secondary, disabled: color.text.disabled },
    divider: color.divider,
    grey: color.grey,
    action: { active: color.action.active, hover: color.action.hover, selected: color.action.selected },
  },
  typography: {
    fontFamily: font.family,
    fontSize: 14,
    h1: { fontSize: 28, fontWeight: 700, lineHeight: 1.25 },
    h2: { fontSize: 22, fontWeight: 700, lineHeight: 1.3 },
    h3: { fontSize: 18, fontWeight: 600, lineHeight: 1.35 },
    subtitle1: { fontSize: 16, fontWeight: 600 },
    body1: { fontSize: font.bodySize, lineHeight: `${font.bodyLineHeight}px` },
    body2: { fontSize: 14, lineHeight: "20px" },
    button: { textTransform: "none", fontWeight: 600 },
    overline: { fontSize: 11, fontWeight: 600, letterSpacing: 0.6 },
  },
  components: {
    MuiButton: { defaultProps: { disableElevation: true }, styleOverrides: { root: { borderRadius: radius.lg } } },
    MuiCard: {
      defaultProps: { elevation: 0 },
      styleOverrides: { root: { borderRadius: radius.lg, boxShadow: elevation.base, border: `1px solid ${color.divider}` } },
    },
    MuiPaper: { styleOverrides: { rounded: { borderRadius: radius.lg } } },
    MuiChip: { styleOverrides: { root: { borderRadius: radius.md, fontWeight: 500 } } },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: radius.lg,
          "&.Mui-selected": { backgroundColor: color.primary.main, color: color.primary.contrast },
          "&.Mui-selected:hover": { backgroundColor: color.primary.dark },
          "&.Mui-selected .MuiListItemIcon-root": { color: color.primary.contrast },
        },
      },
    },
    MuiTooltip: { styleOverrides: { tooltip: { fontSize: 12, backgroundColor: color.grey[800] } } },
  },
});
