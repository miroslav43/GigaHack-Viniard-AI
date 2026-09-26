// The only place colours, radii, shadows and fonts are defined (src/Web/CLAUDE.md §7).
// Components read them through the MUI theme or mapPalette — never hardcode colours elsewhere.
// Colours that scripts bake into raster files live in rasterColors.json (same folder) and are re-exported here.
import rasterColors from "./rasterColors.json";

export const font = {
  family: '"Inter Variable", Inter, -apple-system, "Segoe UI", sans-serif',
  bodySize: 16,
  bodyLineHeight: 24,
};

export const spacingUnit = 4;

export const radius = { sm: 4, md: 6, lg: 8, xl: 12, pill: 9999 };

export const color = {
  primary: { 50: "#EEF2FF", 100: "#E0E7FF", 200: "#C7D2FE", light: "#6366F1", main: "#4F46E5", dark: "#4338CA", contrast: "#FFFFFF" },
  secondary: { light: "#929292", main: "#575757", dark: "#202020" },
  background: { default: "#F6F6F6", paper: "#FFFFFF" },
  text: { primary: "#202020", secondary: "#575757", disabled: "#B3B3B3" },
  divider: "#EAEAEA",
  grey: { 50: "#FAFAFA", 100: "#F6F6F6", 200: "#EAEAEA", 300: "#D2D2D2", 400: "#B3B3B3", 500: "#929292", 600: "#575757", 700: "#333333", 800: "#202020", 900: "#101010" },
  success: { 50: "#F0FDFA", main: "#0D9488", dark: "#09675F" },
  warning: { 50: "#FEF3C7", main: "#D97706", dark: "#975304" },
  error: { 50: "#FEF2F2", main: "#C10007", light: "#FF6467", dark: "#870004" },
  info: { 50: "#DFF2FE", main: "#00A6F4", dark: "#0074AA" },
  action: { active: "rgba(0,0,0,.54)", hover: "rgba(0,0,0,.04)", selected: "rgba(0,0,0,.08)" },
  white: "#FFFFFF",
  // hero panel of the login / brand areas (dark indigo, as in docs/design/marcaj_login.png)
  brandNight: "#12103A",
  // map-only colours (ADR-012): chosen for contrast on brown soil, shadows and grass
  map: {
    canopyFill: "#2DD4BF",
    rowDisrupted: "#F59E0B",
    interrowBareSoil: "#D6C3A1",
    interrowMixed: "#A7D7C5",
    interrowVegetation: "#3DA99F",
    background: "#E9E7E1",
    mask: "#202020",
    // hillshade light side of the synthetic canopy relief (3D view): translucent, so the orthophoto shows through
    reliefHighlight: "rgba(255,255,255,0.35)",
    // survey tile footprints: a tile the model left for completion in Marcaj (review_status set)
    tileToComplete: "#F97316",
    // pipeline vegetation mask (magenta: the complement of the green it marks, visible on vines and grass alike)
    vegMask: rasterColors.vegMask.color,
    vegMaskAlpha: rasterColors.vegMask.alpha,
    // roads (roads.geojson): public + field roads share a strong orange (redder than the amber of disrupted rows),
    // a farm's internal roads a light lemon yellow (dashed), both apart from the teal / indigo / soil vine layers
    roadNetwork: "#EA580C",
    roadInternal: "#FDE047",
    // farm outlines (farms.geojson): pink, away from the indigo blocks and the red waste; dark label on a white halo
    farmLine: "#EC4899",
    farmLabel: "#831843",
  },
};

export const elevation = {
  sm: "0 1px 2px 0 rgba(0,0,0,.05)",
  base: "0 1px 2px 0 rgba(0,0,0,.06), 0 1px 3px 0 rgba(0,0,0,.10)",
  md: "0 2px 4px -1px rgba(0,0,0,.06), 0 4px 6px -1px rgba(0,0,0,.10)",
  lg: "0 4px 6px -2px rgba(0,0,0,.05), 0 10px 15px -3px rgba(0,0,0,.10)",
};
