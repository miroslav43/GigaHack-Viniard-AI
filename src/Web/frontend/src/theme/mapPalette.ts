// Map layer colours, derived from tokens (src/Web/CLAUDE.md §7.1). Same in light and dark: the basemap is the orthophoto.
import { color } from "./tokens";

export const mapPalette = {
  background: color.map.background,
  casing: color.white,
  geofence: color.primary.main,
  geofenceMask: color.map.mask,
  studyArea: color.grey[900],
  canopyFill: color.map.canopyFill,
  canopyLine: color.success.main,
  row: {
    regular: color.primary.light,
    disrupted: color.map.rowDisrupted,
    unassessable: color.grey[300],
  },
  interrow: {
    bare_soil: color.map.interrowBareSoil,
    mixed: color.map.interrowMixed,
    vegetation: color.map.interrowVegetation,
    unassessable: color.grey[300],
  },
  block: color.primary[200],
  waste: color.error.main,
  target: color.info.main,
  targetOffRoute: color.grey[600],
  targetText: color.white,
  route: color.primary.main,
  start: color.grey[900],
  passage: color.grey[400],
  forbidden: color.error.main,
  selected: color.error.light,
  // hillshade of the synthetic canopy relief (3D view): neutral light and shade over the orthophoto
  relief: {
    shadow: color.grey[900],
    highlight: color.map.reliefHighlight,
    accent: color.grey[800],
  },
  // survey tile footprints, by status; toComplete wins over the status (tiles.geojson review_status)
  tile: {
    vineyard: color.success.main,
    no_vineyard: color.grey[400],
    toComplete: color.map.tileToComplete,
  },
  vegMask: color.map.vegMask,
} as const;

export type TileStatus = "vineyard" | "no_vineyard";

export type RowStructure = keyof typeof mapPalette.row;
export type InterrowCover = keyof typeof mapPalette.interrow;
