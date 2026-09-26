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
} as const;

export type RowStructure = keyof typeof mapPalette.row;
export type InterrowCover = keyof typeof mapPalette.interrow;
