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
  // roads.geojson: one colour for public and field roads (public drawn wider), another for a farm's internal roads
  road: {
    network: color.map.roadNetwork,
    internal: color.map.roadInternal,
    casing: color.white,
    internalCasing: color.grey[900],
  },
  // farms.geojson: outline + very light fill of the same colour, label text on a halo
  farm: {
    line: color.map.farmLine,
    fill: color.map.farmLine,
    label: color.map.farmLabel,
    halo: color.white,
  },
  // live cadastre: the WMS parcel lines (drawn by the server, legend only) and the parcel picked by a click
  cadastre: {
    parcel: color.map.cadastreParcel,
    highlight: color.map.cadastreHighlight,
  },
} as const;

export type TileStatus = "vineyard" | "no_vineyard";

export type RowStructure = keyof typeof mapPalette.row;
export type InterrowCover = keyof typeof mapPalette.interrow;
