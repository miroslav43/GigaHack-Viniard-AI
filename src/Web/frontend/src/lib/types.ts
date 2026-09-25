import type { InterrowCover, RowStructure } from "@/theme/mapPalette";

export interface BlockSummary {
  vineyard_id: string;
  outline_area_m2: number;
  row_count: number;
  row_length_m: number;
  canopy_count: number;
  canopy_area_m2: number;
  interrow_area_m2: number;
  plant_count: number;
  disrupted_rows: number;
}

export interface SurveySummary {
  survey: {
    id: string;
    name: string;
    captured_at: string;
    gsd_m: number;
    crs: string;
    source: string;
    license: string;
    stage: string;
    mock: boolean;
    tiles_total: number;
    tiles_annotated: number;
    surveyed_area_ha: number;
  };
  uat: { name: string; district: string; country: string; osm_relation_id: number; area_ha: number };
  totals: {
    block_count: number;
    row_count: number;
    row_length_m: number;
    canopy_count: number;
    canopy_area_m2: number;
    canopy_area_ha: number;
    interrow_count: number;
    interrow_area_m2: number;
    interrow_area_ha: number;
    plant_count: number;
    disrupted_rows: number;
    waste_count: number;
    target_count: number;
  };
  structure_counts: Partial<Record<RowStructure, number>>;
  cover_counts: Partial<Record<InterrowCover, number>>;
  route: { mock: boolean; length_m: number; duration_min: number; baseline_length_m: number; speed_kmh: number };
  blocks: BlockSummary[];
}

export interface RowRecord {
  row_id: string;
  vineyard_id: string;
  row_structure: RowStructure;
  length_m: number;
  plant_count: number;
  max_gap_m: number;
  tile_structures: Record<string, RowStructure>;
}

export interface TargetProps {
  target_id: string;
  type: string;
  vineyard_id: string;
  row_id: string | null;
  gap_length_m?: number;
  tile: string;
  route_order: number;
}
