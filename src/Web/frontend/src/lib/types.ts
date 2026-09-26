import type { InterrowCover, RowStructure, TileStatus } from "@/theme/mapPalette";

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
    // provenance of a real AI survey (manifest.json of the pipeline bundle); absent on the mock
    model_version?: string;
    run_id?: string;
    generated_at?: string;
    pipeline_version?: string;
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
  /** the survey tiles (tiles.geojson of an AI bundle, src/Web/CLAUDE.md §6.3); absent on the mock and older bundles */
  tiles?: TilesSummary;
}

export interface TilesSummary {
  total: number;
  vineyard: number;
  no_vineyard: number;
  /** tiles listed in src/AI/configs/tile_review.csv: to complete in Marcaj (any status) */
  to_complete: number;
}

export type TileReviewStatus = "missed" | "partial" | "verify";

/** A feature of tiles.geojson (the survey tile footprints, src/Web/CLAUDE.md §6.3). */
export interface TileProps {
  tile: string;
  status: TileStatus;
  n_rows: number;
  n_canopies: number;
  n_interrows: number;
  n_waste: number;
  veg_frac: number | null;
  nodata_frac: number | null;
  review_priority: number | null;
  review_note: string | null;
  review_status: TileReviewStatus | null;
  has_mask: boolean;
}

/** Optional files of a published survey (web bundle v3): what the map may offer as extra layers. */
export interface OverlayFiles {
  tiles: boolean;
  masks: boolean;
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
  /** null only for a waste target more than 10 m from every block (src/Web/CLAUDE.md §6.3) */
  vineyard_id: string | null;
  row_id: string | null;
  gap_length_m?: number;
  tile: string;
  /** null = the route does not visit this target (see skip_reason) */
  route_order: number | null;
  // AI bundle extras (src/Web/CLAUDE.md §6.3); absent on the mock
  reachable?: boolean;
  kind?: string;
  note?: string | null;
  waste_id?: string | null;
  source_target_id?: string;
  priority?: number | null;
  route_role?: string | null;
  skip_reason?: string | null;
}

export interface WasteProps {
  waste_id: string;
  vineyard_id: string | null;
  tile: string;
  confidence?: number | null;
  category?: string | null;
}

export interface InterrowProps {
  interrow_id: string;
  vineyard_id: string;
  interrow_cover: InterrowCover;
  tile: string;
  area_m2: number;
  /** area of the whole inter-row (union of its pieces across tiles); absent on the mock */
  interrow_total_m2?: number;
}
