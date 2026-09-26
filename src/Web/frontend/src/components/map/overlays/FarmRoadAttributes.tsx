"use client";

// Attribute-panel bodies of a farm (farms.geojson + its summary.json "farms" entry) and of a road (roads.geojson).
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { mapPalette } from "@/theme/mapPalette";
import { useFormat } from "@/lib/useFormat";
import type { FarmSummary, RoadClass } from "@/lib/types";
import { ColorChip, Field } from "../AttributePanel";
import { CadastreSnapshot } from "./CadastreSnapshot";

const ROAD_SOURCES = ["osm", "detected", "cadastre"] as const;
type RoadSource = (typeof ROAD_SOURCES)[number];
const roadSourceOf = (v: string | null): RoadSource | null => (ROAD_SOURCES.includes(v as RoadSource) ? (v as RoadSource) : null);

/** MapLibre hands feature properties back as plain values; null ones may come back missing. */
const strOrNull = (v: unknown) => (typeof v === "string" && v !== "" ? v : null);
const numOrNull = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const ROAD_CLASSES: RoadClass[] = ["public", "field", "internal"];
const roadClassOf = (v: unknown): RoadClass | null => (ROAD_CLASSES.includes(v as RoadClass) ? (v as RoadClass) : null);

export const farmTitle = (props: Record<string, unknown>) => String(props.farm_id ?? "—");
export const roadTitle = (props: Record<string, unknown>) => strOrNull(props.name) ?? String(props.road_id ?? "—");

export function FarmAttributes({
  props,
  farm,
  onPickBlock,
}: {
  props: Record<string, unknown>;
  /** the farm's summary.json entry (sums of its blocks) */
  farm: FarmSummary | undefined;
  onPickBlock?: (vineyardId: string) => void;
}) {
  const t = useTranslations("map.farm");
  const f = useFormat();
  const area = farm?.area_m2 ?? numOrNull(props.area_m2);
  return (
    <Box data-testid="farm-attributes">
      <Field label={t("id")}>{farmTitle(props)}</Field>
      {farm && (
        <>
          <Typography variant="body2" color="text.secondary" sx={{ pt: 1.25 }}>
            {t("blocks", { n: farm.n_blocks })}
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, py: 1 }}>
            {farm.vineyard_ids.map((id) =>
              onPickBlock ? <Chip key={id} size="small" label={id} onClick={() => onPickBlock(id)} clickable /> : <Chip key={id} size="small" label={id} />,
            )}
          </Box>
        </>
      )}
      {area != null && <Field label={t("area")}>{f.area(area)}</Field>}
      {farm && (
        <>
          <Field label={t("rows")}>{f.int(farm.row_count)}</Field>
          <Field label={t("rowLength")}>{f.length(farm.row_length_m)}</Field>
          <Field label={t("canopyArea")}>{f.area(farm.canopy_area_m2)}</Field>
          <Field label={t("interrowArea")}>{f.area(farm.interrow_area_m2)}</Field>
          <Field label={t("plants")}>{f.int(farm.plant_count)}</Field>
          <Field label={t("targets")}>{f.int(farm.target_count)}</Field>
        </>
      )}
      <CadastreSnapshot props={props} parcels={farm?.n_parcels} />
      {farm && (
        <>
          <Typography variant="caption" color="text.secondary" component="p" sx={{ pt: 1 }}>
            {t("note")}
          </Typography>
        </>
      )}
    </Box>
  );
}

export function RoadAttributes({ props, onPickFarm }: { props: Record<string, unknown>; onPickFarm?: (farmId: string) => void }) {
  const t = useTranslations("map.road");
  const f = useFormat();
  const roadClass = roadClassOf(props.road_class);
  const highway = strOrNull(props.highway), source = strOrNull(props.source), farmId = strOrNull(props.farm_id);
  const length = numOrNull(props.length_m);
  return (
    <Box data-testid="road-attributes">
      <Field label="road_id">{String(props.road_id ?? "—")}</Field>
      <Field label={t("class")}>
        {roadClass ? (
          <ColorChip label={t(`classes.${roadClass}`)} color={roadClass === "internal" ? mapPalette.road.internal : mapPalette.road.network} />
        ) : (
          "—"
        )}
      </Field>
      <Field label={t("highway")}>{highway === "cross_path" ? t("crossPath") : (highway ?? "—")}</Field>
      <Field label={t("name")}>{strOrNull(props.name) ?? "—"}</Field>
      <Field label={t("surface")}>{strOrNull(props.surface) ?? "—"}</Field>
      <Field label={t("length")}>{length == null ? "—" : f.length(length)}</Field>
      <Field label={t("farm")}>
        {farmId ? (onPickFarm ? <Chip size="small" label={farmId} onClick={() => onPickFarm(farmId)} clickable /> : farmId) : "—"}
      </Field>
      {source && <Field label={t("source")}>{roadSourceOf(source) ? t(`sources.${roadSourceOf(source)}`) : source}</Field>}
      {props.cadastral === true && (
        <Field label={t("cadastre")}>
          <Chip size="small" color="success" variant="outlined" label={t("cadastralConfirmed")} />
        </Field>
      )}
    </Box>
  );
}
