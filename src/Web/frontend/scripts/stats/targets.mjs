// Inspection targets (targets.geojson) → counts by type / kind / priority / skip reason, and per-block damage.
import { countBy, round, sum } from "./util.mjs";

/** The kinds that count as missing plants for the block health index (older bundles only have type "missing"). */
const isMissingPlant = (p) => (p.kind != null ? p.kind === "missing_plant" : p.type === "missing");

/** A section is null when the bundle predates the property (the mock has no kind / priority / skip_reason). */
const countOrNull = (props, key) => (props.some((p) => p[key] != null) ? countBy(props, key) : null);

export const targetStats = (targets) => {
  const props = targets.features.map((f) => f.properties);
  const onRoute = props.filter((p) => p.route_order != null).length;
  return {
    total: props.length,
    by_type: countBy(props, "type"),
    by_kind: countOrNull(props, "kind"),
    by_priority: countOrNull(props, "priority"),
    on_route: onRoute,
    off_route: props.length - onRoute,
    skip_reasons: countOrNull(props, "skip_reason"),
    gap_total_m: round(sum(props.map((p) => p.gap_length_m ?? 0))),
  };
};

/** vineyard_id → { missing, gap_m }: missing plants and summed gap length of the block's targets. */
export const damageByBlock = (targets) => {
  const out = new Map();
  for (const { properties: p } of targets.features) {
    if (!p.vineyard_id) continue;
    const d = out.get(p.vineyard_id) ?? { missing: 0, gap_m: 0 };
    out.set(p.vineyard_id, {
      missing: d.missing + (isMissingPlant(p) ? 1 : 0),
      gap_m: d.gap_m + (p.gap_length_m ?? 0),
    });
  }
  return out;
};
