// Block health index (orientative, docs/STATISTICI.md): 0–100 from disrupted rows, missing plants and gap length.
//   score = 100 × (1 − W.disrupted·d − W.missing·min(1, m/M) − W.gap·min(1, g/G))
//   d = disrupted rows / rows, m = missing plants per 100 m of row, g = gap metres per 100 m of row,
//   M, G = the NORM_QUANTILE of m, g over the eligible blocks (blocks with at least MIN_ROW_LENGTH_M of rows).
import { quantile, round } from "./util.mjs";

export const HEALTH = {
  weights: { disrupted: 0.4, missing: 0.3, gap: 0.3 },
  min_row_length_m: 100,
  norm_quantile: 0.95,
};

const per100m = (v, lengthM) => (lengthM > 0 ? (v / lengthM) * 100 : 0);
const normalised = (v, norm) => (norm > 0 ? Math.min(1, v / norm) : 0);

/** blocks: summary.blocks; damage: vineyard_id → { missing, gap_m }; farmOf: vineyard_id → farm_id (optional). */
export const healthStats = (blocks, damage, farmOf = new Map()) => {
  const rows = blocks.map((b) => {
    const d = damage.get(b.vineyard_id) ?? { missing: 0, gap_m: 0 };
    return {
      vineyard_id: b.vineyard_id,
      farm_id: farmOf.get(b.vineyard_id) ?? null,
      row_length_m: b.row_length_m,
      disrupted_share: b.row_count > 0 ? b.disrupted_rows / b.row_count : 0,
      missing_per_100m: per100m(d.missing, b.row_length_m),
      gap_m_per_100m: per100m(d.gap_m, b.row_length_m),
      eligible: b.row_length_m >= HEALTH.min_row_length_m,
    };
  });
  const eligible = rows.filter((r) => r.eligible);
  const norm = {
    missing_per_100m: round(quantile(eligible.map((r) => r.missing_per_100m), HEALTH.norm_quantile)),
    gap_m_per_100m: round(quantile(eligible.map((r) => r.gap_m_per_100m), HEALTH.norm_quantile)),
  };
  const w = HEALTH.weights;
  const score = (r) =>
    100 *
    (1 -
      w.disrupted * r.disrupted_share -
      w.missing * normalised(r.missing_per_100m, norm.missing_per_100m) -
      w.gap * normalised(r.gap_m_per_100m, norm.gap_m_per_100m));
  return {
    ...HEALTH,
    norm,
    blocks: rows.map(({ eligible: ok, ...r }) => ({
      ...r,
      disrupted_share: round(r.disrupted_share, 4),
      missing_per_100m: round(r.missing_per_100m),
      gap_m_per_100m: round(r.gap_m_per_100m),
      row_length_m: round(r.row_length_m),
      score: ok ? round(score(r), 1) : null,
    })),
  };
};
