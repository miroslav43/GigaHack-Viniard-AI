"use client";

import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { LinkChip } from "@/components/common/links";
import type { BlockHealth } from "@/lib/types";
import { useFormat } from "@/lib/useFormat";
import { chartPalette } from "@/theme/chartPalette";
import { BarList, type BarItem } from "./BarList";

/** Below this health index a block is flagged for attention (bar colour; the label carries the number). */
export const HEALTH_ATTENTION = 60;
const SIDE_MAX = 8;
const SCORE_MAX = 100;

/** The weakest and the strongest eligible blocks (score != null), up to SIDE_MAX each, without overlap. */
export function splitRanking(blocks: BlockHealth[]): { worst: BlockHealth[]; best: BlockHealth[] } {
  const scored = blocks.filter((b) => b.score != null).sort((a, b) => (a.score ?? 0) - (b.score ?? 0));
  const worstN = Math.min(SIDE_MAX, Math.ceil(scored.length / 2));
  const bestN = Math.min(SIDE_MAX, scored.length - worstN);
  return { worst: scored.slice(0, worstN), best: scored.slice(scored.length - bestN).reverse() };
}

function RankingColumn({ title, blocks }: { title: string; blocks: BlockHealth[] }) {
  const f = useFormat();
  const items: BarItem[] = blocks.map((b) => ({
    key: b.vineyard_id,
    value: b.score ?? 0,
    valueText: f.num(b.score ?? 0, 0),
    color: (b.score ?? 0) < HEALTH_ATTENTION ? chartPalette.healthLow : chartPalette.healthOk,
    label: (
      <Box component="span" sx={{ display: "inline-flex", alignItems: "center", gap: 1.5 }}>
        <LinkChip size="small" label={b.vineyard_id} href={`/harta?bloc=${b.vineyard_id}`} />
        {b.farm_id && (
          <Typography component="span" variant="caption" color="text.secondary">
            {b.farm_id}
          </Typography>
        )}
      </Box>
    ),
  }));
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>
        {title}
      </Typography>
      <BarList items={items} max={SCORE_MAX} ariaLabel={title} />
    </Box>
  );
}

/** Two columns: weakest blocks first (where to look), strongest blocks second. */
export function HealthRanking({ blocks }: { blocks: BlockHealth[] }) {
  const t = useTranslations("stats.health");
  const { worst, best } = splitRanking(blocks);
  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" }, gap: 5 }}>
      <RankingColumn title={t("rankingWorst")} blocks={worst} />
      <RankingColumn title={t("rankingBest")} blocks={best} />
    </Box>
  );
}
