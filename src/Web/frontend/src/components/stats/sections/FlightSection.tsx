"use client";

import { useTranslations } from "next-intl";
import Typography from "@mui/material/Typography";
import { CountUp } from "@/components/common/CountUp";
import { useFormat } from "@/lib/useFormat";
import { TileGrid } from "../TileGrid";
import { ChartCard, StatSection, StatTile } from "../parts";
import type { SectionProps } from "../types";

const CM_PER_M = 100;

export function FlightSection({ letter, stats, survey, uatAreaHa }: SectionProps) {
  const t = useTranslations("stats.flight");
  const f = useFormat();
  const tiles = stats.tiles;
  const gsdCm = survey.gsd_m * CM_PER_M;
  const counts = tiles && {
    total: tiles.items.length,
    vineyard: tiles.items.filter((x) => x.status === "vineyard").length,
    noVineyard: tiles.items.filter((x) => x.status !== "vineyard").length,
    toComplete: tiles.items.filter((x) => x.to_complete).length,
  };
  const gsd = f.num(gsdCm, 1);

  return (
    <StatSection
      id="zbor"
      letter={letter}
      title={t("title")}
      lead={
        counts
          ? t("lead", {
              total: f.int(counts.total),
              gsd,
              vineyard: f.int(counts.vineyard),
              noVineyard: f.int(counts.noVineyard),
              toComplete: f.int(counts.toComplete),
            })
          : t("leadNoTiles", { total: f.int(survey.tiles_total), gsd })
      }
    >
      {tiles && (
        <ChartCard title={t("gridTitle")} subtitle={t("gridSubtitle")} wide minHeight={360}>
          <TileGrid
            tiles={tiles}
            labels={{
              aria: t("gridAria", { total: f.int(tiles.items.length) }),
              tileTitle: (x) =>
                t(x.status === "vineyard" ? "tileVineyard" : "tileNone", { tile: x.tile, veg: f.pct(x.veg_frac) }) +
                (x.to_complete ? t("tileToComplete") : ""),
              legendVeg: t("legendVeg"),
              legendNone: t("legendNone"),
              legendToComplete: t("legendToComplete"),
              pct: (v) => f.pct(v),
            }}
          />
        </ChartCard>
      )}

      <StatTile
        label={t("tilesLabel")}
        value={<CountUp value={counts?.total ?? survey.tiles_total} />}
        hint={
          counts
            ? t("tilesHint", { vineyard: f.int(counts.vineyard), noVineyard: f.int(counts.noVineyard), toComplete: f.int(counts.toComplete) })
            : undefined
        }
      />
      <StatTile
        label={t("gsdLabel")}
        value={<CountUp value={gsdCm} format="num" digits={1} suffix={` ${f.units.cmpx}`} />}
        hint={t("gsdHint", { date: f.date(survey.captured_at), source: survey.source })}
        delayMs={60}
      />
      <StatTile
        label={t("areaLabel")}
        value={<CountUp value={survey.surveyed_area_ha} format="num" digits={1} suffix={` ${f.units.ha}`} />}
        hint={uatAreaHa > 0 ? t("areaHint", { share: f.pct(survey.surveyed_area_ha / uatAreaHa) }) : undefined}
        delayMs={120}
      />
      {tiles && (
        <StatTile
          label={t("vegLabel")}
          value={<CountUp value={tiles.veg_frac_mean} format="pct" />}
          hint={t("vegHint")}
          delayMs={180}
        />
      )}
      {survey.model_version && (
        <StatTile
          label={t("modelLabel")}
          value={
            <Typography component="span" sx={{ fontSize: 15, fontWeight: 600, fontFamily: "monospace", wordBreak: "break-all" }}>
              {survey.model_version}
            </Typography>
          }
          hint={survey.run_id ? t("modelRun", { run: survey.run_id }) : undefined}
          delayMs={240}
        />
      )}
    </StatSection>
  );
}
