"use client";

import { useTranslations } from "next-intl";
import { useFormat } from "@/lib/useFormat";
import type { TaskKind } from "@/lib/tasks";

export type Titled = { kind: TaskKind; row_id: string | null; vineyard_id: string | null; gap_length_m: number | null };

/** Localized task title (the stored title is Romanian); shared by the tasks table and the notifications. */
export function useTaskTitle() {
  const t = useTranslations("tasks");
  const f = useFormat();
  return (x: Titled) => {
    if (x.kind === "gap") return x.gap_length_m ? t("titleGap", { m: f.m(x.gap_length_m, 1), row: x.row_id ?? "?" }) : t("titleGapNoLen", { row: x.row_id ?? "?" });
    if (x.kind === "missing") return t("titleMissing", { row: x.row_id ?? "?" });
    if (x.kind === "waste") return t("titleWaste", { block: x.vineyard_id ?? "?" });
    return t("titleOther");
  };
}
