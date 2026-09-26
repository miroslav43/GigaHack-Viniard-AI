"use client";

// Cadastre snapshot of a farm or a block (optional n_parcels, landuse_counts, cadastral_codes of farms.geojson /
// blocks.geojson): the parcel count, the land-use breakdown and the codes, collapsed and capped. Renders nothing
// when the survey has no snapshot.
import { useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { useTranslations } from "next-intl";
import { useFormat } from "@/lib/useFormat";
import { Field } from "../AttributePanel";

/** cadastral codes shown when expanded; the rest is a "+K" */
const MAX_CODES = 20;

/** MapLibre serialises nested arrays / objects of feature properties to JSON strings: read either form. */
const nested = (v: unknown): unknown => {
  if (typeof v !== "string" || !/^[[{]/.test(v)) return v;
  try {
    return JSON.parse(v) as unknown;
  } catch {
    return null;
  }
};
const countOf = (v: unknown) => (typeof v === "number" && Number.isInteger(v) && v >= 0 ? v : null);
const codesOf = (v: unknown) => {
  const x = nested(v);
  return Array.isArray(x) ? x.filter((c): c is string => typeof c === "string") : null;
};
const landuseOf = (v: unknown) => {
  const x = nested(v);
  if (!x || typeof x !== "object" || Array.isArray(x)) return null;
  const entries = Object.entries(x as Record<string, unknown>).flatMap(([k, n]) => (countOf(n) === null ? [] : [[k, n as number] as const]));
  return entries.sort((a, b) => b[1] - a[1]);
};

export function CadastreSnapshot({ props, parcels }: { props: Record<string, unknown>; parcels?: number | null }) {
  const t = useTranslations("map.cadastreSnapshot");
  const f = useFormat();
  const [open, setOpen] = useState(false);
  const n = countOf(parcels ?? props.n_parcels);
  const landuse = landuseOf(props.landuse_counts);
  const codes = codesOf(props.cadastral_codes);
  if (n === null && !landuse?.length && !codes?.length) return null;
  const shown = codes?.slice(0, MAX_CODES) ?? [];
  const more = (codes?.length ?? 0) - shown.length;
  return (
    <Box data-testid="cadastre-snapshot">
      {n !== null && <Field label={t("parcels")}>{f.int(n)}</Field>}
      {landuse?.map(([use, count]) => (
        <Field key={use} label={t("landuse", { use })}>
          {f.int(count)}
        </Field>
      ))}
      {codes && codes.length > 0 && (
        <Box sx={{ py: 1 }}>
          <Button size="small" onClick={() => setOpen((v) => !v)} aria-expanded={open} sx={{ px: 0 }}>
            {t(open ? "hideCodes" : "showCodes", { n: codes.length })}
          </Button>
          {open && (
            <Typography variant="caption" component="p" sx={{ wordBreak: "break-word" }}>
              {shown.join(", ")}
              {more > 0 && ` ${t("more", { n: more })}`}
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
}
