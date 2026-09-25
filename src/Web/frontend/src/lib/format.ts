// Number formatting — every displayed figure goes through here (unit always shown, locale-aware).
// Server components: `makeFormat(await getLocale())`; client components: `useFormat()` from ./useFormat.

const INTL_LOCALE: Record<string, string> = { ro: "ro-RO", en: "en-GB", ru: "ru-RU" };

const UNITS: Record<string, { m: string; m2: string; ha: string; km: string; min: string; kmh: string; cmpx: string }> = {
  ro: { m: "m", m2: "m²", ha: "ha", km: "km", min: "min", kmh: "km/h", cmpx: "cm/px" },
  en: { m: "m", m2: "m²", ha: "ha", km: "km", min: "min", kmh: "km/h", cmpx: "cm/px" },
  ru: { m: "м", m2: "м²", ha: "га", km: "км", min: "мин", kmh: "км/ч", cmpx: "см/пкс" },
};

export function makeFormat(locale: string) {
  const intl = INTL_LOCALE[locale] ?? INTL_LOCALE.ro;
  const u = UNITS[locale] ?? UNITS.ro;
  const nf = (min: number, max: number) => new Intl.NumberFormat(intl, { minimumFractionDigits: min, maximumFractionDigits: max });

  const num = (v: number, digits = 1) => nf(digits, digits).format(v);
  const int = (v: number) => nf(0, 0).format(v);
  const m = (v: number, digits = 1) => `${num(v, digits)} ${u.m}`;
  const km = (v: number) => `${num(v / 1000, 2)} ${u.km}`;
  const m2 = (v: number, digits = 1) => `${num(v, digits)} ${u.m2}`;
  const ha = (vM2: number, digits?: number) => {
    const h = vM2 / 1e4;
    return `${num(h, digits ?? (h < 1 ? 4 : 2))} ${u.ha}`;
  };
  return {
    units: u,
    num,
    int,
    m,
    km,
    /** metres below 1 km, kilometres above */
    length: (v: number) => (v >= 1000 ? km(v) : m(v)),
    m2,
    ha,
    /** m² with hectares alongside, e.g. "536,2 m² · 0,0536 ha" */
    area: (vM2: number) => `${m2(vM2)} · ${ha(vM2)}`,
    min: (v: number) => `${int(Math.round(v))} ${u.min}`,
    pct: (v: number) => `${num(v * 100, 0)} %`,
    date: (iso: string) => new Intl.DateTimeFormat(intl, { day: "2-digit", month: "2-digit", year: "numeric" }).format(new Date(iso)),
  };
}

export type Format = ReturnType<typeof makeFormat>;
