"use client";

import { useState, type FormEvent } from "react";
import { useLocale, useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import MenuItem from "@mui/material/MenuItem";
import TextField from "@mui/material/TextField";
import { requestDemo, type LeadResult } from "@/app/[locale]/prezentare/actions";
import { INSTITUTION_TYPES } from "@/lib/leads";

/** "Request a pilot" form of the presentation site (server action → public.lead). */
export function LeadForm() {
  const t = useTranslations("landing.contact");
  const locale = useLocale();
  // the fill time is measured from the first render (bots submit instantly)
  const [startedAt] = useState(() => String(Date.now()));
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<LeadResult | null>(null);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBusy(true);
    const res = await requestDemo(new FormData(e.currentTarget)).catch((): LeadResult => ({ ok: false, error: "generic" }));
    setResult(res);
    setBusy(false);
  };

  if (result?.ok) return <Alert severity="success">{t("success")}</Alert>;

  return (
    <Box component="form" onSubmit={onSubmit} sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" }, gap: 4 }}>
      <input type="hidden" name="locale" value={locale} />
      <input type="hidden" name="started_at" value={startedAt} />
      {/* honeypot: hidden from people, filled by bots */}
      <Box aria-hidden sx={{ position: "absolute", left: -10000, width: 1, height: 1, overflow: "hidden" }}>
        <label>
          Website <input type="text" name="website" tabIndex={-1} autoComplete="off" />
        </label>
      </Box>
      <TextField name="name" label={t("name")} autoComplete="name" required fullWidth />
      <TextField name="position" label={t("position")} autoComplete="organization-title" fullWidth />
      <TextField name="institution" label={t("institution")} autoComplete="organization" required fullWidth />
      <TextField name="institution_type" label={t("institutionType")} select required fullWidth defaultValue="primarie">
        {INSTITUTION_TYPES.map((v) => (
          <MenuItem key={v} value={v}>
            {t(`types.${v}`)}
          </MenuItem>
        ))}
      </TextField>
      <TextField name="email" type="email" label={t("email")} autoComplete="email" required fullWidth />
      <TextField name="phone" type="tel" label={t("phone")} autoComplete="tel" fullWidth />
      <TextField
        name="message"
        label={t("message")}
        placeholder={t("messagePlaceholder")}
        multiline
        minRows={3}
        fullWidth
        sx={{ gridColumn: { sm: "1 / -1" } }}
        slotProps={{ htmlInput: { maxLength: 2000 } }}
      />
      <FormControlLabel
        sx={{ gridColumn: { sm: "1 / -1" }, alignItems: "flex-start", "& .MuiCheckbox-root": { mt: -1 } }}
        control={<Checkbox name="consent" required />}
        label={t("consent")}
      />
      {result && !result.ok && (
        <Alert severity={result.error === "invalid" ? "warning" : "error"} sx={{ gridColumn: { sm: "1 / -1" } }}>
          {t(`errors.${result.error}`)}
        </Alert>
      )}
      <Box sx={{ gridColumn: { sm: "1 / -1" } }}>
        <Button type="submit" variant="contained" size="large" disabled={busy} sx={{ px: 8, py: 3 }}>
          {t("submit")}
        </Button>
      </Box>
    </Box>
  );
}
