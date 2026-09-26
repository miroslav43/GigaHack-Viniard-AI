"use client";

import { useState, type FormEvent } from "react";
import { useLocale, useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import ArrowBack from "@mui/icons-material/ArrowBack";
import MarkEmailReadOutlined from "@mui/icons-material/MarkEmailReadOutlined";
import { getPathname } from "@/i18n/routing";
import { createRecoveryClient } from "@/lib/supabase/client";
import { AUTH_ENABLED } from "@/lib/supabase/config";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * "Forgot password": Supabase emails a one-time link to /parola-noua, where the owner picks a new password.
 * The answer is the same whether or not the address has an account, so the form cannot be used to find accounts.
 * The link only works if this host is in Supabase → Auth → URL Configuration → Redirect URLs.
 */
export function ForgotPasswordForm({ initialEmail, onBack }: { initialEmail: string; onBack: () => void }) {
  const t = useTranslations("auth");
  const locale = useLocale();
  const [email, setEmail] = useState(initialEmail);
  const [busy, setBusy] = useState(false);
  const [sentTo, setSentTo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const valid = EMAIL_RE.test(email.trim());

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!AUTH_ENABLED || !valid) return;
    setBusy(true);
    setError(null);
    const address = email.trim().toLowerCase();
    const redirectTo = `${window.location.origin}${getPathname({ href: "/parola-noua", locale })}?flow=reset`;
    const { error: err } = await createRecoveryClient().auth.resetPasswordForEmail(address, { redirectTo });
    setBusy(false);
    if (err) {
      setError(err.code === "over_email_send_rate_limit" || err.status === 429 ? t("forgotRateLimit") : t("forgotError"));
      return;
    }
    setSentTo(address);
  };

  const back = (
    <Button variant="text" startIcon={<ArrowBack />} onClick={onBack} sx={{ alignSelf: "flex-start" }}>
      {t("backToSignIn")}
    </Button>
  );

  if (sentTo) {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
        <Box>
          <Box sx={{ color: "primary.main", "& svg": { fontSize: 44 } }}>
            <MarkEmailReadOutlined />
          </Box>
          <Typography variant="h1" component="h1" sx={{ fontSize: 34, mt: 2 }}>
            {t("forgotSentTitle")}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
            {t("forgotSent", { email: sentTo })}
          </Typography>
        </Box>
        <Alert severity="info">{t("forgotSentHint")}</Alert>
        <Button variant="outlined" size="large" onClick={() => setSentTo(null)} sx={{ py: 3 }}>
          {t("forgotResend")}
        </Button>
        {back}
      </Box>
    );
  }

  return (
    <Box component="form" onSubmit={onSubmit} noValidate sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <Box>
        <Typography variant="h1" component="h1" sx={{ fontSize: 34 }}>
          {t("forgotTitle")}
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
          {t("forgotSubtitle")}
        </Typography>
      </Box>
      {!AUTH_ENABLED && <Alert severity="info">{t("unavailable")}</Alert>}
      {error && <Alert severity="error">{error}</Alert>}
      <TextField
        label={t("email")}
        placeholder={t("emailPlaceholder")}
        type="email"
        autoComplete="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        disabled={!AUTH_ENABLED || busy}
        autoFocus
        required
        fullWidth
      />
      <Button type="submit" variant="contained" size="large" disabled={!AUTH_ENABLED || busy || !valid} sx={{ py: 3.5, fontSize: 16 }}>
        {t("forgotSubmit")}
      </Button>
      {back}
    </Box>
  );
}
