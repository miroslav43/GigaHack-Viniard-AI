"use client";

import { useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import MuiLink from "@mui/material/Link";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import Visibility from "@mui/icons-material/VisibilityOutlined";
import VisibilityOff from "@mui/icons-material/VisibilityOffOutlined";
import ArrowForward from "@mui/icons-material/ArrowForward";
import { useRouter } from "@/i18n/routing";
import { createClient } from "@/lib/supabase/client";
import { AUTH_ENABLED, DEMO_COOKIE } from "@/lib/supabase/config";
import { ForgotPasswordForm } from "./ForgotPasswordForm";

/** Only same-app relative paths are accepted as a post-login target. */
const safeNext = (v: string | null) => (v && v.startsWith("/") && !v.startsWith("//") ? v : "/");

export function LoginForm() {
  const t = useTranslations("auth");
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(AUTH_ENABLED ? null : t("unavailable"));
  // ?forgot=1 (from an expired reset link) opens the "forgot password" step directly
  const [forgot, setForgot] = useState(params.get("forgot") === "1");

  const go = () => {
    router.replace(safeNext(params.get("next")));
    router.refresh();
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!AUTH_ENABLED) return;
    setBusy(true);
    setError(null);
    const { error: err } = await createClient().auth.signInWithPassword({ email: email.trim(), password });
    if (err) {
      setError(err.code === "invalid_credentials" ? t("error") : err.code === "user_banned" ? t("banned") : t("errorGeneric"));
      setBusy(false);
      return;
    }
    document.cookie = `${DEMO_COOKIE}=; Max-Age=0; path=/; SameSite=Lax`;
    go();
  };

  const startDemo = () => {
    document.cookie = `${DEMO_COOKIE}=1; Max-Age=${60 * 60 * 24 * 7}; path=/; SameSite=Lax`;
    go();
  };

  if (forgot) return <ForgotPasswordForm initialEmail={email} onBack={() => setForgot(false)} />;

  return (
    <Box component="form" onSubmit={onSubmit} noValidate sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <Box>
        <Typography variant="h1" component="h1" sx={{ fontSize: 34 }}>
          {t("welcome")}
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
          {t("subtitle")}
        </Typography>
      </Box>
      {error && <Alert severity={AUTH_ENABLED ? "error" : "info"}>{error}</Alert>}
      <TextField
        label={t("email")}
        placeholder={t("emailPlaceholder")}
        type="email"
        autoComplete="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        disabled={!AUTH_ENABLED || busy}
        required
        fullWidth
      />
      <TextField
        label={t("password")}
        placeholder={t("passwordPlaceholder")}
        type={show ? "text" : "password"}
        autoComplete="current-password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        disabled={!AUTH_ENABLED || busy}
        required
        fullWidth
        slotProps={{
          input: {
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={() => setShow((v) => !v)} edge="end" aria-label={show ? t("hidePassword") : t("showPassword")}>
                  {show ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />
      <MuiLink
        component="button"
        type="button"
        variant="body2"
        underline="hover"
        onClick={() => setForgot(true)}
        disabled={!AUTH_ENABLED}
        sx={{ alignSelf: "flex-end", mt: -3, fontWeight: 500 }}
      >
        {t("forgot")}
      </MuiLink>
      <Button type="submit" variant="contained" size="large" disabled={!AUTH_ENABLED || busy || !email || !password} sx={{ py: 3.5, fontSize: 16 }}>
        {t("signIn")}
      </Button>
      <Divider>
        <Typography variant="body2" color="text.secondary">
          {t("or")}
        </Typography>
      </Divider>
      <Button variant="outlined" size="large" endIcon={<ArrowForward />} onClick={startDemo} sx={{ py: 3 }}>
        {t("demo")}
      </Button>
      <Typography variant="caption" color="text.secondary" sx={{ textAlign: "center", mt: -3 }}>
        {t("demoHint")}
      </Typography>
    </Box>
  );
}
