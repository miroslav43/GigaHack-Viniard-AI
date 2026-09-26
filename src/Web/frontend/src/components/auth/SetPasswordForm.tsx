"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { useTranslations } from "next-intl";
import type { EmailOtpType } from "@supabase/supabase-js";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import Visibility from "@mui/icons-material/VisibilityOutlined";
import VisibilityOff from "@mui/icons-material/VisibilityOffOutlined";
import { Link, useRouter } from "@/i18n/routing";
import { createClient } from "@/lib/supabase/client";
import { AUTH_ENABLED, DEMO_COOKIE } from "@/lib/supabase/config";

const MIN_LENGTH = 10;

type State = { kind: "loading" } | { kind: "invalid"; expired: boolean } | { kind: "ready"; email: string } | { kind: "done" };

/**
 * Landing page of the invitation email. Supabase's default link returns the session in the URL fragment
 * (#access_token=…), which the server never sees, so it is read here; a customised template that sends
 * ?token_hash=…&type=invite is verified here too. Any signed-in user can also change the password here.
 */
export function SetPasswordForm() {
  const t = useTranslations("setPassword");
  const router = useRouter();
  const [state, setState] = useState<State>(AUTH_ENABLED ? { kind: "loading" } : { kind: "invalid", expired: false });
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    // the link's tokens are single-use: run once, also under StrictMode's double effect
    if (started.current || !AUTH_ENABLED) return;
    started.current = true;
    const supabase = createClient();
    const hash = new URLSearchParams(window.location.hash.slice(1));
    const query = new URLSearchParams(window.location.search);
    // drop the tokens from the address bar and the history
    if (window.location.hash || query.has("token_hash")) window.history.replaceState(null, "", window.location.pathname);

    (async () => {
      if (hash.has("error") || hash.has("error_code")) {
        setState({ kind: "invalid", expired: true });
        return;
      }
      const accessToken = hash.get("access_token");
      const refreshToken = hash.get("refresh_token");
      const tokenHash = query.get("token_hash");
      if (accessToken && refreshToken) {
        const { error: err } = await supabase.auth.setSession({ access_token: accessToken, refresh_token: refreshToken });
        if (err) return setState({ kind: "invalid", expired: true });
      } else if (tokenHash) {
        const { error: err } = await supabase.auth.verifyOtp({ type: (query.get("type") ?? "invite") as EmailOtpType, token_hash: tokenHash });
        if (err) return setState({ kind: "invalid", expired: true });
      }
      const { data } = await supabase.auth.getUser();
      setState(data.user ? { kind: "ready", email: data.user.email ?? "" } : { kind: "invalid", expired: false });
    })();
  }, []);

  const tooShort = password.length > 0 && password.length < MIN_LENGTH;
  const mismatch = confirm.length > 0 && confirm !== password;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (password.length < MIN_LENGTH || password !== confirm) return;
    setBusy(true);
    setError(null);
    const { error: err } = await createClient().auth.updateUser({ password });
    if (err) {
      setError(err.code === "same_password" ? t("samePassword") : err.code === "weak_password" ? t("weak") : t("error"));
      setBusy(false);
      return;
    }
    document.cookie = `${DEMO_COOKIE}=; Max-Age=0; path=/; SameSite=Lax`;
    setState({ kind: "done" });
    router.replace("/");
    router.refresh();
  };

  const heading = (
    <Box>
      <Typography variant="h1" component="h1" sx={{ fontSize: 34 }}>
        {t("title")}
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
        {state.kind === "ready" ? t("subtitle", { email: state.email }) : t("subtitleGeneric")}
      </Typography>
    </Box>
  );

  if (state.kind === "loading" || state.kind === "done") {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {heading}
        <Box sx={{ display: "flex", alignItems: "center", gap: 3, color: "text.secondary" }}>
          <CircularProgress size={20} />
          <Typography variant="body2">{state.kind === "done" ? t("saved") : t("checking")}</Typography>
        </Box>
      </Box>
    );
  }

  if (state.kind === "invalid") {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {heading}
        <Alert severity="warning">{state.expired ? t("expired") : t("invalid")}</Alert>
        <Button component={Link} href="/login" variant="outlined" size="large" sx={{ py: 3 }}>
          {t("toLogin")}
        </Button>
      </Box>
    );
  }

  const visibility = (
    <InputAdornment position="end">
      <IconButton onClick={() => setShow((v) => !v)} edge="end" aria-label={show ? t("hide") : t("show")}>
        {show ? <VisibilityOff /> : <Visibility />}
      </IconButton>
    </InputAdornment>
  );

  return (
    <Box component="form" onSubmit={onSubmit} noValidate sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      {heading}
      {error && <Alert severity="error">{error}</Alert>}
      {/* lets password managers store the new password under the right account */}
      <input type="email" name="email" autoComplete="username" value={state.email} readOnly hidden />
      <TextField
        label={t("password")}
        type={show ? "text" : "password"}
        autoComplete="new-password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        error={tooShort}
        helperText={t("hint", { n: MIN_LENGTH })}
        disabled={busy}
        autoFocus
        required
        fullWidth
        slotProps={{ input: { endAdornment: visibility } }}
      />
      <TextField
        label={t("confirm")}
        type={show ? "text" : "password"}
        autoComplete="new-password"
        value={confirm}
        onChange={(e) => setConfirm(e.target.value)}
        error={mismatch}
        helperText={mismatch ? t("mismatch") : " "}
        disabled={busy}
        required
        fullWidth
      />
      <Button
        type="submit"
        variant="contained"
        size="large"
        disabled={busy || password.length < MIN_LENGTH || password !== confirm}
        sx={{ py: 3.5, fontSize: 16 }}
      >
        {t("submit")}
      </Button>
    </Box>
  );
}
