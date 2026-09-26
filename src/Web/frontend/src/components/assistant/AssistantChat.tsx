"use client";

import { useEffect, useRef, useState, useSyncExternalStore, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { useLocale, useTranslations } from "next-intl";
import ReactMarkdown, { type Components } from "react-markdown";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Fab from "@mui/material/Fab";
import IconButton from "@mui/material/IconButton";
import MuiLink from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import AutoAwesomeOutlined from "@mui/icons-material/AutoAwesomeOutlined";
import CloseOutlined from "@mui/icons-material/CloseOutlined";
import RestartAltOutlined from "@mui/icons-material/RestartAltOutlined";
import SendOutlined from "@mui/icons-material/SendOutlined";
import { Link, usePathname } from "@/i18n/routing";
import type { UatRole } from "@/lib/viewer";

type Message = { role: "user" | "assistant"; text: string; error?: boolean };

const STORE = "solemtrix.assistant";
const MAX_QUESTION = 1000;

type Stored = { open: boolean; messages: Message[] };
const EMPTY: Stored = { open: false, messages: [] };

/**
 * Conversation + open state, outside React: survives navigation between the app and admin shells, and a reload
 * of the tab (sessionStorage, this tab only). The server snapshot is always empty, so hydration never mismatches.
 */
const store = (() => {
  let snapshot: Stored | null = null;
  const listeners = new Set<() => void>();
  const get = () => {
    if (!snapshot) {
      try {
        snapshot = JSON.parse(sessionStorage.getItem(STORE) ?? "null") ?? EMPTY;
      } catch {
        snapshot = EMPTY;
      }
    }
    return snapshot!;
  };
  return {
    get,
    server: () => EMPTY,
    set(fn: (s: Stored) => Stored) {
      snapshot = fn(get());
      try {
        sessionStorage.setItem(STORE, JSON.stringify(snapshot));
      } catch {}
      listeners.forEach((l) => l());
    },
    subscribe(l: () => void) {
      listeners.add(l);
      return () => void listeners.delete(l);
    },
  };
})();

/** Only in-app paths become links (client navigation, so this panel stays open); anything else is plain text. */
const isInternal = (href: string | undefined): href is string => !!href && href.startsWith("/") && !href.startsWith("//");

const FAB = 56;
const GAP = 12;
/** Mobile bottom navigation height (AppShell) — the button sits above it. */
const BOTTOM_NAV = 64;

/**
 * Where the round button sits: bottom-right on every page; on the map, left of the zoom buttons and the ruler
 * (MapLibre NavigationControl + MeasureTool, both at the right edge) and above the attribution line.
 */
function fabPosition(mobile: boolean, onMap: boolean, bottomNav: boolean) {
  if (mobile) return { right: onMap ? 64 : 16, bottom: (bottomNav ? BOTTOM_NAV : 0) + (onMap ? 64 : 16) };
  return { right: onMap ? 64 : 24, bottom: onMap ? 36 : 24 };
}

/**
 * The assistant of the app shell: a round button, always at the bottom right, opens a non-modal panel (the page stays usable) where
 * Gemini explains how to do things in Solemtrix, with links to the right page. The shell layout persists across
 * navigation, so following a link keeps the conversation open.
 */
export function AssistantChat({ role, demo, bottomNav = true }: { role: UatRole | null; demo: boolean; bottomNav?: boolean }) {
  const t = useTranslations("assistant");
  const locale = useLocale();
  const pathname = usePathname();
  const theme = useTheme();
  const mobile = useMediaQuery(theme.breakpoints.down("md"));
  const state = useSyncExternalStore(store.subscribe, store.get, store.server);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { open, messages } = state;
  const fab = fabPosition(mobile, pathname.startsWith("/harta"), bottomNav);
  const panelBottom = fab.bottom + FAB + GAP;

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages, open]);

  const setMessages = (fn: (m: Message[]) => Message[]) => store.set((s) => ({ ...s, messages: fn(s.messages) }));
  const setOpen = (value: boolean) => store.set((s) => ({ ...s, open: value }));

  const ask = async (question: string) => {
    const q = question.trim().slice(0, MAX_QUESTION);
    if (!q || busy) return;
    const history = [...messages.filter((m) => !m.error), { role: "user" as const, text: q }];
    setMessages(() => [...history, { role: "assistant", text: "" }]);
    setInput("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const fail = (key: string) =>
      setMessages((m) => [...m.slice(0, -1), { role: "assistant", text: t(`errors.${key}`), error: true }]);
    try {
      const res = await fetch("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.map(({ role: r, text }) => ({ role: r, text })), locale, path: pathname }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const code = (await res.json().catch(() => ({}))).error;
        fail(["not_configured", "rate_limit", "sign_in"].includes(code) ? code : "generic");
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let text = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        setMessages((m) => [...m.slice(0, -1), { role: "assistant", text }]);
      }
      if (!text.trim()) fail("generic");
    } catch (e) {
      if ((e as Error).name !== "AbortError") fail("generic");
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(input);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void ask(input);
    }
  };
  const reset = () => {
    abortRef.current?.abort();
    setMessages(() => []);
  };

  const suggestions: string[] = demo
    ? t.raw("suggestions.demo")
    : role === "uat_admin"
      ? t.raw("suggestions.admin")
      : role === "platform_admin"
        ? t.raw("suggestions.platform")
        : t.raw("suggestions.member");

  const markdown: Components = {
    a: ({ href, children }) =>
      isInternal(href) ? (
        <MuiLink component={Link} href={href} sx={{ fontWeight: 600 }}>
          {children}
        </MuiLink>
      ) : (
        <>{children}</>
      ),
    p: ({ children }) => <Typography variant="body2" sx={{ my: 1.5 }}>{children}</Typography>,
    ol: ({ children }) => <Box component="ol" sx={{ pl: 5, my: 1.5, "& li": { mb: 1 } }}>{children}</Box>,
    ul: ({ children }) => <Box component="ul" sx={{ pl: 5, my: 1.5, "& li": { mb: 1 } }}>{children}</Box>,
    li: ({ children }) => <Typography component="li" variant="body2">{children}</Typography>,
    img: () => null,
  };

  const bubble = (m: Message, i: number): ReactNode => (
    <Box
      key={i}
      sx={{
        alignSelf: m.role === "user" ? "flex-end" : "stretch",
        maxWidth: m.role === "user" ? "85%" : "100%",
        px: 3,
        py: m.role === "user" ? 2 : 0.5,
        borderRadius: 2,
        bgcolor: m.role === "user" ? "primary.main" : "transparent",
        color: m.role === "user" ? "primary.contrastText" : "text.primary",
        "& > :first-of-type": { mt: 0 },
        "& > :last-child": { mb: 0 },
      }}
    >
      {m.role === "user" ? (
        <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
          {m.text}
        </Typography>
      ) : m.error ? (
        <Alert severity="warning">{m.text}</Alert>
      ) : m.text ? (
        <ReactMarkdown components={markdown} skipHtml>
          {m.text}
        </ReactMarkdown>
      ) : (
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, color: "text.secondary", py: 1 }}>
          <CircularProgress size={16} />
          <Typography variant="body2">{t("thinking")}</Typography>
        </Box>
      )}
    </Box>
  );

  return (
    <>
      <Tooltip title={open ? t("close") : t("open")} placement="left">
        <Fab
          color="primary"
          onClick={() => setOpen(!open)}
          aria-label={open ? t("close") : t("open")}
          aria-expanded={open}
          sx={{ position: "fixed", zIndex: (th) => th.zIndex.modal - 1, right: fab.right, bottom: fab.bottom, width: FAB, height: FAB }}
        >
          {open ? <CloseOutlined /> : <AutoAwesomeOutlined />}
        </Fab>
      </Tooltip>

      {open && (
        <Paper
          role="dialog"
          aria-label={t("title")}
          elevation={8}
          sx={{
            position: "fixed",
            zIndex: (th) => th.zIndex.modal - 1,
            right: mobile ? 8 : fab.right,
            left: mobile ? 8 : "auto",
            bottom: panelBottom,
            width: mobile ? "auto" : 420,
            height: `min(${mobile ? 560 : 640}px, calc(100dvh - ${panelBottom + (mobile ? 72 : 24)}px))`,
            display: "flex",
            flexDirection: "column",
            borderRadius: 3,
            overflow: "hidden",
          }}
        >
          <Box sx={{ px: 4, py: 3, display: "flex", alignItems: "center", gap: 2, borderBottom: 1, borderColor: "divider" }}>
            <AutoAwesomeOutlined color="primary" />
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="subtitle1" component="h2" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
                {t("title")}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {t("subtitle")}
              </Typography>
            </Box>
            <Tooltip title={t("reset")}>
              <span>
                <IconButton size="small" onClick={reset} disabled={messages.length === 0} aria-label={t("reset")}>
                  <RestartAltOutlined fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
            <IconButton size="small" onClick={() => setOpen(false)} aria-label={t("close")}>
              <CloseOutlined fontSize="small" />
            </IconButton>
          </Box>

          <Box ref={listRef} aria-live="polite" sx={{ flex: 1, overflow: "auto", px: 3, py: 3, display: "flex", flexDirection: "column", gap: 3 }}>
            {messages.length === 0 ? (
              <Box sx={{ px: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  {t("intro")}
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, mt: 4 }}>
                  {suggestions.map((s) => (
                    <Chip key={s} label={s} variant="outlined" onClick={() => void ask(s)} sx={{ height: "auto", py: 1, "& .MuiChip-label": { whiteSpace: "normal" } }} />
                  ))}
                </Box>
              </Box>
            ) : (
              messages.map(bubble)
            )}
          </Box>

          <Box component="form" onSubmit={onSubmit} sx={{ p: 3, borderTop: 1, borderColor: "divider", display: "flex", gap: 2, alignItems: "flex-end" }}>
            <TextField
              value={input}
              onChange={(e) => setInput(e.target.value.slice(0, MAX_QUESTION))}
              onKeyDown={onKey}
              placeholder={t("placeholder")}
              aria-label={t("placeholder")}
              multiline
              maxRows={4}
              size="small"
              fullWidth
              autoFocus={!mobile}
            />
            <IconButton type="submit" color="primary" disabled={busy || !input.trim()} aria-label={t("send")}>
              <SendOutlined />
            </IconButton>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ px: 4, pb: 2, mt: -1 }}>
            {t("disclaimer")}
          </Typography>
        </Paper>
      )}
    </>
  );
}
