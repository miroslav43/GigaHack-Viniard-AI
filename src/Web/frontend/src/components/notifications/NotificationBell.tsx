"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import Badge from "@mui/material/Badge";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import Popover from "@mui/material/Popover";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import NotificationsOutlined from "@mui/icons-material/NotificationsOutlined";
import AssignmentIndOutlined from "@mui/icons-material/AssignmentIndOutlined";
import { Link, usePathname, useRouter } from "@/i18n/routing";
import { createClient } from "@/lib/supabase/client";
import { useFormat } from "@/lib/useFormat";
import type { TaskKind } from "@/lib/tasks";
import { useTaskTitle } from "@/components/tasks/useTaskTitle";

/** Row of public.notification (RLS: only the signed-in user's rows). Written by a trigger on public.task. */
interface Notification {
  id: number;
  kind: "task_assigned";
  task_id: number | null;
  payload: {
    title?: string;
    task_kind?: TaskKind;
    target_id?: string | null;
    row_id?: string | null;
    vineyard_id?: string | null;
    gap_length_m?: number | null;
    due_date?: string | null;
    actor_email?: string | null;
    actor_name?: string | null;
  };
  read_at: string | null;
  created_at: string;
}

const COLUMNS = "id,kind,task_id,payload,read_at,created_at";
const LIMIT = 20;

/** "3 min ago" in the UI language (kept outside render: it reads the clock). */
function timeAgo(iso: string, locale: string) {
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const s = Math.round((new Date(iso).getTime() - Date.now()) / 1000);
  const abs = Math.abs(s);
  if (abs < 60) return rtf.format(s, "second");
  if (abs < 3600) return rtf.format(Math.round(s / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(s / 3600), "hour");
  return rtf.format(Math.round(s / 86400), "day");
}

/**
 * The bell of the app shell: unread count, the latest notifications, and a toast when one arrives.
 * Live through Supabase Realtime (postgres_changes on public.notification, filtered by RLS and user_id);
 * also reloads when the tab becomes visible again, in case the socket was asleep.
 */
export function NotificationBell({ userId }: { userId: string }) {
  const t = useTranslations("notifications");
  const locale = useLocale();
  const f = useFormat();
  const taskTitle = useTaskTitle();
  const router = useRouter();
  const pathname = usePathname();
  const pathRef = useRef(pathname);
  const [supabase] = useState(createClient);
  const [items, setItems] = useState<Notification[]>([]);
  const [unread, setUnread] = useState(0);
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const [toast, setToast] = useState<Notification | null>(null);

  useEffect(() => {
    pathRef.current = pathname;
  }, [pathname]);

  const load = useCallback(async () => {
    const [list, count] = await Promise.all([
      supabase.from("notification").select(COLUMNS).order("created_at", { ascending: false }).limit(LIMIT),
      supabase.from("notification").select("id", { count: "exact", head: true }).is("read_at", null),
    ]);
    if (!list.error) setItems((list.data ?? []) as Notification[]);
    if (!count.error) setUnread(count.count ?? 0);
  }, [supabase]);

  useEffect(() => {
    const filter = `user_id=eq.${userId}`;
    const channel = supabase
      .channel(`notifications:${userId}`)
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "notification", filter }, (msg) => {
        const n = msg.new as Notification;
        setItems((prev) => [n, ...prev.filter((x) => x.id !== n.id)].slice(0, LIMIT));
        setUnread((c) => c + 1);
        setToast(n);
        // the task list is server-rendered: show the new task without a manual reload
        if (pathRef.current.startsWith("/sarcini")) router.refresh();
      })
      // read in another tab / device
      .on("postgres_changes", { event: "UPDATE", schema: "public", table: "notification", filter }, () => void load())
      // first load once listening (nothing slips between the two); also when the socket cannot connect
      .subscribe((status) => {
        if (status === "SUBSCRIBED" || status === "CHANNEL_ERROR" || status === "TIMED_OUT") void load();
      });
    const onVisible = () => document.visibilityState === "visible" && void load();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      void supabase.removeChannel(channel);
    };
  }, [supabase, userId, load, router]);

  const markRead = async (ids: number[] | "all") => {
    const now = new Date().toISOString();
    const q = supabase.from("notification").update({ read_at: now });
    const { error } = await (ids === "all" ? q.is("read_at", null) : q.in("id", ids).is("read_at", null));
    if (error) return;
    setItems((prev) => prev.map((x) => (x.read_at || (ids !== "all" && !ids.includes(x.id)) ? x : { ...x, read_at: now })));
    setUnread((c) => (ids === "all" ? 0 : Math.max(0, c - items.filter((x) => ids.includes(x.id) && !x.read_at).length)));
  };

  const open = (n: Notification) => {
    setAnchor(null);
    setToast(null);
    if (!n.read_at) void markRead([n.id]);
    if (n.task_id) router.push({ pathname: "/sarcini", query: { sarcina: String(n.task_id) } });
  };

  const title = (n: Notification) =>
    n.payload.task_kind
      ? taskTitle({
          kind: n.payload.task_kind,
          row_id: n.payload.row_id ?? null,
          vineyard_id: n.payload.vineyard_id ?? null,
          gap_length_m: n.payload.gap_length_m ?? null,
        })
      : (n.payload.title ?? "");
  const actor = (n: Notification) => n.payload.actor_name || n.payload.actor_email;

  return (
    <>
      <Tooltip title={t("title")}>
        <IconButton
          size="small"
          onClick={(e) => setAnchor(e.currentTarget)}
          aria-label={unread ? t("openUnread", { n: unread }) : t("title")}
          aria-haspopup="dialog"
        >
          <Badge badgeContent={unread} color="error" max={99}>
            <NotificationsOutlined />
          </Badge>
        </IconButton>
      </Tooltip>

      <Popover
        open={Boolean(anchor)}
        anchorEl={anchor}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
        transformOrigin={{ vertical: "top", horizontal: "left" }}
        slotProps={{ paper: { role: "dialog", "aria-label": t("title"), sx: { width: 380, maxWidth: "calc(100vw - 32px)", maxHeight: 520, display: "flex", flexDirection: "column" } } }}
      >
        <Box sx={{ px: 4, py: 3, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}>
          <Typography variant="subtitle1" component="h2" sx={{ fontWeight: 600 }}>
            {t("title")}
          </Typography>
          <Button size="small" onClick={() => void markRead("all")} disabled={unread === 0}>
            {t("markAll")}
          </Button>
        </Box>
        <Divider />
        {items.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ px: 4, py: 6, textAlign: "center" }}>
            {t("empty")}
          </Typography>
        ) : (
          <List dense disablePadding sx={{ overflow: "auto" }}>
            {items.map((n) => (
              <ListItemButton
                key={n.id}
                onClick={() => open(n)}
                sx={{ alignItems: "flex-start", gap: 3, px: 4, py: 3, bgcolor: n.read_at ? "transparent" : "action.hover" }}
              >
                <AssignmentIndOutlined color={n.read_at ? "disabled" : "primary"} sx={{ mt: 0.5 }} />
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: n.read_at ? 500 : 700 }}>
                    {t("taskAssigned")}
                  </Typography>
                  <Typography variant="body2">{[n.payload.target_id, title(n)].filter(Boolean).join(" · ")}</Typography>
                  <Typography variant="caption" color="text.secondary" component="p">
                    {[actor(n) && t("by", { name: actor(n)! }), n.payload.due_date && t("due", { date: f.date(n.payload.due_date) }), timeAgo(n.created_at, locale)]
                      .filter(Boolean)
                      .join(" · ")}
                  </Typography>
                </Box>
                {!n.read_at && <Box aria-label={t("unread")} sx={{ width: 8, height: 8, mt: 2, borderRadius: "50%", bgcolor: "primary.main", flexShrink: 0 }} />}
              </ListItemButton>
            ))}
          </List>
        )}
        <Divider />
        <Button component={Link} href="/sarcini" onClick={() => setAnchor(null)} sx={{ m: 2 }}>
          {t("allTasks")}
        </Button>
      </Popover>

      <Snackbar
        open={Boolean(toast)}
        autoHideDuration={8000}
        onClose={(_, reason) => reason !== "clickaway" && setToast(null)}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
      >
        <Alert
          severity="info"
          variant="filled"
          icon={<AssignmentIndOutlined />}
          onClose={() => setToast(null)}
          action={
            toast?.task_id ? (
              <Button color="inherit" size="small" onClick={() => toast && open(toast)}>
                {t("view")}
              </Button>
            ) : undefined
          }
          sx={{ alignItems: "center" }}
        >
          {toast && t("toast", { title: [toast.payload.target_id, title(toast)].filter(Boolean).join(" · ") })}
        </Alert>
      </Snackbar>
    </>
  );
}
