import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { getLocale, getTranslations } from "next-intl/server";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Link from "@mui/material/Link";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import { createClient } from "@/lib/supabase/server";
import { ADMIN_API_ENABLED } from "@/lib/supabase/admin";
import { AUTH_ENABLED, SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL } from "@/lib/supabase/config";
import { SURVEY_ID } from "@/lib/data";
import type { Viewer } from "@/lib/viewer";

type Status = "ok" | "warn" | "fail";
interface Check {
  label: string;
  status: Status;
  value: string;
  ms?: number;
}

const PROJECT_REF = SUPABASE_URL.match(/https:\/\/([a-z0-9]+)\.supabase\.co/)?.[1] ?? "";
const REPO = "https://github.com/miroslav43/GigaHack-Viniard-AI";

async function timed<T>(fn: () => Promise<T>): Promise<[T | null, number, string | null]> {
  const t0 = performance.now();
  try {
    return [await fn(), Math.round(performance.now() - t0), null];
  } catch (e) {
    return [null, Math.round(performance.now() - t0), e instanceof Error ? e.message : String(e)];
  }
}

const mask = (s: string) => (s.length > 16 ? `${s.slice(0, 14)}…${s.slice(-4)}` : s ? "•••" : "");

export async function SystemPanel({ viewer, jwtExp }: { viewer: Viewer; jwtExp: number | null }) {
  const t = await getTranslations("superAdmin.system");
  const tr = await getTranslations("auth.roles");
  const locale = await getLocale();
  const dt = (d: Date) => new Intl.DateTimeFormat(locale, { dateStyle: "short", timeStyle: "medium" }).format(d);
  const root = process.cwd();

  const [authHealth, authMs, authErr] = await timed(async () => {
    const res = await fetch(`${SUPABASE_URL}/auth/v1/health`, { headers: { apikey: SUPABASE_PUBLISHABLE_KEY }, cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as { version?: string; name?: string };
  });
  const [dbCount, dbMs, dbErr] = await timed(async () => {
    const supabase = await createClient();
    const { count, error } = await supabase.from("uat").select("*", { count: "exact", head: true });
    if (error) throw new Error(error.message);
    return count ?? 0;
  });
  const [summary, , dataErr] = await timed(async () => {
    const file = path.join(root, "public", "data", SURVEY_ID, "summary.json");
    const [json, st] = await Promise.all([readFile(file, "utf8"), stat(file)]);
    return { s: JSON.parse(json), at: st.mtime };
  });
  const [orthoTiles] = await timed(async () => (await readdir(path.join(root, "public", "ortho", "tiles"))).length);
  const pkg = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")) as { dependencies: Record<string, string>; devDependencies: Record<string, string> };

  const checks: Check[] = [
    { label: t("supabase"), status: AUTH_ENABLED ? "ok" : "fail", value: AUTH_ENABLED ? `${SUPABASE_URL} · ${mask(SUPABASE_PUBLISHABLE_KEY)}` : "—" },
    { label: t("secretKey"), status: ADMIN_API_ENABLED ? "ok" : "warn", value: ADMIN_API_ENABLED ? "SUPABASE_SECRET_KEY ✓" : "SUPABASE_SECRET_KEY —" },
    { label: t("authHealth"), status: authErr ? "fail" : "ok", value: authErr ?? `${authHealth?.name ?? "GoTrue"} ${authHealth?.version ?? ""}`, ms: authMs },
    { label: t("db"), status: dbErr ? "fail" : "ok", value: dbErr ?? `public.uat: ${dbCount}`, ms: dbMs },
    {
      label: t("session"),
      status: "ok",
      value: t("sessionValue", { email: viewer.email ?? "—", role: viewer.role ? tr(viewer.role) : "—", exp: jwtExp ? dt(new Date(jwtExp * 1000)) : "—" }),
    },
    {
      label: t("data", { survey: SURVEY_ID }),
      status: dataErr ? "fail" : "ok",
      value: dataErr ?? t("dataValue", { rows: summary!.s.totals.row_count, targets: summary!.s.totals.target_count, at: dt(summary!.at) }),
    },
    { label: t("ortho"), status: orthoTiles ? "ok" : "warn", value: orthoTiles ? t("orthoValue", { tiles: orthoTiles }) : t("orthoMissing") },
  ];

  const versions: [string, string][] = [
    ["Node.js", process.version],
    ["NODE_ENV", process.env.NODE_ENV ?? ""],
    ...(["next", "react", "@mui/material", "maplibre-gl", "next-intl", "@supabase/supabase-js", "@supabase/ssr"] as const).map(
      (p): [string, string] => [p, pkg.dependencies[p] ?? pkg.devDependencies[p] ?? "—"],
    ),
  ];

  const dash = `https://supabase.com/dashboard/project/${PROJECT_REF}`;
  const links: [string, string][] = [
    [t("linkDashboard"), dash],
    [t("linkUsers"), `${dash}/auth/users`],
    [t("linkSql"), `${dash}/sql/new`],
    [t("linkTables"), `${dash}/editor`],
    [t("linkKeys"), `${dash}/settings/api-keys`],
    [t("linkAdvisors"), `${dash}/advisors/security`],
    [t("linkRepo"), REPO],
    [t("linkActions"), `${REPO}/actions`],
  ];

  const commands: [string, string][] = [
    ["pnpm data", "ortofoto + survey siret3-mock → public/"],
    ["pnpm dev", "http://localhost:3000"],
    ["pnpm lint && pnpm typecheck && pnpm e2e", "CI local"],
    ["node scripts/gen-seed-sql.mjs", "→ src/Web/supabase/seed/uats_surveys.sql"],
    ["src/Web/supabase/migrations/", "schema (RLS) — aplicată prin MCP / supabase db push"],
    ["src/Web/supabase/seed/demo_users.sql", "conturi demo (__DEMO_PASSWORD__)"],
  ];

  const color: Record<Status, "success" | "warning" | "error"> = { ok: "success", warn: "warning", fail: "error" };

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "3fr 2fr" }, gap: 5, alignItems: "start" }}>
      <Card>
        <Typography variant="h3" sx={{ px: 5, py: 4 }}>
          {t("checks")}
        </Typography>
        <Table size="small">
          <TableBody>
            {checks.map((c) => (
              <TableRow key={c.label}>
                <TableCell sx={{ width: 110 }}>
                  <Chip size="small" color={color[c.status]} label={t(c.status)} />
                </TableCell>
                <TableCell sx={{ fontWeight: 600 }}>{c.label}</TableCell>
                <TableCell sx={{ wordBreak: "break-all" }}>
                  {c.value}
                  {c.ms !== undefined && (
                    <Typography component="span" variant="caption" color="text.secondary">
                      {` · ${c.ms} ms`}
                    </Typography>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
        <Card sx={{ p: 5 }}>
          <Typography variant="h3" sx={{ mb: 2 }}>
            {t("links")}
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
            {links.map(([label, href]) => (
              <Link key={href} href={href} target="_blank" rel="noreferrer" underline="hover">
                {label} ↗
              </Link>
            ))}
          </Box>
        </Card>
        <Card sx={{ p: 5 }}>
          <Typography variant="h3" sx={{ mb: 2 }}>
            {t("versions")}
          </Typography>
          <Table size="small">
            <TableBody>
              {versions.map(([k, v]) => (
                <TableRow key={k}>
                  <TableCell>{k}</TableCell>
                  <TableCell align="right">
                    <code>{v}</code>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
        <Card sx={{ p: 5 }}>
          <Typography variant="h3" sx={{ mb: 2 }}>
            {t("commands")}
          </Typography>
          {commands.map(([cmd, note]) => (
            <Box key={cmd} sx={{ py: 1 }}>
              <Box component="code" sx={{ display: "block", fontSize: 13, bgcolor: "grey.100", px: 2, py: 1, borderRadius: 1 }}>
                {cmd}
              </Box>
              <Typography variant="caption" color="text.secondary">
                {note}
              </Typography>
            </Box>
          ))}
        </Card>
      </Box>
    </Box>
  );
}
