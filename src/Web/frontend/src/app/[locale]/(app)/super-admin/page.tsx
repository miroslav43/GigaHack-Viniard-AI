import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { AccessDenied } from "@/components/common/AccessDenied";
import { AdminTabs } from "@/components/admin/common";
import { UatPanel } from "@/components/admin/UatPanel";
import { UsersPanel } from "@/components/admin/UsersPanel";
import { SurveysPanel } from "@/components/admin/SurveysPanel";
import { SystemPanel } from "@/components/admin/SystemPanel";
import { AuditPanel, type AuditRow } from "@/components/admin/AuditPanel";
import { ADMIN_TABS, type AdminSurvey, type AdminTab, type AdminUat, type AdminUser, type LocalBundle, type Role } from "@/components/admin/types";
import { createClient } from "@/lib/supabase/server";
import { ADMIN_API_ENABLED, createAdminClient } from "@/lib/supabase/admin";
import type { UatRow } from "@/lib/uats";
import { getViewer, isPlatformAdmin } from "@/lib/viewer";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: PageProps<"/[locale]/super-admin">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("superAdmin.title")} · Solemtrix`, robots: { index: false } };
}

type UatSurveyLink = { uat_key: string; survey_id: string; survey_name: string; overlap_ha: number };

async function listAuthUsers() {
  if (!ADMIN_API_ENABLED) return null;
  const { data, error } = await createAdminClient().auth.admin.listUsers({ perPage: 1000 });
  if (error) throw new Error(error.message);
  return data.users;
}

export default async function SuperAdminPage({ params, searchParams }: PageProps<"/[locale]/super-admin">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  // proxy.ts already answers 403 for anyone else; this check is defense in depth
  if (!isPlatformAdmin(viewer)) return <AccessDenied />;

  const t = await getTranslations("superAdmin");
  const raw = (await searchParams).tab;
  const tab: AdminTab = ADMIN_TABS.includes(raw as AdminTab) ? (raw as AdminTab) : "uat";
  const supabase = await createClient();

  let panel: React.ReactNode = null;

  if (tab === "uat") {
    const [{ data: rows }, { data: links }, users] = await Promise.all([
      supabase.from("uat_public").select("*").order("name"),
      supabase.from("uat_survey").select("uat_key, survey_id, overlap_ha"),
      listAuthUsers(),
    ]);
    const uats: AdminUat[] = ((rows ?? []) as UatRow[]).map((r) => ({
      ...r,
      area_ha: Number(r.area_ha),
      surveys: ((links ?? []) as UatSurveyLink[]).filter((l) => l.uat_key === r.key).map((l) => ({ id: l.survey_id, overlap_ha: Number(l.overlap_ha) })),
      users: users ? users.filter((u) => u.app_metadata?.uat === r.key).length : null,
    }));
    panel = <UatPanel uats={uats} />;
  }

  if (tab === "users") {
    const [users, { data: uatRows }] = await Promise.all([listAuthUsers(), supabase.from("uat").select("key, name").order("name")]);
    const list: AdminUser[] = (users ?? [])
      .map((u) => ({
        id: u.id,
        email: u.email ?? "",
        name: (u.user_metadata?.full_name as string | undefined) ?? null,
        uat: (u.app_metadata?.uat as string | undefined) ?? null,
        role: (u.app_metadata?.uat_role as Role | undefined) ?? null,
        created_at: u.created_at,
        last_sign_in_at: u.last_sign_in_at ?? null,
        banned: Boolean(u.banned_until && new Date(u.banned_until) > new Date()),
      }))
      .sort((a, b) => a.email.localeCompare(b.email));
    panel = <UsersPanel users={list} uats={uatRows ?? []} adminApi={ADMIN_API_ENABLED} meId={viewer.userId} />;
  }

  if (tab === "surveys") {
    const [{ data: rows }, { data: links }, { data: uatRows }] = await Promise.all([
      supabase.from("survey_public").select("id, name, captured_at, area_ha, data_path").order("id"),
      supabase.from("uat_survey").select("uat_key, survey_id, overlap_ha"),
      supabase.from("uat").select("key, name"),
    ]);
    const names = new Map((uatRows ?? []).map((u) => [u.key as string, u.name as string]));
    const surveys: AdminSurvey[] = (rows ?? []).map((s) => ({
      ...(s as Omit<AdminSurvey, "uats">),
      area_ha: Number(s.area_ha),
      uats: ((links ?? []) as UatSurveyLink[])
        .filter((l) => l.survey_id === s.id)
        .map((l) => ({ key: l.uat_key, name: names.get(l.uat_key) ?? l.uat_key, overlap_ha: Number(l.overlap_ha) })),
    }));
    const dataDir = path.join(process.cwd(), "public", "data");
    const dirs = (await readdir(dataDir, { withFileTypes: true }).catch(() => [])).filter((d) => d.isDirectory() && d.name !== "ref");
    const bundles: LocalBundle[] = [];
    for (const d of dirs) {
      const summary = await readFile(path.join(dataDir, d.name, "summary.json"), "utf8").catch(() => null);
      if (summary) bundles.push({ id: d.name, name: JSON.parse(summary).survey.name, registered: surveys.some((s) => s.id === d.name) });
    }
    panel = <SurveysPanel surveys={surveys} bundles={bundles} />;
  }

  if (tab === "system") {
    const { data } = await supabase.auth.getClaims();
    panel = <SystemPanel viewer={viewer} jwtExp={(data?.claims.exp as number | undefined) ?? null} />;
  }

  if (tab === "audit") {
    const { data } = await supabase.from("admin_audit_log").select("*").order("at", { ascending: false }).limit(200);
    panel = <AuditPanel rows={(data ?? []) as AuditRow[]} />;
  }

  return (
    <Page>
      <PageHeader title={t("title")} subtitle={t("subtitle")} />
      <AdminTabs current={tab} />
      {panel}
    </Page>
  );
}
