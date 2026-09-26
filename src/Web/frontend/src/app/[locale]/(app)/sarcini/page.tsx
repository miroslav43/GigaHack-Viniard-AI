import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Alert from "@mui/material/Alert";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { AccessDenied } from "@/components/common/AccessDenied";
import { TasksPanel, type Assignable, type TaskRole } from "@/components/tasks/TasksPanel";
import { createClient } from "@/lib/supabase/server";
import { ADMIN_API_ENABLED, createAdminClient } from "@/lib/supabase/admin";
import { readTargets, type TaskRow } from "@/lib/tasks";
import { listUatUsers } from "@/lib/team";
import { getViewer } from "@/lib/viewer";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: PageProps<"/[locale]/sarcini">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("tasks.title")} · Solemtrix` };
}

export default async function TasksPage({ params }: PageProps<"/[locale]/sarcini">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("tasks");
  const viewer = await getViewer();

  if (viewer.kind !== "user") {
    return (
      <Page>
        <PageHeader title={t("title")} />
        <Alert severity="info">{t("demo")}</Alert>
      </Page>
    );
  }
  const role = viewer.role as TaskRole;
  if (!viewer.uat || !["uat_admin", "inspector", "viewer"].includes(role)) return <AccessDenied />;

  const supabase = await createClient();
  // RLS: only this municipality's tasks come back
  const { data } = await supabase.from("task_public").select("*").order("status").order("priority").order("created_at", { ascending: false });
  const tasks = (data ?? []) as TaskRow[];

  let targets: Awaited<ReturnType<typeof readTargets>> = [];
  let members: Assignable[] = [];
  if (role === "uat_admin") {
    const taken = new Set(tasks.map((x) => `${x.survey_id}/${x.target_id}`));
    targets = (await readTargets(viewer.uat.surveys)).filter((x) => !taken.has(`${x.survey_id}/${x.target_id}`)).sort((a, b) => (a.route_order ?? 0) - (b.route_order ?? 0));
    if (ADMIN_API_ENABLED) {
      const users = await listUatUsers(createAdminClient(), viewer.uat.key);
      members = users
        .filter((u) => ["inspector", "uat_admin"].includes(u.app_metadata?.uat_role) && !(u.banned_until && new Date(u.banned_until) > new Date()))
        .map((u) => ({ id: u.id, email: u.email ?? "", name: (u.user_metadata?.full_name as string | undefined) ?? null }))
        .sort((a, b) => a.email.localeCompare(b.email));
    }
  }

  return (
    <Page>
      <PageHeader title={t("title")} subtitle={role === "uat_admin" ? t("subtitleAdmin") : t("subtitleMember")} />
      <TasksPanel role={role} meId={viewer.userId!} tasks={tasks} targets={targets} members={members} adminApi={ADMIN_API_ENABLED} />
    </Page>
  );
}
