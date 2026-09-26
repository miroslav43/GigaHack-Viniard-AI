import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Alert from "@mui/material/Alert";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { AccessDenied } from "@/components/common/AccessDenied";
import { TeamPanel, type TeamRow } from "@/components/team/TeamPanel";
import { createClient } from "@/lib/supabase/server";
import { ADMIN_API_ENABLED, createAdminClient } from "@/lib/supabase/admin";
import { listUatUsers, toMember } from "@/lib/team";
import type { TaskRow } from "@/lib/tasks";
import { getViewer } from "@/lib/viewer";

export const dynamic = "force-dynamic";

/** Tasks completed in the last `days` days (request time; the page is dynamic). */
function doneWithinDays(tasks: Pick<TaskRow, "status" | "completed_at">[], days: number) {
  const since = Date.now() - days * 24 * 3600 * 1000;
  return tasks.filter((x) => x.status === "done" && x.completed_at && new Date(x.completed_at).getTime() > since).length;
}

export async function generateMetadata({ params }: PageProps<"/[locale]/echipa">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("team.title")} · Solemtrix`, robots: { index: false } };
}

export default async function TeamPage({ params }: PageProps<"/[locale]/echipa">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const viewer = await getViewer();
  // proxy.ts already answers 403 for anyone but a municipality admin; defense in depth
  if (viewer.kind !== "user" || viewer.role !== "uat_admin" || !viewer.uat) return <AccessDenied />;

  const t = await getTranslations();
  const uatName = t("uat.townHall", { name: viewer.uat.name });
  const header = <PageHeader title={t("team.title")} subtitle={t("team.subtitle", { uat: uatName })} />;

  if (!ADMIN_API_ENABLED) {
    return (
      <Page>
        {header}
        <Alert severity="warning">{t("team.noKey")}</Alert>
      </Page>
    );
  }

  const supabase = await createClient();
  const [users, { data: taskData }] = await Promise.all([
    listUatUsers(createAdminClient(), viewer.uat.key),
    supabase.from("task_public").select("assignee, status, completed_at"),
  ]);
  const tasks = (taskData ?? []) as Pick<TaskRow, "assignee" | "status" | "completed_at">[];

  const rows: TeamRow[] = users
    .map(toMember)
    .map((m) => {
      const mine = tasks.filter((x) => x.assignee === m.id);
      return {
        ...m,
        isMe: m.id === viewer.userId,
        tasks: {
          open: mine.filter((x) => x.status === "open").length,
          in_progress: mine.filter((x) => x.status === "in_progress").length,
          done: mine.filter((x) => x.status === "done").length,
        },
      };
    })
    // the admin first, then inspectors, then viewers
    .sort((a, b) => Number(b.isMe) - Number(a.isMe) || (a.role ?? "").localeCompare(b.role ?? "") || a.email.localeCompare(b.email));

  return (
    <Page>
      {header}
      <TeamPanel
        uatName={uatName}
        rows={rows}
        openTotal={tasks.filter((x) => x.status === "open" || x.status === "in_progress").length}
        doneWeek={doneWithinDays(tasks, 7)}
      />
    </Page>
  );
}
