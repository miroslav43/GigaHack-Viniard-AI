"use server";

// Field tasks. Writes go through the caller's session, so RLS + the task_guard trigger decide:
// the UAT admin creates/assigns/deletes in its municipality; an inspector only moves its own tasks forward.
import { revalidatePath } from "next/cache";
import { createClient } from "@/lib/supabase/server";
import { getViewer } from "@/lib/viewer";
import { readTargets, taskTitle, type TaskStatus } from "@/lib/tasks";
import { requireUatAdmin } from "@/lib/team";

export type ActionResult<T = null> = { ok: true; data: T } | { ok: false; error: string };

const STATUSES: TaskStatus[] = ["open", "in_progress", "done", "cancelled"];
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

const run = async <T,>(fn: () => Promise<T>): Promise<ActionResult<T>> => {
  try {
    const data = await fn();
    revalidatePath("/[locale]/sarcini", "page");
    revalidatePath("/[locale]/echipa", "page");
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
};

/** Resolves an assignee id to its email after checking it belongs to the admin's municipality. */
async function checkAssignee(assignee: string | null) {
  if (!assignee) return { assignee: null, assignee_email: null };
  const { admin, uatKey } = await requireUatAdmin();
  const { data } = await admin.auth.admin.getUserById(assignee);
  const u = data.user;
  if (!u || u.app_metadata?.uat !== uatKey || !["inspector", "uat_admin"].includes(u.app_metadata?.uat_role)) throw new Error("invalid_assignee");
  return { assignee: u.id, assignee_email: u.email ?? null };
}

/** Creates one task per inspection target (skipping targets that already have one). */
export async function createTasksFromTargets(input: {
  targetIds: string[];
  assignee: string | null;
  dueDate: string | null;
  priority: number;
}): Promise<ActionResult<number>> {
  return run(async () => {
    const viewer = await getViewer();
    if (viewer.kind !== "user" || viewer.role !== "uat_admin" || !viewer.uat) throw new Error("forbidden");
    if (input.dueDate && !DATE_RE.test(input.dueDate)) throw new Error("invalid_date");
    const priority = [1, 2, 3].includes(input.priority) ? input.priority : 2;
    const who = await checkAssignee(input.assignee);
    const wanted = new Set(input.targetIds);
    const targets = (await readTargets(viewer.uat.surveys)).filter((t) => wanted.has(`${t.survey_id}/${t.target_id}`));
    if (targets.length === 0) return 0;
    const supabase = await createClient();
    const rows = targets.map((t) => ({
      uat_key: viewer.uat!.key,
      survey_id: t.survey_id,
      target_id: t.target_id,
      kind: t.kind,
      title: taskTitle(t),
      vineyard_id: t.vineyard_id,
      row_id: t.row_id,
      gap_length_m: t.gap_length_m,
      location: `SRID=4326;POINT(${t.lon} ${t.lat})`,
      priority,
      due_date: input.dueDate,
      ...who,
    }));
    const { data, error } = await supabase
      .from("task")
      .upsert(rows, { onConflict: "uat_key,survey_id,target_id", ignoreDuplicates: true })
      .select("id");
    if (error) throw new Error(error.message);
    return data?.length ?? 0;
  });
}

export async function assignTasks(input: { ids: number[]; assignee: string | null; dueDate: string | null; priority: number | null }): Promise<ActionResult> {
  return run(async () => {
    const viewer = await getViewer();
    if (viewer.kind !== "user" || viewer.role !== "uat_admin") throw new Error("forbidden");
    if (input.dueDate && !DATE_RE.test(input.dueDate)) throw new Error("invalid_date");
    const who = await checkAssignee(input.assignee);
    const patch: Record<string, unknown> = { ...who, due_date: input.dueDate };
    if (input.priority && [1, 2, 3].includes(input.priority)) patch.priority = input.priority;
    const supabase = await createClient();
    const { error } = await supabase.from("task").update(patch).in("id", input.ids);
    if (error) throw new Error(error.message);
    return null;
  });
}

export async function setTaskStatus(input: { id: number; status: TaskStatus; note: string }): Promise<ActionResult> {
  return run(async () => {
    const viewer = await getViewer();
    if (viewer.kind !== "user") throw new Error("forbidden");
    if (!STATUSES.includes(input.status)) throw new Error("invalid_status");
    const supabase = await createClient();
    const { data, error } = await supabase
      .from("task")
      .update({ status: input.status, resolution_note: input.note.trim() || null })
      .eq("id", input.id)
      .select("id");
    if (error) throw new Error(error.code === "42501" ? "forbidden" : error.message);
    if (!data?.length) throw new Error("forbidden");
    return null;
  });
}

export async function deleteTasks(ids: number[]): Promise<ActionResult<number>> {
  return run(async () => {
    const viewer = await getViewer();
    if (viewer.kind !== "user" || viewer.role !== "uat_admin") throw new Error("forbidden");
    const supabase = await createClient();
    const { data, error } = await supabase.from("task").delete().in("id", ids).select("id");
    if (error) throw new Error(error.message);
    return data?.length ?? 0;
  });
}
