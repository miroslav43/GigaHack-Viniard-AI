// Municipality (UAT) team: the UAT admin manages inspectors and viewers of its own municipality only.
// Uses the Admin API (secret key, server-only); the caller's role and municipality are re-read from Auth.
import "server-only";
import type { User } from "@supabase/supabase-js";
import { ADMIN_API_ENABLED, createAdminClient } from "./supabase/admin";
import { getViewer } from "./viewer";

export const TEAM_ROLES = ["inspector", "viewer"] as const;
export type TeamRole = (typeof TEAM_ROLES)[number];

export interface TeamMember {
  id: string;
  email: string;
  name: string | null;
  role: string | null;
  banned: boolean;
  /** invited by email and has not set a password yet */
  invited: boolean;
  created_at: string;
  last_sign_in_at: string | null;
}

export const toMember = (u: User): TeamMember => ({
  id: u.id,
  email: u.email ?? "",
  name: (u.user_metadata?.full_name as string | undefined) ?? null,
  role: (u.app_metadata?.uat_role as string | undefined) ?? null,
  banned: Boolean(u.banned_until && new Date(u.banned_until) > new Date()),
  invited: Boolean(u.invited_at && !u.email_confirmed_at),
  created_at: u.created_at,
  last_sign_in_at: u.last_sign_in_at ?? null,
});

/** The signed-in UAT admin, confirmed from Auth (not only from the JWT). Throws "forbidden" otherwise. */
export async function requireUatAdmin() {
  const viewer = await getViewer();
  if (viewer.kind !== "user" || viewer.role !== "uat_admin" || !viewer.uat || !viewer.userId) throw new Error("forbidden");
  if (!ADMIN_API_ENABLED) throw new Error("secret_key_missing");
  const admin = createAdminClient();
  const { data, error } = await admin.auth.admin.getUserById(viewer.userId);
  const meta = data.user?.app_metadata;
  if (error || meta?.uat_role !== "uat_admin" || meta?.uat !== viewer.uat.key) throw new Error("forbidden");
  return { viewer, admin, uatKey: viewer.uat.key };
}

/** All accounts of one municipality (the Admin API has no server-side filter on app_metadata). */
export async function listUatUsers(admin: ReturnType<typeof createAdminClient>, uatKey: string) {
  const users: User[] = [];
  for (let page = 1; page < 50; page++) {
    const { data, error } = await admin.auth.admin.listUsers({ page, perPage: 1000 });
    if (error) throw new Error(error.message);
    users.push(...data.users.filter((u) => u.app_metadata?.uat === uatKey));
    if (data.users.length < 1000) break;
  }
  return users;
}

/** A member the UAT admin may manage: same municipality, team role (never another admin). */
export async function requireTeamMember(admin: ReturnType<typeof createAdminClient>, id: string, uatKey: string) {
  const { data, error } = await admin.auth.admin.getUserById(id);
  const u = data.user;
  if (error || !u || u.app_metadata?.uat !== uatKey || !TEAM_ROLES.includes(u.app_metadata?.uat_role)) throw new Error("forbidden");
  return u;
}

/** Audit entry written with the secret key (the audit table is insertable only by platform admins via RLS). */
export async function auditAsUatAdmin(
  admin: ReturnType<typeof createAdminClient>,
  actor: { id: string; email: string | null },
  uatKey: string,
  action: string,
  entityId: string | null,
  details: Record<string, unknown> = {},
) {
  await admin
    .from("admin_audit_log")
    .insert({ actor: actor.id, actor_email: actor.email, action, entity: "user", entity_id: entityId, details: { uat: uatKey, ...details } });
}
