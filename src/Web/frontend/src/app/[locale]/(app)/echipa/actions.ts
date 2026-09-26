"use server";

// Team management for the municipality (UAT) admin — strictly limited to its own municipality and to
// team roles (inspector, viewer). Other admins and other municipalities are never reachable.
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { getLocale } from "next-intl/server";
import type { AuthError } from "@supabase/supabase-js";
import { routing } from "@/i18n/routing";
import { auditAsUatAdmin, requireTeamMember, requireUatAdmin, TEAM_ROLES, type TeamRole } from "@/lib/team";

export type ActionResult<T = null> = { ok: true; data: T } | { ok: false; error: string };

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const run = async <T,>(fn: () => Promise<T>): Promise<ActionResult<T>> => {
  try {
    const data = await fn();
    revalidatePath("/[locale]/echipa", "page");
    revalidatePath("/[locale]/sarcini", "page");
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
};

const checkRole = (role: string): TeamRole => {
  if (!TEAM_ROLES.includes(role as TeamRole)) throw new Error("invalid_role");
  return role as TeamRole;
};

/** Absolute URL of the set-password page, on the host the admin is using (unless SITE_URL pins it).
 *  Supabase only honours it if it is in Auth → URL Configuration → Redirect URLs; otherwise it falls back to the Site URL. */
async function setPasswordUrl() {
  const locale = await getLocale();
  const path = `${locale === routing.defaultLocale ? "" : `/${locale}`}/parola-noua`;
  if (process.env.SITE_URL) return `${process.env.SITE_URL.replace(/\/$/, "")}${path}`;
  const h = await headers();
  const host = h.get("x-forwarded-host") ?? h.get("host") ?? "localhost:3000";
  const proto = h.get("x-forwarded-proto") ?? (/^(localhost|127\.)/.test(host) ? "http" : "https");
  return `${proto}://${host}${path}`;
}

const inviteError = (e: AuthError) =>
  e.code === "email_exists" || e.code === "user_already_exists"
    ? "email_exists"
    : e.code === "over_email_send_rate_limit"
      ? "email_rate_limit"
      : e.code === "email_address_not_authorized"
        ? "email_not_authorized"
        : e.message;

/** Invites a member by email: Supabase sends the link, the member sets the password on /parola-noua. */
export async function inviteMember(input: { email: string; fullName: string; role: string }): Promise<ActionResult<string>> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const email = input.email.trim().toLowerCase();
    if (!EMAIL_RE.test(email)) throw new Error("invalid_email");
    const role = checkRole(input.role);
    const { data, error } = await admin.auth.admin.inviteUserByEmail(email, {
      data: { full_name: input.fullName.trim() || undefined },
      redirectTo: await setPasswordUrl(),
    });
    if (error) throw new Error(inviteError(error));
    // the invite API only takes user_metadata; municipality and role are app_metadata (server-set)
    const { error: metaError } = await admin.auth.admin.updateUserById(data.user.id, { app_metadata: { uat: uatKey, uat_role: role } });
    if (metaError) {
      await admin.auth.admin.deleteUser(data.user.id);
      throw new Error(metaError.message);
    }
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, "team.invite", data.user.id, { email, role });
    return email;
  });
}

/** Sends the invitation again (links expire) to a member who has not set a password yet. */
export async function resendInvite(id: string): Promise<ActionResult<string>> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const member = await requireTeamMember(admin, id, uatKey);
    if (member.email_confirmed_at || member.last_sign_in_at) throw new Error("already_active");
    const { error } = await admin.auth.admin.inviteUserByEmail(member.email!, { redirectTo: await setPasswordUrl() });
    if (error) throw new Error(inviteError(error));
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, "team.resend_invite", id, { email: member.email });
    return member.email!;
  });
}

export async function updateMember(input: { id: string; fullName: string; role: string }): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const member = await requireTeamMember(admin, input.id, uatKey);
    const role = checkRole(input.role);
    const { error } = await admin.auth.admin.updateUserById(input.id, {
      user_metadata: { full_name: input.fullName.trim() || null },
      // the municipality never changes here — only the team role
      app_metadata: { uat: uatKey, uat_role: role },
    });
    if (error) throw new Error(error.message);
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, "team.update", input.id, { email: member.email, role });
    return null;
  });
}

export async function resetMemberPassword(id: string, password: string): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const member = await requireTeamMember(admin, id, uatKey);
    if (password.length < 10) throw new Error("weak_password");
    const { error } = await admin.auth.admin.updateUserById(id, { password });
    if (error) throw new Error(error.message);
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, "team.reset_password", id, { email: member.email });
    return null;
  });
}

export async function setMemberBanned(id: string, banned: boolean): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const member = await requireTeamMember(admin, id, uatKey);
    const { error } = await admin.auth.admin.updateUserById(id, { ban_duration: banned ? "876000h" : "none" });
    if (error) throw new Error(error.message);
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, banned ? "team.disable" : "team.enable", id, { email: member.email });
    return null;
  });
}

export async function deleteMember(id: string): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const member = await requireTeamMember(admin, id, uatKey);
    // ban first so no refresh can happen; tokens already issued expire within the hour
    await admin.auth.admin.updateUserById(id, { ban_duration: "876000h" });
    const { error } = await admin.auth.admin.deleteUser(id);
    if (error) throw new Error(error.message);
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, "team.delete", id, { email: member.email });
    return null;
  });
}
