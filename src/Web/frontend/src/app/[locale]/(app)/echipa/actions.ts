"use server";

// Team management for the municipality (UAT) admin — strictly limited to its own municipality and to
// team roles (inspector, viewer). Other admins and other municipalities are never reachable.
import { revalidatePath } from "next/cache";
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

export async function createMember(input: { email: string; fullName: string; password: string; role: string }): Promise<ActionResult<string>> {
  return run(async () => {
    const { admin, viewer, uatKey } = await requireUatAdmin();
    const email = input.email.trim().toLowerCase();
    if (!EMAIL_RE.test(email)) throw new Error("invalid_email");
    if (input.password.length < 10) throw new Error("weak_password");
    const role = checkRole(input.role);
    const { data, error } = await admin.auth.admin.createUser({
      email,
      password: input.password,
      email_confirm: true,
      user_metadata: { full_name: input.fullName.trim() || undefined },
      app_metadata: { uat: uatKey, uat_role: role },
    });
    if (error) throw new Error(error.code === "email_exists" ? "email_exists" : error.message);
    await auditAsUatAdmin(admin, { id: viewer.userId!, email: viewer.email }, uatKey, "team.create", data.user.id, { email, role });
    return data.user.id;
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
